"""
FastAPI + LangGraph Agent with Remote MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler
MongoDB-backed per-user conversation memory (last 5 message pairs)
Auto Deploy enabled using deploy.yml file
"""

import os
import httpx
import asyncio
import uuid
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as PydanticBaseModel, Field, create_model

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

# ============================================================
# Environment
# ============================================================

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "agrigpt-backend-agent"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "https://newapi.alumnx.com/agrigpt/mcp/")
MCP_API_KEY = os.getenv("MCP_API_KEY")
MCP_TIMEOUT = 30.0

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "conversations")

MAX_PAIRS = 5  # sliding window: last 5 human+AI pairs = 10 messages max

# WHATSAPP: Uncomment and populate when Meta credentials are ready
# WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")   # your custom verify token set in Meta dashboard
# WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")   # permanent or temporary access token from Meta
# WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")  # from Meta Business > WhatsApp > API Setup


# ============================================================
# MongoDB Setup
# ============================================================

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB]
conversations: Collection = db[MONGODB_COLLECTION]

# Ensure phone_number is indexed as primary lookup key
conversations.create_index([("phone_number", ASCENDING)], unique=True)


# ============================================================
# MongoDB Memory Helpers
# ============================================================

def load_history(phone_number: str) -> list:
    """
    Load stored messages for a user and reconstruct LangChain message objects.
    Returns up to MAX_PAIRS * 2 messages (soft window — never cuts mid-pair).
    """
    doc = conversations.find_one({"phone_number": phone_number})
    if not doc or "messages" not in doc:
        return []

    raw_messages = doc["messages"]
    reconstructed = []
    for m in raw_messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "human":
            reconstructed.append(HumanMessage(content=content))
        elif role == "ai":
            reconstructed.append(AIMessage(content=content))
        elif role == "system":
            reconstructed.append(SystemMessage(content=content))

    return reconstructed


def save_history(phone_number: str, messages: list):
    """
    Persist conversation history to MongoDB.
    Only stores HumanMessage and AIMessage (strips tool calls, tool results).
    Enforces soft sliding window of last MAX_PAIRS pairs.
    """
    # Filter to only human and final AI messages
    storable = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            storable.append({"role": "human", "content": msg.content if isinstance(msg.content, str) else str(msg.content)})
        elif isinstance(msg, AIMessage):
            # Skip intermediate AI messages that only contain tool_calls with no text content
            content = msg.content
            if isinstance(content, str) and content.strip():
                storable.append({"role": "ai", "content": content})
            elif isinstance(content, list):
                # Extract text blocks only
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                joined = " ".join(t for t in text_parts if t.strip())
                if joined.strip():
                    storable.append({"role": "ai", "content": joined})

    # Pair-aware sliding window:
    # Walk backwards to collect complete pairs without breaking a pair
    pairs_collected = 0
    cutoff_index = 0
    i = len(storable) - 1
    while i >= 0 and pairs_collected < MAX_PAIRS:
        if storable[i]["role"] == "ai":
            # Look for the preceding human message
            if i > 0 and storable[i - 1]["role"] == "human":
                pairs_collected += 1
                cutoff_index = i - 1
                i -= 2
            else:
                i -= 1
        else:
            i -= 1

    window = storable[cutoff_index:] if pairs_collected > 0 else storable

    conversations.update_one(
        {"phone_number": phone_number},
        {"$set": {"messages": window}},
        upsert=True
    )


# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        self.client = httpx.Client(timeout=MCP_TIMEOUT)

    def list_tools(self) -> List[Dict[str, Any]]:
        print(f"Calling MCP server: {self.base_url}/getToolsList")
        response = self.client.post(
            f"{self.base_url}/getToolsList",
            headers=self.headers,
            json={}
        )
        response.raise_for_status()
        tools = response.json().get("tools", [])
        print(f"Received {len(tools)} tools from MCP server")
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        print(f"Calling MCP tool: {name} with args: {arguments}")
        response = self.client.post(
            f"{self.base_url}/callTool",
            headers=self.headers,
            json={"name": name, "arguments": arguments}
        )
        response.raise_for_status()
        result = response.json().get("result")
        print(f"MCP tool result: {result}")
        return result


# ============================================================
# State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# Agent Builder
# ============================================================

def build_agent():
    mcp_client = MCPClient(MCP_BASE_URL, MCP_API_KEY)

    print("Fetching tools from remote MCP...")
    remote_tools = mcp_client.list_tools()

    if not remote_tools:
        raise RuntimeError("No tools found on remote MCP server.")

    print(f"Loaded {len(remote_tools)} tools: {[t['name'] for t in remote_tools]}")

    dynamic_tools = []

    for tool_schema in remote_tools:
        tool_name = tool_schema["name"]
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})

        def create_tool(name: str, desc: str, schema: Dict[str, Any]):
            properties = schema.get("properties", {})
            field_definitions = {}

            for prop_name, prop_details in properties.items():
                prop_type = prop_details.get("type", "string")
                type_mapping = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }
                py_type = type_mapping.get(prop_type, str)
                description_text = prop_details.get("description", "")
                required = prop_name in schema.get("required", [])

                if required:
                    field_definitions[prop_name] = (py_type, Field(..., description=description_text))
                else:
                    default_val = prop_details.get("default", None)
                    field_definitions[prop_name] = (py_type, Field(default=default_val, description=description_text))

            ArgsSchema = create_model(f"{name}_args", **field_definitions)

            def remote_tool_func(**kwargs) -> str:
                cleaned = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    result = mcp_client.call_tool(name, cleaned)
                    return str(result)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"Remote MCP error: {str(e)}"

            return StructuredTool.from_function(
                func=remote_tool_func,
                name=name,
                description=desc,
                args_schema=ArgsSchema
            )

        dynamic_tools.append(create_tool(tool_name, description, input_schema))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    llm_with_tools = llm.bind_tools(dynamic_tools)

    def agent_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(dynamic_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================
# Startup
# ============================================================

print("\nBUILDING AGENT AT STARTUP...")
app_agent = build_agent()
print("AGENT BUILD COMPLETE\n")


# ============================================================
# Core Agent Invocation (shared by all entry points)
# ============================================================

def extract_final_answer(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                return msg.content
            elif isinstance(msg.content, list) and msg.content:
                block = msg.content[0]
                if isinstance(block, dict) and block.get("text", "").strip():
                    return block["text"]
                elif str(block).strip():
                    return str(block)
    return "No response generated."


def run_agent(phone_number: str, user_message: str) -> str:
    """
    Load memory, invoke agent, save updated memory, return final answer.
    This is the single entry point for agent execution regardless of channel.
    """
    history = load_history(phone_number)
    history.append(HumanMessage(content=user_message))

    result = app_agent.invoke({"messages": history})

    final_answer = extract_final_answer(result)

    save_history(phone_number, result["messages"])

    return final_answer


# ============================================================
# WhatsApp Sender (WHATSAPP: uncomment when credentials ready)
# ============================================================

# WHATSAPP: Uncomment this entire function when Meta credentials are ready
# async def send_whatsapp_message(to_phone: str, message: str):
#     """Send a text reply back to the user via WhatsApp Cloud API."""
#     url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
#     headers = {
#         "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "messaging_product": "whatsapp",
#         "to": to_phone,
#         "type": "text",
#         "text": {"body": message}
#     }
#     async with httpx.AsyncClient(timeout=10.0) as client:
#         response = await client.post(url, headers=headers, json=payload)
#         if response.status_code != 200:
#             print(f"Failed to send WhatsApp message: {response.text}")


# ============================================================
# Background Task: Process WhatsApp message and reply
# ============================================================

async def process_and_reply(phone_number: str, user_message: str):
    """
    Runs in background after 200 OK is returned to WhatsApp webhook.
    Invokes agent synchronously in a thread pool to avoid blocking event loop.
    Then sends reply via WhatsApp API.
    """
    try:
        loop = asyncio.get_event_loop()
        final_answer = await loop.run_in_executor(
            None, run_agent, phone_number, user_message
        )

        print(f"Agent reply for {phone_number}: {final_answer[:100]}")

        # WHATSAPP: Uncomment the line below when Meta credentials are ready
        # await send_whatsapp_message(phone_number, final_answer)

        # LOCAL TEST: Remove this block when going live with WhatsApp
        print(f"[LOCAL MODE] Reply not sent via WhatsApp. Answer: {final_answer}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing message for {phone_number}: {e}")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="AgriGPT WhatsApp Agent")


# ============================================================
# WhatsApp Webhook Verification (GET)
# Meta calls this once when you register the webhook in the dashboard.
# ============================================================

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    # WHATSAPP: Replace the hardcoded token check with env var when going live:
    # if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:

    # LOCAL TEST: Hardcoded verify token for local testing
    LOCAL_VERIFY_TOKEN = "test_verify_token_123"

    if hub_mode == "subscribe" and hub_verify_token == LOCAL_VERIFY_TOKEN:
        print(f"Webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Webhook verification failed.")


# ============================================================
# WhatsApp Webhook Handler (POST)
# Meta sends all incoming messages here.
# ============================================================

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives WhatsApp webhook events.
    Immediately returns 200 OK, processes message in background.
    """
    payload = await request.json()
    print(f"Incoming webhook payload: {payload}")

    try:
        # WHATSAPP: This parsing is correct for Meta Cloud API payload structure.
        # It works as-is for real WhatsApp webhooks.
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            # Could be a status update (delivered, read) — ignore
            return {"status": "ok"}

        message = messages[0]
        msg_type = message.get("type")

        if msg_type != "text":
            # Non-text messages (image, audio, etc.) — not handled yet
            print(f"Ignoring non-text message type: {msg_type}")
            return {"status": "ok"}

        phone_number = message.get("from")         # e.g. "911234567890"
        user_message = message["text"].get("body", "").strip()

        if not phone_number or not user_message:
            return {"status": "ok"}

        print(f"Message from {phone_number}: {user_message}")

        # Fire and forget — respond 200 immediately, process in background
        background_tasks.add_task(process_and_reply, phone_number, user_message)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error parsing webhook: {e}")
        # Still return 200 to prevent WhatsApp from retrying
        return {"status": "ok"}

    return {"status": "ok"}


# ============================================================
# LOCAL TEST ENDPOINT — Remove or disable before going live
# Bypasses WhatsApp entirely. Calls agent directly and returns response.
# ============================================================

class TestChatRequest(BaseModel):
    phone_number: str   # simulate a WhatsApp user by phone number
    message: str

class TestChatResponse(BaseModel):
    phone_number: str
    response: str


@app.post("/test/chat", response_model=TestChatResponse)
def test_chat(request: TestChatRequest):
    """
    LOCAL TESTING ONLY.
    Simulates a WhatsApp message without needing Meta credentials.
    Shares the same agent + MongoDB memory as the real webhook handler.
    Remove this endpoint before deploying to production.
    """
    print(f"\n[TEST] phone={request.phone_number} | message={request.message}")

    try:
        final_answer = run_agent(request.phone_number, request.message)
        print(f"[TEST] Response: {final_answer[:100]}")
        return TestChatResponse(phone_number=request.phone_number, response=final_answer)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
    print("Done")
