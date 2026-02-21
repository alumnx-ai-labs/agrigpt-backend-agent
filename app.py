"""
FastAPI + LangGraph Agent with Remote MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

MongoDB Storage Schema (one document per chat session):
  {
    "chat_id":      "uuid-from-frontend",   ← unique index (primary key)
    "phone_number": "911234567890",          ← indexed, non-unique (1 user → N chats)
    "created_at":   ISODate,
    "updated_at":   ISODate,
    "messages": [
      { "role": "human", "content": "..." },
      { "role": "ai",    "content": "..." }
    ]                                        ← capped at MAX_MESSAGES (20)
  }

New Chat flow:
  - Frontend generates a new UUID on "New Chat" click and sends it as chat_id.
  - Backend finds no history for that chat_id → agent starts fresh.
  - MongoDB creates the document automatically on first save.
  - Same chat_id on subsequent messages → history is loaded and agent remembers.

Auto Deploy enabled using deploy.yml file
"""

import os
import httpx
import asyncio
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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
MCP_BASE_URL   = os.getenv("MCP_BASE_URL", "https://newapi.alumnx.com/agrigpt/mcp/")
MCP_API_KEY    = os.getenv("MCP_API_KEY")
MCP_TIMEOUT    = 30.0

MONGODB_URI        = os.getenv("MONGODB_URI")
MONGODB_DB         = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chats")  # fresh collection, new schema

# Max messages stored per chat_id (human + AI combined = 10 full turns).
# The LLM receives ALL stored messages as context on every invocation.
MAX_MESSAGES = 20

# WHATSAPP: Uncomment when Meta credentials are ready
# WHATSAPP_VERIFY_TOKEN    = os.getenv("WHATSAPP_VERIFY_TOKEN")
# WHATSAPP_ACCESS_TOKEN    = os.getenv("WHATSAPP_ACCESS_TOKEN")
# WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# ============================================================
# MongoDB Setup
# ============================================================
mongo_client   = MongoClient(MONGODB_URI)
db             = mongo_client[MONGODB_DB]
chat_sessions: Collection = db[MONGODB_COLLECTION]

# Indexes
# chat_id      → unique  (one document per conversation session)
# phone_number → non-unique (one user can have many chat sessions)
# updated_at   → for future TTL / cleanup
chat_sessions.create_index([("chat_id", ASCENDING)], unique=True)
chat_sessions.create_index([("phone_number", ASCENDING)])
chat_sessions.create_index([("updated_at", ASCENDING)])

print(f"Connected to MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")

# ============================================================
# MongoDB Memory Helpers
# ============================================================

def load_history(chat_id: str) -> list:
    """
    Load stored messages for a chat session and reconstruct LangChain
    message objects.

    Returns all stored messages (up to MAX_MESSAGES). The agent feeds
    ALL of them to the LLM so it can answer new questions with full
    awareness of the entire conversation history for that chat_id.

    If chat_id is new (no document exists) → returns empty list
    → agent starts a fresh conversation automatically.
    """
    doc = chat_sessions.find_one({"chat_id": chat_id})
    if not doc or "messages" not in doc:
        return []

    reconstructed = []
    for m in doc["messages"]:
        role    = m.get("role")
        content = m.get("content", "")
        if role == "human":
            reconstructed.append(HumanMessage(content=content))
        elif role == "ai":
            reconstructed.append(AIMessage(content=content))
        elif role == "system":
            reconstructed.append(SystemMessage(content=content))
    return reconstructed


def save_history(chat_id: str, messages: list, phone_number: str | None = None):
    """
    Persist updated conversation history to MongoDB under chat_id.

    Steps:
      1. Strip ToolMessages and tool-call-only AIMessages (not useful as LLM context).
      2. Apply pair-aware sliding window: keep the last MAX_MESSAGES messages,
         always ending on a complete human+AI pair.
      3. Upsert the document — creates it on first save (new chat),
         updates it on subsequent saves (continuing chat).

    phone_number is stored as a metadata field so you can later query
    all chat sessions for a specific user:
      db.chat_sessions_v2.find({ phone_number: "911234567890" })
    """
    # Step 1: Filter to storable human/ai messages only
    storable = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            storable.append({"role": "human", "content": content})

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                storable.append({"role": "ai", "content": content})
            elif isinstance(content, list):
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ]
                joined = " ".join(t for t in text_parts if t.strip())
                if joined.strip():
                    storable.append({"role": "ai", "content": joined})
        # ToolMessage and other internal types are intentionally skipped

    # Step 2: Pair-aware sliding window
    # Walk backwards collecting complete human+AI pairs until MAX_MESSAGES is reached.
    # This guarantees the stored history always ends on an AI reply (no dangling human turns).
    if len(storable) <= MAX_MESSAGES:
        window = storable
    else:
        pairs_to_collect = MAX_MESSAGES // 2
        pairs_collected  = 0
        cutoff_index     = 0
        i = len(storable) - 1

        while i >= 0 and pairs_collected < pairs_to_collect:
            if storable[i]["role"] == "ai" and i > 0 and storable[i - 1]["role"] == "human":
                pairs_collected += 1
                cutoff_index = i - 1
                i -= 2
            else:
                i -= 1

        window = storable[cutoff_index:] if pairs_collected > 0 else storable[-MAX_MESSAGES:]

    # Step 3: Upsert
    now = datetime.now(timezone.utc)
    update_fields: dict = {
        "messages":   window,
        "updated_at": now,
    }
    if phone_number:
        update_fields["phone_number"] = phone_number

    chat_sessions.update_one(
        {"chat_id": chat_id},
        {
            "$set":         update_fields,
            "$setOnInsert": {"created_at": now},  # only set on first insert
        },
        upsert=True
    )

# ============================================================
# MCP Client
# ============================================================
class MCPClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers  = {"Content-Type": "application/json"}
        self.client   = httpx.Client(timeout=MCP_TIMEOUT)

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
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ============================================================
# Agent Builder
# ============================================================
def build_agent():
    mcp_client   = MCPClient(MCP_BASE_URL, MCP_API_KEY)
    print("Fetching tools from remote MCP...")
    remote_tools = mcp_client.list_tools()
    if not remote_tools:
        raise RuntimeError("No tools found on remote MCP server.")
    print(f"Loaded {len(remote_tools)} tools: {[t['name'] for t in remote_tools]}")

    dynamic_tools = []
    for tool_schema in remote_tools:
        tool_name    = tool_schema["name"]
        description  = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})

        def create_tool(name: str, desc: str, schema: Dict[str, Any]):
            properties        = schema.get("properties", {})
            field_definitions = {}
            type_mapping = {
                "string": str, "integer": int, "number": float,
                "boolean": bool, "array": list, "object": dict
            }
            for prop_name, prop_details in properties.items():
                py_type          = type_mapping.get(prop_details.get("type", "string"), str)
                description_text = prop_details.get("description", "")
                required         = prop_name in schema.get("required", [])
                if required:
                    field_definitions[prop_name] = (py_type, Field(..., description=description_text))
                else:
                    field_definitions[prop_name] = (
                        py_type,
                        Field(default=prop_details.get("default", None), description=description_text)
                    )

            ArgsSchema = create_model(f"{name}_args", **field_definitions)

            def remote_tool_func(**kwargs) -> str:
                cleaned = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    return str(mcp_client.call_tool(name, cleaned))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    return f"Remote MCP error: {str(e)}"

            return StructuredTool.from_function(
                func=remote_tool_func,
                name=name,
                description=desc,
                args_schema=ArgsSchema
            )

        dynamic_tools.append(create_tool(tool_name, description, input_schema))

    llm            = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    llm_with_tools = llm.bind_tools(dynamic_tools)

    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

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
# Core Agent Invocation — shared by ALL channels
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


def run_agent(chat_id: str, user_message: str, phone_number: str | None = None) -> str:
    """
    Single entry point for agent execution across all channels (web, WhatsApp).

    Flow:
      1. Load history for chat_id from MongoDB.
         - New chat_id (from "New Chat" click) → empty history → fresh conversation.
         - Existing chat_id → full history → agent remembers previous context.
      2. Append the new human message.
      3. Invoke the LLM with the full message history as context.
      4. Save updated history back to MongoDB (trimmed to MAX_MESSAGES).
      5. Return the final text answer.

    Parameters
    ----------
    chat_id      : UUID from frontend. New UUID = new conversation. Same UUID = continued conversation.
    user_message : The user's new message text.
    phone_number : Stored as metadata on the MongoDB document (used to query all chats per user).
    """
    print(f"[run_agent] chat_id={chat_id} | phone={phone_number} | msg={user_message[:60]}")

    history = load_history(chat_id)
    print(f"[run_agent] Loaded {len(history)} messages from history.")

    history.append(HumanMessage(content=user_message))

    result       = app_agent.invoke({"messages": history})
    final_answer = extract_final_answer(result)

    save_history(chat_id, result["messages"], phone_number=phone_number)
    print(f"[run_agent] Saved history. Answer: {final_answer[:80]}")

    return final_answer

# ============================================================
# WhatsApp Sender (uncomment when Meta credentials are ready)
# ============================================================
# async def send_whatsapp_message(to_phone: str, message: str):
#     url     = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
#     headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
#     payload = {"messaging_product": "whatsapp", "to": to_phone, "type": "text", "text": {"body": message}}
#     async with httpx.AsyncClient(timeout=10.0) as client:
#         resp = await client.post(url, headers=headers, json=payload)
#         if resp.status_code != 200:
#             print(f"Failed to send WhatsApp message: {resp.text}")

# ============================================================
# Background Task — WhatsApp channel
# ============================================================
async def process_and_reply(phone_number: str, user_message: str):
    """
    For WhatsApp: chat_id == phone_number (one persistent session per number).
    Runs after 200 OK is returned to the WhatsApp webhook.
    """
    try:
        loop         = asyncio.get_event_loop()
        final_answer = await loop.run_in_executor(
            None, run_agent, phone_number, user_message, phone_number
        )
        print(f"[WhatsApp] Reply for {phone_number}: {final_answer[:100]}")
        # await send_whatsapp_message(phone_number, final_answer)
        print(f"[WhatsApp] Send skipped (LOCAL MODE).")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[WhatsApp] Error for {phone_number}: {e}")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Agent")

# ============================================================
# WhatsApp Webhook Verification (GET)
# ============================================================
@app.get("/webhook")
async def verify_webhook(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge:    str = Query(None, alias="hub.challenge")
):
    # WHATSAPP: replace hardcoded token with WHATSAPP_VERIFY_TOKEN env var when going live
    LOCAL_VERIFY_TOKEN = "test_verify_token_123"
    if hub_mode == "subscribe" and hub_verify_token == LOCAL_VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed.")

# ============================================================
# WhatsApp Webhook Handler (POST)
# ============================================================
@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives WhatsApp events. Returns 200 immediately, processes in background."""
    payload = await request.json()
    print(f"[Webhook] Incoming payload: {payload}")
    try:
        entry    = payload.get("entry", [{}])[0]
        changes  = entry.get("changes", [{}])[0]
        value    = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ok"}

        message  = messages[0]
        msg_type = message.get("type")
        if msg_type != "text":
            print(f"[Webhook] Ignoring non-text type: {msg_type}")
            return {"status": "ok"}

        phone_number = message.get("from")
        user_message = message["text"].get("body", "").strip()
        if not phone_number or not user_message:
            return {"status": "ok"}

        print(f"[Webhook] Message from {phone_number}: {user_message}")
        background_tasks.add_task(process_and_reply, phone_number, user_message)

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[Webhook] Parse error: {e}")

    return {"status": "ok"}

# ============================================================
# Chat Endpoint — Web / Mobile Frontend
#
# Frontend contract:
#   • On "New Chat" click → generate a fresh UUID and store it:
#       const chatId = crypto.randomUUID()          // browser
#       import { v4 as uuidv4 } from 'uuid'         // Node / React Native
#
#   • Send chat_id + phone_number + message on every turn of that session.
#   • On next "New Chat" click → generate a new UUID → fresh conversation.
#
# Backend behavior:
#   • New chat_id → no history found → agent starts completely fresh.
#   • Same chat_id → history loaded → agent answers with full context.
#   • MongoDB document created automatically on first message of a new chat.
# ============================================================
class ChatRequest(BaseModel):
    chat_id:      str   # UUID generated by frontend — new UUID = new conversation
    phone_number: str   # user's phone number — stored as metadata
    message:      str   # user's message text

class ChatResponse(BaseModel):
    chat_id:      str
    phone_number: str
    response:     str

@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest):
    """
    Chat endpoint for web / mobile frontends.

    Accepts all three required fields: chat_id, phone_number, message.

    - chat_id   → controls memory isolation (new UUID = blank slate)
    - phone_number → stored as metadata; query all sessions for a user with:
                     db.chat_sessions_v2.find({ phone_number: "911234567890" })
    - message   → the user's input text

    The agent receives the full stored history (up to 20 messages) as LLM
    context so it can answer intelligently based on the entire conversation.
    """
    print(f"\n[/test/chat] chat_id={request.chat_id} | phone={request.phone_number} | msg={request.message}")
    try:
        final_answer = run_agent(
            chat_id=request.chat_id,
            user_message=request.message,
            phone_number=request.phone_number
        )
        return ChatResponse(
            chat_id=request.chat_id,
            phone_number=request.phone_number,
            response=final_answer
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)