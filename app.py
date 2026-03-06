"""
FastAPI + LangGraph Agent with Multi-MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

Connects to MULTIPLE MCP servers simultaneously (e.g. Alumnx + Vignan)
and merges all their tools into one agent dynamically at startup.

FALLBACK FEATURE:
  - If tools don't find relevant information in knowledge base,
    the agent makes a direct Gemini API call for the answer
  - Source information is returned in the sources array
  - Format: ["Knowledge Base: file1.pdf, file2.pdf"] or ["Gemini API"]

New Chat flow:
  - Frontend generates a new UUID on "New Chat" click and sends it as chat_id.
  - Backend finds no history for that chat_id → agent starts fresh.
  - MongoDB creates the document automatically on first save.
  - Same chat_id on subsequent messages → history is loaded and agent remembers.

Auto Deploy enabled using deploy.yml file

CHANGES FROM ORIGINAL:
- Improved source extraction with multiple field name support
- Better KB data validation
- Smarter fallback logic with multiple conditions
- Enhanced error handling
"""

import os
import httpx
import asyncio
import json
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = "agrigpt-backend-agent"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_TIMEOUT    = float(os.getenv("MCP_TIMEOUT", "30"))

# ── Multi-MCP Configuration ──────────────────────────────────────────────────
MCP_SERVERS: List[Dict[str, str]] = [
    {
        "name":    "Alumnx",
        "url":     os.getenv("ALUMNX_MCP_URL", "http://localhost:9000"),
        "api_key": os.getenv("ALUMNX_MCP_API_KEY", ""),
    },
    {
        "name":    "Vignan",
        "url":     os.getenv("VIGNAN_MCP_URL", "http://localhost:8000"),
        "api_key": os.getenv("VIGNAN_MCP_API_KEY", ""),
    },
]

MONGODB_URI        = os.getenv("MONGODB_URI")
MONGODB_DB         = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chats")

# Max messages stored per chat_id (human + AI combined = 10 full turns).
MAX_MESSAGES = 20

# ============================================================
# MongoDB Setup
# ============================================================
mongo_client   = MongoClient(MONGODB_URI)
db             = mongo_client[MONGODB_DB]
chat_sessions: Collection = db[MONGODB_COLLECTION]

chat_sessions.create_index([("chat_id",      ASCENDING)], unique=True)
chat_sessions.create_index([("phone_number", ASCENDING)])
chat_sessions.create_index([("updated_at",   ASCENDING)])

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
    """
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
            "$setOnInsert": {"created_at": now},
        },
        upsert=True
    )


# ============================================================
# Gemini Fallback Handler
# ============================================================
async def get_gemini_fallback_answer(user_question: str) -> str:
    """
    When knowledge base tools don't find information,
    call Gemini directly for a general answer.
    
    Returns: The generated answer text
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,  # Slightly higher for creative answers
            google_api_key=GOOGLE_API_KEY,
        )
        
        response = llm.invoke([
            SystemMessage(content="You are a helpful agricultural assistant. Provide clear, concise answers."),
            HumanMessage(content=user_question)
        ])
        
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"[Gemini Fallback Error] {e}")
        return f"Unable to generate answer: {str(e)}"


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Backend Agent", version="2.0")

global_tools = []
global_tool_results = []

# ============================================================
# Agent State & Graph
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def fetch_mcp_tools(server: Dict[str, str]) -> List[StructuredTool]:
    """
    Fetch tools from a single MCP server.
    Returns a list of LangChain StructuredTool objects.
    """
    try:
        async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
            response = await client.post(
                f"{server['url']}/mcp/tools",
                json={"api_key": server["api_key"]},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            tools_data = response.json()
            if not isinstance(tools_data, list):
                tools_data = tools_data.get("tools", [])
            
            tools = []
            for tool_spec in tools_data:
                tool_name = tool_spec.get("name", "unknown_tool")
                tool_desc = tool_spec.get("description", "")
                tool_input = tool_spec.get("input_schema", {})
                
                properties = tool_input.get("properties", {})
                
                DynamicInput = create_model(
                    f"{tool_name}_Input",
                    **{
                        key: (str, Field(description=prop.get("description", "")))
                        for key, prop in properties.items()
                    }
                )
                
                async def tool_runner(
                    server=server,
                    tool_name=tool_name,
                    **kwargs
                ):
                    try:
                        async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
                            response = await client.post(
                                f"{server['url']}/mcp/run",
                                json={
                                    "tool": tool_name,
                                    "input": kwargs,
                                    "api_key": server["api_key"]
                                },
                                headers={"Content-Type": "application/json"}
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            # Store for later extraction
                            global_tool_results.append({
                                "tool": tool_name,
                                "server": server["name"],
                                "result": result
                            })
                            
                            return json.dumps(result) if isinstance(result, dict) else str(result)
                    except httpx.TimeoutException:
                        return json.dumps({"error": f"Tool {tool_name} timed out"})
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                
                tool = StructuredTool(
                    name=tool_name,
                    description=tool_desc,
                    func=tool_runner,
                    args_schema=DynamicInput
                )
                tools.append(tool)
            
            print(f"[{server['name']}] Loaded {len(tools)} tools")
            return tools
    except Exception as e:
        print(f"[ERROR] Failed to fetch tools from {server['name']}: {e}")
        return []


async def initialize_agent():
    """Initialize the agent with tools from all MCP servers at startup."""
    global global_tools
    
    all_tools = []
    for server in MCP_SERVERS:
        print(f"Connecting to {server['name']} ({server['url']})...")
        server_tools = await fetch_mcp_tools(server)
        all_tools.extend(server_tools)
    
    global_tools = all_tools
    print(f"[Agent Init] Total tools available: {len(global_tools)}")


@app.on_event("startup")
async def startup_event():
    """Initialize agent on app startup."""
    await initialize_agent()


def agent_node(state: AgentState) -> dict:
    """
    Agent decision node.
    Uses all available tools and Gemini model to process messages.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
        )
        
        llm_with_tools = llm.bind_tools(global_tools) if global_tools else llm
        
        response = llm_with_tools.invoke(state["messages"])
        
        return {"messages": [response]}
    except Exception as e:
        print(f"[Agent Node Error] {e}")
        error_msg = f"Error: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


def tool_node_func(state: AgentState) -> dict:
    """Execute tools if LLM requested them."""
    return {"messages": []}


# Build the agent graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(global_tools))

graph_builder.add_edge(START, "agent")

def should_use_tools(state: AgentState) -> str:
    """Route to tools if LLM requested them, otherwise END."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return END


graph_builder.add_conditional_edges("agent", should_use_tools)
graph_builder.add_edge("tools", "agent")

agent_graph = graph_builder.compile()


async def run_agent(chat_id: str, user_message: str, phone_number: str) -> dict:
    """
    Run the agent for a chat session.
    
    Returns:
        {
            "answer": str,
            "tool_results": List[Dict] containing tool execution info
        }
    """
    global global_tool_results
    global_tool_results.clear()
    
    # Load conversation history
    history = load_history(chat_id)
    
    # Prepare initial state with conversation history + new message
    state = AgentState(messages=history + [HumanMessage(content=user_message)])
    
    # Run the agent
    try:
        result = agent_graph.invoke(state, {"recursion_limit": 25})
    except Exception as e:
        print(f"[Agent Error] {e}")
        result = {"messages": [AIMessage(content=f"Error: {str(e)}")]}
    
    # Extract final answer
    messages = result.get("messages", [])
    final_answer = ""
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            final_answer = msg.content
            break
    
    # Save conversation history
    save_history(chat_id, messages, phone_number)
    
    return {
        "answer": final_answer,
        "tool_results": global_tool_results.copy()
    }


# ============================================================
# Source Extraction (IMPROVED)
# ============================================================
def extract_sources_from_tool_results(tool_results: List[Dict[str, Any]]) -> Tuple[List[str], bool]:
    """
    Extract source filenames from tool execution results.
    
    IMPROVED: Handles multiple source field names and validates KB data presence
    
    Returns:
    - sources: List of PDF filenames or empty list
    - has_meaningful_data: Boolean indicating if KB actually found relevant info
    """
    sources = set()
    has_meaningful_data = False
    
    if not tool_results:
        print("[extract_sources] No tool results provided")
        return [], False
    
    print(f"[extract_sources] Processing {len(tool_results)} tool results")
    
    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue
        
        tool_name = tool_result.get("tool", "unknown")
        result_data = tool_result.get("result")
        
        if not result_data:
            print(f"[extract_sources] {tool_name}: No result data")
            continue
        
        # Handle stringified JSON results
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except:
                print(f"[extract_sources] {tool_name}: Could not parse as JSON")
                continue
        
        if not isinstance(result_data, dict):
            continue
        
        print(f"[extract_sources] {tool_name}: Processing result keys: {list(result_data.keys())}")
        
        # Check if KB actually found results
        if result_data.get("found") or result_data.get("success") or result_data.get("matches"):
            has_meaningful_data = True
            print(f"[extract_sources] {tool_name}: Found meaningful KB data")
        
        # Extract from multiple possible source field names
        source_fields = ["sources", "source", "files", "documents", "references", "results"]
        
        for field in source_fields:
            if field not in result_data:
                continue
            
            field_value = result_data[field]
            print(f"[extract_sources] {tool_name}: Checking field '{field}'")
            
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, dict):
                        # Try multiple field names for filename
                        filename = (item.get("filename") or item.get("file") or 
                                   item.get("name") or item.get("source") or 
                                   item.get("path"))
                        if filename and isinstance(filename, str):
                            sources.add(filename.strip())
                            has_meaningful_data = True
                            print(f"[extract_sources] Added source: {filename}")
                    elif isinstance(item, str) and item.strip():
                        sources.add(item.strip())
                        has_meaningful_data = True
                        print(f"[extract_sources] Added source (string): {item}")
            elif isinstance(field_value, dict):
                filename = (field_value.get("filename") or field_value.get("file") or 
                           field_value.get("name") or field_value.get("source") or
                           field_value.get("path"))
                if filename:
                    sources.add(str(filename).strip())
                    has_meaningful_data = True
                    print(f"[extract_sources] Added source (dict): {filename}")
        
        # Check for answer/content that indicates KB found something
        for content_field in ["answer", "content", "data", "response", "result"]:
            if content_field in result_data:
                answer_text = str(result_data.get(content_field, ""))
                if answer_text.strip() and len(answer_text) > 20:  # Non-trivial answer
                    has_meaningful_data = True
                    print(f"[extract_sources] {tool_name}: Found meaningful content in '{content_field}'")
    
    final_sources = sorted(list(sources))
    print(f"[extract_sources] Final result: sources={final_sources}, has_meaningful_data={has_meaningful_data}")
    
    return final_sources, has_meaningful_data


# ============================================================
# Response Validation & Cleaning
# ============================================================
def is_meaningful_response(response: str) -> bool:
    """Check if response is substantial enough to return."""
    if not response or len(response) < 30:
        return False
    
    # Check for bot-like patterns
    generic_phrases = ["i'm not sure", "unable to", "cannot answer", "i don't know"]
    if any(phrase in response.lower() for phrase in generic_phrases):
        return False
    
    return True


def clean_response_text(text: str) -> str:
    """
    Clean and format the response text.
    """
    if not text:
        return ""
    
    # Remove markdown formatting asterisks
    cleaned = text.replace("**", "").replace("*", "")
    
    # Convert escaped newlines to actual newlines
    cleaned = cleaned.replace("\\n", "\n")
    
    # Remove source sections if any
    if "📚 Sources:" in cleaned or "Sources:" in cleaned:
        if "📚 Sources:" in cleaned:
            cleaned = cleaned.split("📚 Sources:")[0]
        else:
            cleaned = cleaned.split("Sources:")[0]
    
    cleaned = cleaned.strip()
    
    return cleaned


# ============================================================
# Chat Endpoint (IMPROVED)
# ============================================================
class ChatRequest(BaseModel):
    chatId:       str
    phone_number: str
    message:      str


class ChatResponse(BaseModel):
    chatId:       str
    phone_number: str
    response:     str
    sources:      List[str] = []


@app.post("/test/chat", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    Chat endpoint for web / mobile frontends with IMPROVED fallback logic.
    
    IMPROVEMENTS:
    - Better source extraction with multiple field names
    - Validation that KB actually found meaningful data
    - Multi-condition fallback logic
    - Consistent response formatting
    """
    global global_tool_results
    
    print(f"\n[/test/chat] chatId={request.chatId} | phone={request.phone_number} | msg={request.message}")
    try:
        global_tool_results.clear()
        
        # Run agent
        agent_result = await run_agent(
            request.chatId,
            request.message,
            request.phone_number
        )
        
        final_answer = agent_result["answer"]
        tool_results = agent_result.get("tool_results", [])
        
        print("[/test/chat] Extracting sources...")
        sources = []
        has_kb_data = False
        
        # Extract sources and check if KB found real data
        if tool_results:
            sources, has_kb_data = extract_sources_from_tool_results(tool_results)
        
        print(f"[/test/chat] KB Data Found: {has_kb_data} | Sources: {sources}")
        
        # Determine if we should use Gemini fallback
        should_use_gemini = False
        reason = ""
        
        if not has_kb_data and tool_results:
            # Tools ran but KB didn't find meaningful data
            should_use_gemini = True
            reason = "Tools ran but KB returned no meaningful data"
        elif not sources and not has_kb_data:
            # No sources extracted and no KB data found
            should_use_gemini = True
            reason = "No sources extracted and no KB data"
        elif any(phrase in final_answer.lower() for phrase in 
                ["not found in knowledge base", "not found", "no information", 
                 "not available", "no data", "no results", "no matches"]):
            # Agent explicitly said KB has nothing
            should_use_gemini = True
            reason = "Agent found no KB results"
        
        if should_use_gemini:
            print(f"[/test/chat] Using Gemini fallback: {reason}")
            gemini_answer = await get_gemini_fallback_answer(request.message)
            
            if is_meaningful_response(gemini_answer):
                final_answer = f"Sorry not found in knowledge base but I can use Gemini API to answer your question\n\n{gemini_answer}"
                sources = ["Gemini API"]
            else:
                final_answer = "I couldn't find information about this topic in my knowledge base or through other sources. Please try a different query."
                sources = ["Unable to find answer"]
        elif not sources and has_kb_data:
            # KB found data but couldn't extract sources
            print("[/test/chat] KB found data but source extraction failed, using Gemini backup...")
            sources = ["Knowledge Base (sources unavailable)"]
        
        # Clean response
        cleaned_response = clean_response_text(final_answer)
        
        print(f"[/test/chat] Final sources: {sources}")
        print(f"[/test/chat] Response length: {len(cleaned_response)} chars")
        
        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=cleaned_response,
            sources=sources,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# WhatsApp Webhook (Existing)
# ============================================================
@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    WhatsApp Business API webhook for incoming messages.
    """
    body = await request.json()
    
    try:
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        messages = changes.get("value", {}).get("messages", [])
        
        if not messages:
            return PlainTextResponse("ok")
        
        message = messages[0]
        from_number = message.get("from", "")
        text = message.get("text", {}).get("body", "")
        
        if not text:
            return PlainTextResponse("ok")
        
        # Use phone number as chat_id for WhatsApp
        background_tasks.add_task(
            whatsapp_reply_async,
            from_number,
            text,
            from_number  # Same as chat_id
        )
        
        return PlainTextResponse("ok")
    except Exception as exc:
        import traceback; traceback.print_exc()
        return PlainTextResponse("ok")


async def whatsapp_reply_async(phone_number: str, user_message: str, chat_id: str):
    """
    Async task to process WhatsApp message and send reply.
    """
    try:
        result = await run_agent(chat_id, user_message, phone_number)
        final_answer = result["answer"]
        print(f"[WhatsApp] Reply for {phone_number}: {final_answer[:100]}")
        print("[WhatsApp] Send skipped (LOCAL MODE).")
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[WhatsApp] Error for {phone_number}: {exc}")


# ============================================================
# Health Check
# ============================================================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tools_loaded": len(global_tools),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)