"""
FastAPI + LangGraph Agent with Remote MCP Tool Discovery
Production-ready distributed MCP architecture
"""

import os
import httpx
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as PydanticBaseModel, Field, create_model

# ============================================================
# Environment
# ============================================================

load_dotenv()
# LangSmith Configuration
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "agrigpt-backend-agent"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "https://agrigpt-backend-mcp.onrender.com")
MCP_API_KEY = os.getenv("MCP_API_KEY")
MCP_TIMEOUT = 30.0  # Increased from 10.0 to 30.0 seconds for better reliability

# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

        #if api_key:
        #    self.headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.Client(timeout=MCP_TIMEOUT)

    def list_tools(self) -> List[Dict[str, Any]]:
        print(f"📡 Calling MCP server: {self.base_url}/getToolsList")
        response = self.client.post(
            f"{self.base_url}/getToolsList",
            headers=self.headers,
            json={}
        )
        response.raise_for_status()
        tools = response.json().get("tools", [])
        print(f"📥 Received {len(tools)} tools from MCP server")
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        print(f"🔧 Calling MCP tool: {name} with args: {arguments}")
        response = self.client.post(
            f"{self.base_url}/callTool",
            headers=self.headers,
            json={
                "name": name,
                "arguments": arguments
            }
        )
        response.raise_for_status()
        result = response.json().get("result")
        print(f"✅ MCP tool result: {result}")
        return result


# ============================================================
# State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent():

    mcp_client = MCPClient(MCP_BASE_URL, MCP_API_KEY)

    print("🔎 Fetching tools from remote MCP...")
    remote_tools = mcp_client.list_tools()

    if not remote_tools:
        raise RuntimeError("No tools found on remote MCP server.")

    print(f"✅ Loaded {len(remote_tools)} tools: {[t['name'] for t in remote_tools]}")

    # Convert MCP tools into LangChain tools dynamically
    dynamic_tools = []

    for tool_schema in remote_tools:
        tool_name = tool_schema["name"]
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})
        print(f"🔨 Creating tool: {tool_name}")
        print(f"   Input schema: {input_schema}")

        # Create closure to capture tool_name and schema
        def create_tool(name: str, desc: str, schema: Dict[str, Any]):
            # Create a Pydantic model for the tool arguments based on the schema
            properties = schema.get("properties", {})
            
            print(f"   Processing properties: {list(properties.keys())}")
            
            # Build field definitions for Pydantic model
            field_definitions = {}
            for prop_name, prop_details in properties.items():
                prop_type = prop_details.get("type", "string")
                
                # Map JSON schema types to Python types
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
                    field_definitions[prop_name] = (py_type, Field(default=None, description=description_text))
            
            # Dynamically create the args schema class using pydantic's create_model
            ArgsSchema = create_model(
                f"{name}_args",
                **field_definitions
            )
            
            print(f"   ArgsSchema fields: {ArgsSchema.model_fields.keys()}")
            
            # Create function that accepts **kwargs to match the schema
            def remote_tool_func(**kwargs) -> str:
                print(f"🎯 Executing remote tool: {name}")
                print(f"📝 Tool arguments received: {kwargs}")
                try:
                    result = mcp_client.call_tool(name, kwargs)
                    print(f"✨ Tool execution successful: {result}")
                    return str(result)
                except Exception as e:
                    error_msg = f"Remote MCP error: {str(e)}"
                    print(f"❌ Tool execution failed: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return error_msg
            
            tool = StructuredTool.from_function(
                func=remote_tool_func,
                name=name,
                description=desc,
                args_schema=ArgsSchema
            )
            
            print(f"   ✅ Tool created successfully")
            return tool

        dynamic_tools.append(create_tool(tool_name, description, input_schema))

    print(f"🧰 Total tools created: {len(dynamic_tools)}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    llm_with_tools = llm.bind_tools(dynamic_tools)
    print("🤖 LLM bound with tools")

    def agent_node(state: State):
        print("\n" + "="*60)
        print("🧠 AGENT NODE CALLED")
        print("="*60)
        print(f"📨 Input messages count: {len(state['messages'])}")
        for i, msg in enumerate(state['messages']):
            print(f"  Message {i}: {type(msg).__name__} - {msg.content[:100] if hasattr(msg, 'content') else msg}")
        
        print("🤔 Invoking LLM...")
        response = llm_with_tools.invoke(state["messages"])
        
        print(f"💬 LLM Response type: {type(response).__name__}")
        print(f"💬 LLM Response content: {response.content}")
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"🔧 Tool calls detected: {len(response.tool_calls)}")
            for i, tc in enumerate(response.tool_calls):
                print(f"  Tool call {i}: {tc}")
        else:
            print("ℹ️  No tool calls in response")
        
        return {"messages": [response]}

    def should_continue(state: State):
        print("\n" + "="*60)
        print("🚦 SHOULD_CONTINUE CHECK")
        print("="*60)
        last_message = state["messages"][-1]
        print(f"🔍 Last message type: {type(last_message).__name__}")
        
        if hasattr(last_message, "tool_calls"):
            print(f"🔍 Has tool_calls attribute: {last_message.tool_calls}")
            if last_message.tool_calls:
                print("➡️  Routing to TOOLS node")
                return "tools"
        else:
            print("ℹ️  No tool_calls attribute")
        
        print("🏁 Routing to END")
        return END

    workflow = StateGraph(State)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(dynamic_tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    print("📊 Workflow compiled successfully")
    return workflow.compile()

# Build agent once at startup
print("\n🚀 BUILDING AGENT AT STARTUP...")
app_agent = build_agent()
print("✅ AGENT BUILD COMPLETE\n")

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="MCP Powered LangGraph Agent")


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print("\n" + "🌟"*30)
    print(f"📥 NEW CHAT REQUEST: {request.message}")
    print("🌟"*30 + "\n")

    try:
        result = app_agent.invoke({
           "messages": [HumanMessage(content=request.message)]
        })

        print("\n" + "="*60)
        print("📊 AGENT EXECUTION COMPLETE")
        print("="*60)
        print(f"📝 Total messages in result: {len(result['messages'])}")

        final_answer = "Hi, this is from agent response. MCP integration is in progress"

        for i, msg in enumerate(result["messages"]):
            print(f"Message {i}: {type(msg).__name__}")
            if isinstance(msg, AIMessage):
                # Handle both string and list responses from LLM
                if isinstance(msg.content, str):
                    # Content is already a string
                    final_answer = msg.content
                    print(f"  ✅ Final answer extracted (string): {final_answer[:100]}...")
                elif isinstance(msg.content, list) and len(msg.content) > 0:
                    # Content is a list of content blocks
                    if isinstance(msg.content[0], dict) and 'text' in msg.content[0]:
                        # Extract text from first content block
                        final_answer = msg.content[0]['text']
                        print(f"  ✅ Final answer extracted (from list): {final_answer[:100]}...")
                    else:
                        # Fallback: convert to string
                        final_answer = str(msg.content[0])
                        print(f"  ✅ Final answer extracted (converted): {final_answer[:100]}...")
                else:
                    # Fallback: convert entire content to string
                    final_answer = str(msg.content)
                    print(f"  ✅ Final answer extracted (fallback): {final_answer[:100]}...")

        print(f"\n🎉 Returning response: {str(final_answer)[:100]}\n")
        return ChatResponse(response=final_answer)

    except Exception as e:
        print(f"\n❌ ERROR in chat endpoint: {str(e)}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)