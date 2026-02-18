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

# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
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
            json={"name": name, "arguments": arguments}
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

    dynamic_tools = []

    for tool_schema in remote_tools:
        tool_name = tool_schema["name"]
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})
        print(f"🔨 Creating tool: {tool_name}")

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
                print(f"🎯 Executing remote tool: {name} with args: {kwargs}")
                # Strip None values so MCP server uses its own defaults
                cleaned = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    result = mcp_client.call_tool(name, cleaned)
                    print(f"✨ Tool execution successful: {result}")
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
        response = llm_with_tools.invoke(state["messages"])
        print(f"💬 LLM Response content: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"🔧 Tool calls detected: {len(response.tool_calls)}")
        return {"messages": [response]}

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("➡️  Routing to TOOLS node")
            return "tools"
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
    print(f"\n📥 NEW CHAT REQUEST: {request.message}\n")

    try:
        result = app_agent.invoke({
            "messages": [HumanMessage(content=request.message)]
        })

        # ✅ FIX: grab the LAST AIMessage with non-empty content
        final_answer = "No response generated."
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content.strip():
                    final_answer = msg.content
                    break
                elif isinstance(msg.content, list) and msg.content:
                    block = msg.content[0]
                    if isinstance(block, dict) and block.get("text", "").strip():
                        final_answer = block["text"]
                        break
                    elif str(block).strip():
                        final_answer = str(block)
                        break

        print(f"\n🎉 Returning response: {str(final_answer)[:100]}\n")
        return ChatResponse(response=final_answer)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)