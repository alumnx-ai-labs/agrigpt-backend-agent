# AgriGPT Backend Agent — Claude Code Instructions

## Project Overview

FastAPI + LangGraph agent that connects to multiple MCP servers (Alumnx, Vignan), discovers tools dynamically at startup, and answers agricultural queries via WhatsApp and a web chat endpoint. MongoDB stores per-session conversation history.

## Architecture

```
app.py              — Single-file app: all models, routes, agent logic
tests/
  test_api.py       — FastAPI endpoint tests (mocked dependencies)
  test_utils.py     — Pure function tests (extract_final_answer, sources, etc.)
  test_history.py   — MongoDB history helper tests
pytest.ini          — testpaths = tests, pythonpath = .
requirements.txt    — Runtime deps only (no test deps listed)
.github/workflows/
  ci.yml            — Run tests on every PR
  deploy.yml        — Deploy to EC2 on push to main (after tests pass)
```

## Key Modules in app.py

| Symbol                                | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `MCPClient`                           | REST client for MCP servers (`/list-tools`, `/callTool`)     |
| `build_agent()`                       | Discovers tools from all MCP servers, builds LangGraph graph |
| `run_agent()`                         | Entry point for agent execution (loads/saves history)        |
| `load_history()` / `save_history()`   | MongoDB conversation memory helpers                          |
| `extract_final_answer()`              | Gets last AIMessage text from result dict                    |
| `extract_sources_from_result()`       | Finds PDF filenames in AI response text                      |
| `extract_sources_from_tool_results()` | Finds PDF filenames from raw tool results                    |
| `clean_response_text()`               | Strips markdown from LLM output                              |
| `get_gemini_fallback()`               | Calls Gemini Web Search when tools return no data            |
| `has_meaningful_tool_results()`       | Checks if tool results have useful content                   |
| `is_agriculture_query()`              | Keyword-based topic detection                                |

## API Endpoints

| Method | Path         | Description                                                        |
| ------ | ------------ | ------------------------------------------------------------------ |
| GET    | `/hello`     | Health check — returns `{"message": "hello claude!!"}`             |
| GET    | `/webhook`   | WhatsApp verification (hub.verify_token = `test_verify_token_123`) |
| POST   | `/webhook`   | WhatsApp message handler (background task)                         |
| POST   | `/test/chat` | Main chat endpoint — tool-first + Gemini fallback                  |

## Environment Variables

```
GOOGLE_API_KEY          — Required for Gemini LLM
MONGODB_URI             — MongoDB connection string
MONGODB_DB              — Database name (default: agrigpt)
MONGODB_COLLECTION      — Collection name (default: chats)
ALUMNX_MCP_URL          — Alumnx MCP server URL
ALUMNX_MCP_API_KEY      — Alumnx API key (optional)
VIGNAN_MCP_URL          — Vignan MCP server URL
VIGNAN_MCP_API_KEY      — Vignan API key (optional)
MCP_TIMEOUT             — Tool call timeout in seconds (default: 30)
LANGSMITH_API_KEY       — Optional: enables LangSmith tracing
```

## Testing Guidelines

- All tests mock external dependencies (MongoDB, LangGraph, MCP servers, Gemini)
- Import mocks must be set up in `sys.modules` BEFORE importing `app`
- `build_agent` is always patched since it makes real MCP network calls at module load
- Run tests: `pytest` (uses `pytest.ini` config)
- Test files follow pattern `tests/test_*.py`
- Do NOT write tests that make real network calls or hit a real MongoDB

## Development Workflow

1. Write tests first (or alongside code changes)
2. Run `pytest` to verify all tests pass locally
3. Push branch → GitHub Actions runs CI (`ci.yml`) automatically
4. Open PR → Claude Code can review and suggest changes
5. Merge PR to `main` → `deploy.yml` deploys to EC2 automatically

## Code Style

- Match existing patterns in `app.py` (single-file, no separate modules)
- Use `print()` for logging (no logging framework)
- Type hints: use `str | None` (Python 3.10+ syntax)
- Keep functions focused; prefer reading existing code before editing
- No docstrings needed on simple functions; keep existing docs intact

## Deployment

Auto-deploy on push to `main` via `.github/workflows/deploy.yml`:

- SSH into EC2 (`/agrigpt/agent`)
- `git pull origin main`
- `pip install -r requirements.txt`
- `sudo systemctl restart agrigpt-agent.service`

EC2 host, user, and SSH key stored in GitHub Secrets:
`EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`
