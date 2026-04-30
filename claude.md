# Research Assistant with Iterative Refinement

## Project Overview

A local research assistant app that takes a complex question, breaks it into sub-queries, searches for information, evaluates result quality, re-searches if needed, and synthesizes findings into a structured report. Built with **LangGraph** (Python) for orchestration, **Streamlit** for the UI, and **uv** for dependency management.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11+** | Runtime |
| **uv** | Dependency management and virtual environment |
| **LangGraph** | Stateful agent workflow orchestration |
| **LangChain** | LLM integrations and tool abstractions |
| **Anthropic Claude** | LLM provider (via `langchain-anthropic`) |
| **Tavily** | Web search tool for research (via `tavily-python`) |
| **Streamlit** | Frontend UI |

---

## Project Structure

```
research-assistant/
├── CLAUDE.md              # This file - project guide
├── pyproject.toml          # uv project config and dependencies
├── .env                    # API keys (ANTHROPIC_API_KEY, TAVILY_API_KEY)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── app.py              # Streamlit entry point
│   ├── graph.py            # LangGraph workflow definition
│   ├── nodes.py            # Individual node functions
│   ├── state.py            # Graph state schema
│   └── prompts.py          # All LLM prompt templates
```

---

## Setup Instructions

### 1. Initialize the project with uv

```bash
uv init research-assistant
cd research-assistant
```

### 2. Add dependencies

```bash
uv add langgraph langchain langchain-anthropic langchain-community tavily-python streamlit python-dotenv
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

> **Getting API keys:**
> - Anthropic: https://console.anthropic.com/
> - Tavily: https://tavily.com/ (free tier available, good for development)

### 4. Run the app

```bash
uv run streamlit run src/app.py
```

---

## Architecture

### Graph Workflow

The LangGraph workflow follows this flow:

```
[START]
   │
   ▼
┌──────────────┐
│  Decompose   │  ← Break question into 2-4 sub-queries
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Research    │  ← Run web searches for each sub-query
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Evaluate   │────▶│   Research   │  ← Loop back if gaps found
│   Quality    │ gap  │  (re-search) │    (max 2 refinement loops)
└──────┬───────┘     └──────────────┘
       │ sufficient
       ▼
┌──────────────┐
│  Synthesize  │  ← Combine findings into structured report
└──────┬───────┘
       │
       ▼
   [END]
```

### State Schema

The shared state object passed between all nodes:

```python
class ResearchState(TypedDict):
    question: str                    # Original user question
    sub_queries: list[str]           # Decomposed sub-queries
    search_results: list[dict]       # Raw search results
    evaluation: str                  # Quality evaluation ("sufficient" or "gaps: ...")
    refinement_count: int            # Track loop iterations (max 2)
    report: str                      # Final synthesized report
```

---

## Implementation Guide

### Node Specifications

#### 1. `decompose_question` node
- **Input:** `question` from state
- **Action:** Call Claude to break the question into 2-4 focused sub-queries
- **Output:** Update `sub_queries` in state
- **Prompt strategy:** Ask the LLM to identify distinct aspects of the question that need separate research

#### 2. `research` node
- **Input:** `sub_queries` from state
- **Action:** Use Tavily search API for each sub-query, collect results
- **Output:** Update `search_results` in state
- **Note:** Each search should return top 3-5 results. Store the query, title, URL, and content snippet for each result

#### 3. `evaluate_quality` node
- **Input:** `question`, `sub_queries`, `search_results` from state
- **Action:** Call Claude to assess whether search results adequately cover the original question
- **Output:** Update `evaluation` in state with either `"sufficient"` or a string starting with `"gaps:"` describing what's missing
- **Prompt strategy:** Ask the LLM to check for completeness, identify missing perspectives, and flag contradictions

#### 4. `synthesize_report` node
- **Input:** `question`, `search_results` from state
- **Action:** Call Claude to produce a structured report from all collected results
- **Output:** Update `report` in state
- **Report format:** Title, executive summary, key findings (organized by theme), sources list

### Conditional Edge Logic

After `evaluate_quality`, route based on:

```python
def should_continue_research(state: ResearchState) -> str:
    if state["refinement_count"] >= 2:
        return "synthesize"  # Hit max loops, move on
    if state["evaluation"].startswith("gaps:"):
        return "research"    # Gaps found, research more
    return "synthesize"      # Quality sufficient
```

When looping back to research, the `gaps:` string from the evaluation should be used to generate new, targeted sub-queries that fill the identified gaps.

---

## Streamlit UI Specifications

### Layout

Keep it simple and functional:

1. **Header** — App title and brief description
2. **Input area** — Text input or text area for the research question, plus a "Research" button
3. **Progress section** — While the graph runs, show which node is currently executing. Use `st.status` or `st.expander` to show:
   - Sub-queries generated
   - Search results found (count + brief preview)
   - Evaluation result (sufficient or gaps found)
   - Refinement loop status (e.g., "Refining... iteration 2/2")
4. **Report section** — Display the final synthesized report in markdown

### Streamlit Tips

- Use `st.session_state` to persist the research results across reruns
- Use `st.spinner` or `st.status` for progress indication during graph execution
- Display intermediate state updates using `st.expander` so users can inspect each step
- The graph should run synchronously from Streamlit's perspective — invoke it and display results when done

### Example UI Code Pattern

```python
import streamlit as st
from graph import build_graph

st.title("Research Assistant")

question = st.text_area("What would you like to research?", height=100)

if st.button("Research", type="primary"):
    if question:
        graph = build_graph()
        with st.status("Researching...", expanded=True) as status:
            # Invoke graph and stream state updates
            result = graph.invoke({"question": question, "refinement_count": 0})
            status.update(label="Research complete!", state="complete")
        
        st.markdown("## Report")
        st.markdown(result["report"])
```

---

## LangGraph Implementation Pattern

### Building the Graph

```python
from langgraph.graph import StateGraph, END
from state import ResearchState

def build_graph():
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("decompose", decompose_question)
    workflow.add_node("research", research)
    workflow.add_node("evaluate", evaluate_quality)
    workflow.add_node("synthesize", synthesize_report)

    # Set entry point
    workflow.set_entry_point("decompose")

    # Add edges
    workflow.add_edge("decompose", "research")
    workflow.add_edge("research", "evaluate")

    # Conditional edge after evaluation
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_research,
        {
            "research": "research",
            "synthesize": "synthesize",
        }
    )

    workflow.add_edge("synthesize", END)

    return workflow.compile()
```

### LLM Initialization

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096,
)
```

### Tavily Search Setup

```python
import os
from tavily import TavilyClient

search_tool = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
# Use search_tool.search(query, max_results=3, search_depth="basic")
```

---

## Key Implementation Rules

1. **Always increment `refinement_count`** when looping back through research to prevent infinite loops
2. **Cap refinement loops at 2** — diminishing returns after that, and it keeps response times reasonable
3. **Handle search failures gracefully** — if Tavily returns no results for a sub-query, note it in the state rather than crashing
4. **Keep prompts in `prompts.py`** — don't scatter prompt strings across node functions
5. **Use `python-dotenv`** to load `.env` at the top of `app.py` before any LangChain imports
6. **Each node function** should take `state: ResearchState` as its only argument and return a `dict` with only the state keys it updates
7. **Don't over-engineer** — this is a simple app. No database, no auth, no caching. Just the graph, the UI, and the LLM calls

---

## .gitignore

```
.env
__pycache__/
*.pyc
.venv/
.python-version
```

---

## Development Workflow

1. **Start with `state.py`** — define the `ResearchState` TypedDict
2. **Write `prompts.py`** — draft all prompt templates
3. **Build `nodes.py`** — implement each node function, test individually
4. **Wire up `graph.py`** — connect nodes with edges and conditional logic
5. **Build `app.py`** — create the Streamlit UI and connect it to the graph
6. **Test end-to-end** — run with a simple question first, then try complex multi-part questions

---

## Git Workflow

### Branching

- **Never commit directly to `main`** — all changes must go through a feature branch
- Branch naming: `feat/<short-description>`, `fix/<short-description>`, `chore/<short-description>`

```bash
git checkout -b feat/my-feature
```

### Commit Often

Commit after each logical unit of work — don't batch unrelated changes into one commit. Good checkpoints:
- After adding a new file
- After making a node or edge work correctly
- After fixing a bug
- After updating dependencies

### Conventional Commits

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<optional scope>): <short description>

<optional body>
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `chore` | Dependency updates, config changes, tooling |
| `refactor` | Code restructuring with no behaviour change |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `style` | Formatting, whitespace (no logic change) |

**Examples:**

```
feat(graph): add checkpointer support to build_graph

fix(nodes): handle empty Tavily results without crashing

chore: add langgraph-cli as dev dependency

docs(claude.md): add git workflow guidelines
```

### Merging

Open a pull request to merge your branch into `main`. Do not merge locally unless there is a specific reason to.

---

## Example Test Questions

Use these to validate the app works correctly:

- **Simple:** "What is the current state of fusion energy research?"
- **Multi-faceted:** "Compare the economic, environmental, and social impacts of remote work adoption since 2020"
- **Requires refinement:** "How are different countries approaching AI regulation, and what are the key differences between the EU AI Act and the US executive order on AI?"

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure you're running with `uv run` not plain `python` |
| Tavily returns empty results | Check your API key, check query isn't too long/vague |
| Graph loops infinitely | Verify `refinement_count` is being incremented in the research node |
| Streamlit reruns clear results | Store results in `st.session_state` |
| Rate limits from Anthropic | Add a small `time.sleep(1)` between LLM calls in nodes |