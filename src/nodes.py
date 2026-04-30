import os

from tavily import TavilyClient
from langchain_anthropic import ChatAnthropic

from state import ResearchState
from prompts import (
    DECOMPOSE_QUESTION_PROMPT,
    EVALUATE_QUALITY_PROMPT,
    SYNTHESIZE_REPORT_PROMPT,
    REFINE_QUERIES_PROMPT,
)

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096,
)

# Smaller token budget for short-output nodes (decompose, evaluate)
llm_short = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=512,
)


def decompose_question(state: ResearchState) -> dict:
    prompt = DECOMPOSE_QUESTION_PROMPT.format(question=state["question"])
    response = llm_short.invoke(prompt)
    lines = [
        line.strip()
        for line in response.content.strip().splitlines()
        if line.strip()
    ]
    # Strip leading list markers like "1. ", "2. ", etc.
    sub_queries = []
    for line in lines:
        if line[1:3] in (". ", ") "):
            sub_queries.append(line[3:].strip())
        else:
            sub_queries.append(line)
    return {"sub_queries": sub_queries}


_tavily_key = os.environ.get("TAVILY_API_KEY")
print(f"[nodes] TAVILY_API_KEY loaded: {'YES (' + _tavily_key[:12] + '...)' if _tavily_key else 'NO - key is None!'}")
search_tool = TavilyClient(api_key=_tavily_key)


def research(state: ResearchState) -> dict:
    sub_queries = state["sub_queries"]
    refinement_count = state.get("refinement_count", 0)
    evaluation = state.get("evaluation", "")

    if evaluation.startswith("gaps:"):
        gaps = evaluation[len("gaps:"):].strip()
        previous_queries_str = "\n".join(f"- {q}" for q in sub_queries)
        prompt = REFINE_QUERIES_PROMPT.format(
            question=state["question"],
            gaps=gaps,
            previous_queries=previous_queries_str,
        )
        response = llm.invoke(prompt)
        lines = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
        sub_queries = []
        for line in lines:
            if line[1:3] in (". ", ") "):
                sub_queries.append(line[3:].strip())
            else:
                sub_queries.append(line)
        refinement_count += 1

    new_results = []
    for query in sub_queries:
        try:
            response = search_tool.search(query, max_results=3, search_depth="basic")
            results_list = response.get("results", [])
            print(f"[research] Query '{query[:60]}': {len(results_list)} results returned")
            for r in results_list:
                new_results.append({
                    "query": query,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],  # truncate to stay within token limits
                })
        except Exception as e:
            print(f"[research] Search FAILED for query '{query}': {type(e).__name__}: {e}")
            new_results.append({
                "query": query,
                "title": "",
                "url": "",
                "content": "Search failed for this query.",
            })

    existing_results = state.get("search_results") or []
    return {
        "sub_queries": sub_queries,
        "search_results": existing_results + new_results,
        "refinement_count": refinement_count,
    }


def synthesize_report(state: ResearchState) -> dict:
    search_results_str = "\n\n".join(
        f"Query: {r['query']}\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in state["search_results"]
    )
    prompt = SYNTHESIZE_REPORT_PROMPT.format(
        question=state["question"],
        search_results=search_results_str,
    )
    response = llm.invoke(prompt)
    return {"report": response.content.strip()}


def evaluate_quality(state: ResearchState) -> dict:
    sub_queries_str = "\n".join(f"- {q}" for q in state["sub_queries"])
    search_results_str = "\n\n".join(
        f"Query: {r['query']}\nTitle: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in state["search_results"]
    )
    prompt = EVALUATE_QUALITY_PROMPT.format(
        question=state["question"],
        sub_queries=sub_queries_str,
        search_results=search_results_str,
    )
    response = llm_short.invoke(prompt)
    return {"evaluation": response.content.strip()}
