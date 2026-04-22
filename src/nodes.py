from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

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


def decompose_question(state: ResearchState) -> dict:
    prompt = DECOMPOSE_QUESTION_PROMPT.format(question=state["question"])
    response = llm.invoke(prompt)
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
