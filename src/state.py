from typing import TypedDict


class ResearchState(TypedDict):
    question: str           # Original user question
    sub_queries: list[str]  # Decomposed sub-queries
    search_results: list[dict]  # Raw search results
    evaluation: str         # Quality evaluation ("sufficient" or "gaps: ...")
    refinement_count: int   # Track loop iterations (max 2)
    report: str             # Final synthesized report
