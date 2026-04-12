DECOMPOSE_QUESTION_PROMPT = """You are a research assistant. Break the following question into 2-4 focused sub-queries that together cover all aspects of the question.

Each sub-query should:
- Address a distinct aspect of the original question
- Be specific enough to return useful search results
- Be phrased as a search query (not a question to an LLM)

Return ONLY a numbered list of sub-queries, one per line. No preamble, no explanation.

Question: {question}"""


EVALUATE_QUALITY_PROMPT = """You are a research quality evaluator. Assess whether the search results adequately answer the original question.

Original question: {question}

Sub-queries researched:
{sub_queries}

Search results collected:
{search_results}

Evaluate the results for:
- Completeness: are all aspects of the question covered?
- Missing perspectives: are there important angles not addressed?
- Contradictions: are there significant conflicting claims that need clarification?

If the results are sufficient to write a comprehensive answer, respond with exactly:
sufficient

If there are gaps, respond with exactly:
gaps: <brief description of what is missing or needs clarification>

Respond with one of those two options only. No other text."""


SYNTHESIZE_REPORT_PROMPT = """You are a research analyst. Using the search results below, write a structured research report that answers the original question.

Original question: {question}

Search results:
{search_results}

Write the report in this format:

# <Title>

## Executive Summary
<2-3 sentence overview of the key findings>

## Key Findings

<Organize findings into thematic sections with subheadings. Cite sources inline where relevant.>

## Sources
<Numbered list of sources: title and URL>

Be factual, balanced, and concise. Do not speculate beyond what the sources support."""


REFINE_QUERIES_PROMPT = """You are a research assistant. A first round of research was conducted but gaps were identified.

Original question: {question}

Gaps identified: {gaps}

Generate 2-3 new, targeted search queries specifically designed to fill these gaps. Do not repeat queries that have already been searched.

Already searched:
{previous_queries}

Return ONLY a numbered list of new sub-queries, one per line. No preamble, no explanation."""
