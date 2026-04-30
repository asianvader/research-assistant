from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from graph import build_graph


def _handle_updates(chunk, status, research_placeholder, all_search_results, refinement_count):
    report_text = None

    if "decompose" in chunk:
        sub_queries = chunk["decompose"].get("sub_queries", [])
        with st.expander("Sub-queries", expanded=False):
            for q in sub_queries:
                st.write(f"- {q}")
        status.update(label="Searching the web...")

    if "research" in chunk:
        # Node returns the full accumulated list, so assign (not extend).
        all_search_results = chunk["research"].get("search_results", all_search_results)
        refinement_count = chunk["research"].get("refinement_count", refinement_count)
        with research_placeholder.container():
            with st.expander(
                f"Search results ({len(all_search_results)} found)", expanded=False
            ):
                for r in all_search_results:
                    st.markdown(
                        f"**{r['title']}**  \n{r['url']}  \n{r['content'][:200]}..."
                    )
        if refinement_count > 0:
            status.update(label=f"Refining search... iteration {refinement_count}/2")
        else:
            status.update(label="Evaluating results...")

    if "evaluate" in chunk:
        evaluation = chunk["evaluate"].get("evaluation", "")
        with st.expander("Evaluation", expanded=False):
            st.write(evaluation)
        if evaluation.startswith("gaps:"):
            status.update(label=f"Refining search... iteration {refinement_count + 1}/2")
        else:
            status.update(label="Synthesizing report...")

    if "synthesize" in chunk:
        report_text = chunk["synthesize"].get("report", "")
        if refinement_count > 0:
            with st.expander("Refinement", expanded=False):
                st.write(f"Completed {refinement_count}/2 refinement iteration(s).")

    return all_search_results, refinement_count, report_text


st.title("Research Assistant")
st.caption("Ask a complex question and get a structured research report.")

question = st.text_area("What would you like to research?", height=100)

if st.button("Research", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        graph = build_graph()
        all_search_results = []
        refinement_count = 0
        report_text = ""
        initial_state = {"question": question, "refinement_count": 0}

        try:
            with st.status("Decomposing question...", expanded=True) as status:
                research_placeholder = st.empty()
                streaming_report_placeholder = st.empty()

                try:
                    for mode, chunk in graph.stream(
                        initial_state, stream_mode=["updates", "messages"]
                    ):
                        if mode == "updates":
                            all_search_results, refinement_count, new_report = _handle_updates(
                                chunk, status, research_placeholder,
                                all_search_results, refinement_count,
                            )
                            if new_report is not None:
                                report_text = new_report

                        elif mode == "messages":
                            msg_chunk, metadata = chunk
                            if (
                                metadata.get("langgraph_node") == "synthesize"
                                and msg_chunk.content
                            ):
                                report_text += msg_chunk.content
                                streaming_report_placeholder.markdown(
                                    f"*Generating report...*\n\n{report_text}"
                                )

                except TypeError:
                    # Fallback: list stream_mode unsupported; node-level updates only.
                    all_search_results = []
                    refinement_count = 0
                    report_text = ""
                    research_placeholder = st.empty()
                    for chunk in graph.stream(initial_state, stream_mode="updates"):
                        all_search_results, refinement_count, new_report = _handle_updates(
                            chunk, status, research_placeholder,
                            all_search_results, refinement_count,
                        )
                        if new_report is not None:
                            report_text = new_report

                streaming_report_placeholder.empty()
                status.update(label="Research complete!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Research failed: {e}")
            st.stop()

        st.markdown("## Report")
        st.markdown(report_text if report_text else "No report generated.")
