from dotenv import load_dotenv
load_dotenv()

import uuid
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from graph import build_graph

st.title("Research Assistant")
st.caption("Ask a complex question and get a structured research report.")

question = st.text_area("What would you like to research?", height=100)

if st.button("Research", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        if "checkpointer" not in st.session_state:
            st.session_state.checkpointer = MemorySaver()

        graph = build_graph(checkpointer=st.session_state.checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        with st.status("Researching...", expanded=True) as status:
            result = graph.invoke(
                {"question": question, "refinement_count": 0},
                config=config,
            )

            with st.expander("Sub-queries", expanded=False):
                for q in result.get("sub_queries", []):
                    st.write(f"- {q}")

            with st.expander(f"Search results ({len(result.get('search_results', []))} found)", expanded=False):
                for r in result.get("search_results", []):
                    st.markdown(f"**{r['title']}**  \n{r['url']}  \n{r['content'][:200]}...")

            evaluation = result.get("evaluation", "")
            with st.expander("Evaluation", expanded=False):
                st.write(evaluation)

            refinement_count = result.get("refinement_count", 0)
            if refinement_count > 0:
                with st.expander("Refinement", expanded=False):
                    st.write(f"Completed {refinement_count}/2 refinement iteration(s).")

            status.update(label="Research complete!", state="complete")

        st.markdown("## Report")
        st.markdown(result.get("report", "No report generated."))
