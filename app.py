import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from io import StringIO

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="Candidate Recommendation Engine",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stExpander"] > details {
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 10px;
  padding: .25rem .75rem;
  background: rgba(49,51,63,0.02);
}
[data-testid="stDataFrame"] thead tr th { font-weight: 600; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Caching & resources
# -------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    )
    return embs

@st.cache_data(show_spinner=False)
def summarize_fit_cached(resume_text: str, job_desc: str) -> str:
    try:
        import openai
        openai.api_key = st.secrets.get("OPENAI_API_KEY", "sk-REPLACE")
        prompt = f"""
        You are an assistant that explains why a resume is a good fit for a job.

        Job Description:
        {job_desc}

        Candidate Resume:
        {resume_text}

        Provide 2-3 sentences summarizing why this candidate is a strong fit for the job.
        """
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"

# -------------------------------
# Core logic
# -------------------------------

def rank_resumes(job_desc: str, resumes: List[str]) -> List[Tuple[int, float]]:
    job_emb = embed_texts([job_desc])[0]             
    resume_embs = embed_texts(resumes)               
    scores = resume_embs @ job_emb                   
    order = np.argsort(-scores)                      
    return [(int(i), float(scores[i])) for i in order]

# -------------------------------
# Streamlit app
# -------------------------------
def main():
    st.title("🔍 Candidate Recommendation Engine")
    st.caption("Rank uploaded resumes against a job description, then generate short fit summaries.")

    with st.sidebar:
        st.header("Settings")
        generate_summaries = st.toggle("Generate summaries", value=True, help="Turn off for fastest results.")
        default_topk = 10
        top_k = st.number_input(
            "Top K candidates",
            min_value=1, max_value=50, value=default_topk, step=1,
            help="How many candidates to display after ranking."
        )
        with st.expander("Advanced"):
            show_cards = st.toggle("Show candidate cards", value=True, help="Expandable per-candidate cards under the table.")
            show_table = st.toggle("Show results table", value=True, help="Sortable table with progress bars.")
            st.caption("Both views can be shown at once.")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        job_desc = st.text_area(
            "Job Description",
            placeholder="Paste the role overview, key responsibilities, and requirements…",
            height=220
        )

    with col_right:
        uploaded_files = st.file_uploader(
            "Upload Resumes (.txt)",
            type=["txt"], accept_multiple_files=True,
            help="Drag and drop multiple .txt resumes."
        )
        if uploaded_files:
            total_bytes = sum(getattr(f, "size", 0) or 0 for f in uploaded_files)
            st.metric("Files uploaded", f"{len(uploaded_files)}")
            st.caption(f"Total size: ~{total_bytes/1024:.1f} KB")

        if job_desc:
            words = len(job_desc.split())
            st.metric("Job description length", f"{words} words")
            st.progress(min(words / 1500, 1.0))

    st.divider()

    if not job_desc or not uploaded_files:
        st.info("Add a job description and upload one or more .txt resumes to begin.")
        st.stop()

    resumes, ids = [], []
    for f in uploaded_files:
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        resumes.append(text or "")
        ids.append(f.name)

    with st.status("Computing similarities…", expanded=False) as status:
        ranked = rank_resumes(job_desc, resumes)
        k = min(int(top_k), len(ranked))
        status.update(label=f"Ranking {len(resumes)} resumes…", state="running")

        results = []
        if generate_summaries:
            prog = st.progress(0, text="Creating summaries…")
        for n, (idx, score) in enumerate(ranked[:k], start=1):
            summary = summarize_fit_cached(resumes[idx], job_desc) if generate_summaries else ""
            results.append({
                "Candidate": ids[idx],
                "Similarity": float(score),
                "Why a good fit": summary
            })
            if generate_summaries:
                prog.progress(n / k, text=f"Creating summaries… ({n}/{k})")

        status.update(label="Done!", state="complete")

    st.success(f"Top {k} candidate{'s' if k != 1 else ''}:")

    df = pd.DataFrame(results)

    if st.session_state.get("show_table", True) if False else True:
        pass

    if 'show_table' not in locals():
        show_table = True
    if 'show_cards' not in locals():
        show_cards = True

    if show_table:
        score_min = float(df["Similarity"].min()) if not df.empty else 0.0
        score_max = float(df["Similarity"].max()) if not df.empty else 1.0
        denom = (score_max - score_min) if (score_max - score_min) > 1e-9 else 1.0
        df["_score_norm"] = ((df["Similarity"] - score_min) / denom).clip(0, 1)

        st.dataframe(
            df[["Candidate", "_score_norm", "Why a good fit", "Similarity"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Candidate": st.column_config.TextColumn(help="Uploaded file name"),
                "_score_norm": st.column_config.ProgressColumn(
                    "Match",
                    help="Similarity scaled within the shown set",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=1.0
                ),
                "Why a good fit": st.column_config.TextColumn(width="medium"),
                "Similarity": st.column_config.NumberColumn(format="%.3f", help="Raw cosine similarity")
            }
        )

    if show_cards:
        st.subheader("Candidate cards")
        for row in df.itertuples(index=False):
            with st.expander(f"{row.Candidate}  •  similarity {row.Similarity:.3f}", expanded=False):
                try:
                    score_norm = float(getattr(row, "_score_norm", 0.0))
                except Exception:
                    score_norm = 0.0
                st.progress(score_norm)
                if row._asdict().get("Why a good fit"):
                    st.markdown(row._asdict()["Why a good fit"])
                with st.popover("Preview resume (first 1200 chars)"):
                    idx = ids.index(row.Candidate)
                    preview = resumes[idx][:1200]
                    st.code(preview or "[empty]", language="markdown")

    csv_buf = StringIO()
    (df.drop(columns=["_score_norm"], errors="ignore")).to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_buf.getvalue(),
        file_name="candidate_rankings.csv",
        mime="text/csv"
    )
    st.caption("Tip: For faster runs, disable summaries. For large batches, keep only the table view.")

if __name__ == "__main__":
    main()