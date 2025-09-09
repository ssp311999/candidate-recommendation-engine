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
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- minimal CSS polish ----------
st.markdown("""
<style>
.main .block-container { padding-top: 1.6rem; max-width: 1300px; }
.card { background: var(--secondary-background-color); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 16px 18px; }
.card h4 { margin: 0 0 0.4rem 0; }
.stButton>button { 
  border-radius: 10px; padding: 10px 16px; font-weight: 600;
  background: linear-gradient(135deg,#6aa0ff,#7c9cff,#a58bff);
  border: none;
}
.stButton>button:hover { filter: brightness(1.05); }
section[data-testid="stFileUploaderDropzone"] { min-height: 160px; }
.small { opacity: .75; font-size: .9rem; }
.kpi { font-size: 0.95rem; opacity: .8; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Model + helpers
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    # L2-normalized embeddings -> dot product == cosine similarity
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False
    )
    return embs

def rank_resumes(job_desc: str, resumes: List[str]) -> List[Tuple[int, float]]:
    job_emb = embed_texts([job_desc])[0]
    resume_embs = embed_texts(resumes)
    scores = resume_embs @ job_emb  # cosine similarity because normalized
    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order]

@st.cache_data(show_spinner=False)
def summarize_fit_cached(resume_text: str, job_desc: str) -> str:
    try:
        from openai import OpenAI
        import streamlit as st

        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return "[Summary unavailable: missing OPENAI_API_KEY in secrets]"

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are an assistant that explains why a resume is a good fit for a job.

Job Description:
{job_desc}

Candidate Resume:
{resume_text}

Provide 2–3 sentences summarizing why this candidate is a strong fit.
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"[Summary unavailable: {e}]"

# -------------------------------
# App
# -------------------------------
def main():
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Top‑K candidates", 5, 50, 10, step=5)
    generate_summaries = st.sidebar.toggle("Generate AI fit summaries", value=False, help="Uses your API key if enabled")
    show_cards = st.sidebar.toggle("Show candidate cards", value=True)
    show_table = st.sidebar.toggle("Show results table", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: you can drag‑drop multiple .txt resumes.")

    # Header
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown("### 🕵️ Candidate Recommendation Engine")
        st.caption("Rank uploaded resumes against a job description using semantic similarity. Optionally generate short fit summaries.")
    # with col2:
    #     st.metric("Status", "Idle", help="Will update after ranking")

    st.markdown("")

    # Inputs (two columns)
    left, right = st.columns([1.6, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Job Description")
        job_desc = st.text_area(
            "Paste the role overview, responsibilities, and requirements…",
            label_visibility="collapsed",
            height=260,
            placeholder="Paste the job description here…"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Upload Resumes (.txt)")
        uploaded_files = st.file_uploader(
            "Drag and drop files here or click Browse",
            type=["txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            st.markdown("**Selected files:**")
            for f in uploaded_files[:8]:
                st.markdown(f"- ✅ `{f.name}`")
            if len(uploaded_files) > 8:
                st.markdown(f"*…and {len(uploaded_files)-8} more*")
        else:
            st.caption("No files yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Action
    action_col, _ = st.columns([1, 3])
    with action_col:
        run = st.button("Rank Candidates", use_container_width=False)

    st.markdown("---")

    # Guard
    if not run:
        st.info("Add a job description and upload one or more `.txt` resumes, then click **Rank Candidates**.")
        return
    if not job_desc or not uploaded_files:
        st.warning("Please provide both a job description and at least one resume.")
        return

    # Read resumes
    resumes, ids = [], []
    for f in uploaded_files:
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        resumes.append(text or "")
        ids.append(f.name)

    # Ranking + (optional) summaries
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

    # Normalize score for progress bars
    score_min = float(df["Similarity"].min()) if not df.empty else 0.0
    score_max = float(df["Similarity"].max()) if not df.empty else 1.0
    denom = (score_max - score_min) if (score_max - score_min) > 1e-9 else 1.0
    df["_score_norm"] = ((df["Similarity"] - score_min) / denom).clip(0, 1)

    if show_table:
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
                st.progress(float(getattr(row, "_score_norm", 0.0)))
                if getattr(row, "Why a good fit", ""):
                    st.markdown(getattr(row, "Why a good fit"))
                # quick preview
                try:
                    idx = ids.index(row.Candidate)
                    preview = resumes[idx][:1200]
                except Exception:
                    preview = ""
                with st.popover("Preview resume (first 1200 chars)"):
                    st.code(preview or "[empty]", language="markdown")

    # Download
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
