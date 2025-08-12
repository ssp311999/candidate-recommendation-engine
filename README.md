## Candidate Recommendation Engine

A **Streamlit** web application that ranks uploaded candidate resumes against a given job description using semantic similarity.  
Optionally, it can generate **AI-powered fit summaries** explaining why each candidate is a strong match.

---

## Approach

- **Job Description Input** — Paste the full role overview, responsibilities, and requirements.
- **Resume Upload** — Upload multiple .txt resumes in bulk.
- **Semantic Ranking** — Uses SentenceTransformers (all-MiniLM-L6-v2) to generate embeddings and compute cosine similarity.
- **Top-K Selection** — Adjustable from 5 to 50 candidates.
- **Match Visualization** — Similarity scores displayed as progress bars in a sortable table.
- **Candidate Cards** — Expandable views showing:
    ~ AI-generated fit summary (optional)
    ~ Match score progress bar
    ~ Resume preview (first 1,200 characters)
- **AI Summaries (Optional)** — Uses OpenAI GPT models to generate 2–3 sentence “Why a good fit” explanations.
- **Downloadable Results** — Export rankings to CSV.

---

## How It Works

- **Embedding Generation** — The job description and resumes are converted into dense vector representations using the all-MiniLM-L6-v2 model.
- **Cosine Similarity** — Scores are computed by taking the dot product of normalized embeddings, giving a similarity score from -1.0 to 1.0 .
- **Sorting & Display** — Candidates are sorted by similarity (highest first) and displayed in a table and/or card format.
- **Optional Summaries** — If enabled, the app uses your OpenAI API key to call GPT models for short “fit summaries.”

---

## Approach

- **Resume Format**: Only plain text `.txt` files are supported (no PDF/DOCX parsing).
- **Language**: Works best with English resumes and job descriptions.
- **Content Quality**: The quality of ranking depends on the completeness and relevance of resume text.
- **API Key**: OpenAI API key required only for AI summaries. Store it in .streamlit/secrets.toml:
- **Embedding Model**: `all-MiniLM-L6-v2` is lightweight and fast; larger models could improve accuracy at the cost of performance.

---

## Notes on Performance & Scalability

- Embedding computations are cached for repeated runs with the same inputs.
- Optimized for small-to-medium batches (<100 resumes).
- Larger datasets may require:
    ~ Batch processing
    ~ More powerful models
    ~ Async/multiprocessing pipelines

---

## Deployment

- Works locally and on Streamlit Community Cloud.

---