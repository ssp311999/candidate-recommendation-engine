## Candidate Recommendation Engine

A **Streamlit** web application that ranks uploaded candidate resumes against a given job description using semantic similarity.  
Optionally, it can generate **AI-powered fit summaries** to explain why each candidate is a good match.

---

## Approach

- **Job Description Input** — Paste the full role overview, responsibilities, and requirements.
- **Resume Upload** — Upload multiple `.txt` files containing candidate resumes.
- **Semantic Ranking** — Uses [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) to generate embeddings and compute cosine similarity.
- **Top-K Selection** — View the top 5–50 ranked candidates (default: 10).
- **Match Visualization** — See similarity scores as progress bars in a sortable table.
- **Candidate Cards** — Expandable sections showing fit summary, match score, and resume preview.
- **AI Summaries (Optional)** — Generate 2–3 sentence "Why a good fit" explanations using OpenAI GPT models.
- **Downloadable Results** — Export rankings to CSV for offline review.

---

## How It Works

- **Embedding Generation** — Converts job description and resumes into dense vector representations.
- **Cosine Similarity** — Measures semantic alignment between job and resume embeddings.
- **Sorting & Display** — Ranks candidates by similarity score; displays results in table and card formats.
- **Optional Summaries** — GPT-generated explanations provide context on why a candidate matches.

---

## Approach

- **Resume Format**: Only plain text `.txt` files are supported (no PDF/DOCX parsing).
- **Language**: Assumes resumes and job descriptions are in English.
- **Content Quality**: The quality of ranking depends on the completeness and relevance of resume text.
- **API Key**: OpenAI API key must be stored in `.streamlit/secrets.toml` for summaries; otherwise, summaries are skipped.
- **Embedding Model**: `all-MiniLM-L6-v2` is lightweight and fast; larger models could improve accuracy at the cost of performance.

---

## Notes

- **Performance**: Embedding computation is cached for repeated runs with identical inputs.
- **Scalability**: Designed for small-to-medium batches (e.g., <100 resumes). Large datasets may require batching or async processing.
- **Deployment**: Works locally and on Streamlit Community Cloud.
- **Export**: Rankings can be downloaded as CSV.

---
