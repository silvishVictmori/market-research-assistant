# Market Research Assistant (Wikipedia + Streamlit)

A lightweight Streamlit web app that helps you:
1. **Validate** whether a user input looks like a real “industry” (basic keyword/signal check)
2. **Retrieve and display** the **top 5 relevant Wikipedia pages (URLs)** for that industry (grounding)
3. Generate a **short industry report (<500 words)** based on Wikipedia snippets, including:
   - Quick definitions (one-line summaries)
   - Key themes (scope, value chain, enablers, ecosystem)
   - Current dynamics (demand/supply/differentiation)
   - Grounding extracts (short snippets from the retrieved pages)

## How it works (3-step flow)
- **Step 1 — Input:** Enter an industry term (e.g., *automotive industry*, *fintech*, *pharmaceuticals*).
- **Step 2 — Sources:** The app retrieves Wikipedia matches and shows **five distinct URLs**.
- **Step 3 — Report:** The app builds a structured report from the retrieved page text and enforces a **500-word limit**.

## Features / improvements implemented
- **Query disambiguation:** broad inputs are expanded (e.g., `"car"` → `"car industry"`).
- **Industry-signal filtering:** results are filtered using industry-related keywords to reduce irrelevant pages.
- **Text cleaning:** removes Wikipedia artifacts (e.g., spaced letters/digits).
- **Consistent outputs:** docs and URLs are reused across steps using `st.session_state`.
- **Word limit control:** report is trimmed to stay under 500 words without cutting mid-sentence.

## Run locally
### 1) Install dependencies
```bash
pip install streamlit langchain-community
```

### 2) Start the app
```bash
streamlit run ML.py
```

## Deploy on Streamlit Community Cloud
1. Push your code to GitHub (public repo is simplest; private repos require granting Streamlit access).
2. In Streamlit Cloud, click **Deploy an app**.
3. Select:
   - **Repository:** `YOUR_USERNAME/market-research-assistant`
   - **Branch:** `main` (or your repo’s default branch)
   - **Main file path:** `ML.py`

## Notes / limitations
- Keyword heuristics may misclassify ambiguous inputs.
- Wikipedia relevance varies, especially for broad queries.
- Custom CSS spacing may render differently across Streamlit versions/themes.
- Retrieval can be slow depending on network/Wikipedia response (caching recommended).
