import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever

st.set_page_config(page_title="Industry Wikipedia Assistant", layout="wide")

with st.sidebar:
    st.markdown("## Settings")
    st.caption("Provide your own API keys (stored only for this session).")

    # LangSmith (optional). Safe default: app still works without it.
    langsmith_key = st.text_input(
        "LangSmith API key (optional)",
        type="password",
        placeholder="ls__...",
        help="Used only for tracing/monitoring. Not required to run the app.",
    )
    langsmith_project = st.text_input(
        "LangSmith project name (optional)",
        value="industry-assistant",
        help="Only used if a LangSmith API key is provided.",
    )
    enable_tracing = st.toggle(
        "Enable LangSmith tracing",
        value=False,
        help="Only applies when LangSmith API key is provided.",
    )

    st.divider()
    st.markdown("### Wikipedia settings")
    top_k = st.slider(
        "Retriever top_k_results",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Higher values may improve relevance but can be slower.",
    )

if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    os.environ["LANGSMITH_PROJECT"] = langsmith_project or "industry-assistant"
    os.environ["LANGSMITH_TRACING"] = "true" if enable_tracing else "false"
else:
    # Ensure these defaults don't break runs on machines without LangSmith
    os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ.pop("LANGSMITH_PROJECT", None)
    os.environ.pop("LANGSMITH_TRACING", None)
    
# validate input

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def has_real_text(s: str) -> bool:
    return bool(s and re.search(r"[A-Za-z0-9]", s.strip()))

# validate if it is a real industry

INDUSTRY_SIGNALS = {
    # core
    "industry", "industries", "industrial", "sector", "sectors", "market", "markets",
    "business", "commerce", "commercial", "trade",

    # value chain / operations
    "supply chain", "value chain", "manufacturing", "production", "distribution",
    "wholesale", "retail", "logistics", "procurement",

    # finance / market structure
    "competition", "competitor", "pricing", "revenue", "profit", "cost",
    "demand", "supply", "market share",

    # regulation / standards
    "regulation", "regulated", "compliance", "standard", "standards",

    # common sector labels
    "services", "service industry", "financial services", "healthcare", "insurance",
    "banking", "telecommunications", "energy", "oil and gas", "mining",
    "construction", "transport", "transportation", "automotive", "pharmaceutical",
    "biotechnology", "aerospace", "defense", "agriculture", "food industry",
    "hospitality", "tourism",
}


def score_industry_signals(text: str) -> int:
    #Count how many signal phrases appear in the given text.
    t = normalize(text)
    return sum(1 for sig in INDUSTRY_SIGNALS if sig in t)

def validate_industry_with_wikipedia(
    industry: str,
    retriever: WikipediaRetriever,
    min_results: int = 5,
    min_signal_docs: int = 2,
    min_signal_hits_per_doc: int = 1,
):
    """
    Returns (is_valid, message, docs, title_url_pairs)

    Valid if:
    - non-empty input
    - retriever returns >= min_results docs
    - we can extract exactly 5 distinct URLs (Step 2 requirement)
    - "industry-ness": at least `min_signal_docs` of the top docs have >= `min_signal_hits_per_doc`
      signal matches in title+snippet OR the query itself contains signals.

    Improvements:
    - Disambiguates broad queries by appending "industry" when needed
    - Filters out irrelevant docs using signal hits
    """
    if not has_real_text(industry):
        return False, "Step 1: Please enter an industry (required).", [], []

    # Disambiguate broad queries
    norm = normalize(industry)
    query = industry
    if all(k not in norm for k in ["industry", "sector", "market"]):
        query = f"{industry} industry"

    # Retrieve
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)

    if not docs:
        return (
            False,
            f"Step 1: No Wikipedia results for '{industry}'. Try a clearer industry term.",
            [],
            [],
        )

    # Filter docs for industry relevance (title + snippet)
    filtered = []
    for d in docs:
        md = d.metadata or {}
        title = md.get("title") or md.get("page_title") or ""
        snippet = (d.page_content or "")[:500]
        if score_industry_signals(f"{title} {snippet}") >= min_signal_hits_per_doc:
            filtered.append(d)

    # Use filtered if it gives enough results; otherwise fall back to original docs
    docs_to_use = filtered if len(filtered) >= min_results else docs

    if len(docs_to_use) < min_results:
        return (
            False,
            f"Step 1: Not enough Wikipedia results for '{industry}'. Try a clearer industry term "
            "(e.g., 'semiconductor industry', 'banking sector').",
            docs_to_use,
            [],
        )

    # Extract exactly 5 distinct URLs (Step 2 requirement)
    title_url_pairs = extract_5_urls(docs_to_use)
    if len(title_url_pairs) < 5:
        return (
            False,
            f"Step 1: I couldn’t extract 5 distinct Wikipedia URLs for '{industry}'. "
            "Try adding 'industry', 'sector', or a more specific term.",
            docs_to_use,
            title_url_pairs,
        )

    # Signal check in query
    query_score = score_industry_signals(industry)

    # Signal check in top docs (title + first part of page content)
    signal_docs = 0
    for d in docs_to_use[:min_results]:
        md = d.metadata or {}
        title = md.get("title") or md.get("page_title") or ""
        snippet = (d.page_content or "")[:500]
        hits = score_industry_signals(f"{title} {snippet}")
        if hits >= min_signal_hits_per_doc:
            signal_docs += 1

    if query_score == 0 and signal_docs < min_signal_docs:
        return (
            False,
            f"Step 1: '{industry}' doesn’t look like an industry term based on Wikipedia results. "
            "Try phrasing it like 'X industry' / 'X sector' / 'X market'.",
            docs_to_use,
            title_url_pairs,
        )

    return True, "OK", docs_to_use, title_url_pairs

#Return up to 5 distinct URLs from doc metadata.
def extract_5_urls(docs) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen_urls = set()

    for d in docs:
        md = d.metadata or {}
        url = md.get("source") or md.get("url")
        title = md.get("title") or md.get("page_title") or "Wikipedia page"
        if url and url not in seen_urls:
            out.append((title, url))
            seen_urls.add(url)
        if len(out) == 5:
            break
    
    return out

# Utility functions for processing text and building the report
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)

def word_count(text: str) -> int:
    return len((text or "").split())

def add_sentences_up_to_budget(existing: str, candidate: str, remaining_words: int) -> str:
    if remaining_words <= 0:
        return existing

    out = existing
    for s in split_sentences(candidate):
        s = s.strip()
        if not s:
            continue

        sep = "\n" if out and not out.endswith("\n") else ""
        attempt = out + sep + s
        if word_count(attempt) <= word_count(out) + remaining_words:
            out = attempt
        else:
            break

    return out


def enforce_lt_500_words_complete(report_sections: list[str], max_words: int = 500) -> str:
    report = ""
    for section in report_sections:
        remaining = max_words - word_count(report)
        report = add_sentences_up_to_budget(report, section, remaining)
        if word_count(report) >= max_words:
            break
    return report.strip()

def take_bullets_up_to_words(bullets: list[str], max_words: int) -> str:
    out_lines = []
    current = 0
    for b in bullets:
        b_words = len(b.split())
        # +1 word buffer (roughly) for newline separation
        if current + b_words <= max_words:
            out_lines.append(b)
            current += b_words
        else:
            break
    return "\n".join(out_lines)

def clean_wiki_text(text: str) -> str:
    t = (text or "").strip()

    # normalize all whitespace
    t = re.sub(r"\s+", " ", t)

    # collapse spaces between digits and punctuation in numbers
    t = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", t)
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)

    # join sequences of single-letter tokens even if mixed with stray spaces:
    def join_spelled(m):
        return m.group(0).replace(" ", "")

    t = re.sub(r"(?:\b[A-Za-z]\b\s+){2,}\b[A-Za-z]\b", join_spelled, t)

    # clean spaces before punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)

    return t

def doc_text(d) -> str:
    md = d.metadata or {}
    # Try common metadata fields first (if present)
    for k in ["summary", "description", "snippet", "content"]:
        if md.get(k):
            return str(md.get(k))
    # Fallback to page_content
    return d.page_content or ""
    
def build_report(industry: str, docs) -> str:
    titles: list[str] = []
    quick_defs: list[str] = []
    extracts: list[str] = []

    for d in docs[:5]:
        md = d.metadata or {}
        title = md.get("title") or md.get("page_title") or "Wikipedia page"
        titles.append(title)

        content = clean_wiki_text(doc_text(d))
        if not content:
            continue

        # Prefer first sentence for definition (more "complete" than N words)
        sentences = split_sentences(content)
        if sentences:
            first_sentence = sentences[0].strip()
            # Keep definition concise
            def_words = first_sentence.split()[:28]
            quick_defs.append(f"- {title}: {' '.join(def_words).rstrip('.')}.")
        else:
            # fallback
            quick_defs.append(f"- {title}: {' '.join(content.split()[:28]).rstrip('.') }.")

        # Extracts: take 1–2 sentences to keep coherence
        if len(sentences) >= 2:
            extract_text = f"{sentences[0].strip()} {sentences[1].strip()}"
        elif sentences:
            extract_text = sentences[0].strip()
        else:
            extract_text = content
        extracts.append(f"- {title}: {' '.join(extract_text.split()[:60]).rstrip('.') }.")


    defs_block = take_bullets_up_to_words(quick_defs, max_words=130)
    extracts_block = take_bullets_up_to_words(extracts, max_words=220)

    narrative = (
        f"**Industry report:** {industry}\n\n"

        f"**Overview**\n\n"
        f"Grounded Wikipedia topics (top matches): {', '.join(titles)}.\n\n"
        "This report summarizes the industry’s definition, typical structure, and recurring themes as described by those pages.\n\n"
    
        "**Notable subtopics (quick definitions)**\n\n"
        f"{defs_block if defs_block else '- (No extract text available.)'}\n\n"
    
        "**Key themes**\n\n"
        "- Scope & definition: What the industry includes, its core activities, and common boundaries.\n"
        "- Value chain: Typical upstream inputs, production/service delivery, and downstream customers/end users.\n"
        "- Enablers: Technologies, infrastructure, standards, and processes that commonly show up across the pages.\n"
        "- Ecosystem: Adjacent sectors, institutions, and public/private actors that influence outcomes.\n\n"
    
        "**Current dynamics**\n\n"
        "- Demand-side: The use-cases and adoption contexts implied by the descriptions (who uses the outputs and why).\n"
        "- Supply-side: Inputs, operational complexity, scaling constraints, and typical bottlenecks suggested by the coverage.\n"
        "- Differentiation: Where competition tends to focus (cost, performance, reliability, compliance, and distribution).\n\n"
    
        "**Grounding extracts**\n\n"
        f"{extracts_block if extracts_block else '- (No extract text available.)'}"
    )

    final_report = narrative.strip()

    words = final_report.split()
    if len(words) > 500:
        final_report = " ".join(words[:500])

    return final_report


#streamlit UI construction

st.set_page_config(page_title="Industry Wikipedia Assistant", layout="wide")

st.markdown(
    """
<style>
.main-title {font-size: 2rem; font-weight: 800; margin: 0 0 0.25rem 0;}
.subtle {color: #6b7280; margin: 0 0 0.75rem 0;}
.card {padding: 1rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #ffffff;}
.small {font-size: 0.9rem;}
/* Reduce empty space above widgets */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextInput"]) {margin-top: -0.5rem;}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown('<div class="main-title">Market Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Wikipedia Retriever + Streamlit web UI</p>', unsafe_allow_html=True)


# Purpose: keep track of which step the user is currently on.
if "step" not in st.session_state:
    st.session_state.step = 1

if "docs" not in st.session_state:
    st.session_state.docs = None
if "title_url_pairs" not in st.session_state:
    st.session_state.title_url_pairs = None
if "industry" not in st.session_state:
    st.session_state.industry = ""

# Layout: left column inputs, right column outputs.
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Step 1 — Enter an industry (required)")
    industry_input = st.text_input(
        "Industry",
        value=st.session_state.industry,
        placeholder="e.g., automotive industry, fintech, pharmaceuticals",
        label_visibility="collapsed",
    )

    run = st.button("Find sources (Step 2)", type="primary", use_container_width=True)

    st.markdown("### Status")
    step_text = st.empty()
    progress = st.progress(0)
    step_text.write(f"Current step: {st.session_state.step} / 3")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")

    # -----------------
    # Clicking run moves to Step 2 (URLs page)
    # -----------------
    if run:
        st.session_state.industry = industry_input
        st.session_state.step = 1
        step_text.write(f"Current step: {st.session_state.step} / 3")
        progress.progress(0)

        retriever = WikipediaRetriever(lang="en", top_k_results=10)
        is_valid, msg, docs, title_url_pairs = validate_industry_with_wikipedia(
            st.session_state.industry, retriever, min_results=5
        )
        if not is_valid:
            st.error(msg)
        else:
            st.session_state.docs = docs
            st.session_state.title_url_pairs = title_url_pairs
            st.session_state.step = 2

 
        # STEP 2: Show URLs of five most relevant pages
    if st.session_state.step == 1:
        step_text.write("Current step: 1 / 3")
        progress.progress(0)
        st.info("Enter an industry on the left and click **Find sources (Step 2)**.")

    elif st.session_state.step == 2:
        step_text.write("Current step: 2 / 3")
        progress.progress(40)

        st.subheader("Step 2 — Five most relevant Wikipedia pages (URLs)")
        pairs = st.session_state.title_url_pairs or []
        for i, (title, url) in enumerate(pairs[:5], start=1):
            st.write(f"{i}. [{title}]({url}) — {url}")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Proceed to detailed industry report (Step 3)", type="primary", use_container_width=True):
                st.session_state.step = 3
        with col_b:
            if st.button("Back to Step 1", use_container_width=True):
                st.session_state.step = 1

# Build report based on the retrieved pages
    elif st.session_state.step == 3:
        step_text.write("Current step: 3 / 3")
        progress.progress(75)

        st.subheader("Step 3 — Industry report (<500 words)")
        docs = st.session_state.docs or []
        report = build_report(st.session_state.industry, docs[:5])

        progress.progress(100)
        st.caption(f"Word count: {len(report.split())}")
        st.markdown(report)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Back to Wikipedia URLs (Step 2)", use_container_width=True):
                st.session_state.step = 2
        with col_b:
            if st.button("Start over (Step 1)", use_container_width=True):
                st.session_state.step = 1

    st.markdown("</div>", unsafe_allow_html=True)