from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from datetime import date
import argparse
import os
import re
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv
import openai

from relevancy import generate_relevance_score
from download_new_papers import get_papers


topics = {
    "Physics": "",  # special-cased below
    "Mathematics": "math",
    "Computer Science": "cs",
    "Quantitative Biology": "q-bio",
    "Quantitative Finance": "q-fin",
    "Statistics": "stat",
    "Electrical Engineering and Systems Science": "eess",
    "Economics": "econ",
}

physics_topics = {
    "Astrophysics": "astro-ph",
    "Condensed Matter": "cond-mat",
    "General Relativity and Quantum Cosmology": "gr-qc",
    "High Energy Physics - Experiment": "hep-ex",
    "High Energy Physics - Lattice": "hep-lat",
    "High Energy Physics - Phenomenology": "hep-ph",
    "High Energy Physics - Theory": "hep-th",
    "Mathematical Physics": "math-ph",
    "Nonlinear Sciences": "nlin",
    "Nuclear Experiment": "nucl-ex",
    "Nuclear Theory": "nucl-th",
    "Physics": "physics",
    "Quantum Physics": "quant-ph",
}


def is_valid_arxiv_id(s: str) -> bool:
    """Valid arXiv IDs: YYMM.NNNNN (or YYMM.NNNN) with optional vN; or archive/YYMMNNN with optional vN."""
    s = (s or "").strip()
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s):
        return True
    if re.fullmatch(r"[a-z\-]+/\d{7}(v\d+)?", s):
        return True
    return False


def clean_title(raw: str) -> str:
    t = (raw or "").strip()
    # remove repeated "Title:" prefixes (sometimes duplicated)
    while True:
        new = re.sub(r"^\s*Title:\s*", "", t).strip()
        if new == t:
            break
        t = new
    return t


def safe_link_for_paper(paper: dict) -> str:
    """
    ALWAYS returns a working link.
    - If we can extract a VALID arXiv id -> https://arxiv.org/abs/<id>
    - Otherwise -> arXiv title search URL
    """
    title = clean_title(paper.get("title", ""))
    url = (paper.get("main_page") or "").strip()

    # Normalize relative URLs
    if url.startswith("/"):
        url = "https://arxiv.org" + url

    # If it's an arXiv abs/pdf URL, validate the ID part
    m = re.search(r"arxiv\.org/(abs|pdf)/([^?#/]+)", url)
    if m:
        arxiv_id = m.group(2).replace(".pdf", "").strip()
        if is_valid_arxiv_id(arxiv_id):
            return "https://arxiv.org/abs/" + arxiv_id
        # If it's arXiv but NOT a valid id, DO NOT return it (this was the bug)
        url = ""

    # If url is a bare arXiv id
    if url and not url.startswith("http") and is_valid_arxiv_id(url):
        return "https://arxiv.org/abs/" + url

    # Final fallback: title search (never gives "Invalid article identifier")
    return "https://arxiv.org/search/?query=" + quote_plus(title) + "&searchtype=title"


def render_paper_html(p: dict, include_score: bool) -> str:
    title = clean_title(p.get("title", ""))
    authors = p.get("authors", "")
    link = safe_link_for_paper(p)

    extra = ""
    if include_score:
        score = p.get("Relevancy score", "")
        reason = p.get("Reasons for match", "")
        if score != "" or reason != "":
            extra = f"<br>Score: {score}<br>Reason: {reason}"

    return f'Title: <a href="{link}">{title}</a><br>Authors: {authors}{extra}'


def generate_body(topic, categories, interest, threshold, fallback_n=15):
    categories = categories or []
    threshold = int(threshold)

    # ---- Download papers ----
    if topic == "Physics":
        # Map human-readable -> physics.* codes
        physics_map = {
            "Applied Physics": "physics.app-ph",
            "Physics Education": "physics.ed-ph",
            "History and Philosophy of Physics": "physics.hist-ph",
            "Instrumentation and Detectors": "physics.ins-det",
            "Optics": "physics.optics",
        }
        if not categories:
            raise RuntimeError(
                "For topic 'Physics', you must provide at least one category (e.g. ['Optics'])."
            )

        physics_codes = []
        for c in categories:
            if c in physics_map:
                physics_codes.append(physics_map[c])
            elif isinstance(c, str) and c.startswith("physics."):
                physics_codes.append(c)
            else:
                raise RuntimeError(
                    f"Unknown Physics category '{c}'. Use one of {list(physics_map.keys())}."
                )

        # Pull directly from the subpages so we don't mix in unrelated physics
        papers = []
        seen = set()
        for code in physics_codes:
            for p in get_papers(code):
                key = (p.get("main_page") or "") + "||" + (p.get("title") or "")
                if key not in seen:
                    seen.add(key)
                    papers.append(p)

    elif topic in physics_topics:
        papers = get_papers(physics_topics[topic])

    elif topic in topics:
        papers = get_papers(topics[topic])

    else:
        raise RuntimeError(f"Invalid topic {topic}")

    if not papers:
        return "<p>No papers found for this run.</p>"

    # ---- Scoring / fallback ----
    if interest and str(interest).strip():
        used_fallback = False
        scored = []

        try:
            scored, _ = generate_relevance_score(
                papers,
                query={"interest": interest},
                threshold_score=threshold,
                num_paper_in_prompt=4,  # you already reduced; keep it
            )
        except Exception:
            # If LLM output/parsing is flaky, do NOT crash — fallback
            used_fallback = True
            scored = []

        final_list = scored
        if not final_list:
            used_fallback = True
            final_list = papers[:fallback_n]

        parts = []
        if used_fallback:
            parts.append(
                f"No papers exceeded the relevance threshold ({threshold}) for this run. "
                f"Showing the {fallback_n} most recent papers instead.<br><br>"
            )

        parts.append(
            "<br><br>".join(render_paper_html(p, include_score=not used_fallback) for p in final_list)
        )
        return "".join(parts)

    # No interest => raw list
    return "<br><br>".join(render_paper_html(p, include_score=False) for p in papers)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="yaml config file to use", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("No openai api key found")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    topic = config.get("topic", "")
    categories = config.get("categories", [])
    threshold = config.get("threshold", 7)
    interest = config.get("interest", "")

    body = generate_body(topic, categories, interest, threshold)

    with open("digest.html", "w", encoding="utf-8") as f:
        f.write(body)

    # This sendgrid block is mainly for local runs; in your GitHub Action you send later.
    if os.environ.get("SENDGRID_API_KEY"):
        sg = SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
        from_email = Email(os.environ.get("FROM_EMAIL"))
        to_email = To(os.environ.get("TO_EMAIL"))
        subject = date.today().strftime("Personalized arXiv Digest, %d %b %Y")
        content = Content("text/html", body)
        mail = Mail(from_email, to_email, subject, content)

        resp = sg.client.mail.send.post(request_body=mail.get())
        if not (200 <= resp.status_code <= 299):
            raise RuntimeError(f"SendGrid failed: {resp.status_code} {resp.body}")
    else:
        print("No sendgrid api key found. Skipping email")
