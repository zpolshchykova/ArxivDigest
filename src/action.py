from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from datetime import date
import argparse
import yaml
import os
import re
from urllib.parse import quote_plus

from dotenv import load_dotenv
import openai

from relevancy import generate_relevance_score
from download_new_papers import get_papers


# Hackathon quality code. Don't judge too harshly.
# Feel free to submit pull requests to improve the code.

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
    """Validate arXiv identifiers. Accepts new and old formats, optional vN."""
    s = (s or "").strip()
    # new style: YYMM.NNNNN or YYMM.NNNN, optional vN
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s):
        return True
    # old style: archive/YYMMNNN, optional vN
    if re.fullmatch(r"[a-z\-]+/\d{7}(v\d+)?", s):
        return True
    return False


def make_clickable_arxiv_link(paper: dict) -> str:
    """
    Return a safe URL:
    - If paper['main_page'] contains a valid arXiv abs/pdf URL with a real ID -> use it.
    - If paper['main_page'] is a bare valid arXiv ID -> turn into https://arxiv.org/abs/<ID>
    - Otherwise -> link to arXiv title search (always works; never "Invalid article identifier")
    """
    title = (paper.get("title") or "").replace("Title:", "").strip()
    url = (paper.get("main_page") or "").strip()

    # Normalize relative URL like "/abs/2501.01234"
    if url.startswith("/"):
        url = "https://arxiv.org" + url

    # If it looks like an arxiv abs/pdf URL, verify the id segment
    if url.startswith("http"):
        m = re.search(r"arxiv\.org/(abs|pdf)/([^?#/]+)", url)
        if m:
            arxiv_id = m.group(2).replace(".pdf", "")
            if is_valid_arxiv_id(arxiv_id):
                # normalize pdf->abs for nicer clicking
                return "https://arxiv.org/abs/" + arxiv_id
            # bad id embedded in URL => discard
        # any other http URL: keep it
        return url

    # If it isn't http, maybe it's a bare arXiv id
    if url and is_valid_arxiv_id(url):
        return "https://arxiv.org/abs/" + url

    # Final fallback: title search
    return "https://arxiv.org/search/?query=" + quote_plus(title) + "&searchtype=title"


def render_paper_html(p: dict, include_score: bool) -> str:
    title = (p.get("title") or "").replace("Title:", "").strip()
    authors = p.get("authors", "")
    link = make_clickable_arxiv_link(p)

    extra = ""
    if include_score:
        score = p.get("Relevancy score", "")
        reason = p.get("Reasons for match", "")
        if score != "" or reason != "":
            extra = f"<br>Score: {score}<br>Reason: {reason}"

    return (
        f'Title: <a href="{link}">{title}</a><br>'
        f"Authors: {authors}{extra}"
    )


def generate_body(topic, categories, interest, threshold, fallback_n=15):
    """
    Builds the HTML digest.

    Key behavior:
    - Physics topic: pulls directly from each physics.* subpage you chose (so no random physics).
    - If LLM scoring fails or yields nothing: falls back to most recent papers (so never empty).
    - Links are always safe (never invalid arXiv ID links).
    """
    categories = categories or []
    threshold = int(threshold)

    # ---- Determine what to download ----
    if topic == "Physics":
        # Map human names -> physics.* codes
        physics_map = {
            "Applied Physics": "physics.app-ph",
            "Physics Education": "physics.ed-ph",
            "History and Philosophy of Physics": "physics.hist-ph",
            "Instrumentation and Detectors": "physics.ins-det",
            "Optics": "physics.optics",
        }

        if not categories:
            raise RuntimeError(
                "For topic 'Physics', you must provide at least one category "
                "(e.g. ['Optics', 'Applied Physics'])."
            )

        # Normalize categories to codes
        physics_codes = []
        for c in categories:
            if c in physics_map:
                physics_codes.append(physics_map[c])
            elif isinstance(c, str) and c.startswith("physics.") and len(c) > len("physics."):
                physics_codes.append(c)
            else:
                raise RuntimeError(
                    f"Unknown Physics category '{c}'. Use one of: {list(physics_map.keys())} "
                    f"or a physics.* code like 'physics.optics'."
                )

        # Pull directly from each selected subpage (THIS fixes the random CME/biology/etc.)
        papers = []
        seen = set()
        for code in physics_codes:
            for p in get_papers(code):
                key = (p.get("main_page") or "") + "||" + (p.get("title") or "")
                if key not in seen:
                    seen.add(key)
                    papers.append(p)

    elif topic in physics_topics:
        abbr = physics_topics[topic]
        papers = get_papers(abbr)

    elif topic in topics:
        abbr = topics[topic]
        papers = get_papers(abbr)

    else:
        raise RuntimeError(f"Invalid topic {topic}")

    if not papers:
        return "<html><body><p>No papers found for this run.</p></body></html>"

    # ---- LLM scoring (optional) ----
    if interest and str(interest).strip():
        used_fallback = False
        scored = []
        hallucinated = False

        try:
            scored, hallucinated = generate_relevance_score(
                papers,
                query={"interest": interest},
                threshold_score=threshold,
                num_paper_in_prompt=4,  # you already set this low; keep it
            )
        except Exception:
            # If the LLM/parsing code is flaky, DON'T DIE. Just fallback.
            used_fallback = True
            scored = []
            hallucinated = False

        # If nothing passes threshold, fallback to recent
        final_list = scored
        if not final_list:
            used_fallback = True
            final_list = papers[:fallback_n]

        body_parts = []
        if used_fallback:
            body_parts.append(
                f"No papers exceeded the relevance threshold ({threshold}) for this run. "
                f"Showing the {fallback_n} most recent papers instead.<br><br>"
            )
        else:
            # Only show hallucination warning if we actually used scored results
            if hallucinated:
                body_parts.append(
                    "Warning: the model output was partially malformed. Scores/reasons may be imperfect.<br><br>"
                )

        body_parts.append(
            "<br><br>".join(render_paper_html(p, include_score=not used_fallback) for p in final_list)
        )
        return "".join(body_parts)

    # ---- No interest => raw list ----
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

    from_email = os.environ.get("FROM_EMAIL")
    to_email = os.environ.get("TO_EMAIL")

    body = generate_body(topic, categories, interest, threshold)

    with open("digest.html", "w", encoding="utf-8") as f:
        f.write(body)

    if os.environ.get("SENDGRID_API_KEY"):
        sg = SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
        from_e = Email(from_email)  # must be verified sender in SendGrid
        to_e = To(to_email)
        subject = date.today().strftime("Personalized arXiv Digest, %d %b %Y")
        content = Content("text/html", body)
        mail = Mail(from_e, to_e, subject, content)

        response = sg.client.mail.send.post(request_body=mail.get())
        if 200 <= response.status_code <= 299:
            print("SendGrid: Success")
        else:
            raise RuntimeError(f"SendGrid: Failure ({response.status_code}) {response.body}")
    else:
        print("No sendgrid api key found. Skipping email")
