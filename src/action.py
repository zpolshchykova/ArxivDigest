from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# OpenAI (the repo uses the older openai python package style: openai.ChatCompletion.create)
try:
    import openai  # type: ignore
except Exception:
    openai = None

# SendGrid (optional)
try:
    from sendgrid import SendGridAPIClient  # type: ignore
    from sendgrid.helpers.mail import Content, Email, Mail, To  # type: ignore
except Exception:
    SendGridAPIClient = None


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

PHYSICS_HUMAN_TO_CODE = {
    "Applied Physics": "physics.app-ph",
    "Physics Education": "physics.ed-ph",
    "History and Philosophy of Physics": "physics.hist-ph",
    "Instrumentation and Detectors": "physics.ins-det",
    "Optics": "physics.optics",
}

# A practical keyword list so the fallback is NOT “random recent papers”
KEYWORDS = [
    # Platforms/materials
    "lithium niobate", "linbo3", "liNbO3", "lnoi", "tfln", "thin-film lithium niobate", "ppln",
    "silicon photonics", "silicon nitride", "sin", "si/siN",
    # Devices/components
    "electro-optic", "pockels", "modulator", "microring", "micro-ring", "microresonator", "resonator",
    "waveguide", "photonic integrated circuit", "integrated photonics", "nanophotonics",
    "high-q", "whispering-gallery", "wgm", "photonic crystal",
    # Nonlinear/quantum optics
    "chi(2)", "χ(2)", "second harmonic", "shg", "opo", "frequency comb", "comb",
    "frequency conversion", "sum-frequency", "difference-frequency",
    "spdc", "sfwm", "squeezing", "entanglement", "single-photon", "single photon",
    "quantum photonics", "quantum optics", "quantum information",
    # Experimental/measurement-ish
    "packaging", "fiber-to-chip", "grating coupler", "edge coupling", "interferometer",
    "spectroscopy", "metrology", "detector",
]

AUTHOR_BOOST_KEYWORDS = [
    "grange",  # Rachel Grange (if it appears in the author list)
]


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    categories: List[str]
    published: datetime
    abs_url: str
    pdf_url: str
    keyword_score: int = 0
    llm_score: Optional[int] = None
    llm_reason: Optional[str] = None


def _parse_arxiv_dt(s: str) -> datetime:
    # Example: 2025-12-12T18:22:11Z
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _normalize_text(s: str) -> str:
    return " ".join(s.split()).strip()


def _extract_base_id(arxiv_id_or_url: str) -> str:
    # Accept: "2512.01234v1" or "2512.01234" or "http://arxiv.org/abs/2512.01234v1"
    s = arxiv_id_or_url.strip()
    if "/abs/" in s:
        s = s.rsplit("/abs/", 1)[-1]
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"v\d+$", "", s)
    return s


def resolve_category_codes(topic: str, categories: List[str]) -> List[str]:
    topic = (topic or "").strip()

    # If user already provided category codes, pass them through
    if categories and all(
        (c.startswith("physics.") or c in ("quant-ph",) or re.match(r"^[a-z\-]+\.[A-Z\-]+$", c) or re.match(r"^[a-z\-]+$", c))
        for c in categories
    ):
        return categories

    if topic == "Physics":
        if not categories:
            raise RuntimeError(
                "For topic 'Physics', you must set categories (e.g. Optics / Applied Physics / Instrumentation and Detectors)."
            )
        codes = []
        for c in categories:
            c = c.strip()
            if c in PHYSICS_HUMAN_TO_CODE:
                codes.append(PHYSICS_HUMAN_TO_CODE[c])
            elif c.startswith("physics."):
                codes.append(c)
            else:
                raise RuntimeError(
                    f"Unknown Physics category '{c}'. Use one of: {list(PHYSICS_HUMAN_TO_CODE.keys())} "
                    f"or direct codes like physics.optics."
                )
        return codes

    if topic in ("Quantum Physics", "quant-ph"):
        # quant-ph is its own arXiv category
        return ["quant-ph"]

    # Fallback: treat topic as a category code if it looks like one
    if re.match(r"^[a-z\-]+(\.[A-Z\-]+)?$", topic):
        return [topic]

    raise RuntimeError(
        f"Unsupported topic '{topic}' for this simplified action.py. "
        f"Use topic 'Physics' with physics.* categories, or topic 'Quantum Physics'."
    )


def fetch_recent_papers(category_codes: List[str], days_back: int, max_results_per_cat: int = 200) -> List[Paper]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    seen: set = set()
    out: List[Paper] = []

    for cat in category_codes:
        params = {
            "search_query": f"cat:{cat}",
            "start": 0,
            "max_results": max_results_per_cat,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = requests.get(
            ARXIV_API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "ArxivDigestBot/1.0 (personal use)"},
        )
        r.raise_for_status()

        # Parse Atom XML
        import xml.etree.ElementTree as ET

        root = ET.fromstring(r.text)
        for entry in root.findall("atom:entry", ATOM_NS):
            abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
            published_s = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
            if not abs_url or not published_s:
                continue

            published = _parse_arxiv_dt(published_s)
            # Since results are sorted newest->oldest, we can stop once we go past cutoff
            if published < cutoff:
                break

            base_id = _extract_base_id(abs_url)
            if base_id in seen:
                continue
            seen.add(base_id)

            title = _normalize_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
            summary = _normalize_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))

            authors = []
            for a in entry.findall("atom:author", ATOM_NS):
                name = a.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
                if name:
                    authors.append(name)

            cats = []
            for c in entry.findall("atom:category", ATOM_NS):
                term = c.attrib.get("term", "").strip()
                if term:
                    cats.append(term)

            abs_link = f"https://arxiv.org/abs/{base_id}"
            pdf_link = f"https://arxiv.org/pdf/{base_id}.pdf"

            out.append(
                Paper(
                    arxiv_id=base_id,
                    title=title,
                    authors=authors,
                    summary=summary,
                    categories=cats,
                    published=published,
                    abs_url=abs_link,
                    pdf_url=pdf_link,
                )
            )

    # newest first
    out.sort(key=lambda p: p.published, reverse=True)
    return out


def compute_keyword_score(p: Paper) -> int:
    text = (p.title + " " + p.summary).lower()
    score = 0

    for kw in KEYWORDS:
        if kw.lower() in text:
            score += 2

    author_blob = " ".join(p.authors).lower()
    for ak in AUTHOR_BOOST_KEYWORDS:
        if ak in author_blob:
            score += 6

    # small boost if it's clearly in one of your target categories
    if any(c in ("physics.optics", "physics.app-ph", "quant-ph") for c in p.categories):
        score += 1

    return score


def llm_score_papers(papers: List[Paper], interest: str, threshold: int) -> List[Paper]:
    """
    Robust scoring:
    - If OpenAI fails OR response is not parseable, we just return papers with llm_score=None (no crash).
    """
    if not interest.strip():
        return papers

    if openai is None:
        return papers

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return papers

    openai.api_key = api_key
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Keep cost controlled
    max_llm = int(os.environ.get("MAX_LLM_PAPERS", "40"))
    papers_to_score = papers[:max_llm]

    # Build prompt
    items = []
    for i, p in enumerate(papers_to_score, start=1):
        items.append(
            f"{i}. Title: {p.title}\n"
            f"   Authors: {', '.join(p.authors)}\n"
            f"   Abstract: {p.summary}\n"
        )

    system = (
        "You are a careful research assistant. "
        "You MUST only score the papers given. Do NOT invent papers. "
        "Output must be machine-parseable."
    )
    user = (
        "Task:\n"
        "For each paper, give a relevance score from 1 to 10 (integer), where 10 is extremely relevant.\n"
        f"A paper is relevant if it matches ANY ONE of the interests below. It does NOT need to match all.\n"
        f"Use the interests as soft preferences (de-prioritize means lower score, not automatically zero).\n\n"
        f"Interests:\n{interest.strip()}\n\n"
        "Output format:\n"
        "Return EXACTLY one JSON object per paper, one per line, in the SAME ORDER.\n"
        'Each JSON object must have keys: "Relevancy score" (integer 1-10) and "Reasons for match" (1-2 sentences).\n'
        "Do not add any extra text.\n\n"
        "Papers:\n"
        + "\n".join(items)
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        # Don't crash the workflow
        return papers

    # Parse JSON objects (very forgiving)
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    parsed: List[Tuple[int, str]] = []

    for ln in lines:
        # accept "1. {...}" or "{...}"
        m = re.search(r"(\{.*\})", ln)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
            score = int(obj.get("Relevancy score"))
            reason = str(obj.get("Reasons for match", "")).strip()
            parsed.append((score, reason))
        except Exception:
            continue

    if len(parsed) != len(papers_to_score):
        # If the model output is malformed, do NOT crash; just skip LLM scoring
        return papers

    for p, (s, r) in zip(papers_to_score, parsed):
        p.llm_score = s
        p.llm_reason = r

    return papers


def build_html(papers: List[Paper], threshold: int, days_back: int, title: str) -> str:
    # Pick “relevant”
    relevant = [p for p in papers if (p.llm_score is not None and p.llm_score >= threshold)]

    # Fallback: if none exceed threshold, show top by keyword_score (not random newest)
    used_fallback = False
    if not relevant:
        used_fallback = True
        ranked = sorted(papers, key=lambda p: (p.keyword_score, p.published), reverse=True)
        # If keyword scores are all zero, then show newest
        if ranked and ranked[0].keyword_score == 0:
            ranked = sorted(papers, key=lambda p: p.published, reverse=True)
        relevant = ranked[:15]

    def paper_block(p: Paper) -> str:
        score_part = ""
        if p.llm_score is not None:
            score_part = f"<div><b>Relevance:</b> {p.llm_score}/10</div>"
        reason_part = ""
        if p.llm_reason:
            reason_part = f"<div><b>Why:</b> {html.escape(p.llm_reason)}</div>"

        cats = ", ".join(p.categories)
        authors = ", ".join(p.authors)
        pub = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")

        # IMPORTANT: link directly to PDF
        return (
            "<div style='margin: 14px 0; padding: 10px; border: 1px solid #ddd; border-radius: 8px;'>"
            f"<div style='font-size: 16px;'><b>Title:</b> "
            f"<a href='{html.escape(p.pdf_url)}'>{html.escape(p.title)}</a></div>"
            f"<div><b>Authors:</b> {html.escape(authors)}</div>"
            f"<div><b>Published:</b> {pub}</div>"
            f"<div><b>Categories:</b> {html.escape(cats)}</div>"
            f"{score_part}"
            f"{reason_part}"
            f"<div style='margin-top: 6px;'><a href='{html.escape(p.abs_url)}'>Abstract page</a></div>"
            "</div>"
        )

    header = f"<h2>{html.escape(title)}</h2>"
    meta = f"<div><i>Looking back {days_back} day(s). Threshold = {threshold}.</i></div>"

    if used_fallback:
        meta += (
            f"<div style='margin-top:6px; color:#a00;'><b>"
            f"No papers exceeded the relevance threshold ({threshold}) for this run. "
            f"Showing 15 keyword-ranked papers instead."
            f"</b></div>"
        )

    blocks = "\n".join(paper_block(p) for p in relevant)
    return header + meta + blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="YAML config to load")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    topic = str(cfg.get("topic", "")).strip()
    categories = cfg.get("categories") or []
    categories = [str(c).strip() for c in categories]
    threshold = int(cfg.get("threshold", 6))
    interest = str(cfg.get("interest", "") or "")
    days_back = int(cfg.get("days_back", 4))

    cat_codes = resolve_category_codes(topic, categories)

    papers = fetch_recent_papers(cat_codes, days_back=days_back, max_results_per_cat=200)
    for p in papers:
        p.keyword_score = compute_keyword_score(p)

    # Sort candidates by (keyword_score, recency) before LLM scoring
    papers.sort(key=lambda p: (p.keyword_score, p.published), reverse=True)

    papers = llm_score_papers(papers, interest=interest, threshold=threshold)

    # Build digest HTML
    full = ["<html><body>"]
    full.append("<h1>Personalized arXiv Digest</h1>")
    full.append(build_html(papers, threshold=threshold, days_back=days_back, title=os.path.basename(args.config)))
    full.append("</body></html>")
    digest_html = "\n".join(full)

    with open("digest.html", "w", encoding="utf-8") as f:
        f.write(digest_html)

    # Send email if SendGrid is configured
    sg_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    from_email = os.environ.get("FROM_EMAIL", "").strip()
    to_email = os.environ.get("TO_EMAIL", "").strip()

    if sg_key and SendGridAPIClient is not None and from_email and to_email:
        try:
            sg = SendGridAPIClient(api_key=sg_key)
            subject = f"Personalized arXiv Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            mail = Mail(
                Email(from_email),
                To(to_email),
                subject,
                Content("text/html", digest_html),
            )
            sg.client.mail.send.post(request_body=mail.get())
            print("Sent digest via SendGrid.")
        except Exception as e:
            print(f"SendGrid send failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
