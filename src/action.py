from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple, Optional

import requests
import yaml

from password_gate import gate_body_end, gate_body_start, gate_head

# OpenAI (repo uses old-style openai.ChatCompletion.create)
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
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_RETRY_ATTEMPTS = 4
ARXIV_RETRY_BACKOFF_SECONDS = (10, 30, 60)
CROSSREF_API_URL = "https://api.crossref.org/works"
CROSSREF_RETRY_ATTEMPTS = 3
CROSSREF_RETRY_BACKOFF_SECONDS = (5, 15, 30)

PHYSICS_HUMAN_TO_CODE = {
    "Applied Physics": "physics.app-ph",
    "Physics Education": "physics.ed-ph",
    "History and Philosophy of Physics": "physics.hist-ph",
    "Instrumentation and Detectors": "physics.ins-det",
    "Optics": "physics.optics",
}

# Keyword fallback so “no LLM matches” still shows relevant-ish stuff
KEYWORDS = [
    # Platforms/materials
    "lithium niobate", "linbo3", "linbo", "liNbO3", "lnoi", "tfln", "thin-film lithium niobate", "ppln",
    "silicon photonics", "silicon nitride", "sin", "si/sin",
    # Devices/components
    "electro-optic", "pockels", "modulator",
    "microring", "micro-ring", "microresonator", "resonator",
    "waveguide", "photonic integrated circuit", "integrated photonics", "nanophotonics",
    "high-q", "whispering-gallery", "wgm", "photonic crystal",
    # Nonlinear/quantum optics
    "chi(2)", "χ(2)", "second harmonic", "shg", "opo", "frequency comb", "comb",
    "frequency conversion", "sum-frequency", "difference-frequency",
    "spdc", "sfwm", "squeezing", "entanglement", "single-photon", "single photon",
    "quantum photonics", "quantum optics", "quantum information",
    # Experimental/measurement-ish
    "packaging", "fiber-to-chip", "grating coupler", "edge coupling",
    "interferometer", "spectroscopy", "metrology", "detector",
]

AUTHOR_BOOST_KEYWORDS = ["grange"]  # small boost if author list includes "grange"

DEFAULT_PUBLISHED_JOURNALS = [
    # Science / AAAS family
    {"name": "Science", "issn": "0036-8075"},
    {"name": "Science Advances", "issn": "2375-2548"},
    {"name": "Science Robotics", "issn": "2470-9476"},
    # Nature and broad/relevant Nature-family journals
    {"name": "Nature", "issn": "1476-4687"},
    {"name": "Nature Communications", "issn": "2041-1723"},
    {"name": "Nature Physics", "issn": "1745-2481"},
    {"name": "Nature Photonics", "issn": "1749-4893"},
    {"name": "Nature Nanotechnology", "issn": "1748-3395"},
    {"name": "Nature Materials", "issn": "1476-4660"},
    {"name": "Nature Electronics", "issn": "2520-1131"},
    {"name": "Nature Reviews Physics", "issn": "2522-5820"},
    {"name": "Communications Physics", "issn": "2399-3650"},
    {"name": "Communications Materials", "issn": "2662-4443"},
    {"name": "npj Quantum Information", "issn": "2056-6387"},
    {"name": "npj Nanophotonics", "issn": "2948-1509"},
    # Optica Publishing Group
    {"name": "Optica", "issn": "2334-2536"},
    {"name": "Optics Express", "issn": "1094-4087"},
    {"name": "Optics Letters", "issn": "1539-4794"},
    {"name": "Journal of the Optical Society of America B", "issn": "1520-8540"},
    {"name": "Photonics Research", "issn": "2327-9125"},
    {"name": "Light: Science & Applications", "issn": "2047-7538"},
    {"name": "Advanced Photonics", "issn": "2577-5421"},
    {"name": "Laser & Photonics Reviews", "issn": "1863-8899"},
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
    source: str = "arxiv"
    venue: str = ""
    keyword_score: int = 0
    llm_score: Optional[int] = None
    llm_reason: Optional[str] = None


def working_days_cutoff(now_utc: datetime, working_days_back: int) -> datetime:
    """Return cutoff datetime N working days back (skips Sat/Sun)."""
    if working_days_back <= 0:
        return now_utc
    d = now_utc
    remaining = working_days_back
    while remaining > 0:
        d = d - timedelta(days=1)
        if d.weekday() < 5:  # Mon=0..Fri=4
            remaining -= 1
    return d


def _parse_arxiv_dt(s: str) -> datetime:
    # Example: 2025-12-12T18:22:11Z
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _normalize_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(_normalize_text(s))


def _extract_base_id(abs_url: str) -> str:
    # abs_url like https://arxiv.org/abs/2512.10462v1
    s = abs_url.strip()
    if "/abs/" in s:
        s = s.rsplit("/abs/", 1)[-1]
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"v\d+$", "", s)
    return s


def resolve_category_codes(topic: str, categories: List[str]) -> List[str]:
    topic = (topic or "").strip()

    if topic == "Physics":
        if not categories:
            raise RuntimeError("For topic 'Physics', you must set categories (e.g. Optics).")
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
                    f"or codes like physics.optics."
                )
        return codes

    if topic in ("Quantum Physics", "quant-ph"):
        return ["quant-ph"]

    # allow raw category code as topic
    if re.match(r"^[a-z\-]+(\.[A-Z\-]+)?$", topic):
        return [topic]

    raise RuntimeError("Unsupported topic. Use 'Physics' or 'Quantum Physics'.")


def _get_arxiv_response(params: dict) -> requests.Response:
    last_error = None
    for attempt in range(1, ARXIV_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                timeout=(10, 60),
                headers={"User-Agent": "ArxivDigestBot/1.0 (personal use)"},
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == ARXIV_RETRY_ATTEMPTS:
                break
            wait_seconds = ARXIV_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(ARXIV_RETRY_BACKOFF_SECONDS) - 1)
            ]
            print(
                f"arXiv request failed on attempt {attempt}/{ARXIV_RETRY_ATTEMPTS}: "
                f"{exc}. Retrying in {wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"arXiv request failed after {ARXIV_RETRY_ATTEMPTS} attempts"
    ) from last_error


def _get_crossref_response(params: dict) -> requests.Response:
    headers = {"User-Agent": "ArxivDigestBot/1.0 (personal weekly digest)"}
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    if mailto:
        headers["User-Agent"] += f" (mailto:{mailto})"

    last_error = None
    for attempt in range(1, CROSSREF_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                CROSSREF_API_URL,
                params=params,
                timeout=(10, 60),
                headers=headers,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == CROSSREF_RETRY_ATTEMPTS:
                break
            wait_seconds = CROSSREF_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(CROSSREF_RETRY_BACKOFF_SECONDS) - 1)
            ]
            print(
                f"Crossref request failed on attempt {attempt}/{CROSSREF_RETRY_ATTEMPTS}: "
                f"{exc}. Retrying in {wait_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Crossref request failed after {CROSSREF_RETRY_ATTEMPTS} attempts"
    ) from last_error


def fetch_recent_papers(
    category_codes: List[str],
    cutoff: datetime,
    end_cutoff: Optional[datetime] = None,
    max_results_per_cat: int = 200,
) -> List[Paper]:
    seen = set()
    out: List[Paper] = []

    for cat in category_codes:
        params = {
            "search_query": f"cat:{cat}",
            "start": 0,
            "max_results": max_results_per_cat,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = _get_arxiv_response(params)

        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)

        for entry in root.findall("atom:entry", ATOM_NS):
            abs_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
            published_s = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
            if not abs_url or not published_s:
                continue

            published = _parse_arxiv_dt(published_s)
            if end_cutoff is not None and published >= end_cutoff:
                continue
            if published < cutoff:
                break  # newest->oldest

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
                term = (c.attrib.get("term", "") or "").strip()
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

    out.sort(key=lambda p: p.published, reverse=True)
    return out


def _date_parts_to_datetime(value: Optional[dict]) -> Optional[datetime]:
    if not value:
        return None
    parts = value.get("date-parts") or []
    if not parts or not parts[0]:
        return None
    year = int(parts[0][0])
    month = int(parts[0][1]) if len(parts[0]) > 1 else 1
    day = int(parts[0][2]) if len(parts[0]) > 2 else 1
    return datetime(year, month, day, tzinfo=timezone.utc)


def _crossref_published_date(item: dict) -> Optional[datetime]:
    for key in ("published-online", "published-print", "published", "issued"):
        published = _date_parts_to_datetime(item.get(key))
        if published is not None:
            return published
    return None


def _crossref_authors(item: dict) -> List[str]:
    authors = []
    for author in item.get("author") or []:
        given = str(author.get("given", "") or "").strip()
        family = str(author.get("family", "") or "").strip()
        name = " ".join(x for x in (given, family) if x).strip()
        if not name:
            name = str(author.get("name", "") or "").strip()
        if name:
            authors.append(name)
    return authors


def _journal_specs(config_value: object) -> List[Dict[str, str]]:
    if not config_value:
        return DEFAULT_PUBLISHED_JOURNALS
    if isinstance(config_value, str) and config_value.strip().lower() in {"default", "default_photonics"}:
        return DEFAULT_PUBLISHED_JOURNALS
    if not isinstance(config_value, list):
        raise RuntimeError("published_journals must be a list or 'default_photonics'.")

    specs = []
    for item in config_value:
        if isinstance(item, str):
            specs.append({"name": item, "issn": item})
        elif isinstance(item, dict):
            issn = str(item.get("issn", "") or "").strip()
            name = str(item.get("name", "") or issn).strip()
            if issn:
                specs.append({"name": name, "issn": issn})
        else:
            raise RuntimeError("Each published_journals entry must be a string or {name, issn}.")
    return specs


def fetch_published_papers(
    journals: Iterable[Dict[str, str]],
    cutoff: datetime,
    end_cutoff: Optional[datetime] = None,
    max_results_per_journal: int = 50,
) -> List[Paper]:
    seen = set()
    out: List[Paper] = []
    until = (end_cutoff - timedelta(days=1)) if end_cutoff else datetime.now(timezone.utc)

    for journal in journals:
        issn = str(journal.get("issn", "") or "").strip()
        fallback_name = str(journal.get("name", "") or issn).strip()
        if not issn:
            continue

        params = {
            "filter": (
                "type:journal-article,"
                f"issn:{issn},"
                f"from-pub-date:{cutoff.date().isoformat()},"
                f"until-pub-date:{until.date().isoformat()}"
            ),
            "rows": max_results_per_journal,
            "sort": "published",
            "order": "desc",
            "select": (
                "DOI,title,author,abstract,published-online,published-print,"
                "published,issued,container-title,URL,subject,ISSN"
            ),
        }
        response = _get_crossref_response(params)
        items = response.json().get("message", {}).get("items", [])

        for item in items:
            doi = str(item.get("DOI", "") or "").strip().lower()
            if not doi or doi in seen:
                continue

            published = _crossref_published_date(item)
            if published is None:
                continue
            if published < cutoff:
                continue
            if end_cutoff is not None and published >= end_cutoff:
                continue

            title_values = item.get("title") or []
            title = _strip_html(str(title_values[0] if title_values else ""))
            if not title:
                continue

            seen.add(doi)
            venue_values = item.get("container-title") or []
            venue = _normalize_text(venue_values[0] if venue_values else fallback_name)
            url = str(item.get("URL", "") or f"https://doi.org/{doi}").strip()
            subjects = [str(s).strip() for s in (item.get("subject") or []) if str(s).strip()]
            summary = _strip_html(str(item.get("abstract", "") or ""))

            out.append(
                Paper(
                    arxiv_id=doi,
                    title=title,
                    authors=_crossref_authors(item),
                    summary=summary or f"Published article in {venue}.",
                    categories=[venue] + subjects,
                    published=published,
                    abs_url=url,
                    pdf_url=url,
                    source="published",
                    venue=venue,
                )
            )

        time.sleep(1)

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

    # small bonus for target categories / curated venues
    if any(c in ("physics.optics", "physics.app-ph", "physics.ins-det", "quant-ph") for c in p.categories):
        score += 1
    if p.source == "published":
        score += 1

    return score


def llm_score_papers(papers: List[Paper], interest: str) -> List[Paper]:
    """Best-effort scoring. Never crashes the workflow."""
    if not interest.strip():
        return papers
    if openai is None:
        return papers

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return papers
    openai.api_key = api_key

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_llm = int(os.environ.get("MAX_LLM_PAPERS", "40"))
    papers_to_score = papers[:max_llm]

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
        "For each paper, give a relevance score from 1 to 10 (integer).\n"
        "Use a strict personal-relevance scale, not a general scientific-interest scale:\n"
        "10 = must-read; directly about the user's core experimental work, platform, or device class.\n"
        "9 = very strong match with clear direct use for the user's current research.\n"
        "7-8 = relevant and worth opening, but not necessarily central.\n"
        "5-6 = adjacent background or broad topical overlap.\n"
        "3-4 = only weakly related, even if it is in quantum optics or photonics.\n"
        "1-2 = not relevant.\n"
        "Do NOT give 9 or 10 for generic quantum, optics, AI, theory, sensing, or materials papers unless the abstract clearly connects to the user's specific experimental platforms or devices.\n"
        "A paper may be somewhat relevant if it matches one interest below, but broad category overlap alone should score low.\n\n"
        f"Interests:\n{interest.strip()}\n\n"
        "Output format:\n"
        "Return EXACTLY one JSON object per paper, one per line, in the SAME ORDER.\n"
        'Each JSON object must have keys: "Relevancy score" (integer 1-10) and "Reasons for match" (1-2 sentences).\n'
        "Do not add any extra text.\n\n"
        "Papers:\n" + "\n".join(items)
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        return papers

    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    parsed: List[Tuple[int, str]] = []

    for ln in lines:
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
        # malformed output: ignore LLM scores
        return papers

    for p, (s, r) in zip(papers_to_score, parsed):
        p.llm_score = s
        p.llm_reason = r

    return papers


def build_html(papers: List[Paper], threshold: int, lookback_label: str, title: str) -> str:
    header = f"<h2>{html.escape(title)}</h2>"
    meta = f"<div><i>{html.escape(lookback_label)}. Base threshold = {threshold}.</i></div>"

    if not papers:
        return header + meta + "<div><b>No papers found.</b></div>"

    # Sort by best-first for display
    papers_sorted = sorted(papers, key=lambda p: (p.keyword_score, p.published), reverse=True)

    # --- Auto-raise threshold to keep max 15 papers ---
    base_threshold = int(threshold)
    used_threshold = base_threshold

    def relevant_for(t: int) -> List[Paper]:
        rel = [p for p in papers_sorted if (p.llm_score is not None and p.llm_score >= t)]
        # sort relevant by (llm_score desc, keyword_score desc, recency desc)
        rel.sort(key=lambda p: ((p.llm_score or 0), p.keyword_score, p.published), reverse=True)
        return rel

    relevant = relevant_for(used_threshold)
    while len(relevant) > 15 and used_threshold < 10:
        used_threshold += 1
        relevant = relevant_for(used_threshold)

    if used_threshold != base_threshold:
        meta += (
            f"<div><b>Auto-adjust:</b> threshold raised from {base_threshold} to {used_threshold} "
            f"to keep ≤ 15 papers.</div>"
        )

    # If still > 15 even at threshold 10, cap to 15
    if len(relevant) > 15:
        relevant = relevant[:15]
        meta += "<div><b>Note:</b> still > 15 at threshold 10, capped to 15.</div>"

    # --- Fallback if none pass threshold (or no LLM scores exist) ---
    used_fallback = False
    if not relevant:
        used_fallback = True
        ranked = sorted(papers_sorted, key=lambda p: (p.keyword_score, p.published), reverse=True)
        if ranked and ranked[0].keyword_score == 0:
            ranked = sorted(papers_sorted, key=lambda p: p.published, reverse=True)
        relevant = ranked[:15]
        meta += (
            f"<div style='margin-top:6px; color:#a00;'><b>"
            f"No papers exceeded the relevance threshold ({used_threshold}) for this run. "
            f"Showing 15 keyword-ranked papers instead."
            f"</b></div>"
        )

    def paper_block(p: Paper) -> str:
        authors = ", ".join(p.authors)
        pub = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        cats = ", ".join(p.categories)
        source_label = "arXiv preprint" if p.source == "arxiv" else "Published article"
        venue_part = f"<div><b>Venue:</b> {html.escape(p.venue)}</div>" if p.venue else ""

        score_part = ""
        if p.llm_score is not None and not used_fallback:
            score_part = f"<div><b>Relevance:</b> {p.llm_score}/10</div>"

        reason_part = ""
        if p.llm_reason and not used_fallback:
            reason_part = f"<div><b>Why:</b> {html.escape(p.llm_reason)}</div>"

        return (
            "<div style='margin: 14px 0; padding: 10px; border: 1px solid #ddd; border-radius: 8px;'>"
            f"<div style='font-size: 16px;'><b>Title:</b> "
            f"<a href='{html.escape(p.pdf_url)}'>{html.escape(p.title)}</a></div>"
            f"<div><b>Authors:</b> {html.escape(authors)}</div>"
            f"<div><b>Source:</b> {source_label}</div>"
            f"{venue_part}"
            f"<div><b>Published:</b> {pub}</div>"
            f"<div><b>Categories:</b> {html.escape(cats)}</div>"
            f"{score_part}{reason_part}"
            f"<div style='margin-top: 6px;'><a href='{html.escape(p.abs_url)}'>Abstract page</a></div>"
            "</div>"
        )

    preprints = [p for p in relevant if p.source == "arxiv"]
    published = [p for p in relevant if p.source == "published"]
    sections = []
    if preprints:
        sections.append("<h3>arXiv preprints</h3>" + "\n".join(paper_block(p) for p in preprints))
    if published:
        sections.append("<h3>Published papers</h3>" + "\n".join(paper_block(p) for p in published))
    return header + meta + "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="YAML config to load")
    parser.add_argument(
        "--target-date",
        default="",
        help="UTC publication date to digest, formatted YYYY-MM-DD. Overrides date_window.",
    )
    parser.add_argument("--start-date", default="", help="Inclusive UTC start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Inclusive UTC end date, YYYY-MM-DD.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    topic = str(cfg.get("topic", "")).strip()
    categories = [str(c).strip() for c in (cfg.get("categories") or [])]
    threshold = int(cfg.get("threshold", 6))
    interest = str(cfg.get("interest", "") or "")
    sources = [str(s).strip().lower() for s in (cfg.get("sources") or ["arxiv"]) if str(s).strip()]
    if not sources:
        sources = ["arxiv"]

    date_window = str(cfg.get("date_window", "") or "").strip()
    working_days_back = cfg.get("working_days_back", None)
    days_back = cfg.get("days_back", None)

    now = datetime.now(timezone.utc)
    end_cutoff = None
    allow_latest_fallback = True

    if bool(args.start_date) != bool(args.end_date):
        raise RuntimeError("Set both --start-date and --end-date, or neither.")

    if args.start_date:
        start_day = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_day = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_day < start_day:
            raise RuntimeError("--end-date must be on or after --start-date.")
        cutoff = start_day
        end_cutoff = end_day + timedelta(days=1)
        allow_latest_fallback = False
        lookback_label = f"Showing papers published from {args.start_date} through {args.end_date} UTC"
    elif args.target_date:
        target_day = datetime.strptime(args.target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff = target_day
        end_cutoff = target_day + timedelta(days=1)
        allow_latest_fallback = False
        lookback_label = f"Showing papers published on {args.target_date} UTC"
    elif date_window == "previous_day":
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = today - timedelta(days=1)
        end_cutoff = today
        allow_latest_fallback = False
        lookback_label = f"Showing papers published on {cutoff.strftime('%Y-%m-%d')} UTC"
    elif working_days_back is not None:
        wd = int(working_days_back)
        cutoff = working_days_cutoff(now, wd)
        lookback_label = f"Looking back {wd} working day(s)"
    else:
        dd = int(days_back) if days_back is not None else 1
        cutoff = now - timedelta(days=dd)
        lookback_label = f"Looking back {dd} day(s)"

    # IMPORTANT: start cutoff at midnight UTC so you don't miss earlier papers in the cutoff day
    cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    papers: List[Paper] = []

    if "arxiv" in sources:
        cat_codes = resolve_category_codes(topic, categories)
        papers.extend(fetch_recent_papers(cat_codes, cutoff=cutoff, end_cutoff=end_cutoff, max_results_per_cat=200))

    if "published" in sources:
        max_crossref = int(cfg.get("max_crossref_results_per_journal", 50))
        papers.extend(
            fetch_published_papers(
                _journal_specs(cfg.get("published_journals")),
                cutoff=cutoff,
                end_cutoff=end_cutoff,
                max_results_per_journal=max_crossref,
            )
        )

    # Absolute fallback: if the window is empty, show latest available in these categories
    if not papers and allow_latest_fallback and "arxiv" in sources:
        very_old = datetime(1900, 1, 1, tzinfo=timezone.utc)
        papers = fetch_recent_papers(cat_codes, cutoff=very_old, max_results_per_cat=200)
        lookback_label += " — no results in window, showing latest available instead"

    for p in papers:
        p.keyword_score = compute_keyword_score(p)

    # Pre-rank before LLM to keep costs sane
    papers.sort(key=lambda p: (p.keyword_score, p.published), reverse=True)

    papers = llm_score_papers(papers, interest=interest)

    full = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Daily reading notes</title>",
        gate_head(),
        "</head>",
        "<body>",
        gate_body_start(),
    ]
    full.append("<h1>Personalized Research Digest</h1>")
    full.append(build_html(papers, threshold=threshold, lookback_label=lookback_label, title=os.path.basename(args.config)))
    full.append(gate_body_end())
    full.append("</body></html>")
    digest_html = "\n".join(full)

    with open("digest.html", "w", encoding="utf-8") as f:
        f.write(digest_html)

    # Optional local sending (your GitHub workflow sends combined email separately)
    sg_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    from_email = os.environ.get("FROM_EMAIL", "").strip()
    to_email = os.environ.get("TO_EMAIL", "").strip()

    if sg_key and SendGridAPIClient is not None and from_email and to_email:
        try:
            sg = SendGridAPIClient(api_key=sg_key)
            subject = f"Personalized arXiv Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            mail = Mail(Email(from_email), To(to_email), subject, Content("text/html", digest_html))
            sg.client.mail.send.post(request_body=mail.get())
            print("Sent digest via SendGrid.")
        except Exception as e:
            print(f"SendGrid send failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
