"""
Rebuilds docs/index.html from all archived digests in docs/digests/.

Expects digest files named like:
    docs/digests/2026-07-07.html
    docs/digests/2026-07-07_quant-ph.html   (optional suffix if you run several configs)

For each digest it counts papers (unique arxiv.org/abs links) and extracts
relevancy scores, then aggregates stats by day of week.

Run from the repo root:  python src/build_site.py
"""

import re
import html
import calendar
from collections import defaultdict
from datetime import date
from pathlib import Path

DIGEST_DIR = Path("docs/digests")
INDEX_PATH = Path("docs/index.html")

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
SCORE_RE = re.compile(r"[Ss]core\D{0,20}?(\d{1,2})\b")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def analyze_digest(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    paper_ids = set(ARXIV_ID_RE.findall(text))
    scores = [int(s) for s in SCORE_RE.findall(text) if 0 <= int(s) <= 10]
    return {
        "n_papers": len(paper_ids),
        "scores": scores,
    }


def digest_label(path: Path, day: str) -> str:
    return path.stem[len(day):].lstrip("_-") or "digest"


def build_calendar(by_date):
    if not by_date:
        return "<p class='dim'>No saved days yet.</p>"

    months = sorted({d[:7] for d in by_date}, reverse=True)
    blocks = []

    for month in months:
        year, month_num = map(int, month.split("-"))
        title = f"{MONTH_NAMES[month_num - 1]} {year}"
        weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month_num)
        rows = []

        for week in weeks:
            cells = []
            for day_date in week:
                day = day_date.isoformat()
                if day_date.month != month_num:
                    cells.append("<td class='cal-empty'></td>")
                    continue

                files = by_date.get(day, [])
                if files:
                    links = " ".join(
                        f'<a href="digests/{html.escape(f.name)}">{html.escape(digest_label(f, day))}</a>'
                        for f in files
                    )
                    cells.append(
                        f"<td class='cal-hit'><div class='cal-day'>{day_date.day}</div>"
                        f"<div class='cal-links'>{links}</div></td>"
                    )
                else:
                    cells.append(f"<td><div class='cal-day dim'>{day_date.day}</div></td>")

            rows.append(f"<tr>{''.join(cells)}</tr>")

        blocks.append(
            f"""
<section class="month">
<h3>{html.escape(title)}</h3>
<table class="calendar">
<tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th></tr>
{''.join(rows)}
</table>
</section>
"""
        )

    return "\n".join(blocks)


def main():
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)

    # Group digest files by date (a date can have several files if you run multiple configs)
    by_date = defaultdict(list)
    for f in sorted(DIGEST_DIR.glob("*.html")):
        m = DATE_RE.match(f.name)
        if not m:
            continue
        by_date[m.group(1)].append(f)

    dates = sorted(by_date.keys(), reverse=True)
    calendar_html = build_calendar(by_date)

    # ---- per-day rows + weekday aggregation ----
    day_rows = []
    wk_days = defaultdict(int)          # weekday -> number of digest days
    wk_papers = defaultdict(int)        # weekday -> total papers
    wk_scores = defaultdict(list)       # weekday -> all scores
    wk_hits = defaultdict(int)          # weekday -> papers with score >= 7

    for d in dates:
        y, mo, dy = map(int, d.split("-"))
        weekday = WEEKDAYS[date(y, mo, dy).weekday()]
        n_papers, scores = 0, []
        links = []
        for f in by_date[d]:
            info = analyze_digest(f)
            n_papers += info["n_papers"]
            scores += info["scores"]
            label = digest_label(f, d)
            links.append(f'<a href="digests/{html.escape(f.name)}">{html.escape(label)}</a>')

        avg = f"{sum(scores)/len(scores):.1f}" if scores else "–"
        hits = sum(1 for s in scores if s >= 7)
        day_rows.append(
            f"<tr><td>{d}</td><td>{weekday[:3]}</td><td>{n_papers}</td>"
            f"<td>{avg}</td><td>{hits}</td><td>{' · '.join(links)}</td></tr>"
        )

        wk_days[weekday] += 1
        wk_papers[weekday] += n_papers
        wk_scores[weekday] += scores
        wk_hits[weekday] += hits

    # ---- weekday stats table with inline bars ----
    max_avg_papers = max(
        (wk_papers[w] / wk_days[w] for w in WEEKDAYS if wk_days[w]), default=1
    ) or 1
    wk_rows = []
    for w in WEEKDAYS:
        if not wk_days[w]:
            wk_rows.append(f"<tr><td>{w}</td><td colspan=4 class='dim'>no digests yet</td></tr>")
            continue
        avg_papers = wk_papers[w] / wk_days[w]
        avg_score = (
            f"{sum(wk_scores[w])/len(wk_scores[w]):.2f}" if wk_scores[w] else "–"
        )
        avg_hits = wk_hits[w] / wk_days[w]
        bar_w = int(100 * avg_papers / max_avg_papers)
        wk_rows.append(
            f"<tr><td>{w}</td><td>{wk_days[w]}</td>"
            f"<td>{avg_papers:.1f} <span class='bar' style='width:{bar_w}px'></span></td>"
            f"<td>{avg_score}</td><td>{avg_hits:.1f}</td></tr>"
        )

    total_papers = sum(wk_papers.values())
    all_scores = [s for w in WEEKDAYS for s in wk_scores[w]]
    overall_avg = f"{sum(all_scores)/len(all_scores):.2f}" if all_scores else "–"

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Daily reading notes</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1c1c1c; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2.2rem; }}
  h3 {{ font-family: Helvetica, Arial, sans-serif; font-size: 0.9rem; margin: 1.2rem 0 0.4rem; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #ddd; vertical-align: top; }}
  th {{ font-family: Helvetica, Arial, sans-serif; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #666; }}
  .calendar {{ table-layout: fixed; margin-bottom: 1.5rem; }}
  .calendar th, .calendar td {{ width: 14.28%; height: 4.4rem; padding: 0.35rem; border: 1px solid #e3e3e3; }}
  .calendar th {{ height: auto; text-align: center; }}
  .calendar td {{ background: #fafafa; }}
  .calendar .cal-hit {{ background: #fff; }}
  .cal-day {{ font-family: Helvetica, Arial, sans-serif; font-size: 0.78rem; margin-bottom: 0.35rem; }}
  .cal-links a {{ display: block; font-size: 0.78rem; line-height: 1.35; overflow-wrap: anywhere; }}
  .cal-empty {{ background: #f4f4f4; }}
  .bar {{ display: inline-block; height: 0.6em; background: #4a6fa5; vertical-align: middle; margin-left: 6px; }}
  .dim {{ color: #999; }}
  .latest {{ font-size: 1.05rem; margin: 1rem 0; }}
  footer {{ margin-top: 3rem; font-size: 0.8rem; color: #888; }}
</style>
</head>
<body>
<h1>Daily reading notes</h1>
"""
    if dates:
        latest = dates[0]
        latest_links = " · ".join(
            f'<a href="digests/{html.escape(f.name)}">{html.escape(digest_label(f, latest))}</a>'
            for f in by_date[latest]
        )
        page += f'<p class="latest">Latest: <b>{latest}</b> — {latest_links}</p>\n'
    else:
        page += "<p>No digests archived yet.</p>\n"

    page += f"""
<h2>Calendar</h2>
{calendar_html}

<h2>Stats by day of week</h2>
<p>{len(dates)} digest days · {total_papers} papers total · overall avg relevancy {overall_avg}</p>
<table>
<tr><th>Weekday</th><th>Digests</th><th>Avg papers/day</th><th>Avg relevancy</th><th>Avg papers ≥7/day</th></tr>
{''.join(wk_rows)}
</table>

<h2>All days</h2>
<table>
<tr><th>Date</th><th>Day</th><th>Papers</th><th>Avg score</th><th>≥7</th><th>Digest</th></tr>
{''.join(day_rows)}
</table>

<footer>Rebuilt automatically after each update. Some days are expected to be quiet.</footer>
</body>
</html>
"""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(page, encoding="utf-8")
    print(f"Wrote {INDEX_PATH} ({len(dates)} days, {total_papers} papers)")


if __name__ == "__main__":
    main()
