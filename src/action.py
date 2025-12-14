from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from datetime import date

import argparse
import yaml
import os
from dotenv import load_dotenv
import openai

from relevancy import generate_relevance_score, process_subject_fields
from download_new_papers import get_papers


# Hackathon quality code. Don't judge too harshly.
# Feel free to submit pull requests to improve the code.

topics = {
    "Physics": "",
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


# TODO: surely theres a better way
category_map = {
    "Astrophysics": [
        "Astrophysics of Galaxies",
        "Cosmology and Nongalactic Astrophysics",
        "Earth and Planetary Astrophysics",
        "High Energy Astrophysical Phenomena",
        "Instrumentation and Methods for Astrophysics",
        "Solar and Stellar Astrophysics",
    ],
    "Condensed Matter": [
        "Disordered Systems and Neural Networks",
        "Materials Science",
        "Mesoscale and Nanoscale Physics",
        "Other Condensed Matter",
        "Quantum Gases",
        "Soft Condensed Matter",
        "Statistical Mechanics",
        "Strongly Correlated Electrons",
        "Superconductivity",
    ],
    "General Relativity and Quantum Cosmology": ["None"],
    "High Energy Physics - Experiment": ["None"],
    "High Energy Physics - Lattice": ["None"],
    "High Energy Physics - Phenomenology": ["None"],
    "High Energy Physics - Theory": ["None"],
    "Mathematical Physics": ["None"],
    "Nonlinear Sciences": [
        "Adaptation and Self-Organizing Systems",
        "Cellular Automata and Lattice Gases",
        "Chaotic Dynamics",
        "Exactly Solvable and Integrable Systems",
        "Pattern Formation and Solitons",
    ],
    "Nuclear Experiment": ["None"],
    "Nuclear Theory": ["None"],
    "Physics": [
        "Accelerator Physics",
        "Applied Physics",
        "Atmospheric and Oceanic Physics",
        "Atomic and Molecular Clusters",
        "Atomic Physics",
        "Biological Physics",
        "Chemical Physics",
        "Classical Physics",
        "Computational Physics",
        "Data Analysis, Statistics and Probability",
        "Fluid Dynamics",
        "General Physics",
        "Geophysics",
        "History and Philosophy of Physics",
        "Instrumentation and Detectors",
        "Medical Physics",
        "Optics",
        "Physics and Society",
        "Physics Education",
        "Plasma Physics",
        "Popular Physics",
        "Space Physics",
    ],
    "Quantum Physics": ["None"],
    "Mathematics": [
        "Algebraic Geometry",
        "Algebraic Topology",
        "Analysis of PDEs",
        "Category Theory",
        "Classical Analysis and ODEs",
        "Combinatorics",
        "Commutative Algebra",
        "Complex Variables",
        "Differential Geometry",
        "Dynamical Systems",
        "Functional Analysis",
        "General Mathematics",
        "General Topology",
        "Geometric Topology",
        "Group Theory",
        "History and Overview",
        "Information Theory",
        "K-Theory and Homology",
        "Logic",
        "Mathematical Physics",
        "Metric Geometry",
        "Number Theory",
        "Numerical Analysis",
        "Operator Algebras",
        "Optimization and Control",
        "Probability",
        "Quantum Algebra",
        "Representation Theory",
        "Rings and Algebras",
        "Spectral Theory",
        "Statistics Theory",
        "Symplectic Geometry",
    ],
    "Computer Science": [
        "Artificial Intelligence",
        "Computation and Language",
        "Computational Complexity",
        "Computational Engineering, Finance, and Science",
        "Computational Geometry",
        "Computer Science and Game Theory",
        "Computer Vision and Pattern Recognition",
        "Computers and Society",
        "Cryptography and Security",
        "Data Structures and Algorithms",
        "Databases",
        "Digital Libraries",
        "Discrete Mathematics",
        "Distributed, Parallel, and Cluster Computing",
        "Emerging Technologies",
        "Formal Languages and Automata Theory",
        "General Literature",
        "Graphics",
        "Hardware Architecture",
        "Human-Computer Interaction",
        "Information Retrieval",
        "Information Theory",
        "Logic in Computer Science",
        "Machine Learning",
        "Mathematical Software",
        "Multiagent Systems",
        "Multimedia",
        "Networking and Internet Architecture",
        "Neural and Evolutionary Computing",
        "Numerical Analysis",
        "Operating Systems",
        "Other Computer Science",
        "Performance",
        "Programming Languages",
        "Robotics",
        "Social and Information Networks",
        "Software Engineering",
        "Sound",
        "Symbolic Computation",
        "Systems and Control",
    ],
    "Quantitative Biology": [
        "Biomolecules",
        "Cell Behavior",
        "Genomics",
        "Molecular Networks",
        "Neurons and Cognition",
        "Other Quantitative Biology",
        "Populations and Evolution",
        "Quantitative Methods",
        "Subcellular Processes",
        "Tissues and Organs",
    ],
    "Quantitative Finance": [
        "Computational Finance",
        "Economics",
        "General Finance",
        "Mathematical Finance",
        "Portfolio Management",
        "Pricing of Securities",
        "Risk Management",
        "Statistical Finance",
        "Trading and Market Microstructure",
    ],
    "Statistics": [
        "Applications",
        "Computation",
        "Machine Learning",
        "Methodology",
        "Other Statistics",
        "Statistics Theory",
    ],
    "Electrical Engineering and Systems Science": [
        "Audio and Speech Processing",
        "Image and Video Processing",
        "Signal Processing",
        "Systems and Control",
    ],
    "Economics": ["Econometrics", "General Economics", "Theoretical Economics"],
}


def generate_body(topic, categories, interest, threshold, fallback_n=15):
    """
    fallback_n: if LLM filtering returns nothing above threshold, include top N newest papers anyway
    """
    categories = categories or []

    # ---- Determine arXiv archive abbrev and normalize categories ----
    if topic == "Physics":
        # "physics" is the archive prefix for physics.*
        abbr = "physics"

        # map human-readable -> physics.* codes
        physics_map = {
            "Applied Physics": "physics.app-ph",
            "Physics Education": "physics.ed-ph",
            "History and Philosophy of Physics": "physics.hist-ph",
            "Instrumentation and Detectors": "physics.ins-det",
            "Optics": "physics.optics",
            # you can add more mappings here later if you want
        }
        allowed_codes = set(physics_map.values())

        if not categories:
            raise RuntimeError(
                "For topic 'Physics', you must provide at least one subtopic "
                "(e.g. ['Optics', 'Applied Physics'])."
            )

        normalized = []
        for c in categories:
            if c in physics_map:
                normalized.append(physics_map[c])
            elif c in allowed_codes:
                normalized.append(c)
            else:
                raise RuntimeError(
                    f"Unknown Physics category '{c}'. Use one of: {list(physics_map.keys())} "
                    f"or one of: {sorted(list(allowed_codes))}"
                )
        categories = normalized

    elif topic in physics_topics:
        abbr = physics_topics[topic]
        if categories == ["None"]:
            categories = []

    elif topic in topics:
        abbr = topics[topic]

    else:
        raise RuntimeError(f"Invalid topic {topic}")

    # ---- Download and filter papers ----
    papers = get_papers(abbr)

    if categories:
        # For non-Physics topics, categories are human-readable; validate against category_map
        if topic != "Physics":
            for category in categories:
                if category not in category_map.get(topic, []):
                    raise RuntimeError(f"{category} is not a category of {topic}")

        # Filter by subject fields (usually codes like physics.optics, quant-ph, cs.AI)
                # Filter by subject fields (usually codes like physics.optics, quant-ph, cs.AI)
        filtered = [
            t
            for t in papers
            if bool(set(process_subject_fields(t["subjects"])) & set(categories))
        ]

        # If filtering returns nothing (can happen due to subject-format mismatch),
        # fall back to the unfiltered recent list so the digest isn't empty.
        papers = filtered if filtered else papers


    # ---- Relevance scoring and HTML body ----
    if interest:
        relevancy, hallucination = generate_relevance_score(
            papers,
            query={"interest": interest},
            threshold_score=threshold,
            num_paper_in_prompt=16,
        )

        # Fallback: if nothing passes threshold, include top N newest papers anyway
        used_fallback = False
        final_list = relevancy
        if not final_list:
            used_fallback = True
            final_list = papers[:fallback_n]

        # Render
        body_parts = []

        if hallucination:
            body_parts.append(
                "Warning: the model hallucinated some papers. We have tried to remove them, "
                "but the scores may not be accurate.<br><br>"
            )

        if used_fallback:
            body_parts.append(
                f"No papers exceeded the relevance threshold ({threshold}) for this run. "
                f"Showing the {fallback_n} most recent papers instead.<br><br>"
            )

        def render_paper(p):
            # If it came from relevancy results, it likely has score + reason fields
            score = p.get("Relevancy score", "")
            reason = p.get("Reasons for match", "")
            extra = ""
            if score != "" or reason != "":
                extra = f"<br>Score: {score}<br>Reason: {reason}"
            return (
                f'Title: <a href="{p["main_page"]}">{p["title"]}</a><br>'
                f'Authors: {p["authors"]}{extra}'
            )

        body_parts.append("<br><br>".join(render_paper(p) for p in final_list))
        body = "".join(body_parts)

    else:
        # No interest means no filtering/scoring; just list papers
        body = "<br><br>".join(
            [
                f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}'
                for paper in papers
            ]
        )

    return body


if __name__ == "__main__":
    # Load the .env file (optional locally; GitHub Actions uses env vars directly)
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="yaml config file to use", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("No openai api key found")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    topic = config["topic"]
    categories = config.get("categories", [])
    threshold = config.get("threshold", 7)
    interest = config.get("interest", "")

    from_email = os.environ.get("FROM_EMAIL")
    to_email = os.environ.get("TO_EMAIL")

    body = generate_body(topic, categories, interest, threshold)

    with open("digest.html", "w") as f:
        f.write(body)

    if os.environ.get("SENDGRID_API_KEY", None):
        sg = SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
        from_email = Email(from_email)  # must be verified in SendGrid
        to_email = To(to_email)
        subject = date.today().strftime("Personalized arXiv Digest, %d %b %Y")
        content = Content("text/html", body)
        mail = Mail(from_email, to_email, subject, content)

        response = sg.client.mail.send.post(request_body=mail.get())
        if 200 <= response.status_code <= 299:
            print("Send test email: Success!")
        else:
            print(f"Send test email: Failure ({response.status_code}, {response.body})")
    else:
        print("No sendgrid api key found. Skipping email")
