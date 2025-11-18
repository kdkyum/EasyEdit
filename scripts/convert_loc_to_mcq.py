import json
import os
import re

FILE_PATH = "/u/kdkyum/ptmp_link/workdir/iclr2026_rebuttal_bilinear/EasyEdit/data/counterfact_city-country_test.json"

# Person -> correct country mapping
PERSON_COUNTRY = {
    "Adele": "United Kingdom",
    "Marlene Dietrich": "Germany",
    "John Lennon": "United Kingdom",
    "Francesco Totti": "Italy",
    "Anne Frank": "Germany",
    "Marilyn Monroe": "United States",
    "Donald Trump": "United States",
    "Angela Merkel": "Germany",
    "Pau Gasol": "Spain",
    "Naomi Osaka": "Japan",
    "Hayao Miyazaki": "Japan",
    "Sun Yat-sen": "China",
    "Yao Ming": "China",
    "Xi Jinping": "China",
    "Drake": "Canada",
}

# Preferred fallback order to avoid alt == loc_ans
FALLBACK_PERSON_ORDER = [
    "Adele",
    "Marlene Dietrich",
    "Francesco Totti",
    "John Lennon",
    "Marilyn Monroe",
    "Pau Gasol",
    "Naomi Osaka",
    "Sun Yat-sen",
    "Yao Ming",
    "Xi Jinping",
    "Drake",
    "Angela Merkel",
    "Anne Frank",
    "Hayao Miyazaki",
]


def options_for_country(correct: str):
    if correct == "United Kingdom":
        return ["United Kingdom", "Germany", "France", "Italy"]
    if correct == "Germany":
        return ["Germany", "United Kingdom", "France", "Italy"]
    if correct == "France":
        return ["France", "United Kingdom", "Germany", "Italy"]
    if correct == "Italy":
        return ["Italy", "Spain", "Germany", "France"]
    if correct == "Spain":
        return ["Spain", "Italy", "Germany", "France"]
    if correct == "United States":
        return ["United States", "Canada", "United Kingdom", "Germany"]
    if correct == "Canada":
        return ["Canada", "United States", "United Kingdom", "Germany"]
    if correct == "Japan":
        return ["Japan", "China", "South Korea", "United States"]
    if correct == "China":
        return ["China", "Japan", "South Korea", "United States"]
    if correct == "South Korea":
        return ["South Korea", "Japan", "China", "United States"]
    # default broad set if an unexpected value appears
    pool = [
        "United Kingdom",
        "Germany",
        "France",
        "Italy",
        "Spain",
        "United States",
        "Canada",
        "Japan",
        "China",
        "South Korea",
    ]
    opts = [correct] + [c for c in pool if c != correct][:3]
    return opts


def extract_person(loc: str):
    # Expect pattern: "nq question: In which country was {NAME} born? ..."
    m = re.search(r"In which country was\s+(.+?)\s+born\?", loc)
    return m.group(1) if m else None


def build_mcq(person: str, correct: str) -> str:
    options = options_for_country(correct)
    opts_text = " ".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))
    return f"nq question: In which country was {person} born? {opts_text}"


def main():
    if not os.path.exists(FILE_PATH):
        raise SystemExit(f"File not found: {FILE_PATH}")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        alt = item.get("alt", "")
        loc = item.get("loc", "")
        loc_ans = item.get("loc_ans", "")

        person = extract_person(loc) or "Adele"
        # Determine correct from mapping if available
        correct = PERSON_COUNTRY.get(person, loc_ans)

        # If alt conflicts with correct answer, pick a fallback person
        if alt == correct or not correct:
            for cand in FALLBACK_PERSON_ORDER:
                cand_country = PERSON_COUNTRY[cand]
                if cand_country != alt:
                    person = cand
                    correct = cand_country
                    break

            # Update loc_ans to new correct if changed
            item["loc_ans"] = correct
        else:
            # Keep original loc_ans if it matches mapping; else sync to mapping
            if loc_ans != correct:
                item["loc_ans"] = correct

        # Rebuild MCQ `loc`
        item["loc"] = build_mcq(person, item["loc_ans"])

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
