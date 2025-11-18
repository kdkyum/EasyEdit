import json
import os

FILE_PATH = "/u/kdkyum/ptmp_link/workdir/iclr2026_rebuttal_bilinear/EasyEdit/data/counterfact_city-country_test.json"
TARGET_LOC = (
    "multiple-choice question: In which country was Francesco Totti born? "
    "Italy; Spain; Germany; France. Answer:"
)

def main():
    if not os.path.exists(FILE_PATH):
        raise SystemExit(f"File not found: {FILE_PATH}")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["loc"] = TARGET_LOC

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated {len(data)} entries to uniform loc text.")

if __name__ == "__main__":
    main()
