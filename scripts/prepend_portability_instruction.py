import json
from pathlib import Path

PREFIX = "Answer the following question directly, without any other text before or after your answer. "


def process_file(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] Failed to read {path}: {e}")
        return False

    if not isinstance(data, list):
        print(f"[SKIP] {path} is not a list of records.")
        return False

    changed = False
    for rec in data:
        if not isinstance(rec, dict):
            continue
        port = rec.get("portability")
        if not isinstance(port, dict):
            continue
        nq = port.get("New Question")
        if not isinstance(nq, str):
            continue
        if not nq.startswith(PREFIX):
            port["New Question"] = PREFIX + nq
            changed = True

    if changed:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.write("\n")
            print(f"[OK] Updated {path}")
            return True
        except Exception as e:
            print(f"[ERR] Failed to write {path}: {e}")
            return False
    else:
        print(f"[OK] No changes needed for {path}")
        return False


def main():
    root = Path(__file__).resolve().parents[1]  # project root
    data_dir = root / "data"
    targets = sorted(data_dir.glob("*.json"))
    if not targets:
        print(f"No JSON files found in {data_dir}")
        return
    updated = 0
    for p in targets:
        if process_file(p):
            updated += 1
    print(f"Done. Files updated: {updated}/{len(targets)}")


if __name__ == "__main__":
    main()
