import json
import random
import requests
import concurrent.futures
import time
import os

# File paths
# Assuming the script is run from the root directory or we need to handle paths relative to where it is.
# The user moved it to bilinear_probe/datasets/
# But the test file is likely in bilinear_probe/datasets/ too?
# Let's check where the test file is.
# The user said @[bilinear_probe/datasets/counterfact_person-city_test_wikipedia.json]
# So if the script is in bilinear_probe/datasets/, the file is in the same dir.

CITY_TEST_FILE = "counterfact_person-city_test_wikipedia.json"
OUTPUT_FILE = "counterfact_person-birthyear_test.json"

def load_json(filepath):
    # Try to find the file in current dir or absolute path
    if not os.path.exists(filepath):
        # Try looking in bilinear_probe/datasets/ if running from root
        alt_path = os.path.join("bilinear_probe", "datasets", filepath)
        if os.path.exists(alt_path):
            return json.load(open(alt_path, "r", encoding="utf-8"))
        
        # Try looking in current dir if running from bilinear_probe/datasets/
        # (Already covered by first check)
        
        raise FileNotFoundError(f"Could not find {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_birth_year(subject):
    """
    Fetches the birth year of a subject using Wikidata API.
    """
    try:
        # Step 1: Search for the entity
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action": "wbsearchentities",
            "search": subject,
            "language": "en",
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "WikipediaBirthYearCrawler/1.0 (mailto:your_email@example.com)"
        }
        
        search_response = requests.get(search_url, params=search_params, headers=headers)
        search_data = search_response.json()
        
        if not search_data.get("search"):
            return None
            
        qid = search_data["search"][0]["id"]
        
        # Step 2: Get entity details (Property P569 is Date of Birth)
        entity_url = "https://www.wikidata.org/w/api.php"
        entity_params = {
            "action": "wbgetclaims",
            "entity": qid,
            "property": "P569",
            "format": "json"
        }
        
        entity_response = requests.get(entity_url, params=entity_params, headers=headers)
        entity_data = entity_response.json()
        
        claims = entity_data.get("claims", {}).get("P569", [])
        if not claims:
            return None
            
        # Extract date string
        date_str = claims[0]["mainsnak"]["datavalue"]["value"]["time"]
        
        # Parse year
        if date_str.startswith("+") or date_str.startswith("-"):
            clean_date_str = date_str.lstrip("+")
            parts = clean_date_str.split("-")
            year = parts[0]
            return year
            
    except Exception:
        pass
        
    return None

def main():
    print("Loading datasets...")
    try:
        city_test_data = load_json(CITY_TEST_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # Get list of subjects in the test set
    test_subjects = list(set(item["subject"] for item in city_test_data))
    print(f"Found {len(test_subjects)} unique subjects in the city test set.")

    subject_to_birthyear = {}
    
    print("Fetching birth years from Wikidata...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_subject = {executor.submit(get_birth_year, subject): subject for subject in test_subjects}
        
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                birth_year = future.result()
                if birth_year:
                    subject_to_birthyear[subject] = birth_year
            except Exception as e:
                print(f"Error fetching {subject}: {e}")
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(test_subjects)} subjects. Found {len(subject_to_birthyear)} birth years.", end="\r")

    print(f"\nFound birth years for {len(subject_to_birthyear)} of the test subjects.")

    new_dataset = []
    
    for i, city_item in enumerate(city_test_data):
        subject = city_item["subject"]
        
        # Skip if we don't have birth year data
        if subject not in subject_to_birthyear:
            continue

        real_birthyear_str = subject_to_birthyear[subject]
        try:
            real_birthyear = int(real_birthyear_str)
        except ValueError:
            continue

        # Generate counterfactual birth year
        while True:
            target_birthyear = random.randint(1950, 2000)
            if target_birthyear != real_birthyear:
                break
        
        target_birthyear_str = str(target_birthyear)

        # Create Portability Question: Age Calculation
        reference_year = target_birthyear + random.randint(10, 50)
        expected_age = reference_year - target_birthyear
        
        # Construct the entry
        entry = {
            "case_id": len(new_dataset),
            "subject": subject,
            "src": f"What year was {subject} born in?",
            "rephrase": f"In which year was {subject} born?",
            "answers": [real_birthyear_str],
            "alt": target_birthyear_str,
            "portability": {
                "Recalled Relation": f"({subject}, birth year, {target_birthyear_str})",
                "New Question": f"How old was {subject} in {reference_year}?",
                "New Answer": str(expected_age)
            }
        }

        # Extract description from city dataset's Two-hop Question if available
        if "portability" in city_item and "Two-hop Question" in city_item["portability"]:
            city_two_hop = city_item["portability"]["Two-hop Question"]
            prefix = "In which country was "
            suffix = " born?"
            if city_two_hop.startswith(prefix) and city_two_hop.endswith(suffix):
                description = city_two_hop[len(prefix):-len(suffix)]
                entry["portability"]["Two-hop Question"] = f"In which year was {description} born?"

        new_dataset.append(entry)

    print(f"Generated {len(new_dataset)} entries.")
    save_json(new_dataset, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
