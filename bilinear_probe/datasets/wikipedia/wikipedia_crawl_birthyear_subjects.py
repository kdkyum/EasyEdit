import requests
import json
import time
import concurrent.futures
from datetime import datetime

# Load subjects from the provided JSON file
INPUT_FILE = "counterfact_person-city_train_wikipedia.json"
OUTPUT_FILE = "counterfact_person-birthyear_train.json"

print(f"Loading subjects from {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Extract unique subjects
subjects = list(set(item["subject"] for item in input_data))
print(f"Found {len(subjects)} unique subjects.")

dataset = []
processed_count = 0

def get_birth_year(subject):
    """
    Fetches the birth year of a subject using Wikidata API.
    1. Search for the subject to get the Wikidata ID (QID).
    2. Fetch the entity data to get the birth date (P569).
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
        # The date format in Wikidata is usually like "+1990-01-01T00:00:00Z"
        date_str = claims[0]["mainsnak"]["datavalue"]["value"]["time"]
        
        # Parse year
        # Handle cases like "+1990-..." or "-0551-..." (BC)
        if date_str.startswith("+") or date_str.startswith("-"):
            # Simple parsing: extract the year part
            # +1990-01-01... -> 1990
            # -0551-00-00... -> 551 BC (but we probably just want the year number or string)
            # Let's try to be robust.
            
            # Remove the leading +
            clean_date_str = date_str.lstrip("+")
            
            # Split by - to get year
            parts = clean_date_str.split("-")
            year = parts[0]
            
            # If it was negative, it's BC, but usually we just want the year number for now?
            # Or maybe the user wants "1990".
            # Let's assume standard positive years for now as most people in the dataset are modern.
            
            return year
            
    except Exception as e:
        # print(f"Error fetching data for {subject}: {e}")
        pass
        
    return None

print("Starting to fetch birth years...")

# Use ThreadPoolExecutor for concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all tasks
    future_to_subject = {executor.submit(get_birth_year, subject): subject for subject in subjects}
    
    for i, future in enumerate(concurrent.futures.as_completed(future_to_subject)):
        subject = future_to_subject[future]
        try:
            birth_year = future.result()
            if birth_year:
                # Create dataset entry
                # We need to handle the year format.
                # If it's "0000", it might be invalid.
                # Wikidata years are zero-padded, e.g. "1990".
                
                # Remove leading zeros if any, but keep 0 if it is 0 (unlikely for birth year)
                birth_year_int = int(birth_year)
                birth_year_str = str(birth_year_int)
                
                entry = {
                    "src": f"What year was {subject} born in?",
                    "answers": [birth_year_str],
                    "case_id": len(dataset),
                    "rephrase": f"In which year was {subject} born?",
                    "subject": subject,
                }
                dataset.append(entry)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(subjects)} subjects. Found {len(dataset)} birth years.", end="\r")

print(f"\nFinished. Processed {processed_count} subjects. Total entries: {len(dataset)}")

# Sort by case_id (which effectively is random/insertion order here, but let's just save it)
# Actually, let's re-assign case_ids to be sequential 0..N
for idx, entry in enumerate(dataset):
    entry["case_id"] = idx

print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
