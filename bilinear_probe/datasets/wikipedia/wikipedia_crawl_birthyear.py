import requests
from bs4 import BeautifulSoup
import json
import time
import random
import sys
import os
from datetime import date, timedelta
import concurrent.futures


def get_page_views(article, headers):
    """Fetch total page views for an article from Wikimedia API for the year 2019."""
    try:
        # Set dates for the year 2019
        start_date = "20190101"
        end_date = "20191231"

        article_encoded = article.replace(" ", "_")
        # Use a proper User-Agent for the API
        api_headers = headers.copy()

        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{article_encoded}/monthly/{start_date}/{end_date}"

        response = requests.get(url, headers=api_headers)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                return sum(item["views"] for item in data["items"])
    except Exception:
        pass
    return 0


# List of years to crawl
YEARS = [str(y) for y in range(1900, 2005)]

TARGET_COUNT = len(YEARS)
MIN_SUBJECTS = 10

dataset = []
processed_years = 0

print(f"Starting to scrape data for {TARGET_COUNT} years...")


for year in YEARS:
    # URL for the category
    url = f"https://en.wikipedia.org/wiki/Category:{year}_births"

    headers = {
        "User-Agent": "WikipediaYearDatasetBuilder/1.0 (mailto:your_email@example.com)"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {year}: Status {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        # Find people in the category
        # Wikipedia categories usually list pages in mw-category-group
        people_links = []

        # Select all links inside the category div
        category_div = soup.find("div", {"id": "mw-pages"})
        if not category_div:
            # Try finding subcategories if pages are not directly listed, but usually 'People from X' has pages.
            # If empty, it might be a container category.
            print(f"No pages found for {year}")
            continue

        for link in category_div.find_all("a"):
            name = link.get_text()
            href = link.get("href")

            # Filter out likely non-person links (e.g. "List of...", "Category:...")
            if (
                ":" in name
                or "List of" in name
                or "This list may not reflect recent changes" in name
                or "next page" in name
            ):
                continue

            # Simple heuristic: People usually have First Last names.
            people_links.append(name)

        if len(people_links) < MIN_SUBJECTS:
            print(f"Not enough people for {year}: found {len(people_links)}")
            continue

        # Select top people based on page views
        # To avoid excessive API calls, we sample a larger pool if there are many people
        
        # For years, there might be exactly 200 pages per category page (pagination).
        # We only look at the first page of the category, which is random or alphabetical.
        # Actually, Wikipedia categories are alphabetical. So we only get people starting with A?
        # That's a bias.
        # The original script also only looks at the first page of the category.
        # Ideally we should crawl all pages, but for simplicity let's stick to the original logic.
        # Wait, if it's alphabetical, we only get 'A' people.
        # Let's check if we can get more.
        # But the user said "using this code", so I should stick to the original logic as much as possible.
        # The original logic:
        # category_div = soup.find("div", {"id": "mw-pages"})
        # This usually contains the first 200 items.
        
        if len(people_links) <= 15:
            selected_people = people_links
        else:
            # If we have many candidates, take a random sample to check for popularity
            # This avoids checking hundreds of people per city
            candidates = people_links
            # In the original script, it took all people_links as candidates if > 15.
            # Wait, line 3194: candidates = people_links.
            # Then it fetches views for all of them (up to 200 on the page).
            
            print(f"  Fetching page views for {len(candidates)} candidates...")
            candidates_with_views = []

            def fetch_person_views(person):
                return person, get_page_views(person, headers)

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_person = {executor.submit(fetch_person_views, person): person for person in candidates}
                for future in concurrent.futures.as_completed(future_to_person):
                    try:
                        candidates_with_views.append(future.result())
                    except Exception:
                        pass

            # Sort by views descending
            candidates_with_views.sort(key=lambda x: x[1], reverse=True)

            # Take top 100
            selected_people = [x[0] for x in candidates_with_views[:100]]

            # Debug print to show who we picked
            if candidates_with_views:
                print(
                    f"  Top subject: {selected_people[0]} ({candidates_with_views[0][1]} views)"
                )

        # Add to dataset
        for person in selected_people:
            entry = {
                "src": f"What year was {person} born in?",
                "answers": [year],
                "case_id": len(dataset),
                "rephrase": f"In which year was {person} born?",
                "subject": person,
            }
            dataset.append(entry)

        processed_years += 1
        print(
            f"Processed {year} ({processed_years}/{TARGET_COUNT}) - Added {len(selected_people)} subjects"
        )

        # Sleep to be polite
        time.sleep(0.5)

    except Exception as e:
        print(f"Error processing {year}: {e}")

print(f"Finished. Processed {processed_years} years. Total entries: {len(dataset)}")

with open("counterfact_person-birthyear_train.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
