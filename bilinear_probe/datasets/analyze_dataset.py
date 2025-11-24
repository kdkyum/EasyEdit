import json
from collections import Counter

file_path = '/u/kdkyum/ptmp_link/workdir/iclr2026_rebuttal_bilinear/EasyEdit/bilinear_probe/datasets/counterfact_person-city_train.json'

with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

cities = []
for entry in data:
    # Assuming the first answer is the canonical city name
    cities.append(entry['answers'][0])

city_counts = Counter(cities)
print(f"Total unique cities: {len(city_counts)}")
print("Counts per city:")
for city, count in city_counts.items():
    print(f"{city}: {count}")
