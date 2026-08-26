import json

# Läs in din märkta data
with open("sbert_training_pairs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()
unique_data = []

for item in data:
    # Sortera meningarna så att A-B och B-A räknas som samma dubblett!
    pair = tuple(sorted([item["text1"], item["text2"]]))
    
    if pair not in seen:
        seen.add(pair)
        unique_data.append(item)

# Spara över filen (eller till en ny fil) med enbart unika par
with open("sbert_training_pairs.json", "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=2)

print(f"Rensning klar! Gick från {len(data)} till {len(unique_data)} unika par.")