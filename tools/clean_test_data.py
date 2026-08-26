import json
import os

TEST_FILE = "sbert_training_pairs_test.json"
TRAIN_FILE = "sbert_training_pairs.json"

def clean_and_deduplicate_test_set():
    if not os.path.exists(TEST_FILE):
        print(f"Hittade inte {TEST_FILE}. Kontrollera filnamnet.")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Ladda träningsdatan för att undvika data leakage
    train_pairs = set()
    if os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            train_data = json.load(f)
            for item in train_data:
                pair = tuple(sorted([item["text1"], item["text2"]]))
                train_pairs.add(pair)

    seen_in_test = set()
    unique_test_data = []
    duplicates_count = 0
    leakage_count = 0

    for item in test_data:
        pair = tuple(sorted([item["text1"], item["text2"]]))
        
        # Kolla om den är en dubblett internt i testsetet
        if pair in seen_in_test:
            duplicates_count += 1
            continue
            
        # Kolla om den finns i träningsdatan (så modellen inte "fuskar")
        if pair in train_pairs:
            leakage_count += 1
            continue

        seen_in_test.add(pair)
        unique_test_data.append(item)

    # Spara över den städade testfilen
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_test_data, f, ensure_ascii=False, indent=2)

    print("="*50)
    print(f"RENSNING KLAR AV TEST-DATASETET")
    print("="*50)
    print(f" - Startade med: {len(test_data)} par")
    print(f" - Borttagna interna dubbletter: {duplicates_count}")
    print(f" - Borttagna från träningsdatan (läckage): {leakage_count}")
    print(f" - Slutgiltiga unika testpar: {len(unique_test_data)}")

if __name__ == "__main__":
    clean_and_deduplicate_test_set()