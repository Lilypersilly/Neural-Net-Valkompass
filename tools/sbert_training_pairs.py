import os
import json
import random
import re
import requests

DATASET_DIR = "datasets"
OUTPUT_FILE = "sbert_training_pairs_test.json"
TARGET_GOAL = 500
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def load_all_statements():
    """Loads all party statements and groups them by pillar."""
    pillar_data = {"Economy": [], "Climate": [], "Immigration": [], "Law And Order": []}
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Could not find '{DATASET_DIR}' folder.")
        return pillar_data
        
    for filename in sorted(os.listdir(DATASET_DIR)):
        if filename.endswith(".json") and not filename.startswith("_"):
            filepath = os.path.join(DATASET_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                pillars = data.get("pillars", {})
                for pillar, items in pillars.items():
                    if pillar in pillar_data:
                        if isinstance(items, list):
                            for item in items:
                                text = item if isinstance(item, str) else item.get("stance", "")
                                if text.strip():
                                    pillar_data[pillar].append(text.strip())
                        elif isinstance(items, dict):
                            for _, val in items.items():
                                text = val if isinstance(val, str) else val.get("stance", "")
                                if text.strip():
                                    pillar_data[pillar].append(text.strip())
            except Exception:
                pass
    return pillar_data


def get_random_pair(pillar_data):
    """Generates a pair of sentences with forced keyword overlap to boost 'Yes' rates."""
    pillars = [p for p, texts in pillar_data.items() if len(texts) > 1]
    if not pillars:
        return None, None
        
    rand_val = random.random()
    
    # 30% chans: Olika sakområden (Garanterat 'Nej')
    if rand_val < 0.30:
        p1, p2 = random.sample(pillars, 2)
        text1 = random.choice(pillar_data[p1])
        text2 = random.choice(pillar_data[p2])
        return text1, text2
        
    # 20% chans: Helt slumpmässigt från samma sakområde (Oftast 'Nej', ibland 'Ja')
    elif rand_val < 0.50:
        pillar = random.choice(pillars)
        text1, text2 = random.sample(pillar_data[pillar], 2)
        return text1, text2
        
    # 50% chans: Tvingad nyckelordsmatchning (Hög chans för 'Ja'!)
    else:
        pillar = random.choice(pillars)
        text1 = random.choice(pillar_data[pillar])
        
        # Plocka ut ord som är längre än 4 bokstäver (rensar bort 'och', 'att', 'en', etc.)
        keywords1 = set(w.lower() for w in text1.split() if len(w) > 4)
        
        best_text2 = None
        best_overlap = -1
        
        # Leta efter en annan mening som delar flest nyckelord
        for candidate in pillar_data[pillar]:
            if candidate == text1:
                continue
                
            keywords2 = set(w.lower() for w in candidate.split() if len(w) > 4)
            overlap = len(keywords1.intersection(keywords2))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_text2 = candidate
                
        # Om vi hittar en mening som delar minst ett nyckelord, ta den!
        if best_text2 and best_overlap > 0:
            return text1, best_text2
        else:
            # Fallback om vi inte hittar någon överlappning
            text1, text2 = random.sample(pillar_data[pillar], 2)
            return text1, text2


def get_ai_recommendation(text1: str, text2: str) -> str:
    """Queries local DeepSeek-R1 via Ollama for a labeling recommendation."""
    prompt = f"""You are a Swedish political data curator. We are training a relevance retriever.
Task: Determine if Sentence 1 and Sentence 2 discuss the EXACT SAME specific political wedge/sub-topic (even if taking opposite stances).
- Respond 'y' if they address the same specific issue (e.g., both debate nuclear power vs wind, or both debate benefit caps).
- Respond 'n' if they talk about different issues (e.g., one talks about nuclear power, the other about green industrial subsidies, or one talks about welfare profits and the other ethnic migration).

Sentence 1: "{text1}"
Sentence 2: "{text2}"

Format your final answer strictly as:
RECOMMENDATION: [y/n]
MOTIVATION: [1 short sentence in English explaining why]"""

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if res.status_code == 200:
            raw_response = res.json().get("response", "")
            # Strip R1's internal <think> reasoning tags to keep terminal clean
            clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            return clean_text
        return "Kunde inte ansluta till Ollama (Status: " + str(res.status_code) + ")"
    except Exception as e:
        return f"Ollama offline eller otillgänglig ({e})"


def main():
    pillar_data = load_all_statements()
    
    labeled_data = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)
            
    print("=== SBERT RELEVANCE LABELING TOOL (AI-ASSISTED) ===")
    print("Instructions:")
    print(" - Press 'y' if the sentences are discussing the SAME specific topic/issue.")
    print(" - Press 'n' if they are discussing DIFFERENT topics.")
    print(" - Press 's' to skip.")
    print(" - Press 'q' to quit and save.")
    print(f"Target goal: {TARGET_GOAL} pairs")
    print(f"Currently labeled: {len(labeled_data)}\n")
    
    input("Press Enter to start...")

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        text1, text2 = get_random_pair(pillar_data)
        
        if not text1:
            print("Not enough data in datasets/ to generate pairs.")
            break
            
        total_count = len(labeled_data)
        yes_count = sum(1 for item in labeled_data if item.get("label") == 1)
        no_count = sum(1 for item in labeled_data if item.get("label") == 0)
        progress_pct = (total_count / TARGET_GOAL) * 100 if TARGET_GOAL > 0 else 100.0

        print("=" * 60)
        print(f"PROGRESS: [{total_count}/{TARGET_GOAL}] ({progress_pct:.1f}%) | YES: {yes_count} | NO: {no_count}")
        print("=" * 60)
        print(f"\nSentence 1:\n  \"{text1}\"\n")
        print("-" * 60)
        print(f"\nSentence 2:\n  \"{text2}\"\n")
        print("=" * 60)
        
        print("[DeepSeek-R1 analyserar...]")
        ai_opinion = get_ai_recommendation(text1, text2)
        print(f"\nAI-ASSISTENT:\n{ai_opinion}\n")
        print("=" * 60)
        
        choice = input("Your choice (y/n/s/q) [Enter without text adopts AI or choose manually]: ").strip().lower()
        
        # Shortcut: if you just press enter, extract AI recommendation if available
        if choice == '':
            if 'RECOMMENDATION: y' in ai_opinion.lower() or 'recommendation: y' in ai_opinion:
                choice = 'y'
            elif 'RECOMMENDATION: n' in ai_opinion.lower() or 'recommendation: n' in ai_opinion:
                choice = 'n'
            else:
                continue

        if choice == 'q':
            break
        elif choice == 's':
            continue
        elif choice in ['y', 'n']:
            label = 1 if choice == 'y' else 0
            labeled_data.append({
                "text1": text1,
                "text2": text2,
                "label": label
            })
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(labeled_data, f, ensure_ascii=False, indent=2)

    print(f"\nSession finished. Total labeled pairs saved: {len(labeled_data)} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()