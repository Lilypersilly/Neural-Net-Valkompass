import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

SBERT_MODEL_NAME = "models/fine_tuned_sbert"
DATASET_DIR = "datasets"
EXAMPLE_DIR = "exempel"

def load_structured_datasets(dataset_dir=DATASET_DIR):
    """Loads party JSON files maintaining full prompt/topic/stance structures."""
    datasets_by_pillar = {"Economy": {}, "Climate": {}, "Immigration": {}, "Law And Order": {}}
    party_colors = {}
    
    if not os.path.exists(dataset_dir):
        return datasets_by_pillar, party_colors
        
    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith(".json") and not filename.startswith("_"):
            filepath = os.path.join(dataset_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                info = data.get("party_info", {})
                party_name = info.get("name") or data.get("name")
                if not party_name:
                    continue
                    
                party_colors[party_name] = info.get("color", "#888888")
                pillars_dict = data.get("pillars", {})
                
                for pillar, items in pillars_dict.items():
                    if pillar not in datasets_by_pillar:
                        continue
                    if party_name not in datasets_by_pillar[pillar]:
                        datasets_by_pillar[pillar][party_name] = []
                        
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                topic = item.get("topic", "")
                                prompt = item.get("prompt", "")
                                stance = item.get("stance", "")
                                if stance.strip():
                                    datasets_by_pillar[pillar][party_name].append({
                                        "topic": topic,
                                        "prompt": prompt,
                                        "stance": stance
                                    })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return datasets_by_pillar, party_colors

def evaluate_essay_by_prompts(essay_filepath):
    print(f"Loading evaluation model and datasets...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    
datasets, party_colors = load_structured_datasets()
    
    with open(essay_filepath, "r", encoding="utf-8") as f:
        essay_data = json.load(f)
        
    title = essay_data.get("title", "Unnamed Essay")
    essay_pillars = essay_data.get("pillars", {})
    
    print(f"\n=== EVALUATING ESSAY: {title} ===")
    
    party_cumulative_scores = {}
    party_match_counts = {}

    for pillar, text in essay_pillars.items():
        if not text.strip() or pillar not in datasets:
            continue
            
        # Split user essay text into sentences/chunks
        user_chunks = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        if not user_chunks:
            continue
            
        user_embeddings = model.encode(user_chunks, convert_to_tensor=True)
        parties = datasets[pillar]
        
        print(f"\n--- Pillar: {pillar} ({len(user_chunks)} chunks) ---")
        
        for party, prompt_list in parties.items():
            if not prompt_list:
                continue
                
            # Collect prompt strings or combine prompt + stance for rich matching
            prompt_texts = [f"{p['prompt']} {p['stance']}" for p in prompt_list]
            prompt_embeddings = model.encode(prompt_texts, convert_to_tensor=True)
            
            # Compute similarity matrix between user chunks and party prompt-stances
            cos_matrix = util.cos_sim(user_embeddings, prompt_embeddings)
            
            chunk_scores = []
            for u_idx in range(len(user_chunks)):
                best_match_idx = torch.argmax(cos_matrix[u_idx]).item()
                best_score = cos_matrix[u_idx][best_match_idx].item()
                chunk_scores.append(best_score)
                
            avg_party_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
            
            party_cumulative_scores[party] = party_cumulative_scores.get(party, 0.0) + avg_party_score
            party_match_counts[party] = party_match_counts.get(party, 0) + 1
            print(f"  * {party}: {avg_party_score*100:.1f}% alignment score")

    # Calculate overall podium
    print("\n" + "="*40)
    print("FINAL PROMPT-ALIGNED PODIUM")
    print("="*40)
    final_podium = {
        party: party_cumulative_scores[party] / party_match_counts[party]
        for party in party_cumulative_scores
    }
    sorted_podium = sorted(final_podium.items(), key=lambda x: x[1], reverse=True)
    
    for i, (party, score) in enumerate(sorted_podium):
        print(f"#{i+1} {party}: {score*100:.1f}% Match")

if __name__ == "__main__":
    # Test with your example essay file
    evaluate_essay_by_prompts("exempel/1. Min Babeh.json")