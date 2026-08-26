import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

SBERT_MODEL_NAME = "models/fine_tuned_sbert"
TRAINING_PAIRS_FILE = "sbert_training_pairs_test.json"

def run_calibration():
    print("Loading model and training pairs...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    
    if not os.path.exists(TRAINING_PAIRS_FILE):
        print(f"Error: Could not find {TRAINING_PAIRS_FILE}. Make sure it's in the same directory.")
        return

    with open(TRAINING_PAIRS_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    print(f"Loaded {len(pairs)} pairs. Calculating similarities...")
    
    # Extrahera texter och labels
    texts1 = [item["text1"] for item in pairs]
    texts2 = [item["text2"] for item in pairs]
    labels = np.array([item["label"] for item in pairs]) # 1 = Relevant, 0 = Irrelevant
    
    # Encodas i batcher för effektivitet
    embeddings1 = model.encode(texts1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True, show_progress_bar=True)
    
    # Beräkna cosinuslikhet parvis
    cos_scores = util.cos_sim(embeddings1, embeddings2)
    # Plocka ut diagonalen eftersom element i-1 matchas mot i-2
    similarities = np.array([cos_scores[i][i].item() for i in range(len(pairs))])
    
    pos = similarities[labels == 1]
    neg = similarities[labels == 0]
    
    print("\n" + "="*50)
    print("STATISTICAL DISTRIBUTION SUMMARY (Based on Manual Labels)")
    print("="*50)
    if len(pos) > 0:
        print(f"Relevant Pairs (Label 1)   -> Mean: {pos.mean():.3f} | Std: {pos.std():.3f} | Min: {pos.min():.3f} | Max: {pos.max():.3f}")
    if len(neg) > 0:
        print(f"Irrelevant Pairs (Label 0) -> Mean: {neg.mean():.3f} | Std: {neg.std():.3f} | Min: {neg.min():.3f} | Max: {neg.max():.3f}")
    
    # Sweep thresholds to find the peak F1-score
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.0
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    
    for t in thresholds:
        tp = np.sum((similarities >= t) & (labels == 1))
        fp = np.sum((similarities >= t) & (labels == 0))
        fn = np.sum((similarities < t) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_precision = precision
            best_recall = recall

    print("\n" + "="*50)
    print(f"OPTIMAL RELEVANCE THRESHOLD: {best_thresh:.2f}")
    print("="*50)
    print(f"Peak F1-Score: {best_f1:.3f}")
    print(f"Precision:     {best_precision*100:.1f}%")
    print(f"Recall:        {best_recall*100:.1f}%")

if __name__ == "__main__":
    import os
    run_calibration()