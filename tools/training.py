import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import CosineSimilarityLoss

def train_topic_retriever(json_filepath, model_name="KBLab/sentence-bert-swedish-cased", output_path="models/fine_tuned_sbert"):
    # 1. Load your labeled dataset
    with open(json_filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    train_examples = []
    for item in raw_data:
        text1 = item.get("text1")
        text2 = item.get("text2")
        label = item.get("label") # 1 for same wedge/topic, 0 for different
        
        if text1 and text2 and label is not None:
            # CosineSimilarityLoss expects float scores between 0.0 and 1.0
            train_examples.append(InputExample(texts=[text1, text2], label=float(label)))
            
    # 2. Set up data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    
    # 3. Load base model
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    
    # 4. Define loss function
    train_loss = CosineSimilarityLoss(model)
    
    print(f"Training SBERT model on {len(train_examples)} pairs using device: {device}...")
    
    # 5. Fit the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,
        warmup_steps=100,
        output_path=output_path
    )
    print(f"Training complete! Fine-tuned model saved to: {output_path}")

if __name__ == "__main__":
    # Point this to your saved dataset json file
    train_topic_retriever("sbert_training_pairs.json")