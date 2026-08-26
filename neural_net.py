import os
import json
import re
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SBERT_MODEL_NAME = "models/fine_tuned_sbert"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

RELEVANCE_CUTOFF = 0.57


def load_party_datasets(dataset_dir="datasets"):
    party_colors = {}
    party_metadata = {}
    datasets_by_pillar = {
        "Economy": {},
        "Climate": {},
        "Immigration": {},
        "Law And Order": {}
    }
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        return datasets_by_pillar, party_colors, party_metadata
        
    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith(".json") and not filename.startswith("_"):
            filepath = os.path.join(dataset_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                info = data.get("party_info", {})
                party_name = info.get("name") or data.get("name")
                color = info.get("color") or data.get("color", "#888888")
                
                if party_name:
                    party_colors[party_name] = color
                    party_metadata[party_name] = {
                        "short_name": info.get("short_name", party_name[:2].upper()),
                        "description": info.get("description", "")
                    }
                    
                    pillars_dict = data.get("pillars", {})
                    for pillar, items in pillars_dict.items():
                        if pillar not in datasets_by_pillar:
                            continue
                            
                        statements = []
                        
                        def process_item(item_data):
                            if isinstance(item_data, str) and item_data.strip():
                                return {
                                    "sbert_text": item_data.strip(), 
                                    "nli_text": item_data.strip(), 
                                    "topic": "Allmän ståndpunkt",
                                    "stance": item_data.strip(),
                                    "original": item_data.strip()
                                }
                            elif isinstance(item_data, dict):
                                topic = item_data.get("topic", "").strip()
                                prompt = item_data.get("prompt", "").strip()
                                stance = item_data.get("stance") or item_data.get("statement") or item_data.get("text") or ""
                                if stance.strip():
                                    sbert_text = f"{prompt} {stance.strip()}".strip() if prompt else stance.strip()
                                    return {
                                        "sbert_text": sbert_text, 
                                        "nli_text": stance.strip(),
                                        "topic": topic,
                                        "stance": stance.strip(),
                                        "original": sbert_text
                                    }
                            return None

                        if isinstance(items, list):
                            for item in items:
                                processed = process_item(item)
                                if processed:
                                    statements.append(processed)
                                        
                        elif isinstance(items, dict):
                            for _, val in items.items():
                                processed = process_item(val)
                                if processed:
                                    statements.append(processed)
                                        
                        datasets_by_pillar[pillar][party_name] = statements
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return datasets_by_pillar, party_colors, party_metadata


def load_example_essays(example_dir="exempel"):
    examples = {}
    if not os.path.exists(example_dir):
        os.makedirs(example_dir, exist_ok=True)
        return examples
        
    for filename in sorted(os.listdir(example_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(example_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", filename.replace(".json", ""))
                    pillars = data.get("pillars", {})
                    examples[title] = {
                        "Economy": pillars.get("Economy", ""),
                        "Climate": pillars.get("Climate", ""),
                        "Immigration": pillars.get("Immigration", ""),
                        "Law And Order": pillars.get("Law And Order", "")
                    }
            except Exception as e:
                print(f"Error loading example {filename}: {e}")
                
    return examples


@st.cache_resource
def load_neural_engine():
    try:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
        
        id2label = nli_model.config.id2label
        entail_idx = next(k for k, v in id2label.items() if 'entail' in v.lower())
        contra_idx = next(k for k, v in id2label.items() if 'contradict' in v.lower())
        
        model_package = (sbert_model, nli_model, nli_tokenizer, entail_idx, contra_idx)
        return model_package, device
    except Exception as e:
        st.error(f"Failed to load neural models: {e}")
        st.stop()


def get_sliding_windows(text: str, window_size: int = 2, overlap: int = 1) -> list:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
        
    chunks = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i:i + window_size]
        if window:
            chunks.append(" ".join(window))
        if i + window_size >= len(sentences):
            break
    return chunks


def smart_route_sentences(user_essays_dict: dict, dataset_keys: list, model_package, threshold: float = 0.48) -> dict:
    sbert_model = model_package[0]
    all_chunks = []
    for text in user_essays_dict.values():
        if text.strip():
            all_chunks.extend(get_sliding_windows(text, window_size=2, overlap=1))
            
    all_chunks = list(dict.fromkeys(all_chunks))
    routed_mapping = {pillar: [] for pillar in dataset_keys}
    if not all_chunks:
        return routed_mapping

    pillar_anchors = {
        "Economy": "skatter ekonomi bolagskatt arbete välfärd företag kapitalvinster budget lön",
        "Immigration": "migration asyl invandring gräns flykting integration uppehållstillstånd nyanlända",
        "Law And Order": "brott kriminalitet polisen straff gängkriminalitet trygghet rättsväsende fängelse",
        "Climate": "klimat miljö utsläpp energi grön omställning fossil skog natur tåg"
    }

    chunk_embeddings = sbert_model.encode(all_chunks, convert_to_tensor=True)
    
    for pillar in dataset_keys:
        anchor_text = pillar_anchors.get(pillar, pillar)
        anchor_embedding = sbert_model.encode(anchor_text, convert_to_tensor=True)
        similarities = util.cos_sim(chunk_embeddings, anchor_embedding).squeeze(1)
        
        for idx, score in enumerate(similarities):
            if score.item() >= threshold:
                if all_chunks[idx] not in routed_mapping[pillar]:
                    routed_mapping[pillar].append(all_chunks[idx])

    for pillar, text in user_essays_dict.items():
        box_chunks = get_sliding_windows(text, window_size=2, overlap=1)
        for chunk in box_chunks:
            if pillar in routed_mapping and chunk not in routed_mapping[pillar]:
                routed_mapping[pillar].append(chunk)

    return routed_mapping


def evaluate_routed_profiles(routed_sentences: dict, datasets: dict, model_package, progress_callback=None):
    sbert_model, nli_model, nli_tokenizer, entail_idx, contra_idx = model_package
    device = nli_model.device

    active_routing = {p: chunks for p, chunks in routed_sentences.items() if chunks and p in datasets}
    if not active_routing:
        return {}, {}

    total_steps = sum(
        len(chunks) * len(party_statements)
        for pillar, chunks in active_routing.items()
        for party_statements in datasets[pillar].values()
    )
    current_step = 0

    # Skapar lagring för alla tre skalningsmetoderna samtidigt
    pillar_party_scores = {
        "Standard": {p: {} for p in active_routing},
        "SANN": {p: {} for p in active_routing},
        "Relativ": {p: {} for p in active_routing}
    }
    inference_traces = {}
    
    for pillar, user_chunks in active_routing.items():
        inference_traces[pillar] = {}
        parties = datasets[pillar]
        
        user_embeddings = sbert_model.encode(user_chunks, convert_to_tensor=True)
        
        # --- PASS 1: Beräkna cosinus-matriser för alla partier i förväg ---
        party_matrices = {}
        chunk_party_peaks = {u_idx: [] for u_idx in range(len(user_chunks))}
        
        for party, party_statements in parties.items():
            if not party_statements or not user_chunks:
                continue
            sbert_texts = [p["sbert_text"] for p in party_statements]
            party_embeddings = sbert_model.encode(sbert_texts, convert_to_tensor=True)
            cos_matrix = util.cos_sim(user_embeddings, party_embeddings)
            party_matrices[party] = cos_matrix
            
            for u_idx in range(len(user_chunks)):
                chunk_party_peaks[u_idx].append(torch.max(cos_matrix[u_idx]).item())
        # ------------------------------------------------------------------

        # --- PASS 2: Utvärdera NLI och applicera ALLA skalningar ---
        for party, party_statements in parties.items():
            inference_traces[pillar][party] = []
            if not party_statements or not user_chunks or party not in party_matrices:
                pillar_party_scores["Standard"][pillar][party] = 0.0
                pillar_party_scores["SANN"][pillar][party] = 0.0
                pillar_party_scores["Relativ"][pillar][party] = 0.0
                continue
                
            cos_matrix = party_matrices[party]
            
            # Listor för de tre olika uträkningarna
            scores_std, scores_sann, scores_rel = [], [], []
            
            for u_idx, user_chunk in enumerate(user_chunks):
                best_p_idx = torch.argmax(cos_matrix[u_idx]).item()
                best_match = party_statements[best_p_idx]
                sbert_peak = cos_matrix[u_idx][best_p_idx].item()
                
                if sbert_peak < RELEVANCE_CUTOFF:
                    scores_std.append(0.0)
                    scores_sann.append(0.0)
                    scores_rel.append(0.0)
                    inference_traces[pillar][party].append({
                        "user_chunk": user_chunk,
                        "topic": "Ingen relevant partiståndpunkt hittades.",
                        "stance": "",
                        "party_statement": "Ingen relevant partiståndpunkt hittades.",
                        "scores": {"Standard": 0.0, "SANN": 0.0, "Relativ": 0.0},
                        "nli_status": "Ej Relevant / Utanför Ämnet",
                    })
                    continue
                
                # --- BERÄKNA DE TRE OLIKA SBERT-SKALNINGARNA ---
                # 1. Standard (0.25 - 0.80)
                sbert_std = max(0.0, min(1.0, (sbert_peak - 0.25) / (0.80 - 0.25)))
                
                # 2. SANN (0.0 - 1.0)
                sbert_sann = max(0.0, min(1.0, (sbert_peak - 0.0) / (1.0 - 0.0)))
                
                # 3. Relativ (min_peak - max_peak)
                peaks = chunk_party_peaks[u_idx]
                min_b = min(peaks) if peaks else 0.0
                max_b = max(peaks) if peaks else 1.0
                if max_b == min_b: max_b = min_b + 0.001
                sbert_rel = max(0.0, min(1.0, (sbert_peak - min_b) / (max_b - min_b)))
                
                # --- NLI (Logik) ---
                clean_nli_statement = re.sub(r'^(ja|nej)[.,!?\s]+', '', best_match["nli_text"], flags=re.IGNORECASE)
                inputs = nli_tokenizer([user_chunk], [clean_nli_statement], truncation=True, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = nli_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)[0]
                    
                avg_contra = probs[contra_idx].item()
                avg_entail = probs[entail_idx].item()
                
                contradiction_multiplier = max(0.1, 1.0 - (1.15 * avg_contra))
                entailment_boost = 0.15 * avg_entail if avg_entail > 0.70 else 0.0
                
                # Kombinera NLI med de tre olika sbert-skalningarna
                final_std = max(0.0, min(1.0, (sbert_std * contradiction_multiplier) + entailment_boost))
                final_sann = max(0.0, min(1.0, (sbert_sann * contradiction_multiplier) + entailment_boost))
                final_rel = max(0.0, min(1.0, (sbert_rel * contradiction_multiplier) + entailment_boost))
                
                scores_std.append(final_std)
                scores_sann.append(final_sann)
                scores_rel.append(final_rel)
                
                if avg_contra > 0.40:
                    nli_status = "Konflikt / Motsägelse"
                elif avg_entail > 0.60:
                    nli_status = "Logisk Följd / Stöd"
                else:
                    nli_status = "Neutral / Delvis"

                inference_traces[pillar][party].append({
                    "user_chunk": user_chunk,
                    "topic": best_match["topic"],
                    "stance": best_match["stance"],
                    "party_statement": best_match["original"],
                    "scores": {
                        "Standard": final_std * 100, 
                        "SANN": final_sann * 100, 
                        "Relativ": final_rel * 100
                    },
                    "nli_status": nli_status,
                })
            
            # Spara medelvärdet för partiet i alla tre modes
            pillar_party_scores["Standard"][pillar][party] = (sum(scores_std) / len(scores_std)) if scores_std else 0.0
            pillar_party_scores["SANN"][pillar][party] = (sum(scores_sann) / len(scores_sann)) if scores_sann else 0.0
            pillar_party_scores["Relativ"][pillar][party] = (sum(scores_rel) / len(scores_rel)) if scores_rel else 0.0
            
            current_step += len(user_chunks) * len(party_statements)
            if progress_callback:
                progress_callback(current_step, total_steps, pillar, party)
                
    return pillar_party_scores, inference_traces


def calculate_absolute_podium(pillar_party_scores: dict) -> dict:
    party_totals = {}
    party_counts = {}
    
    for pillar, parties in pillar_party_scores.items():
        for party, score in parties.items():
            party_totals[party] = party_totals.get(party, 0.0) + score
            party_counts[party] = party_counts.get(party, 0) + 1
            
    if not party_totals:
        return {}
        
    absolute_scores = {party: party_totals[party] / party_counts[party] for party in party_totals}
    return dict(sorted(absolute_scores.items(), key=lambda item: item[1], reverse=True))


def generate_execution_log(routed_sentences: dict, pillar_scores: dict, podium: dict) -> str:
    log_lines = []
    log_lines.append("=== NEURAL NET POLITICAL COMPASS - EXECUTION LOG ===")
    
    log_lines.append("\n[1] ROUTED SENTENCES / SLIDING WINDOWS PER PILLAR:")
    for pillar, sentences in routed_sentences.items():
        log_lines.append(f"  - Pillar: {pillar} ({len(sentences)} chunks)")
        for s in sentences:
            log_lines.append(f"    * {s}")
    
    log_lines.append("\n[2] ISOLATED ALIGNMENT SCORES (Strict Per-Chunk Evaluation):")
    for pillar, parties in pillar_scores.items():
        log_lines.append(f"  - Pillar: {pillar}")
        for party, score in sorted(parties.items(), key=lambda x: x[1], reverse=True):
            log_lines.append(f"    * {party}: {score*100:.1f}%")
            
    log_lines.append("\n[3] THE ABSOLUTE IDEOLOGICAL MATCH PODIUM:")
    for i, (party, score) in enumerate(podium.items()):
        log_lines.append(f"  #{i+1} {party}: {score*100:.1f}% Match")
        
    return "\n".join(log_lines)