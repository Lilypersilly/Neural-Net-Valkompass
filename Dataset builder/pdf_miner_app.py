import os
import json
import docx
from sentence_transformers import SentenceTransformer, util

# --- 1. INITIALISERA MODELL ---
print("Laddar SBERT-modellen (kan ta några sekunder)...")
model = SentenceTransformer('../models/fine_tuned_sbert') 

# --- 2. LADDA FRÅGOR FRÅN JSON ---
def load_questions_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        fragor = []
        for pillar, questions in data.get('pillars', {}).items():
            for q in questions:
                fragor.append({
                    "topic": q["topic"],
                    "prompt": q["prompt"]
                })
        return fragor
    except Exception as e:
        print(f"Kunde inte ladda frågor från {json_path}: {e}")
        return []

# --- 3. HJÄLPFUNKTIONER ---
def extract_and_chunk_docx(docx_path, window_size=3, step_size=2):
    try:
        doc = docx.Document(docx_path)
        text = " ".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"  [!] Kunde inte läsa {docx_path}: {e}")
        return []
        
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
    
    chunks = []
    for i in range(0, len(sentences), step_size):
        chunk = ". ".join(sentences[i:i + window_size]) + "."
        if chunk not in chunks:
            chunks.append(chunk)
    return chunks

def save_to_reference_file(party_name, topic, stance, link):
    filename = f"{party_name}_source_tracking.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{topic}]\nStance: {stance}\nKälla: {link}\n\n")

def save_to_json_dataset(party_name, topic, prompt, stance):
    filename = f"{party_name}_dataset.json"
    
    new_entry = {
        "topic": topic,
        "prompt": prompt,
        "stance": stance
    }
    
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
        
    data.append(new_entry)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- 4. HUVUDPROGRAM ---
def main():
    print("\n" + "="*50)
    print(" AUTOPILOT DOCX-MINER (Sparar JSON & TXT)")
    print("="*50 + "\n")
    
    json_file = input("Namn på JSON-fil med frågor (t.ex. questions.json): ").strip()
    fragor = load_questions_from_json(json_file)
    
    if not fragor:
        return
        
    print(f"Laddade in {len(fragor)} frågor från filen!")
    
    party_name = input("Partinamn att spara under (t.ex. SD, C): ").strip()
    
    doc_folder = "pdfer"
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)
        print(f"Skapade mappen '{doc_folder}'. Lägg dina DOCX-filer där och kör igen.")
        return

    docx_files = [f for f in os.listdir(doc_folder) if f.endswith('.docx')]
    if not docx_files:
        print(f"Inga DOCX-filer hittades i mappen '{doc_folder}'.")
        return

    print(f"Hittade {len(docx_files)} DOCX-filer att söka igenom.\n")

    for fraga in fragor:
        topic = fraga["topic"]
        prompt = fraga["prompt"]
        
        collected_count = 0
        filename = f"{party_name}_source_tracking.txt"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                collected_count = f.read().count(f"[{topic}]")
        
        print(f"\n" + "-"*50)
        print(f"BÖRJAR SÖKA EFTER: {topic}")
        print(f"Fråga: {prompt}")
        print(f"Redan insamlade i filen: {collected_count}/10")
        print("-" * 50)
        
        if collected_count >= 10:
            print(f"Redan klar med denna! Hoppar över.")
            continue
        
        for docx_file in docx_files:
            if collected_count >= 10:
                break 
                
            docx_path = os.path.join(doc_folder, docx_file)
            # Vi låter källan bara vara filnamnet
            docx_link = docx_file
            
            print(f"\nLäser in och rankar: {docx_file} ...")
            chunks = extract_and_chunk_docx(docx_path)
            
            if not chunks:
                continue
                
            prompt_emb = model.encode(prompt, convert_to_tensor=True)
            chunk_embs = model.encode(chunks, convert_to_tensor=True)
            hits = util.semantic_search(prompt_emb, chunk_embs, top_k=10)[0] 
            
            for hit in hits:
                if collected_count >= 10:
                    break
                
                # Din säkerhetsgräns för att filtrera bort skräp
                if hit['score'] < 0.55:
                    continue
                    
                chunk_text = chunks[hit['corpus_id']]
                
                print(f"\n[Träff {collected_count+1}/10] SBERT-poäng: {hit['score']:.2f}")
                print(f"> {chunk_text}")
                
                while True:
                    svar = input("Spara? [Y/N/S (Hoppa till nästa fil) / Q (Avbryt)]: ").strip().upper()
                    if svar == 'Y':
                        save_to_reference_file(party_name, topic, chunk_text, docx_link)
                        save_to_json_dataset(party_name, topic, prompt, chunk_text)
                        collected_count += 1
                        print("Sparad i både txt och json!")
                        break
                    elif svar == 'N':
                        print("Kastad.")
                        break
                    elif svar == 'S':
                        print("⏭Hoppar till nästa fil för denna fråga...")
                        break 
                    elif svar == 'Q':
                        print("Avbryter...")
                        return
                    else:
                        print("Skriv Y, N, S eller Q.")
                
                if svar == 'S':
                    break
        
        if collected_count >= 10:
            print(f"\nSnyggt! 10/10 träffar för '{topic}'. Går till nästa fråga...")
        else:
            print(f"\nSlut på filer. Hittade bara {collected_count}/10 träffar för '{topic}'. Går vidare...")

    print("\nHela frågelistan är genomgången! Kolla dina filer.")

if __name__ == "__main__":
    main()