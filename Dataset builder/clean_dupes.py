import os

def remove_duplicate_pdfs(pdfer_folder="pdfer", done_folder="done"):
    print("="*50)
    print(" 🧹 KONTROLLERAR DUPLICERADE FILER")
    print("="*50)
    
    # Kolla om mapparna existerar
    if not os.path.exists(pdfer_folder):
        print(f"❌ Mappen '{pdfer_folder}' hittades inte.")
        return
    if not os.path.exists(done_folder):
        print(f"❌ Mappen '{done_folder}' hittades inte.")
        return

    # Hämta alla filer i båda mapparna (vi gör filnamnen gemener för att undvika problem med stora/små bokstäver)
    pdfer_files = {f.lower(): f for f in os.listdir(pdfer_folder) if os.path.isfile(os.path.join(pdfer_folder, f))}
    done_files = {f.lower(): f for f in os.listdir(done_folder) if os.path.isfile(os.path.join(done_folder, f))}

    # Hitta överlappande filnamn
    duplicates = set(pdfer_files.keys()).intersection(set(done_files.keys()))

    if not duplicates:
        print("✅ Inga dubbletter hittades. Mapparna är skilda åt!")
        return

    print(f"🔍 Hittade {len(duplicates)} dubbletter som finns i både '{pdfer_folder}' och '{done_folder}'.")
    print("Raderar från 'pdfer'...\n")

    deleted_count = 0
    for dup_key in duplicates:
        actual_filename = pdfer_files[dup_key]
        file_path = os.path.join(pdfer_folder, actual_filename)
        
        try:
            os.remove(file_path)
            print(f"🗑️ Raderad från {pdfer_folder}: {actual_filename}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Kunde inte radera {actual_filename}: {e}")

    print(f"\n✨ Klar! Totalt raderades {deleted_count} dubbletter från '{pdfer_folder}'.")

if __name__ == "__main__":
    remove_duplicate_pdfs()