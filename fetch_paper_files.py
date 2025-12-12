import os
import requests
from transformers import GPT2Tokenizer, GPT2Config


OUTPUT_DIR = "gpt2_source_files"
HF_GITHUB_URL = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt2/"


FILES_TO_DOWNLOAD = [
    "modeling_gpt2.py",       
    "configuration_gpt2.py",  
    "tokenization_gpt2.py",   
]

def main():
  
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("--- 1. Downloading Source Code (Reference Implementation) ---")
    for filename in FILES_TO_DOWNLOAD:
        url = HF_GITHUB_URL + filename
        print(f"Fetching {filename}...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(OUTPUT_DIR, filename), "w", encoding='utf-8') as f:
                    f.write(response.text)
                print(f"   ✅ Saved {filename}")
            else:
                print(f"   ❌ Failed to download {filename} (Status: {response.status_code})")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n--- 2. Extracting Vocabulary & Config (Data Files) ---")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("   ✅ Saved vocab.json (The Vocabulary)")
        print("   ✅ Saved merges.txt (The BPE Merge Rules)")
        
        config = GPT2Config.from_pretrained("gpt2")
        config.save_pretrained(OUTPUT_DIR)
        print("   ✅ Saved config.json (The 117M Model Settings)")
        
    except Exception as e:
        print(f"   ❌ Error extracting data files: {e}")

    print(f"\nSUCCESS! All paper-related files are now in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()