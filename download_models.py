import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def download_model(model_name, folder_name):
    print(f"Downloading {model_name} -> {folder_name}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(folder_name)
        
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.save_pretrained(folder_name)
        
        print(f"Successfully saved {model_name} to {folder_name}/")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

def main():
    
    models_to_download = [
        ("gpt2-medium", "gpt2_345M"),
        ("gpt2-large", "gpt2_762M")
    ]
    
    for hf_name, local_folder in models_to_download:
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        download_model(hf_name, local_folder)

if __name__ == "__main__":
    main()
