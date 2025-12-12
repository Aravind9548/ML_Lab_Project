import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

models_to_test = {
    "Original GPT-2": "gpt2",                          
    "Fine-Tuned GPT-2": "gpt2-finetuned-wikitext"    
}

def evaluate_perplexity(model_path):
    print(f"\n--- Evaluating: {model_path} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    except OSError:
        print(f"Error: Could not find folder '{model_path}'. Did you run the training script?")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

for name, path in models_to_test.items():
    score = evaluate_perplexity(path)
    if score:
        print(f"RESULT -> {name} Perplexity: {score:.2f}")