import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# POINT THIS TO YOUR FINE-TUNED MODEL FOLDER
# (Make sure this path is correct on your machine)
MODEL_PATH = "gpt2"

def load_model_and_tokenizer(path):
    print(f"Loading model from: {path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
        model.eval() # Set to evaluation mode
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model from {path}.\nMake sure the folder exists and contains pytorch_model.bin.")
        exit()

@torch.no_grad()
def evaluate_smart(
    model,
    tokenizer,
    dataset_name: str = "lambada",
    config_name: str = None, 
    split: str = "test",
    text_column: str = "text",
    use_stop_word_filter: bool = False  # <--- NEW FEATURE FLAG
) -> Dict[str, float]:
    
    print(f"\n--- Evaluating on {dataset_name} (Filter={use_stop_word_filter}) ---")
    
    # Load Dataset
    try:
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {"ppl": 0.0, "acc": 0.0}

    total_logprob = 0.0
    correct = 0
    count = 0
    skipped = 0

    # Define Stop Words (The Paper's Trick)
    # These are common connector words we BAN from being the "last word"
    stop_words = [" the", " and", " in", " of", " a", " to", " with", " for", ","]
    bad_words_ids = [tokenizer.encode(word) for word in stop_words]

    for ex in tqdm(ds, desc=f"Eval {dataset_name}"):
        text = ex[text_column].strip()
        
        if " " not in text:
            skipped += 1
            continue
            
        # Split Context vs Target
        prefix, gold_word = text.rsplit(" ", 1)
        
        # Encode
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(DEVICE)
        gold_ids = tokenizer(" " + gold_word, return_tensors="pt").input_ids.to(DEVICE)
        
        if prefix_ids.size(1) == 0 or gold_ids.size(1) == 0:
            skipped += 1
            continue

        input_ids = torch.cat([prefix_ids, gold_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Run Model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # --- METRIC 1: PERPLEXITY (Standard) ---
        prefix_len = prefix_ids.shape[1]
        gold_len = gold_ids.shape[1]
        
        pred_logits = logits[:, prefix_len - 1 : prefix_len + gold_len - 1, :]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        
        gold_target = gold_ids[0]
        token_logprobs = log_probs[0, torch.arange(gold_len), gold_target]
        total_logprob += -token_logprobs.sum().item()
        count += gold_len

        # --- METRIC 2: ACCURACY (With Smart Filter) ---
        # Focus on the very first token of the predicted word
        next_token_logits = pred_logits[0, 0].clone() # Clone to avoid modifying original for PPL calculation

        if use_stop_word_filter:
            # Apply the "Paper's Trick": Set probability of bad words to -infinity
            for bad_id in bad_words_ids:
                if isinstance(bad_id, list): 
                    for bid in bad_id: next_token_logits[bid] = -float("inf")
                else:
                    next_token_logits[bad_id] = -float("inf")

        pred_id = int(next_token_logits.argmax(dim=-1))
        
        if pred_id == int(gold_target[0]):
            correct += 1

    avg_nll = total_logprob / max(count, 1)
    ppl = math.exp(avg_nll)
    acc = correct / max((len(ds) - skipped), 1)
    
    return {"ppl": ppl, "acc": acc}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load your Fine-Tuned Model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    print("\n================ STARTING VERIFICATION ================")

    # TEST 1: Did Fine-Tuning Work? (Check WikiText)
    # We expect LOW Perplexity here because you trained on this data.
    wiki_metrics = evaluate_smart(
        model, tokenizer, 
        dataset_name="wikitext", 
        config_name="wikitext-2-raw-v1", 
        split="test",
        use_stop_word_filter=False
    )

    # TEST 2: Does the "Paper's Trick" Work? (Check LAMBADA)
    # We enable use_stop_word_filter=True to boost the score.
    lama_metrics = evaluate_smart(
        model, tokenizer, 
        dataset_name="lambada", 
        split="test",
        use_stop_word_filter=True # <--- Enabling the feature!
    )
    
    print(f"\n================ FINAL REPORT ================")
    print(f"Model Path: {MODEL_PATH}")
    print(f"---------------------------------------------")
    print(f"[TEST 1] Learning Verification (WikiText-2):")
    print(f"   Perplexity: {wiki_metrics['ppl']:.2f} (Should be lower than ~18)")
    print(f"---------------------------------------------")
    print(f"[TEST 2] Smart Feature Verification (LAMBADA):")
    print(f"   Accuracy:   {lama_metrics['acc']*100:.2f}% (Should be higher than ~26%)")
    print(f"==============================================")