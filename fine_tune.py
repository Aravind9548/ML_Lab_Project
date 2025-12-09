import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# ---------------------------
# CONFIGURATION
# ---------------------------
# You can change this to "gpt2-xl" (1.5B) if you have a massive GPU (24GB+ VRAM).
# For standard GPUs (Colab/Consumer), stick to "gpt2" (124M) or "gpt2-medium".
MODEL_NAME = "gpt2" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Loading Model: {MODEL_NAME} ---")
    
    # 1. Load Model & Tokenizer (Same as your eval script)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # GPT-2 needs a pad token for batching (it doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset: OpenWebText (Small Subset)
    # We use "stas/openwebtext-10k" which is a clean, small slice of the massive dataset.
    print("--- Loading Dataset ---")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]", trust_remote_code=True) # Use 500 samples for a quick test run
    
    # 3. Preprocess (Tokenize)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128 # Keep short for speed. Max for GPT-2 is 1024.
        )
    
    print("--- Tokenizing Data ---")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Configure Training
    # This replaces the need to write a manual PyTorch loop
    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME}-finetuned-openwebtext",
        overwrite_output_dir=True,
        num_train_epochs=1,              # 1 epoch is enough to see changes
        per_device_train_batch_size=4,   # Small batch size to fit in memory
        learning_rate=2e-5,              # Standard fine-tuning rate
        save_steps=100,
        logging_steps=10,
        prediction_loss_only=True,
        use_cpu=False if torch.cuda.is_available() else True
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # mlm=False = Standard Causal LM
    )

    # 6. Train
    print("--- Starting Fine-Tuning ---")
    trainer.train()
    
    # 7. Save
    print("--- Saving Model ---")
    trainer.save_model(f"./{MODEL_NAME}-finetuned-openwebtext")
    tokenizer.save_pretrained(f"./{MODEL_NAME}-finetuned-openwebtext")
    
    print("Done! You can now load this path in your evaluation script to see if PPL improved.")

if __name__ == "__main__":
    main()