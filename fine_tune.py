import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


MODEL_NAME = "gpt2" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Loading Model: {MODEL_NAME} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- Loading Dataset ---")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]", trust_remote_code=True) 
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128 
        )
    
    print("--- Tokenizing Data ---")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME}-finetuned-openwebtext",
        overwrite_output_dir=True,
        num_train_epochs=1,              
        per_device_train_batch_size=4,   
        learning_rate=2e-5,              
        save_steps=100,
        logging_steps=10,
        prediction_loss_only=True,
        use_cpu=False if torch.cuda.is_available() else True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), 
    )

    print("--- Starting Fine-Tuning ---")
    trainer.train()
    
    print("--- Saving Model ---")
    trainer.save_model(f"./{MODEL_NAME}-finetuned-openwebtext")
    tokenizer.save_pretrained(f"./{MODEL_NAME}-finetuned-openwebtext")
    
    print("Done! You can now load this path in your evaluation script to see if PPL improved.")

if __name__ == "__main__":
    main()