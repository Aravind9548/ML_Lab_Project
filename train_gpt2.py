# train_gpt2.py
import argparse
import math
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from gpt2_config import MODEL_SIZES
from gpt2_model import GPT2LMHeadModel
from dataset import train_or_load_tokenizer, GPT2TextDataset


def get_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with *.txt files (similar to WebText)")
    parser.add_argument("--model_size", type=str, default="117M",
                        choices=["117M", "345M", "762M", "1542M"])
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10000)

    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tokenizer
    print("Training/loading tokenizer...")
    tokenizer = train_or_load_tokenizer(
        data_dir=args.data_dir,
        vocab_size=50257,
        tokenizer_dir=args.tokenizer_dir,
    )

    # 2. Dataset
    print("Building dataset...")
    full_dataset = GPT2TextDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )
    eval_size = max(1, int(len(full_dataset) * args.eval_ratio))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 3. Model
    config = MODEL_SIZES[args.model_size]
    model = GPT2LMHeadModel(config)
    model.to(device)

    # 4. Optimizer & Scheduler
    no_decay = ["bias", "ln_1", "ln_2", "ln_f", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_steps = min(args.max_steps, args.epochs * len(train_loader))
    scheduler = get_scheduler(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits, loss = model(input_ids=input_ids, labels=labels)

            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})

            if global_step % args.eval_interval == 0:
                eval_loss = evaluate(model, eval_loader, device, fp16=args.fp16)
                print(f"\nStep {global_step}: eval_loss={eval_loss:.4f}, ppl={math.exp(eval_loss):.2f}")

            if global_step % args.save_interval == 0:
                save_path = os.path.join(args.output_dir, f"step_{global_step}")
                save_model(model, config, save_path)

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

    # Final save
    save_model(model, config, os.path.join(args.output_dir, "final"))


def evaluate(model, dataloader, device, fp16: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=fp16):
                _, loss = model(input_ids=input_ids, labels=labels)
            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * (seq_len - 1)
            total_tokens += batch_size * (seq_len - 1)
    model.train()
    return total_loss / total_tokens


def save_model(model, config, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), ckpt_path)
    # Also save a minimal config
    import json
    cfg = {
        "vocab_size": config.vocab_size,
        "n_positions": config.n_positions,
        "n_ctx": config.n_ctx,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()