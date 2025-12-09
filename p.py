import math
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from huggingface_hub import hf_hub_download


# ---------------------------
# CONFIG
# ---------------------------

GPT2_MODELS = {
    "gpt2-117M": "gpt2",          # ~117M
    # "gpt2-345M": "gpt2-medium",  # ~345M
    # "gpt2-762M": "gpt2-large",   # ~762M
    # "gpt2-1542M": "gpt2-xl",       # ~1542M
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ---------------------------
# UTILITIES
# ---------------------------

def load_gpt2(model_name: str):
    """
    Load a GPT-2 model + tokenizer.
    model_name: one of GPT2_MODELS values, e.g. 'gpt2', 'gpt2-medium', ...
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad_token, reuse eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE, dtype=DTYPE)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_lm_loss_per_token(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """
    Compute average negative log-likelihood per token (cross-entropy)
    on the given batch. Returns scalar loss (in nats).
    """
    labels = input_ids.clone()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs.loss.item()


def perplexity_from_loss(loss: float, base: str = "e") -> float:
    """
    Convert average negative log-likelihood to perplexity.
    base='e' if loss is in nats, '2' if in bits.
    """
    if base == "e":
        return math.exp(loss)
    elif base == "2":
        return 2 ** loss
    else:
        raise ValueError("base must be 'e' or '2'")


# ---------------------------
# 1. GENERIC LANGUAGE MODELING (WikiText-2, PTB, etc)
# ---------------------------

def build_lm_dataloader(
    texts: List[str],
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 8,
) -> DataLoader:
    """
    Tokenize a list of raw texts into a packed dataset, then create a DataLoader
    that yields fixed-length blocks (GPT-2 style).
    """
    joined = tokenizer.eos_token.join(texts)

    enc = tokenizer(
        joined,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]

    # Chop into fixed-length blocks
    n_blocks = (len(input_ids) - 1) // max_length
    input_ids = input_ids[: n_blocks * max_length]
    input_ids = input_ids.view(n_blocks, max_length)

    dataset = [{"input_ids": block} for block in input_ids]

    def collate(batch):
        ids = torch.stack([x["input_ids"] for x in batch], dim=0)
        attn = torch.ones_like(ids)
        return {
            "input_ids": ids,
            "attention_mask": attn,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def evaluate_lm_perplexity_on_dataset(
    hf_dataset_name: str,
    split: str,
    model,
    tokenizer,
    text_column: str = "text",
    max_samples: Optional[int] = None,
    max_length: int = 1024,
    batch_size: int = 8,
    config_name: Optional[str] = None,
) -> float:
    """
    Compute perplexity on an HF dataset that has a 'text' column.
    Rough proxy for PTB, WikiText-2, WikiText-103, 1BW, etc.
    """
    if config_name is not None:
        ds = load_dataset(hf_dataset_name, config_name, split=split)
    else:
        ds = load_dataset(hf_dataset_name, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = ds[text_column]

    dataloader = build_lm_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc=f"LM eval on {hf_dataset_name}/{split}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = compute_lm_loss_per_token(model, batch["input_ids"], batch["attention_mask"])
        total_loss += loss
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl = perplexity_from_loss(avg_loss, base="e")
    return ppl


# ---------------------------
# 2. LAMBADA (perplexity + accuracy)
# ---------------------------

@torch.no_grad()
def evaluate_lambada(
    model,
    tokenizer,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate GPT-2 on LAMBADA.
    - Perplexity on the last token (as in paper)
    - Accuracy: whether the most probable next token matches the gold word
    Uses HF dataset: 'lambada' with 'validation'/'test'
    """

    ds = load_dataset("lambada", "plain_text", split=split)  # plain_text version
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    total_logprob = 0.0
    correct = 0
    count = 0

    for ex in tqdm(ds, desc=f"LAMBADA ({split})"):
        text = ex["text"].rstrip()
        if " " not in text:
            continue
        prefix, gold_word = text.rsplit(" ", 1)

        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(DEVICE)
        gold_ids = tokenizer(" " + gold_word, return_tensors="pt").input_ids.to(DEVICE)

        input_ids = torch.cat([prefix_ids, gold_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab]

        prefix_len = prefix_ids.shape[1]
        gold_len = gold_ids.shape[1]

        pred_logits = logits[:, prefix_len - 1 : prefix_len + gold_len - 1, :]
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        gold_target = gold_ids[0]  # [gold_len]
        token_logprobs = log_probs[0, torch.arange(gold_len), gold_target]
        seq_logprob = token_logprobs.sum().item()

        total_logprob += -seq_logprob
        count += gold_len

        # accuracy on first token of gold word
        first_logits = pred_logits[0, 0]
        pred_id = int(first_logits.argmax(dim=-1))
        if pred_id == int(gold_target[0]):
            correct += 1

    avg_nll = total_logprob / max(count, 1)
    ppl = math.exp(avg_nll)
    acc = correct / max(len(ds), 1)
    return {"lambada_ppl": ppl, "lambada_acc": acc}


# ---------------------------
# 3. Children’s Book Test (CBT) – Common Nouns & Named Entities
# ---------------------------

def _cbt_format_example(ex) -> Tuple[str, List[str], str]:
    """
    HF 'cbt' dataset fields (for CN/NE configs):

      - 'sentences': list[str]
      - 'question': str (contains 'XXXXX')
      - 'options': list[str]
      - 'answer': str
    """
    passage = " ".join(ex["sentences"])
    question = ex["question"]
    full_question = passage + " " + question
    candidates = ex["options"]
    answer = ex["answer"]
    return full_question, candidates, answer


@torch.no_grad()
def evaluate_cbt(
    model,
    tokenizer,
    kind: str = "CN",  # "CN" or "NE"
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> float:
    """
    CBT accuracy.
    For each candidate, fill in the blank, compute logprob of the resulting sentence,
    pick the highest.
    """
    assert kind in {"CN", "NE"}
    ds = load_dataset("cbt", kind, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    for ex in tqdm(ds, desc=f"CBT-{kind} ({split})"):
        question, candidates, answer = _cbt_format_example(ex)

        best_logprob = -1e9
        best_cand = None

        for cand in candidates:
            if "XXXXX" in question:
                filled = question.replace("XXXXX", cand)
            else:
                filled = question + " " + cand

            enc = tokenizer(filled, return_tensors="pt")
            input_ids = enc.input_ids.to(DEVICE)
            attn = enc.attention_mask.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            logprob = -outputs.loss.item() * input_ids.shape[1]

            if logprob > best_logprob:
                best_logprob = logprob
                best_cand = cand

        if best_cand == answer:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ---------------------------
# 4. Winograd Schema Challenge (from Parquet)
# ---------------------------

def load_winograd_parquet(config: str = "wsc273"):
    """
    Load Winograd Schema Challenge from Parquet without using the deprecated script.
    Uses the ErnestSDavis/winograd_wsc repo and its Parquet files.
    """
    # This repo has files like: wsc273/test-00000-of-00001.parquet
    parquet_path = hf_hub_download(
        repo_id="ErnestSDavis/winograd_wsc",
        filename=f"{config}/test-00000-of-00001.parquet",
    )
    ds = load_dataset("parquet", data_files={"test": parquet_path})["test"]
    return ds


@torch.no_grad()
def evaluate_winograd(
    model,
    tokenizer,
    config: str = "wsc273",
) -> float:
    """
    Winograd Schema Challenge accuracy.

    Loads Parquet directly (no script), then does LM scoring on each option.
    Dataset fields (see HF README):

      - text: str
      - pronoun: str
      - options: list[str]
      - label: int (index into options)
    """
    ds = load_winograd_parquet(config=config)

    correct = 0
    total = 0

    for ex in tqdm(ds, desc=f"Winograd ({config})"):
        text = ex["text"]
        pronoun = ex["pronoun"]
        options = ex["options"]
        label = int(ex["label"])

        if pronoun not in text:
            continue

        candidate_sents = [
            text.replace(pronoun, opt) for opt in options
        ]

        logprobs = []
        for sent in candidate_sents:
            enc = tokenizer(sent, return_tensors="pt")
            input_ids = enc.input_ids.to(DEVICE)
            attn = enc.attention_mask.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            logprob = -outputs.loss.item() * input_ids.shape[1]
            logprobs.append(logprob)

        pred_label = int(max(range(len(options)), key=lambda i: logprobs[i]))
        if pred_label == label:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ---------------------------
# 5. (Optional) Summarization, Translation, QA – PATTERNS ONLY
# ---------------------------

def generate_tldr_summary(model, tokenizer, article: str, max_new_tokens: int = 100) -> str:
    """
    Roughly follow GPT-2 TL;DR trick: append 'TL;DR:' to article and sample.
    """
    prompt = article.strip() + "\nTL;DR:"
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=2,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    sentences = gen.split(".")
    summary = ".".join(sentences[:3]).strip()
    return summary


def translate_en_fr_fewshot(model, tokenizer, src: str, examples: List[Tuple[str, str]]) -> str:
    """
    Few-shot translation in the paper style:
    'english sentence = french sentence' pairs, then a new english sentence and ask for french.
    """
    context = ""
    for en, fr in examples:
        context += f"english sentence = {en}\nfrench sentence = {fr}\n\n"

    context += f"english sentence = {src}\nfrench sentence ="
    enc = tokenizer(context, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=60,
            do_sample=False,  # greedy
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    translation = gen.split("\n")[0].strip()
    return translation


def qa_fewshot(model, tokenizer, question: str, examples: List[Tuple[str, str]]) -> str:
    """
    Few-shot QA style the paper used for Natural Questions.
    """
    context = ""
    for q, a in examples:
        context += f"Q: {q}\nA: {a}\n\n"
    context += f"Q: {question}\nA:"

    enc = tokenizer(context, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    answer = gen.split("\n")[0].strip()
    return answer


# ---------------------------
# 6. MAIN: RUN EVERYTHING
# ---------------------------

def main():
    results = {}

    for label, hf_name in GPT2_MODELS.items():
        print("=" * 80)
        print(f"Evaluating {label} ({hf_name}) on several GPT-2 paper metrics")
        print("=" * 80)

        model, tokenizer = load_gpt2(hf_name)

        model_results = {}

        # ---- Language modeling (WikiText-2 as proxy) ----
        try:
            wt2_ppl = evaluate_lm_perplexity_on_dataset(
                hf_dataset_name="wikitext",
                config_name="wikitext-2-v1",
                split="test",
                model=model,
                tokenizer=tokenizer,
                text_column="text",
                max_samples=None,
                max_length=1024,
                batch_size=4,
            )
            print(f"[{label}] WikiText-2 PPL: {wt2_ppl:.2f}")
            model_results["wikitext2_ppl"] = wt2_ppl
        except Exception as e:
            print(f"WikiText-2 eval failed for {label}: {e}")

        # ---- LAMBADA ----
        try:
            lambada_metrics = evaluate_lambada(model, tokenizer, split="validation", max_samples=None)
            print(
                f"[{label}] LAMBADA PPL (on target word tokens): {lambada_metrics['lambada_ppl']:.2f}, "
                f"ACC: {100*lambada_metrics['lambada_acc']:.2f}%"
            )
            model_results.update(lambada_metrics)
        except Exception as e:
            print(f"LAMBADA eval failed for {label}: {e}")

        # ---- CBT Common Nouns & Named Entities ----
        for kind in ["CN", "NE"]:
            try:
                acc = evaluate_cbt(model, tokenizer, kind=kind, split="validation", max_samples=2000)
                print(f"[{label}] CBT-{kind} ACC: {100*acc:.2f}%")
                model_results[f"cbt_{kind.lower()}_acc"] = acc
            except Exception as e:
                print(f"CBT-{kind} eval failed for {label}: {e}")

        # ---- Winograd ----
        try:
            winograd_acc = evaluate_winograd(model, tokenizer, config="wsc273")
            print(f"[{label}] Winograd ACC: {100*winograd_acc:.2f}%")
            model_results["winograd_acc"] = winograd_acc
        except Exception as e:
            print(f"Winograd eval failed for {label}: {e}")

        results[label] = model_results

    print("\n================== SUMMARY ==================")
    for label, res in results.items():
        print(f"\nModel: {label}")
        for k, v in res.items():
            if "acc" in k:
                print(f"  {k}: {100*v:.2f}%")
            else:
                print(f"  {k}: {v:.3f}")


if _name_ == "_main_":
    main()