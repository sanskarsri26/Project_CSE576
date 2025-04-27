import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AdamW

from load_model import load_tokenizer, load_trainable_model, load_merged_model

# Config
WG_PATH = "../data/wg_contrastive.tsv"
ADAPTER_IN = "../models/finetuned_lora/"
ADAPTER_OUT = "../models/debiased_lora/"

PROJECTION_STEPS = 30
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
LAMBDA_REG = 0.15
MAX_LEN = 128

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def format_batch(tokenizer, sent_more, sent_less):
    inputs_more = tokenizer(sent_more, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
    inputs_less = tokenizer(sent_less, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
    return inputs_more, inputs_less

def pcgu_unlearn(train_model, ref_model, tokenizer, dataset):
    optimizer = AdamW(filter(lambda p: p.requires_grad, train_model.parameters()), lr=LEARNING_RATE)
    scaler = GradScaler()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: {
            "sent_more": [row["sent_more"] for row in batch],
            "sent_less": [row["sent_less"] for row in batch]
        }
    )

    for step in range(PROJECTION_STEPS):
        train_model.train()
        total_loss = 0
        print(f"\n[PCGU Step {step + 1}/{PROJECTION_STEPS}]")
        for batch in tqdm(dataloader, desc="Unlearning"):
            inputs_more, inputs_less = format_batch(tokenizer, batch["sent_more"], batch["sent_less"])

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out_more = train_model(**inputs_more, labels=inputs_more["input_ids"])
                out_less = train_model(**inputs_less, labels=inputs_less["input_ids"])
                contrastive = torch.clamp(out_less.loss - out_more.loss, min=0)

                # Regularization term
                reg = 0.0
                for t_param, r_param in zip(train_model.parameters(), ref_model.parameters()):
                    if t_param.requires_grad and t_param.shape == r_param.shape:
                        reg += torch.norm(t_param - r_param.detach()) ** 2

                loss = contrastive + LAMBDA_REG * reg

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Step {step + 1} total loss: {total_loss:.4f}")

    train_model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    print(f"\nDebiased LoRA adapter saved to {ADAPTER_OUT}")

def main():
    os.makedirs(ADAPTER_OUT, exist_ok=True)
    tokenizer = load_tokenizer()
    df = pd.read_csv(WG_PATH, sep="\t")
    assert "sent_more" in df.columns and "sent_less" in df.columns
    dataset = df.to_dict("records")

    print("Loading models...")
    train_model = load_trainable_model()
    ref_model = load_merged_model(ADAPTER_IN).eval().cuda()

    print("Running PCGU unlearning...")
    pcgu_unlearn(train_model, ref_model, tokenizer, dataset)

if __name__ == "__main__":
    main()
