import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm

from load_model import load_tokenizer, load_trainable_model

# Configuration
DATA_PATH = "../data/wg.tsv"
SAVE_DIR = "../models/finetuned_lora/"
BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 128
LR = 5e-5
WARMUP_STEPS = 10

# Dataset
class WinogenderDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = [row["sentence"] for _, row in dataframe.iterrows()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0)
        }

# Training Function
def train(model, tokenizer, train_loader):
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(train_loader) * EPOCHS
    )

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        for batch in tqdm(train_loader, desc="Training"):
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"LoRA adapter saved to {SAVE_DIR}")

def main():
    assert os.path.exists(DATA_PATH), f"Missing file: {DATA_PATH}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_trainable_model()

    df = pd.read_csv(DATA_PATH, sep="\t")
    dataset = WinogenderDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    train(model, tokenizer, dataloader)

if __name__ == "__main__":
    main()
