"""
BERT Cross-Encoder for Neural Re-Ranking
Dosen: Zico Pratama Putra
Kelompok: [Nama Anggota 1, Anggota 2, Anggota 3]
"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class CrossEncoderDataset(Dataset):
    def __init__(self, queries, passages, labels, tokenizer, max_length=512):
        self.queries = queries
        self.passages = passages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            self.passages[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTCrossEncoder:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        print(f"✅ Model {model_name} loaded on {self.device}")

    def predict(self, query: str, passage: str) -> float:
        """Inference satu pasangan query-passage"""
        features = self.tokenizer(
            query, passage, truncation=True, padding=True, return_tensors="pt"
        )
        features = {k: v.to(self.device) for k, v in features.items()}

        with torch.no_grad():
            outputs = self.model(**features)
            # Untuk model binary classification
            score = (
                torch.sigmoid(outputs.logits).item()
                if outputs.logits.shape[1] == 1
                else outputs.logits.softmax(dim=1)[:, 1].item()
            )
        return score

    def re_rank(self, query: str, passages: list, batch_size=32, verbose=True):
        """Re-rank list of passages untuk satu query"""
        scores = []
        for i in tqdm(
            range(0, len(passages), batch_size), desc="Re-ranking", disable=not verbose
        ):
            batch_queries = [query] * min(batch_size, len(passages) - i)
            batch_passages = passages[i : i + batch_size]

            batch_scores = [
                self.predict(q, p) for q, p in zip(batch_queries, batch_passages)
            ]
            scores.extend(batch_scores)

        # Return ranked indices + scores
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return ranked_indices, scores

    def train(
        self,
        train_df: pd.DataFrame,
        val_df=None,
        epochs=2,
        batch_size=8,
        lr=2e-5,
        max_length=256
    ):
        train_dataset = CrossEncoderDataset(
            queries=train_df["query"].tolist(),
            passages=train_df["passage"].tolist(),
            labels=train_df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


# ====================== CONTOH PENGGUNAAN ======================
if __name__ == "__main__":
    reranker = BERTCrossEncoder()

    score = reranker.predict(
        query="How to make a good cappuccino?",
        passage="The three steps to make a perfect cappuccino are...",
    )
    print(f"Relevance score: {score:.4f}")
