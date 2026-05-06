import argparse
import os
import random

import numpy as np
from sklearn.metrics import f1_score

from util import dataloader


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def _check_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch and Hugging Face Transformers.\n"
            "Install them with: uv pip install torch transformers tqdm"
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_model_text(df):
    keyword = df["keyword"].fillna("").astype(str)
    location = df["location"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    return ("keyword: " + keyword + " location: " + location + " text: " + text).tolist()


class TweetDataset:
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = self.labels[index]
        return item


def train(args):
    torch, AutoTokenizer, AutoModelForSequenceClassification = _check_dependencies()
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    _set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = dataloader(
        data_dir=args.data_dir,
        valid_size=args.valid_size,
        random_state=args.seed,
    )
    train_df = data.train
    valid_df = data.valid
    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples)
    if args.max_valid_samples is not None:
        valid_df = valid_df.head(args.max_valid_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = TweetDataset(
        build_model_text(train_df),
        train_df["target"].astype("int64").tolist(),
        tokenizer,
        args.max_length,
    )
    valid_dataset = TweetDataset(
        build_model_text(valid_df),
        valid_df["target"].astype("int64").tolist(),
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)

        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss, valid_f1 = evaluate(model, valid_loader, device)
        train_loss = total_loss / max(1, len(train_loader))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} valid_f1={valid_f1:.4f}"
        )

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"saved best checkpoint to {args.output_dir} (f1={best_f1:.4f})")

    return best_f1


def evaluate(model, data_loader, device):
    import torch

    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(batch["labels"].cpu().numpy().tolist())

    return total_loss / max(1, len(data_loader)), f1_score(labels, predictions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models/distilbert-disaster-tweets")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
