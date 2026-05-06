import argparse
import os

from train import DEFAULT_MODEL_NAME, build_model_text
from util import dataloader, download_data


DEFAULT_MODEL_DIR = "models/distilbert-disaster-tweets"
DEFAULT_SUBMISSION_PATH = "submission.csv"


def _check_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires PyTorch and Hugging Face Transformers.\n"
            "Install them with: uv pip install torch transformers tqdm"
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _checkpoint_exists(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer_config.json")
    return os.path.exists(config_path) and os.path.exists(tokenizer_path)


def predict(args):
    torch, AutoTokenizer, AutoModelForSequenceClassification = _check_dependencies()
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    if not os.path.exists(os.path.join(args.data_dir, "train.csv")):
        download_data(args.data_dir)

    if not _checkpoint_exists(args.model_dir):
        raise FileNotFoundError(
            f"Could not find a trained checkpoint in {args.model_dir}.\n"
            f"Run: .venv/bin/python train.py --output-dir {args.model_dir}"
        )

    data = dataloader(data_dir=args.data_dir, valid_size=args.valid_size, random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    class TestDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            encoding = tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            return {key: value.squeeze(0) for key, value in encoding.items()}

    test_loader = DataLoader(
        TestDataset(build_model_text(data.test)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="predict"):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    submission = data.sample_submission.copy()
    submission["target"] = predictions
    submission.to_csv(args.output_path, index=False)
    print(f"saved submission to {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-path", default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
