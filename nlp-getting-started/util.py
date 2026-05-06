import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DisasterTweetData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame

def download_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    kaggle_executable = shutil.which("kaggle")
    if not kaggle_executable:
        raise RuntimeError(
            "Could not find the `kaggle` CLI in PATH. "
            "Install it and make sure your shell can run `kaggle --help`."
        )

    try:
        subprocess.run(
            [
                kaggle_executable,
                "competitions",
                "download",
                "-c",
                "nlp-getting-started",
                "-p",
                data_dir,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        combined_output = "\n".join(part for part in [stdout, stderr] if part)
        if "You must authenticate before you can call the Kaggle API." in combined_output:
            raise RuntimeError(
                "Kaggle API authentication is missing. "
                "Place `kaggle.json` in `~/.kaggle/kaggle.json` and set "
                "`chmod 600 ~/.kaggle/kaggle.json`, then run this script again."
            ) from exc
        raise RuntimeError(
            "Failed to download the competition data with the Kaggle CLI.\n"
            f"{combined_output or exc}"
        ) from exc

    zip_path = os.path.join(data_dir, "nlp-getting-started.zip")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)

def _read_competition_csvs(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    submission_path = os.path.join(data_dir, "sample_submission.csv")

    missing_paths = [
        path
        for path in [train_path, test_path, submission_path]
        if not os.path.exists(path)
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Missing competition data files: {missing}. "
            "Run download_data(data_dir) first or place the Kaggle CSVs there."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission_df = pd.read_csv(submission_path)
    return train_df, test_df, sample_submission_df


def _prepare_text_columns(df):
    df = df.copy()
    for column in ["keyword", "location", "text"]:
        if column in df.columns:
            df[column] = df[column].fillna("")
    return df


def _to_tf_dataset(df, batch_size, shuffle=False, seed=42):
    import tensorflow as tf

    features = {
        "text": df["text"].to_numpy(),
        "keyword": df["keyword"].to_numpy(),
        "location": df["location"].to_numpy(),
    }

    if "target" in df.columns:
        dataset = tf.data.Dataset.from_tensor_slices(
            (features, df["target"].astype("int32").to_numpy())
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(features)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def dataloader(
    data_dir="data",
    valid_size=0.2,
    random_state=42,
    batch_size=None,
    as_tf_dataset=False,
):
    train_df, test_df, sample_submission_df = _read_competition_csvs(data_dir)
    train_df = _prepare_text_columns(train_df)
    test_df = _prepare_text_columns(test_df)

    train_split, valid_split = train_test_split(
        train_df,
        test_size=valid_size,
        random_state=random_state,
        stratify=train_df["target"],
    )

    data = DisasterTweetData(
        train=train_split.reset_index(drop=True),
        valid=valid_split.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        sample_submission=sample_submission_df,
    )

    if not as_tf_dataset:
        return data

    if batch_size is None:
        raise ValueError("batch_size is required when as_tf_dataset=True.")

    return DisasterTweetData(
        train=_to_tf_dataset(data.train, batch_size, shuffle=True, seed=random_state),
        valid=_to_tf_dataset(data.valid, batch_size),
        test=_to_tf_dataset(data.test, batch_size),
        sample_submission=data.sample_submission,
    )
