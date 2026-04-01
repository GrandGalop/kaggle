import os
import shutil
import subprocess
import zipfile

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

def dataloader():
    pass