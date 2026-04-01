import os
from util import download_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def main():
    download_data(DATA_DIR)

if __name__ == "__main__":
    main()