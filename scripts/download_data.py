"""
Step 1 — Download and prepare datasets.
Run: python scripts/download_data.py

Downloads:
  - Amazon Reviews 2023 (McAuley Lab) — Electronics category
  - Saves products.csv and reviews.csv to data/datasets/
"""
import os
import json
import gzip
import requests
import pandas as pd
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

# Amazon Reviews 2023 — Electronics (small subset)
# Full dataset: https://amazon-reviews-2023.github.io
REVIEWS_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
METADATA_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"

DATA_DIR = "./data/datasets"
MAX_REVIEWS = 200_000  # subset for development


def download_file(url: str, path: str):
    logger.info(f"Downloading {url}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def parse_jsonl_gz(path: str, max_rows: int = None) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                rows.append(json.loads(line.strip()))
            except Exception:
                continue
    return rows


def prepare_reviews(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    keep = ["parent_asin", "user_id", "rating", "text", "title", "verified_purchase", "helpful_vote", "timestamp"]
    df = df[[c for c in keep if c in df.columns]].rename(columns={
        "parent_asin": "product_id",
        "text": "reviewText",
        "rating": "overall",
        "helpful_vote": "helpful_votes"
    })
    df = df.dropna(subset=["product_id", "reviewText"])
    return df


def prepare_products(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    keep = ["parent_asin", "title", "price", "main_category", "store", "description", "average_rating", "rating_number"]
    df = df[[c for c in keep if c in df.columns]].rename(columns={
        "parent_asin": "product_id",
        "main_category": "category",
        "store": "brand",
        "average_rating": "avg_rating",
        "rating_number": "review_count"
    })
    df["description"] = df["description"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x or "")
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download reviews
    reviews_gz = f"{DATA_DIR}/Electronics_reviews.jsonl.gz"
    if not os.path.exists(reviews_gz):
        download_file(REVIEWS_URL, reviews_gz)
    logger.info("Parsing reviews...")
    reviews_raw = parse_jsonl_gz(reviews_gz, max_rows=MAX_REVIEWS)
    reviews_df = prepare_reviews(reviews_raw)
    reviews_df.to_csv(f"{DATA_DIR}/reviews.csv", index=False)
    logger.info(f"Saved {len(reviews_df)} reviews → data/datasets/reviews.csv")

    # Download product metadata
    meta_gz = f"{DATA_DIR}/Electronics_meta.jsonl.gz"
    if not os.path.exists(meta_gz):
        download_file(METADATA_URL, meta_gz)
    logger.info("Parsing product metadata...")
    meta_raw = parse_jsonl_gz(meta_gz, max_rows=50_000)
    products_df = prepare_products(meta_raw)

    # Keep only products that have reviews
    valid_ids = set(reviews_df["product_id"].unique())
    products_df = products_df[products_df["product_id"].isin(valid_ids)]
    products_df.to_csv(f"{DATA_DIR}/products.csv", index=False)
    logger.info(f"Saved {len(products_df)} products → data/datasets/products.csv")


if __name__ == "__main__":
    main()
