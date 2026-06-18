"""
Step 3 — Train ABSA and fake review models.
Run: python scripts/train_models.py
     python scripts/train_models.py --model absa
     python scripts/train_models.py --model fake_review
"""
import argparse
from monitoring.mlflow_config import setup_mlflow
from utils.logger import get_logger

logger = get_logger(__name__)


def train_absa():
    from ml.absa_trainer import train
    logger.info("=== Training ABSA model ===")
    results = train()
    logger.info(f"ABSA done: {results}")


def train_fake_review():
    from ml.fake_review_trainer import train
    logger.info("=== Training Fake Review classifier ===")
    results = train()
    logger.info(f"Fake review done: {results}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["absa", "fake_review", "all"], default="all")
    args = parser.parse_args()

    setup_mlflow()

    if args.model in ("absa", "all"):
        train_absa()
    if args.model in ("fake_review", "all"):
        train_fake_review()

    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
