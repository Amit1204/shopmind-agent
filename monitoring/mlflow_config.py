"""
MLflow experiment tracking.
Tracks: model training metrics, hyperparameters, artifacts.
View UI: mlflow ui --port 5000
"""
import mlflow
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENTS = {
    "absa": "shopmind-absa-bert",
    "fake_review": "shopmind-fake-review-xgb",
    "recommender": "shopmind-hybrid-recommender",
}


def setup_mlflow():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info(f"MLflow tracking → {settings.mlflow_tracking_uri}")


def get_or_create_experiment(model_name: str) -> str:
    name = EXPERIMENTS.get(model_name, f"shopmind-{model_name}")
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(name)
    return experiment_id
