"""
Safety layer — validates agent responses before returning to user.
Checks: hallucinated products, price accuracy, review reliability.
"""
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyResult:
    passed: bool
    failed_checks: dict
    message: str = ""


class SafetyValidator:

    def __init__(self, product_db=None):
        self.product_db = product_db  # injected — set in app startup

    def validate(self, products: list[dict], agent_response: str) -> SafetyResult:
        checks = {}

        # 1. Hallucination check — every product in response exists in DB
        checks["hallucination"] = self._check_products_exist(products)

        # 2. Price verification — prices in response match stored prices
        checks["price_verified"] = self._verify_prices(products, agent_response)

        # 3. Review authenticity — min authenticity score across products
        checks["review_authentic"] = self._check_authenticity(products)

        failed = {k: v for k, v in checks.items() if not v}
        passed = len(failed) == 0

        if not passed:
            logger.warning(f"Safety check failed: {failed}")

        return SafetyResult(
            passed=passed,
            failed_checks=failed,
            message="" if passed else f"Safety checks failed: {list(failed.keys())}"
        )

    def _check_products_exist(self, products: list[dict]) -> bool:
        """Verify product IDs exist in the database."""
        if not products or self.product_db is None:
            return True
        ids = [p.get("product_id") for p in products if p.get("product_id")]
        return all(self.product_db.exists(pid) for pid in ids)

    def _verify_prices(self, products: list[dict], response: str) -> bool:
        """
        Check that prices mentioned in the response match stored prices.
        Simple heuristic: extract price patterns from response and cross-check.
        """
        # TODO: implement regex price extraction + DB cross-check
        # For now return True (implement in Week 6)
        return True

    def _check_authenticity(self, products: list[dict], threshold: float = 0.5) -> bool:
        """Ensure avg authenticity score of recommended products is above threshold."""
        scores = [p.get("authenticity_score", 1.0) for p in products]
        if not scores:
            return True
        return (sum(scores) / len(scores)) >= threshold


safety_validator = SafetyValidator()
