"""
Tool 6: Price Compare
Compares prices across platforms. Uses mock/stored data (extend with scraping).
"""
from langchain_core.tools import tool
import pandas as pd
from config.settings import settings
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def price_compare(product_name: str) -> dict:
    """
    Compare prices for a product across platforms (Amazon, Flipkart, etc.)
    Use this when user asks about best price or where to buy cheapest.

    Args:
        product_name: Product name or ID to compare prices for

    Returns:
        dict with platform-wise prices, best deal, and confidence
    """
    logger.info(f"Price compare: {product_name}")

    # TODO Week 5: replace with real scraper using scrapy or requests + BeautifulSoup
    # For now: look up from reference price CSV or return mock data
    try:
        price_df = pd.read_csv("./data/reference/platform_prices.csv")
        matches = price_df[price_df["product_name"].str.lower().str.contains(
            product_name.lower(), na=False
        )]

        if not matches.empty:
            platforms = matches.set_index("platform")["price"].to_dict()
        else:
            # Mock fallback
            platforms = {
                "Amazon": 1999,
                "Flipkart": 1899,
                "Croma": 2099,
                "Reliance Digital": 2149
            }

    except FileNotFoundError:
        platforms = {
            "Amazon": 1999,
            "Flipkart": 1899,
            "Croma": 2099,
            "Reliance Digital": 2149
        }

    best = min(platforms, key=platforms.get)

    return confidence_response(
        result={
            "platforms": platforms,
            "best_deal": {"platform": best, "price": platforms[best]},
            "price_range": {"min": min(platforms.values()), "max": max(platforms.values())}
        },
        confidence=0.7,  # lower since mock data
        source_count=len(platforms)
    )
