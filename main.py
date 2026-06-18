"""
CLI entry point — run ShopMind agent from terminal.
Usage:
  python main.py "find me wireless earphones under 2000"
  python main.py --image /path/to/product.jpg "find similar products"
  python main.py --user-id alice "recommend something for me"
"""
import argparse
import asyncio
from monitoring.langsmith_config import setup_langsmith
from monitoring.mlflow_config import setup_mlflow
from agents.orchestrator import ShopMindAgent
from utils.logger import get_logger

logger = get_logger("main")


async def run_query(query: str, user_id: str, image_path: str | None):
    setup_langsmith()
    setup_mlflow()

    agent = ShopMindAgent()
    print(f"\n🛒 ShopMind Agent")
    print(f"User: {user_id}")
    print(f"Query: {query}")
    if image_path:
        print(f"Image: {image_path}")
    print("─" * 60)

    result = await agent.run(query=query, user_id=user_id, image_path=image_path)

    print(f"\n📢 Answer:\n{result.answer}")
    print(f"\n📊 Metadata:")
    print(f"  Confidence : {result.confidence:.0%}")
    print(f"  Tools used : {', '.join(result.tools_used) or 'none'}")
    print(f"  Safety     : {'✅ passed' if result.safety_passed else '⚠️ warning'}")
    if result.evidence:
        print(f"  Evidence   : {result.evidence[:3]}")


def main():
    parser = argparse.ArgumentParser(description="ShopMind Agent CLI")
    parser.add_argument("query", help="Your shopping query")
    parser.add_argument("--user-id", default="cli_user", help="User ID for personalization")
    parser.add_argument("--image", default=None, help="Path to product image for visual search")
    args = parser.parse_args()

    asyncio.run(run_query(args.query, args.user_id, args.image))


if __name__ == "__main__":
    main()
