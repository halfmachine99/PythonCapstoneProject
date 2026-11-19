"""Create the 5,000-row numeric FreshFlow teaching dataset."""
import argparse
import csv
from pathlib import Path
import numpy as np

COLUMNS = [
    "transaction_id", "day_index", "outlet_id", "product_code",
    "category_code", "quantity", "unit_price", "unit_cost",
    "discount_rate", "stock_before_units", "payment_code", "customer_rating"
]

def generate_dataset(path="data/freshflow_sales.csv", seed=42):
    rng = np.random.default_rng(seed)
    n = 5000
    product_category = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    product_price = np.array([120, 140, 100, 110, 90, 130, 125, 115], dtype=float)
    product_cost = product_price * np.array([.48, .50, .45, .47, .52, .55, .53, .50])
    product = rng.integers(1, 9, n)
    quantity = rng.integers(1, 8, n).astype(float)
    price = product_price[product - 1] * rng.uniform(.94, 1.06, n)
    cost = product_cost[product - 1] * rng.uniform(.96, 1.04, n)
    discount = rng.choice([0, .05, .10, .15, .20], n, p=[.18, .24, .28, .20, .10])
    data = np.column_stack([
        np.arange(100001, 100001 + n), rng.integers(1, 31, n),
        rng.integers(1, 4, n), product, product_category[product - 1],
        quantity, price, cost, discount, rng.integers(10, 100, n),
        rng.integers(1, 5, n), rng.integers(1, 6, n)
    ]).astype(float)

    # Deliberate quality cases from the assignment specification.
    data[[10, 11, 12, 13], 6] = np.nan
    data[[20, 21, 22, 23], 7] = np.nan
    data[[30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
          40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
          50, 51, 52, 53, 54], 11] = np.nan
    data[[60, 61, 62, 63], 5] = [0, -1, 0, -2]
    data[[70, 71], 6] = [-20, -5]
    data[[72, 73], 7] = [-10, -3]
    data[[74, 75], 8] = [-.1, 1.2]
    data[[76, 77, 78], 11] = [0, 6, 9]
    data[[80, 81, 82, 83], 4] = 3 - product_category[product[[80, 81, 82, 83]] - 1]  # mapping mismatches
    data[[84, 85], 2] = [0, 5]                # unknown outlets
    data[[86, 87], 6] = [900, 1200]           # unusually high prices
    data[88, 8] = .85                         # unusually high discount
    data[[89, 90, 91, 92, 93, 94], 9] = 0     # zero stock
    data[95:100, 0] = data[90:95, 0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(data)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FreshFlow numeric CSV")
    parser.add_argument("--output", default="data/freshflow_sales.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.seed)
    print(f"Created {args.output} with 5,000 transactions.")
