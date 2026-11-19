"""FreshFlow end-to-end NumPy analysis."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

COLUMNS = ["transaction_id", "day_index", "outlet_id", "product_code",
           "category_code", "quantity", "unit_price", "unit_cost",
           "discount_rate", "stock_before_units", "payment_code", "customer_rating"]
OUTLETS = {1: "Outlet A", 2: "Outlet B", 3: "Outlet C"}
PRODUCTS = {1: "Orange Juice", 2: "Mango Juice", 3: "Apple Juice",
            4: "Watermelon Juice", 5: "Banana Smoothie", 6: "Mixed Fruit Smoothie",
            7: "Lemonade", 8: "Iced Tea"}
CATEGORIES = {1: "Juice", 2: "Smoothie", 3: "Lemonade", 4: "Tea"}

def load_data(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float,
                         filling_values=np.nan, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} numeric columns; got {data.shape}")
    return data

def validate_data(data):
    n = len(data)
    flags = np.zeros(n, dtype=bool)
    reasons = [[] for _ in range(n)]
    def mark(mask, reason):
        for i in np.where(mask)[0]:
            flags[i] = True
            reasons[i].append(reason)

    ids = data[:, 0]
    _, inverse, counts = np.unique(ids[~np.isnan(ids)], return_inverse=True, return_counts=True)
    repeated = np.zeros(n, dtype=bool)
    repeated[np.where(~np.isnan(ids))[0]] = counts[inverse] > 1
    mark(repeated, "duplicate_transaction_id")
    checks = [
        (np.isnan(data[:, 0]), "missing_transaction_id"),
        (np.isnan(data[:, 1]) | (data[:, 1] < 1) | (data[:, 1] > 30), "invalid_day_index"),
        (~np.isin(data[:, 2], [1, 2, 3]), "unknown_outlet_code"),
        (~np.isin(data[:, 3], np.arange(1, 9)), "unknown_product_code"),
        (~np.isin(data[:, 4], np.arange(1, 5)), "unknown_category_code"),
        (~np.isfinite(data[:, 5]) | (data[:, 5] <= 0), "invalid_quantity"),
        (np.isnan(data[:, 6]), "missing_unit_price"),
        (np.isfinite(data[:, 6]) & (data[:, 6] <= 0), "invalid_unit_price"),
        (np.isnan(data[:, 7]), "missing_unit_cost"),
        (np.isfinite(data[:, 7]) & (data[:, 7] < 0), "invalid_unit_cost"),
        (~np.isfinite(data[:, 8]) | (data[:, 8] < 0) | (data[:, 8] > 1), "invalid_discount_rate"),
        (~np.isfinite(data[:, 9]) | (data[:, 9] < 0), "invalid_stock_before_units"),
        (~np.isin(data[:, 10], [1, 2, 3, 4]), "unknown_payment_code"),
        (np.isfinite(data[:, 11]) & ((data[:, 11] < 1) | (data[:, 11] > 5)), "invalid_customer_rating"),
        (np.isnan(data[:, 11]), "missing_customer_rating"),
    ]
    for mask, reason in checks:
        mark(mask, reason)

    expected = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    product_ok = np.isin(data[:, 3], np.arange(1, 9))
    mismatch = np.zeros(n, dtype=bool)
    mismatch[product_ok] = data[product_ok, 4] != expected[data[product_ok, 3].astype(int) - 1]
    mark(mismatch, "product_category_mismatch")
    mark(data[:, 6] > 500, "unusually_high_unit_price")
    mark(data[:, 8] > .5, "unusually_high_discount")
    mark(data[:, 9] == 0, "zero_stock_before_transaction")

    # Missing ratings and unusual-but-plausible values do not invalidate financial metrics.
    fatal = np.zeros(n, dtype=bool)
    for reason in ("duplicate_transaction_id", "missing_transaction_id", "invalid_day_index",
                   "unknown_outlet_code", "unknown_product_code", "unknown_category_code",
                   "invalid_quantity", "missing_unit_price", "invalid_unit_price",
                   "missing_unit_cost", "invalid_unit_cost",
                   "invalid_discount_rate", "invalid_stock_before_units",
                   "unknown_payment_code", "product_category_mismatch"):
        fatal |= np.array([reason in row for row in reasons])
    return flags, reasons, ~fatal

def calculate_metrics(data, valid):
    rows = data[valid].copy()
    gross = rows[:, 5] * rows[:, 6]
    discount = gross * rows[:, 8]
    net = gross - discount
    cogs = rows[:, 5] * rows[:, 7]
    profit = net - cogs
    return rows, {"gross_sales": gross, "discount_amount": discount,
                  "net_revenue": net, "cost_of_goods_sold": cogs,
                  "gross_profit": profit,
                  "gross_margin_pct": np.divide(profit, net, out=np.full(len(net), np.nan), where=net != 0) * 100,
                  "average_selling_price": np.divide(net, rows[:, 5], out=np.full(len(net), np.nan), where=rows[:, 5] != 0)}

def group_summary(rows, metrics, column, names):
    result = {}
    for code, name in names.items():
        mask = rows[:, column] == code
        result[name] = {"transactions": int(mask.sum()), "units_sold": float(np.sum(rows[mask, 5])),
                        "net_revenue": float(np.sum(metrics["net_revenue"][mask])),
                        "gross_profit": float(np.sum(metrics["gross_profit"][mask]))}
    return result

def daily_summary(rows, metrics):
    daily = []
    for day in range(1, 31):
        mask = rows[:, 1] == day
        daily.append([day, float(np.sum(metrics["net_revenue"][mask])),
                      float(np.sum(metrics["gross_profit"][mask])),
                      float(np.sum(rows[mask, 5]))])
    daily = np.array(daily)
    revenue = daily[:, 1]
    change = np.full(30, np.nan)
    change[1:] = np.diff(revenue)
    pct_change = np.full(30, np.nan)
    pct_change[1:] = np.divide(change[1:], revenue[:-1],
                               out=np.full(29, np.nan), where=revenue[:-1] != 0) * 100
    moving_avg = np.full(30, np.nan)
    for i in range(6, 30):
        moving_avg[i] = np.mean(revenue[i - 6:i + 1])
    q1, q3 = np.percentile(revenue, [25, 75])
    unusual = (revenue < q1 - 1.5 * (q3 - q1)) | (revenue > q3 + 1.5 * (q3 - q1))
    unusual |= np.abs(np.nan_to_num(pct_change, nan=0)) > 30
    return np.column_stack([daily, change, pct_change, moving_avg, unusual.astype(int)])

def descriptive_stats(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    q = np.percentile(values, [25, 50, 75])
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "std_dev": float(np.std(values)),
            "min": float(np.min(values)), "q1": float(q[0]),
            "q3": float(q[2]), "max": float(np.max(values))}

def anomaly_register(rows, metrics):
    values = metrics["net_revenue"]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    z = np.zeros(len(values))
    sd = np.std(values)
    if sd:
        z = (values - np.mean(values)) / sd
    flagged = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr) | (np.abs(z) >= 2)
    return [{"transaction_id": int(rows[i, 0]), "reason": "revenue_outlier_IQR_or_zscore",
             "net_revenue": float(values[i]), "z_score": float(z[i])}
            for i in np.where(flagged)[0]]

def discount_scenarios(rows):
    output = {}
    gross = rows[:, 5] * rows[:, 6]
    cost = rows[:, 5] * rows[:, 7]
    for rate in (.05, .10, .15):
        net = gross * (1 - rate)
        output[f"{rate:.0%}_discount"] = {
            "net_revenue": float(np.sum(net)),
            "gross_profit": float(np.sum(net - cost)),
            "assumed_discount_rate": rate,
            "assumption": "Quantity, unit price and unit cost remain constant; no demand response modeled."
        }
    return output

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def run_analysis(input_path="data/freshflow_sales.csv", output_dir="outputs"):
    data = load_data(input_path)
    if len(data) != 5000:
        raise ValueError(f"Expected 5,000 rows, found {len(data)}")
    if not np.array_equal(np.unique(data[:, 1][np.isfinite(data[:, 1])]), np.arange(1, 31)):
        raise ValueError("Dataset must cover day_index 1 through 30")
    flags, reasons, valid = validate_data(data)
    rows, metrics = calculate_metrics(data, valid)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality = [{"transaction_id": int(data[i, 0]) if np.isfinite(data[i, 0]) else "",
                "flagged": bool(flags[i]), "reasons": ";".join(reasons[i])}
               for i in range(len(data)) if flags[i]]
    write_csv(out / "quality_report.csv", ["transaction_id", "flagged", "reasons"],
              [[x["transaction_id"], x["flagged"], x["reasons"]] for x in quality])
    write_csv(out / "daily_trend.csv", ["day_index", "net_revenue", "gross_profit", "units_sold",
              "revenue_abs_change", "revenue_pct_change", "7_day_moving_average", "unusual_day_flag"],
              daily_summary(rows, metrics))
    metric_names = list(metrics)
    write_csv(out / "cleaned_sales.csv", COLUMNS + metric_names,
              np.column_stack([rows] + [metrics[name] for name in metric_names]))
    anomalies = anomaly_register(rows, metrics)
    write_csv(out / "anomaly_register.csv", ["transaction_id", "reason", "net_revenue", "z_score"],
              [[x["transaction_id"], x["reason"], x["net_revenue"], x["z_score"]] for x in anomalies])

    outlets = group_summary(rows, metrics, 2, OUTLETS)
    products = group_summary(rows, metrics, 3, PRODUCTS)
    categories = group_summary(rows, metrics, 4, CATEGORIES)
    daily = daily_summary(rows, metrics)
    discounts = rows[:, 8]
    bands = {"0-5%": discounts <= .05, "above 5%-10%": (discounts > .05) & (discounts <= .10),
             "above 10%-15%": (discounts > .10) & (discounts <= .15), "above 15%": discounts > .15}
    discount_report = {}
    for label, mask in bands.items():
        band_revenue = float(np.sum(metrics["net_revenue"][mask]))
        band_profit = float(np.sum(metrics["gross_profit"][mask]))
        discount_report[label] = {"transactions": int(mask.sum()),
            "quantity": float(np.sum(rows[mask, 5])), "net_revenue": band_revenue,
            "gross_profit": band_profit,
            "gross_margin_pct": band_profit / band_revenue * 100 if band_revenue else None}

    top_product = max(products, key=lambda k: products[k]["net_revenue"])
    top_outlet = max(outlets, key=lambda k: outlets[k]["net_revenue"])
    most_units = max(products, key=lambda k: products[k]["units_sold"])
    report = {
        "dataset_profile": {"source": str(input_path), "rows_loaded": len(data),
            "valid_financial_rows": len(rows), "flagged_rows": int(flags.sum()),
            "days": 30, "outlets": 3, "products": 8,
            "column_ranges": {COLUMNS[i]: [float(np.nanmin(data[:, i])), float(np.nanmax(data[:, i]))]
                              for i in range(len(COLUMNS))},
            "assumption": "Synthetic teaching data; 30-day period cannot establish seasonality."},
        "data_dictionary": {name: meaning for name, meaning in zip(COLUMNS, [
            "Unique transaction ID", "Observation day (1-30)", "Outlet code (1-3)",
            "Product code (1-8)", "Category code", "Units sold", "Selling price per unit",
            "Cost per unit", "Discount fraction (0-1)", "Stock before transaction",
            "Payment method code (1-4)", "Customer rating (1-5; may be missing)"])},
        "quality": {"flagged_row_count": int(flags.sum()),
                    "issue_counts": {reason: int(sum(reason in r for r in reasons))
                                     for reason in sorted({x for row in reasons for x in row})}},
        "metrics": {key: float(np.sum(value)) for key, value in metrics.items()
                    if key in ("gross_sales", "discount_amount", "net_revenue", "cost_of_goods_sold", "gross_profit")},
        "overall_gross_margin_pct": float(np.sum(metrics["gross_profit"]) /
            np.sum(metrics["net_revenue"]) * 100) if np.sum(metrics["net_revenue"]) else None,
        "outlet_comparison": outlets, "product_comparison": products,
        "category_comparison": categories, "descriptive_statistics": {
            "net_revenue_per_transaction": descriptive_stats(metrics["net_revenue"]),
            "quantity": descriptive_stats(rows[:, 5]), "unit_price": descriptive_stats(rows[:, 6])},
        "correlations": {"quantity_vs_net_revenue": float(np.corrcoef(rows[:, 5], metrics["net_revenue"])[0, 1]),
                         "discount_vs_net_revenue": float(np.corrcoef(rows[:, 8], metrics["net_revenue"])[0, 1])},
        "daily_trend": daily.tolist(), "discount_bands": discount_report,
        "discount_scenarios": discount_scenarios(rows), "anomaly_count": len(anomalies),
        "executive_findings": [
            f"{top_outlet} generated the highest net revenue among outlets.",
            f"{top_product} generated the highest product net revenue; {most_units} sold the most units.",
            f"{int(flags.sum())} rows were flagged for review; {len(rows)} rows were used for financial metrics."
        ],
        "recommendations": [
            f"Review product availability and promotion strategy for {top_product}, the leading revenue product.",
            f"Compare operating practices at {top_outlet} with the other outlets before reallocating stock.",
            "Resolve flagged source records and monitor daily net revenue, gross profit and discount mix."
        ],
        "limitations": ["Only 30 consecutive days are observed; do not infer annual seasonality.",
            "Gross profit excludes operating expenses and is not net profit.",
            "Correlation or scenario calculations do not prove causation or predict demand.",
            "The dataset is synthetic and intended for learning, not actual business decisions."]
    }
    (out / "final_report.json").write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FreshFlow NumPy analysis")
    parser.add_argument("--input", default="data/freshflow_sales.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    result = run_analysis(args.input, args.output)
    print(json.dumps({"rows": result["dataset_profile"]["rows_loaded"],
        "valid_rows": result["dataset_profile"]["valid_financial_rows"],
        "flagged_rows": result["quality"]["flagged_row_count"],
        "report": str(Path(args.output) / "final_report.json")}, indent=2))
