"""Part A: audit raw exports, clean them, store SQLite, and answer the SQL questions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

COMPLETED_REVENUE = 2998668.80


def load_raw():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    return customers, products, orders, tickets


def _is_blank(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.isna()
    text = series.astype(str).str.strip()
    return series.isna() | text.eq("") | text.str.lower().eq("nan")


def _missing_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(_is_blank(df[col]).sum())
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": missing,
            "missing_pct": round(100.0 * missing / n, 2) if n else 0.0,
        })
    return rows


def audit(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame, tickets: pd.DataFrame) -> dict:
    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    gender_raw = customers["gender"]
    blank_gender = gender_raw.isna() | (gender_raw.astype(str).str.strip() == "") | (gender_raw.astype(str) == "nan")
    blank_age = customers["age"].isna() | (customers["age"].astype(str).str.strip() == "") | (customers["age"].astype(str) == "nan")
    blank_delivery = orders["delivery_minutes"].isna() | (orders["delivery_minutes"].astype(str).str.strip() == "") | (orders["delivery_minutes"].astype(str) == "nan")

    report = {
        "shapes": {
            "customers": list(customers.shape),
            "products": list(products.shape),
            "orders": list(orders.shape),
            "tickets": list(tickets.shape),
        },
        "missing": {
            "customers": _missing_table(customers),
            "products": _missing_table(products),
            "orders": _missing_table(orders),
            "tickets": _missing_table(tickets),
        },
        "orders_rows": int(len(orders)),
        "exact_duplicate_rows": int(orders.duplicated().sum()),
        "pending_rows": int((orders["status"] == "pending").sum()),
        "distinct_order_id": int(orders["order_id"].nunique()),
        "non_numeric_amounts": int(amount_num.isna().sum()),
        "negative_delivery_minutes": int((delivery_num < 0).sum()),
        "blank_delivery_minutes": int(blank_delivery.sum()),
        "blank_age": int(blank_age.sum()),
        "age_outside_10_100": int(((age_num < 10) | (age_num > 100)).sum()),
        "distinct_city_spellings": int(customers["city"].nunique()),
        "distinct_nonblank_gender_spellings": int(gender_raw.loc[~blank_gender].nunique()),
        "blank_gender": int(blank_gender.sum()),
    }
    expected = {
        "orders_rows": 13084,
        "exact_duplicate_rows": 119,
        "pending_rows": 656,
        "distinct_order_id": 12309,
        "non_numeric_amounts": 120,
        "negative_delivery_minutes": 106,
        "blank_delivery_minutes": 2541,
        "blank_age": 61,
        "age_outside_10_100": 6,
        "distinct_city_spellings": 24,
        "distinct_nonblank_gender_spellings": 10,
        "blank_gender": 30,
    }
    for key, value in expected.items():
        if report[key] != value:
            raise AssertionError(f"audit {key}: got {report[key]}, expected {value}")
    return report


def _map_gender(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"", "nan"}:
        return "unknown"
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    return "unknown"


def clean_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = customers.copy()
    changes = {}
    original_city = df["city"].astype(str)
    df["city"] = original_city.str.strip().str.title()
    changes["city_values_rewritten"] = int((df["city"] != original_city).sum())
    changes["cities_after"] = df["city"].value_counts().sort_index().to_dict()

    original_gender = df["gender"]
    df["gender"] = original_gender.map(_map_gender)
    blank = original_gender.isna() | (original_gender.astype(str).str.strip() == "") | (original_gender.astype(str) == "nan")
    changes["gender_blank_to_unknown"] = int(blank.sum())
    changes["gender_spellings_remapped"] = int((~blank).sum())
    changes["genders_after"] = df["gender"].value_counts().sort_index().to_dict()

    age = pd.to_numeric(df["age"], errors="coerce")
    outside = (age < 10) | (age > 100)
    changes["ages_outside_10_100_set_missing"] = int(outside.fillna(False).sum())
    age = age.mask(outside)
    changes["ages_missing_before_impute"] = int(age.isna().sum())
    df["age"] = age
    city_median = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_median)
    changes["ages_imputed_with_city_median"] = changes["ages_missing_before_impute"]
    changes["age_min"] = float(df["age"].min())
    changes["age_max"] = float(df["age"].max())
    changes["age_missing_after"] = int(df["age"].isna().sum())

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    changes["signup_dates_parsed"] = int(df["signup_date"].notna().sum())
    return df, changes


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = orders.copy()
    changes = {"rows_start": int(len(df))}
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    changes["order_ts_parsed"] = int(df["order_ts"].notna().sum())

    before = len(df)
    df = df.drop_duplicates()
    changes["exact_duplicate_rows_removed"] = int(before - len(df))

    before = len(df)
    df = df.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    changes["older_status_rows_removed"] = int(before - len(df))

    df["quantity"] = pd.to_numeric(df["quantity"])
    df["discount_pct"] = pd.to_numeric(df["discount_pct"])
    amount = pd.to_numeric(df["order_amount"], errors="coerce")
    changes["unparseable_amounts_set_missing"] = int(amount.isna().sum())
    prices = products.set_index("product_id")["price_inr"]
    filled = (prices.loc[df["product_id"]].to_numpy() * df["quantity"].to_numpy() * (1 - df["discount_pct"].to_numpy() / 100))
    filled = np.round(filled, 2)
    amount = amount.where(amount.notna(), filled)
    changes["amounts_imputed_from_price"] = changes["unparseable_amounts_set_missing"]
    df["order_amount"] = amount

    delivery = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    negative = delivery < 0
    changes["negative_delivery_set_missing"] = int(negative.fillna(False).sum())
    delivery = delivery.mask(negative)
    df["delivery_minutes"] = delivery
    df = df.merge(products[["product_id", "category"]], on="product_id", how="left")
    category_median = df.groupby("category")["delivery_minutes"].transform("median")
    completed = df["status"] == "completed"
    missing_completed = completed & df["delivery_minutes"].isna()
    changes["completed_delivery_imputed"] = int(missing_completed.sum())
    df.loc[completed, "delivery_minutes"] = df.loc[completed, "delivery_minutes"].fillna(category_median.loc[completed])
    df = df.drop(columns=["category"])

    changes["rows_after"] = int(len(df))
    changes["completed"] = int((df["status"] == "completed").sum())
    changes["cancelled"] = int((df["status"] == "cancelled").sum())
    changes["pending"] = int((df["status"] == "pending").sum())
    changes["missing_amount"] = int(df["order_amount"].isna().sum())
    changes["missing_delivery_completed"] = int(df.loc[df["status"] == "completed", "delivery_minutes"].isna().sum())
    changes["missing_delivery_cancelled"] = int(df.loc[df["status"] == "cancelled", "delivery_minutes"].isna().sum())
    changes["completed_amount_sum"] = round(float(df.loc[df["status"] == "completed", "order_amount"].sum()), 2)
    return df.reset_index(drop=True), changes


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    df["price_inr"] = pd.to_numeric(df["price_inr"])
    return df


def _prepare_for_store(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame):
    customers = customers.copy()
    orders = orders.copy()
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return customers, products, orders


def store(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        customers.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        orders.to_sql("orders", con, index=False)


def validate() -> None:
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    checks = []

    def check(name, condition):
        checks.append(name)
        if not condition:
            raise AssertionError(f"checkpoint failed: {name}")

    city_counts = customers["city"].value_counts().to_dict()
    gender_counts = customers["gender"].value_counts().to_dict()
    completed = orders["status"] == "completed"
    cancelled = orders["status"] == "cancelled"
    check("orders_12309", len(orders) == 12309)
    check("completed_11110", int(completed.sum()) == 11110)
    check("cancelled_1199", int(cancelled.sum()) == 1199)
    check("no_pending", int((orders["status"] == "pending").sum()) == 0)
    check("no_missing_amount", int(orders["order_amount"].isna().sum()) == 0)
    check("no_missing_delivery_completed", int(orders.loc[completed, "delivery_minutes"].isna().sum()) == 0)
    check("missing_delivery_cancelled_1199", int(orders.loc[cancelled, "delivery_minutes"].isna().sum()) == 1199)
    revenue = round(float(orders.loc[completed, "order_amount"].sum()), 2)
    check("completed_revenue", revenue == COMPLETED_REVENUE)
    check("six_cities", set(customers["city"]) == {"Pune", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Mumbai"})
    check("city_counts", city_counts == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184})
    check("gender_counts", gender_counts == {"female": 610, "male": 560, "unknown": 30})
    check("ages_present_and_in_range", int(customers["age"].isna().sum()) == 0 and customers["age"].between(18, 60).all())
    check("one_row_per_order", orders["order_id"].nunique() == len(orders))
    if len(checks) < 8:
        raise AssertionError("need at least eight checkpoints")
    print("ALL CHECKPOINTS PASSED")


def load_queries() -> list[tuple[str, str]]:
    text = SQL_PATH.read_text()
    chunks = []
    current_name = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("-- Q"):
            if current_name and buf:
                chunks.append((current_name, "\n".join(buf).strip()))
            current_name = line.split(":")[0].replace("--", "").strip()
            buf = []
        else:
            buf.append(line)
    if current_name and buf:
        chunks.append((current_name, "\n".join(buf).strip()))
    return chunks


def run_sql() -> dict:
    queries = load_queries()
    results = {}
    with sqlite3.connect(DB_PATH) as con:
        for name, sql in queries:
            results[name] = pd.read_sql_query(sql, con)
    q1, q2, q3, q4, q5, q6, q7 = (results[f"Q{i}"] for i in range(1, 8))
    if len(q1) != 30:
        raise AssertionError(f"Q1 expected 30 customers, got {len(q1)}")
    if len(q2) != 30:
        raise AssertionError(f"Q2 expected 30 rows, got {len(q2)}")
    q2_sum = round(float(q2["revenue"].sum()), 2)
    if q2_sum != COMPLETED_REVENUE:
        raise AssertionError(f"Q2 revenue sum {q2_sum}")
    top = q2.sort_values("revenue", ascending=False).iloc[0]
    if top["city"] != "Bengaluru" or top["category"] != "Electronics" or round(float(top["revenue"]), 2) != 233909.45:
        raise AssertionError(f"Q2 top cell unexpected: {top.to_dict()}")
    if len(q3) != 11:
        raise AssertionError(f"Q3 expected 11 products, got {len(q3)}")
    if len(q4) != 25:
        raise AssertionError(f"Q4 expected 25 customers, got {len(q4)}")
    if len(q5) != 550:
        raise AssertionError(f"Q5 expected 550 customers, got {len(q5)}")
    with sqlite3.connect(DB_PATH) as con:
        avg_total = float(pd.read_sql_query(
            "WITH customer_spend AS (SELECT customer_id, SUM(order_amount) AS total_spend "
            "FROM orders WHERE status = 'completed' GROUP BY customer_id) "
            "SELECT AVG(total_spend) AS avg_total FROM customer_spend",
            con,
        )["avg_total"].iloc[0])
    if round(avg_total, 2) != 2552.06:
        raise AssertionError(f"Q5 average {avg_total}")
    tops = q6[q6["rank_in_city"] == 1].set_index("city")
    expected_tops = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    for city, (cid, spend) in expected_tops.items():
        if tops.loc[city, "customer_id"] != cid or round(float(tops.loc[city, "total_spend"]), 2) != spend:
            raise AssertionError(f"Q6 top for {city}: {tops.loc[city].to_dict()}")
    rank_gap = int((q6["rank_in_city"] != q6["row_number_in_city"]).sum())
    if rank_gap != 8:
        raise AssertionError(f"Q6 rank gaps {rank_gap}")
    if len(q7) != 183:
        raise AssertionError(f"Q7 expected 183 days, got {len(q7)}")
    if round(float(q7["cumulative_revenue"].iloc[-1]), 2) != COMPLETED_REVENUE:
        raise AssertionError("Q7 final cumulative revenue mismatch")
    return {
        "frames": {name: df for name, df in results.items()},
        "q1_customers": int(len(q1)),
        "q2_rows": int(len(q2)),
        "q2_sum": q2_sum,
        "q2_top": {"city": top["city"], "category": top["category"], "revenue": round(float(top["revenue"]), 2)},
        "q3_products": int(len(q3)),
        "q4_customers": int(len(q4)),
        "q5_customers": int(len(q5)),
        "q5_average": round(avg_total, 2),
        "q6_tops": {city: {"customer_id": cid, "total_spend": spend} for city, (cid, spend) in expected_tops.items()},
        "q6_rank_ne_row_number": rank_gap,
        "q7_days": int(len(q7)),
        "q7_final_cumulative": round(float(q7["cumulative_revenue"].iloc[-1]), 2),
    }


def exploratory(orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame, q2: pd.DataFrame) -> dict:
    completed = orders.merge(customers[["customer_id", "city"]], on="customer_id").merge(
        products[["product_id", "category"]], on="product_id"
    )
    completed = completed[completed["status"] == "completed"]
    pivot = pd.pivot_table(
        completed,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
    )
    grand = round(float(pivot.loc["Total", "Total"]), 2)
    if grand != round(float(q2["revenue"].sum()), 2):
        raise AssertionError(f"pivot grand total {grand} != SQL {q2['revenue'].sum()}")
    by_city = completed.groupby("city")["order_amount"].agg(sum="sum", mean="mean", count="count")
    by_city["sum"] = by_city["sum"].round(2)
    by_city["mean"] = by_city["mean"].round(2)
    amount = completed["order_amount"]
    q1 = float(amount.quantile(0.25))
    q3 = float(amount.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    corr = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    pairs = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            pairs.append((left, right, float(corr.loc[left, right])))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    bengaluru = by_city.loc["Bengaluru"]
    mumbai = by_city.loc["Mumbai"]
    if round(float(bengaluru["sum"]), 2) != 523643.40 or int(bengaluru["count"]) != 1894:
        raise AssertionError(f"Bengaluru agg unexpected: {bengaluru.to_dict()}")
    if round(float(bengaluru["mean"]), 2) != 276.47:
        raise AssertionError(f"Bengaluru mean {bengaluru['mean']}")
    if round(float(mumbai["sum"]), 2) != 479058.50 or round(float(mumbai["mean"]), 2) != 297.00 or int(mumbai["count"]) != 1613:
        raise AssertionError(f"Mumbai agg unexpected: {mumbai.to_dict()}")
    stats = {
        "pivot_grand_total": grand,
        "sql_total": round(float(q2["revenue"].sum()), 2),
        "by_city": by_city.reset_index().to_dict(orient="records"),
        "mean": round(float(amount.mean()), 2),
        "median": round(float(amount.median()), 2),
        "skewness": round(float(amount.skew()), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "lower_fence": round(lower, 2),
        "upper_fence": round(upper, 2),
        "outliers_above": int((amount > upper).sum()),
        "outliers_below": int((amount < lower).sum()),
        "largest_corr_pair": {"left": pairs[0][0], "right": pairs[0][1], "value": round(pairs[0][2], 3)},
        "correlations": [
            {"left": a, "right": b, "value": round(v, 3)} for a, b, v in pairs
        ],
    }
    if stats["mean"] != 269.91 or stats["median"] != 135.00:
        raise AssertionError(f"distribution mean/median {stats['mean']} {stats['median']}")
    if stats["q1"] != 64.80 or stats["q3"] != 297.00 or stats["upper_fence"] != 645.30:
        raise AssertionError(f"quartiles/fence unexpected: {stats}")
    if stats["outliers_above"] != 897 or stats["outliers_below"] != 0:
        raise AssertionError(f"outlier counts {stats['outliers_above']} {stats['outliers_below']}")
    if abs(stats["skewness"] - 4.61) > 0.05:
        raise AssertionError(f"skewness {stats['skewness']}")
    if stats["largest_corr_pair"]["left"] != "order_amount" or stats["largest_corr_pair"]["right"] != "discount_pct":
        raise AssertionError(f"unexpected correlation pair {stats['largest_corr_pair']}")
    return stats


def run() -> dict:
    customers, products, orders, tickets = load_raw()
    audit_report = audit(customers, products, orders, tickets)
    clean_cust, cust_changes = clean_customers(customers)
    clean_ord, ord_changes = clean_orders(orders, products)
    clean_prod = clean_products(products)
    stored_cust, stored_prod, stored_ord = _prepare_for_store(clean_cust, clean_prod, clean_ord)
    store(stored_cust, stored_prod, stored_ord)
    validate()
    sql = run_sql()
    eda = exploratory(stored_ord, stored_cust, stored_prod, sql["frames"]["Q2"])
    metrics = {
        "audit": {k: v for k, v in audit_report.items() if k != "missing"},
        "audit_missing": audit_report["missing"],
        "audit_shapes": audit_report["shapes"],
        "customer_changes": cust_changes,
        "order_changes": ord_changes,
        "sql": {k: v for k, v in sql.items() if k != "frames"},
        "eda": eda,
    }
    return metrics


def main() -> None:
    metrics = run()
    print("METRICS_JSON_START")
    print(json.dumps(metrics, indent=2, default=str))
    print("METRICS_JSON_END")


if __name__ == "__main__":
    main()
