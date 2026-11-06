"""Part B: RFM segments, K-means, and a leakage-safe lapse model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]
CATEGORICAL = ["city", "gender", "favourite_category"]
FEATURES = NUMERIC + CATEGORICAL

ACTIONS = {
    "Premium": "Offer a high-touch retention call and a personalised electronics coupon before the next expected order.",
    "Loyal Regulars": "Protect the habit with a modest loyalty reward tied to weekly grocery orders.",
    "At Risk": "Send a win-back offer this week; these customers have gone quiet.",
    "Occasional": "Introduce a small basket-building bundle to lift order frequency.",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    return customers, products, orders


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders[orders["status"] == "completed"]
    grouped = completed.groupby("customer_id")
    rfm = pd.DataFrame({
        "customer_id": grouped.size().index,
        "recency": (SNAPSHOT - grouped["order_ts"].max()).dt.days.to_numpy(),
        "frequency": grouped["order_id"].nunique().to_numpy(),
        "monetary": grouped["order_amount"].sum().to_numpy(),
    })
    stats = {
        "customers": int(len(rfm)),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()),
        "recency_max": int(rfm["recency"].max()),
        "frequency_mean": round(float(rfm["frequency"].mean()), 2),
        "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": round(float(rfm["monetary"].mean()), 2),
        "monetary_max": round(float(rfm["monetary"].max()), 2),
    }
    expected = {
        "customers": 1175,
        "recency_median": 19.0,
        "recency_min": 0,
        "recency_max": 180,
        "frequency_mean": 9.46,
        "frequency_max": 33,
        "monetary_mean": 2552.06,
        "monetary_max": 14766.55,
    }
    for key, value in expected.items():
        if stats[key] != value:
            raise AssertionError(f"RFM {key}: got {stats[key]}, expected {value}")
    return rfm, stats


def cluster(rfm: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    fitted = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        inertia = float(model.inertia_)
        silhouette = float(silhouette_score(scaled, labels))
        rows.append({"k": k, "inertia": round(inertia, 1), "silhouette": round(silhouette, 3)})
        fitted[k] = (model, labels)
    reference = {
        2: (1820.8, 0.465),
        3: (1268.2, 0.445),
        4: (743.1, 0.526),
        5: (617.4, 0.481),
        6: (503.2, 0.482),
        7: (408.5, 0.416),
        8: (366.5, 0.412),
    }
    for row in rows:
        exp_inertia, exp_sil = reference[row["k"]]
        if abs(row["inertia"] - exp_inertia) / exp_inertia > 0.05:
            raise AssertionError(f"K={row['k']} inertia {row['inertia']} outside 5% of {exp_inertia}")
        if abs(row["silhouette"] - exp_sil) > 0.05:
            raise AssertionError(f"K={row['k']} silhouette {row['silhouette']} outside 0.05 of {exp_sil}")
    chosen_k = 4
    model, labels = fitted[chosen_k]
    profile_df = rfm.copy()
    profile_df["segment_id"] = labels
    profile = (
        profile_df.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    names = _name_segments(profile)
    profile["segment_name"] = profile["segment_id"].map(names)
    profile["is_premium"] = profile["segment_name"].eq("Premium")
    premium_monetary = float(profile.loc[profile["is_premium"], "monetary"].iloc[0])
    if premium_monetary != float(profile["monetary"].max()):
        raise AssertionError("Premium is not the highest-monetary segment")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled)
    ratios = [round(float(v), 3) for v in pca.explained_variance_ratio_]
    if abs(ratios[0] - 0.634) > 0.02 or abs(ratios[1] - 0.218) > 0.02:
        raise AssertionError(f"PCA ratios {ratios}")
    if abs(sum(pca.explained_variance_ratio_) - 0.852) > 0.02:
        raise AssertionError(f"PCA total {sum(pca.explained_variance_ratio_)}")
    out = rfm.copy()
    out["segment_id"] = labels
    out["segment_name"] = out["segment_id"].map(names)
    out["is_premium"] = out["segment_name"].eq("Premium")
    out = out.sort_values("customer_id")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "customer_segments.csv", index=False)
    return {
        "k_table": rows,
        "chosen_k": chosen_k,
        "profile": [
            {
                "segment_id": int(row.segment_id),
                "segment_name": row.segment_name,
                "size": int(row.size),
                "recency": round(float(row.recency), 2),
                "frequency": round(float(row.frequency), 2),
                "monetary": round(float(row.monetary), 2),
                "is_premium": bool(row.is_premium),
                "action": ACTIONS[row.segment_name],
            }
            for row in profile.itertuples(index=False)
        ],
        "pca_explained_variance_ratio": ratios,
        "pca_total": round(float(sum(pca.explained_variance_ratio_)), 3),
        "premium_customers": int(out["is_premium"].sum()),
    }


def _name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their means. Cluster ids are labels, not a ranking."""
    names: dict[int, str] = {}
    remaining = set(int(v) for v in profile["segment_id"])
    premium_row = profile.loc[profile["monetary"].idxmax()]
    names[int(premium_row["segment_id"])] = "Premium"
    remaining.remove(int(premium_row["segment_id"]))
    rest = profile[profile["segment_id"].isin(remaining)]
    loyal_row = rest.loc[rest["frequency"].idxmax()]
    names[int(loyal_row["segment_id"])] = "Loyal Regulars"
    remaining.remove(int(loyal_row["segment_id"]))
    rest = profile[profile["segment_id"].isin(remaining)]
    risk_row = rest.loc[rest["recency"].idxmax()]
    names[int(risk_row["segment_id"])] = "At Risk"
    remaining.remove(int(risk_row["segment_id"]))
    names[remaining.pop()] = "Occasional"
    return names


def _mode(series: pd.Series) -> str:
    return sorted(series.mode().tolist())[0]


def build_lapse_table(orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pre = orders[orders["order_ts"] < CUTOFF].copy()
    if (pre["order_ts"] >= CUTOFF).any():
        raise AssertionError("pre-cutoff frame contains later orders")
    completed = pre[pre["status"] == "completed"].merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    cohort_ids = completed["customer_id"].drop_duplicates()
    in_september = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < LABEL_END)
    ]["customer_id"].unique()
    label = pd.DataFrame({"customer_id": cohort_ids})
    label["lapsed"] = (~label["customer_id"].isin(in_september)).astype(int)

    grouped = completed.groupby("customer_id")
    features = pd.DataFrame({
        "recency": (CUTOFF - grouped["order_ts"].max()).dt.days,
        "frequency": grouped["order_id"].nunique(),
        "monetary": grouped["order_amount"].sum(),
        "avg_discount_pct": grouped["discount_pct"].mean(),
        "avg_delivery_minutes": grouped["delivery_minutes"].mean(),
        "favourite_category": grouped["category"].agg(_mode),
    })
    final_counts = pre.groupby("customer_id").size()
    cancelled_counts = pre[pre["status"] == "cancelled"].groupby("customer_id").size()
    features["cancellation_rate"] = (cancelled_counts.reindex(features.index).fillna(0) / final_counts.reindex(features.index))
    cust = customers.set_index("customer_id")
    features["city"] = cust["city"]
    features["age"] = cust["age"]
    features["gender"] = cust["gender"]
    features["tenure_days"] = (CUTOFF - cust["signup_date"]).dt.days
    table = features.join(label.set_index("customer_id")).reset_index()
    if table[FEATURES + ["lapsed"]].isna().any().any():
        raise AssertionError("lapse feature table contains missing values")
    stats = {
        "cohort": int(len(table)),
        "lapsed": int(table["lapsed"].sum()),
        "lapsed_rate": round(float(table["lapsed"].mean()), 4),
    }
    if stats["cohort"] != 1156 or stats["lapsed"] != 443:
        raise AssertionError(f"cohort mismatch {stats}")
    return table, stats


def _metrics(y_true, y_pred) -> dict:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _make_pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def train_models(table: pd.DataFrame) -> dict:
    train_df, test_df = train_test_split(
        table, test_size=0.2, stratify=table["lapsed"], random_state=42
    )
    split = {
        "train": int(len(train_df)),
        "test": int(len(test_df)),
        "train_lapsed": int(train_df["lapsed"].sum()),
        "test_lapsed": int(test_df["lapsed"].sum()),
        "train_not_lapsed": int((train_df["lapsed"] == 0).sum()),
    }
    if split["train"] != 924 or split["test"] != 232:
        raise AssertionError(f"split sizes {split}")
    y_train = train_df["lapsed"].to_numpy()
    y_test = test_df["lapsed"].to_numpy()
    x_train = train_df[FEATURES]
    x_test = test_df[FEATURES]
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise AssertionError("a feature column contains a missing value")

    baseline_pred = np.zeros_like(y_test)
    baseline = {
        "accuracy": round(float(accuracy_score(y_test, baseline_pred)), 4),
        "f1": round(float(f1_score(y_test, baseline_pred, zero_division=0)), 4),
    }
    if abs(baseline["accuracy"] - 0.616) > 0.005 or baseline["f1"] != 0.0:
        raise AssertionError(f"baseline {baseline}")

    lr = GridSearchCV(
        _make_pipeline(LogisticRegression(max_iter=1000)),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )
    lr.fit(x_train, y_train)
    rf = GridSearchCV(
        _make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
        param_grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
        cv=5,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )
    rf.fit(x_train, y_train)

    scaler = lr.best_estimator_.named_steps["prep"].named_transformers_["num"]
    freq_index = NUMERIC.index("frequency")
    scaler_mean = round(float(scaler.mean_[freq_index]), 4)
    cohort_mean = round(float(table["frequency"].mean()), 4)

    lr_test = _metrics(y_test, lr.predict(x_test))
    rf_test = _metrics(y_test, rf.predict(x_test))
    if max(lr_test["f1"], rf_test["f1"]) < 0.55:
        raise AssertionError(f"neither model reached F1 0.55: {lr_test['f1']} {rf_test['f1']}")
    selected_name = "random_forest" if rf_test["f1"] >= lr_test["f1"] else "logistic_regression"
    selected = rf.best_estimator_ if selected_name == "random_forest" else lr.best_estimator_

    rng = np.random.RandomState(42)
    lapsed_pos = np.flatnonzero(y_train == 1)
    extra = rng.choice(lapsed_pos, size=split["train_not_lapsed"] - split["train_lapsed"], replace=True)
    balanced_idx = np.concatenate([np.arange(len(train_df)), extra])
    x_bal = x_train.iloc[balanced_idx]
    y_bal = y_train[balanced_idx]
    if int((y_bal == 1).sum()) != int((y_bal == 0).sum()):
        raise AssertionError("balanced training classes are not equal")
    balanced_lr = _make_pipeline(LogisticRegression(max_iter=1000, C=lr.best_params_["model__C"]))
    balanced_lr.fit(x_bal, y_bal)
    balanced_test = _metrics(y_test, balanced_lr.predict(x_test))

    rf_names = selected.named_steps["prep"].get_feature_names_out() if selected_name == "random_forest" else rf.best_estimator_.named_steps["prep"].get_feature_names_out()
    importances = rf.best_estimator_.named_steps["model"].feature_importances_
    top_features = sorted(
        ({"feature": str(name), "importance": round(float(score), 4)} for name, score in zip(rf_names, importances)),
        key=lambda item: item["importance"],
        reverse=True,
    )[:5]
    lr_names = lr.best_estimator_.named_steps["prep"].get_feature_names_out()
    coefs = lr.best_estimator_.named_steps["model"].coef_[0]
    coef_map = {str(name): float(coef) for name, coef in zip(lr_names, coefs)}
    chosen_coefs = []
    for key in ("num__recency", "num__frequency", "num__monetary"):
        chosen_coefs.append({"feature": key, "coefficient": round(coef_map[key], 4), "sign": "positive" if coef_map[key] > 0 else "negative"})

    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    original_pred = selected.predict(x_test)
    reloaded_pred = reloaded.predict(x_test)
    if not np.array_equal(original_pred, reloaded_pred):
        raise AssertionError("reloaded model predictions differ")

    proba = selected.predict_proba(x_test)[:, 1]
    scored = test_df[["customer_id", "lapsed"]].copy()
    scored["lapse_probability"] = proba
    segments = pd.read_csv(OUT / "customer_segments.csv")
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
    retention = scored[scored["customer_id"].isin(premium_ids)].sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True]
    ).head(10)
    retention = retention.rename(columns={"lapsed": "actually_lapsed"})
    retention["lapse_probability"] = retention["lapse_probability"].round(6)
    retention.to_csv(OUT / "retention_list.csv", index=False)

    return {
        "split": split,
        "baseline": baseline,
        "scaler_frequency_mean": scaler_mean,
        "cohort_frequency_mean": cohort_mean,
        "logistic": {
            "best_params": lr.best_params_,
            "best_cv_f1": round(float(lr.best_score_), 4),
            "test": lr_test,
        },
        "random_forest": {
            "best_params": rf.best_params_,
            "best_cv_f1": round(float(rf.best_score_), 4),
            "test": rf_test,
            "top_features": top_features,
        },
        "selected_model": selected_name,
        "logistic_coefficients": chosen_coefs,
        "balanced_training_rows_per_class": int((y_bal == 1).sum()),
        "balanced_logistic_test": balanced_test,
        "reload_matches": True,
        "retention_rows": int(len(retention)),
        "retention_actually_lapsed": int(retention["actually_lapsed"].sum()),
        "retention_preview": retention.to_dict(orient="records"),
    }


def run() -> dict:
    customers, products, orders = _load()
    rfm, rfm_stats = build_rfm(orders)
    segments = cluster(rfm)
    table, cohort = build_lapse_table(orders, customers, products)
    models = train_models(table)
    return {"rfm": rfm_stats, "segments": segments, "cohort": cohort, "models": models}


def main() -> None:
    metrics = run()
    print("METRICS_JSON_START")
    print(json.dumps(metrics, indent=2, default=str))
    print("METRICS_JSON_END")


if __name__ == "__main__":
    main()
