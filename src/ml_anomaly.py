"""
ML & Anomaly Detection module for Claude Code telemetry.

Uses scikit-learn to provide four ML capabilities:
1. **Anomaly Detection** — IsolationForest flags unusual sessions
2. **Practice Classification** — DecisionTree predicts team from usage patterns
3. **Session Clustering** — KMeans groups sessions into behavioral clusters
4. **Cost Forecasting** — LinearRegression predicts future daily cost

ARCHITECTURE:
    ┌─────────────────────────┐
    │  Dashboard (Page 7)     │  ← visualizes results
    └──────────┬──────────────┘
               │ calls
    ┌──────────▼──────────────┐
    │  MLEngine (this file)   │  ← trains models, returns results
    └──────────┬──────────────┘
               │ reads features via
    ┌──────────▼──────────────┐
    │  DatabaseManager        │  ← raw SQL → DataFrames
    └─────────────────────────┘

ALL MODELS ARE CPU-ONLY (scikit-learn). Training takes < 3 seconds on 5,000 sessions.

USAGE:
    from src.ml_anomaly import MLEngine
    ml = MLEngine("data/analytics.db")
    results = ml.run_all()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from src.database import DatabaseManager


# ---------------------------------------------------------------------------
# DATA CLASSES FOR RESULTS
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Results from IsolationForest anomaly detection."""
    features: pd.DataFrame          # session features with anomaly labels
    anomaly_count: int              # number of anomalies detected
    total_sessions: int             # total sessions analyzed
    contamination: float            # contamination parameter used
    anomaly_scores: np.ndarray      # raw decision_function scores
    feature_names: list[str]        # names of features used
    top_anomalies: pd.DataFrame     # top 20 most anomalous sessions


@dataclass
class ClassificationResult:
    """Results from DecisionTree practice classification."""
    accuracy: float                 # overall accuracy (cross-validated)
    cv_scores: np.ndarray           # per-fold accuracy scores
    feature_importances: pd.DataFrame  # feature importance ranking
    tree_rules: str                 # human-readable decision tree rules
    class_report: str               # precision/recall/f1 per class
    predictions: np.ndarray         # predicted labels
    true_labels: np.ndarray         # actual labels
    classes: list[str]              # class names (practices)


@dataclass
class ClusterResult:
    """Results from KMeans session clustering."""
    n_clusters: int                 # number of clusters
    cluster_labels: np.ndarray      # cluster assignment per session
    cluster_profiles: pd.DataFrame  # mean features per cluster
    pca_2d: pd.DataFrame            # 2D PCA projection for visualization
    inertia: float                  # KMeans inertia (within-cluster SS)
    cluster_sizes: dict[int, int]   # number of sessions per cluster


@dataclass
class ForecastResult:
    """Results from LinearRegression cost forecast."""
    historical: pd.DataFrame        # actual daily costs
    forecast: pd.DataFrame          # predicted future costs
    r2_score: float                 # R² on training data
    slope_per_day: float            # daily cost trend ($ per day)
    avg_daily_cost: float           # average daily cost


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "api_call_count",
    "total_cost",
    "total_input_tokens",
    "total_output_tokens",
    "total_cache_read_tokens",
    "prompt_count",
    "tool_use_count",
    "tool_success_count",
    "duration_minutes",
    "cost_per_api_call",
    "tokens_per_api_call",
    "tool_success_rate",
]


def build_session_features(db: DatabaseManager) -> pd.DataFrame:
    """Build a feature matrix from session-level aggregated data.

    Queries the database for per-session metrics and engineers
    additional computed features. Returns one row per session
    with 12 numeric features + metadata columns.

    Args:
        db: DatabaseManager instance connected to analytics.db

    Returns:
        DataFrame with columns:
            session_id, user_email, practice, level, location,
            + all 12 FEATURE_COLUMNS
    """
    # Get raw session stats (uses session_summary view + joins)
    df = db.get_session_stats()

    if df.empty:
        return pd.DataFrame()

    # Fill NaN with 0 (sessions with no tools, no prompts, etc.)
    for col in [
        "prompt_count", "tool_use_count", "tool_success_count",
        "total_cache_read_tokens", "duration_minutes",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure numeric types for aggregation columns
    for col in [
        "api_call_count", "total_cost", "total_input_tokens",
        "total_output_tokens", "total_cache_read_tokens",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---- Engineered features ----
    # Cost efficiency: cost per API call
    df["cost_per_api_call"] = np.where(
        df["api_call_count"] > 0,
        df["total_cost"] / df["api_call_count"],
        0.0,
    )

    # Token density: average tokens per API call
    df["tokens_per_api_call"] = np.where(
        df["api_call_count"] > 0,
        (df["total_input_tokens"] + df["total_output_tokens"]) / df["api_call_count"],
        0.0,
    )

    # Tool success rate (0-100)
    df["tool_success_rate"] = np.where(
        df["tool_use_count"] > 0,
        100.0 * df["tool_success_count"] / df["tool_use_count"],
        0.0,
    )

    # Clip negative duration (edge case from julianday rounding)
    df["duration_minutes"] = df["duration_minutes"].clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# ML ENGINE
# ---------------------------------------------------------------------------

class MLEngine:
    """Orchestrates all ML models for telemetry analysis.

    Builds features once, then runs four independent models:
    1. IsolationForest  → anomaly detection
    2. DecisionTree     → practice classification
    3. KMeans           → session clustering
    4. LinearRegression → cost forecasting

    Usage:
        ml = MLEngine("data/analytics.db")
        results = ml.run_all()
        # results is a dict with keys: anomalies, classification, clusters, forecast
    """

    def __init__(self, db_path: str) -> None:
        self.db = DatabaseManager(db_path)
        self._features: Optional[pd.DataFrame] = None

    @property
    def features(self) -> pd.DataFrame:
        """Lazy-load & cache session features."""
        if self._features is None:
            self._features = build_session_features(self.db)
        return self._features

    def _get_numeric_matrix(self) -> np.ndarray:
        """Extract the numeric feature matrix (N × 12) for ML models."""
        return self.features[FEATURE_COLUMNS].values

    # -------------------------------------------------------------------
    # 1. ANOMALY DETECTION — IsolationForest
    # -------------------------------------------------------------------

    def detect_anomalies(self, contamination: float = 0.05) -> AnomalyResult:
        """Find unusual sessions using IsolationForest.

        HOW IT WORKS:
        - Builds 100 random binary trees by randomly selecting
          features and split values.
        - Anomalies are data points that require FEWER splits to
          isolate (short average path length across all trees).
        - The decision_function returns a score: lower = more anomalous.

        Args:
            contamination: Expected fraction of anomalies (0.01–0.5).
                           Default 0.05 means ~5% of sessions flagged.

        Returns:
            AnomalyResult with labels, scores, and top anomalies.
        """
        df = self.features.copy()
        X = self._get_numeric_matrix()

        # Standardize features so no single feature dominates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train IsolationForest
        model = IsolationForest(
            n_estimators=100,       # 100 random trees
            contamination=contamination,
            random_state=42,        # reproducible results
            n_jobs=-1,              # use all CPU cores
        )
        labels = model.fit_predict(X_scaled)  # +1 = normal, -1 = anomaly
        scores = model.decision_function(X_scaled)  # continuous score

        # Add results to DataFrame
        df["anomaly_label"] = labels
        df["anomaly_score"] = scores
        df["is_anomaly"] = labels == -1

        # Top anomalies (most negative scores = most anomalous)
        top = (
            df[df["is_anomaly"]]
            .sort_values("anomaly_score")
            .head(20)
        )

        return AnomalyResult(
            features=df,
            anomaly_count=int((labels == -1).sum()),
            total_sessions=len(df),
            contamination=contamination,
            anomaly_scores=scores,
            feature_names=FEATURE_COLUMNS,
            top_anomalies=top,
        )

    # -------------------------------------------------------------------
    # 2. PRACTICE CLASSIFICATION — DecisionTreeClassifier
    # -------------------------------------------------------------------

    def classify_practice(self, max_depth: int = 5) -> ClassificationResult:
        """Predict which practice (team) a session belongs to.

        Uses a DecisionTree because:
        - Interpretable: we can print the decision rules in plain text
        - Fast: trains in milliseconds
        - Feature importance: shows which metrics distinguish teams

        The target variable is 'practice' (5 classes:
        ML Eng, Frontend Eng, Data Eng, Backend Eng, Platform Eng).

        Args:
            max_depth: Maximum tree depth (controls complexity).
                       5 is a good balance between accuracy and readability.

        Returns:
            ClassificationResult with accuracy, rules, importance.
        """
        df = self.features.copy()
        X = self._get_numeric_matrix()

        # Encode practice labels as integers
        le = LabelEncoder()
        y = le.fit_transform(df["practice"].values)
        class_names = le.classes_.tolist()

        # 5-fold cross-validation for honest accuracy estimate
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",  # handle any class imbalance
        )
        cv_scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")

        # Train final model on all data (for rules & importance)
        tree.fit(X, y)
        predictions = tree.predict(X)

        # Feature importance ranking
        importances = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": tree.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # Human-readable tree rules
        rules = export_text(
            tree,
            feature_names=FEATURE_COLUMNS,
            class_names=class_names,
            max_depth=max_depth,
        )

        # Classification report (precision, recall, f1 per class)
        report = classification_report(
            y, predictions,
            target_names=class_names,
            zero_division=0,
        )

        return ClassificationResult(
            accuracy=float(cv_scores.mean()),
            cv_scores=cv_scores,
            feature_importances=importances,
            tree_rules=rules,
            class_report=report,
            predictions=le.inverse_transform(predictions),
            true_labels=df["practice"].values,
            classes=class_names,
        )

    # -------------------------------------------------------------------
    # 3. SESSION CLUSTERING — KMeans
    # -------------------------------------------------------------------

    def cluster_sessions(self, n_clusters: int = 4) -> ClusterResult:
        """Group sessions into behavioral clusters.

        Uses KMeans to find natural groupings. Then projects
        to 2D with PCA for visualization.

        Why KMeans:
        - Simple, fast, deterministic (with fixed seed)
        - Produces equal-ish sized clusters
        - Easy to interpret cluster centroids

        Args:
            n_clusters: Number of clusters (default 4).

        Returns:
            ClusterResult with labels, profiles, and 2D projection.
        """
        df = self.features.copy()
        X = self._get_numeric_matrix()

        # Standardize (KMeans is distance-based, needs scaling)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train KMeans
        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,              # 10 random initializations
            max_iter=300,
        )
        cluster_labels = km.fit_predict(X_scaled)

        # Cluster profiles: mean of original (unscaled) features per cluster
        df["cluster"] = cluster_labels
        profiles = (
            df.groupby("cluster")[FEATURE_COLUMNS]
            .mean()
            .round(2)
            .reset_index()
        )

        # PCA projection to 2D for scatter plot
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": coords_2d[:, 0],
            "PC2": coords_2d[:, 1],
            "cluster": cluster_labels,
            "session_id": df["session_id"].values,
            "practice": df["practice"].values,
            "total_cost": df["total_cost"].values,
        })

        # Cluster sizes
        sizes = {int(k): int(v) for k, v in
                 pd.Series(cluster_labels).value_counts().items()}

        return ClusterResult(
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
            cluster_profiles=profiles,
            pca_2d=pca_df,
            inertia=float(km.inertia_),
            cluster_sizes=sizes,
        )

    # -------------------------------------------------------------------
    # 4. COST FORECASTING — LinearRegression
    # -------------------------------------------------------------------

    def forecast_cost(self, days_ahead: int = 30) -> ForecastResult:
        """Predict future daily costs using linear trend.

        Fits a simple linear regression on historical daily cost
        (day_number → daily_cost) and extrapolates forward.

        Why LinearRegression:
        - Transparent: slope tells you $/day trend
        - Fast, no hyperparameters
        - Good enough for a trend indicator

        Args:
            days_ahead: Number of days to forecast into the future.

        Returns:
            ForecastResult with historical data, forecast, and R².
        """
        # Get daily cost from the database
        daily = self.db.get_daily_trends()

        if daily.empty:
            return ForecastResult(
                historical=pd.DataFrame(),
                forecast=pd.DataFrame(),
                r2_score=0.0,
                slope_per_day=0.0,
                avg_daily_cost=0.0,
            )

        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        # X = day number (0, 1, 2, ...), y = daily cost
        daily["day_number"] = (daily["date"] - daily["date"].min()).dt.days
        X = daily[["day_number"]].values
        y = daily["daily_cost"].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        # Historical fitted values
        daily["fitted_cost"] = model.predict(X)

        # Forecast future dates
        last_date = daily["date"].max()
        last_day_num = int(daily["day_number"].max())

        future_days = np.arange(
            last_day_num + 1,
            last_day_num + 1 + days_ahead
        ).reshape(-1, 1)
        future_costs = model.predict(future_days)

        forecast_df = pd.DataFrame({
            "date": pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=days_ahead,
                freq="D",
            ),
            "day_number": future_days.flatten(),
            "predicted_cost": future_costs,
        })

        return ForecastResult(
            historical=daily,
            forecast=forecast_df,
            r2_score=float(r2),
            slope_per_day=float(model.coef_[0]),
            avg_daily_cost=float(y.mean()),
        )

    # -------------------------------------------------------------------
    # RUN ALL MODELS
    # -------------------------------------------------------------------

    def run_all(
        self,
        contamination: float = 0.05,
        max_depth: int = 5,
        n_clusters: int = 4,
        days_ahead: int = 30,
    ) -> dict[str, Any]:
        """Run all four ML models and return results.

        Args:
            contamination: IsolationForest contamination rate.
            max_depth: DecisionTree max depth.
            n_clusters: KMeans number of clusters.
            days_ahead: LinearRegression forecast horizon.

        Returns:
            Dict with keys: anomalies, classification, clusters, forecast
        """
        return {
            "anomalies": self.detect_anomalies(contamination),
            "classification": self.classify_practice(max_depth),
            "clusters": self.cluster_sessions(n_clusters),
            "forecast": self.forecast_cost(days_ahead),
        }


# ---------------------------------------------------------------------------
# CLI — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("ML & Anomaly Detection — Quick Test")
    print("=" * 60)

    ml = MLEngine("data/analytics.db")

    # --- Features ---
    t0 = time.perf_counter()
    feats = ml.features
    t_feat = time.perf_counter() - t0
    print(f"\n✓ Features: {feats.shape[0]} sessions × {feats.shape[1]} columns "
          f"({t_feat:.2f}s)")
    print(f"  Feature columns: {FEATURE_COLUMNS}")

    # --- Anomaly Detection ---
    t0 = time.perf_counter()
    anom = ml.detect_anomalies()
    t_anom = time.perf_counter() - t0
    print(f"\n✓ Anomalies: {anom.anomaly_count}/{anom.total_sessions} flagged "
          f"({anom.contamination:.0%} contamination) ({t_anom:.2f}s)")
    print(f"  Score range: [{anom.anomaly_scores.min():.3f}, "
          f"{anom.anomaly_scores.max():.3f}]")

    # --- Classification ---
    t0 = time.perf_counter()
    clf = ml.classify_practice()
    t_clf = time.perf_counter() - t0
    print(f"\n✓ Classification: {clf.accuracy:.1%} accuracy (5-fold CV) ({t_clf:.2f}s)")
    print(f"  CV scores: {clf.cv_scores.round(3)}")
    print(f"  Classes: {clf.classes}")
    print(f"\n  Top 5 features:")
    for _, row in clf.feature_importances.head(5).iterrows():
        print(f"    {row['feature']:30s} → {row['importance']:.4f}")

    # --- Clustering ---
    t0 = time.perf_counter()
    clust = ml.cluster_sessions()
    t_clust = time.perf_counter() - t0
    print(f"\n✓ Clustering: {clust.n_clusters} clusters ({t_clust:.2f}s)")
    for cid, size in sorted(clust.cluster_sizes.items()):
        print(f"    Cluster {cid}: {size} sessions")

    # --- Forecast ---
    t0 = time.perf_counter()
    fc = ml.forecast_cost()
    t_fc = time.perf_counter() - t0
    print(f"\n✓ Forecast: R²={fc.r2_score:.4f}, trend={fc.slope_per_day:+.4f} $/day "
          f"({t_fc:.2f}s)")
    print(f"  Avg daily cost: ${fc.avg_daily_cost:.2f}")
    if not fc.forecast.empty:
        print(f"  30-day forecast range: "
              f"${fc.forecast['predicted_cost'].min():.2f} – "
              f"${fc.forecast['predicted_cost'].max():.2f}")

    print(f"\n{'=' * 60}")
    print("All models complete!")
