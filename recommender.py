"""
Airbnb Hibrit Öneri Motoru — v2 (Kişiselleştirilmiş)
======================================================
Üç katman:
  1. Content-Based Filtering  – özellik vektörleri arası cosine similarity
  2. Kişiselleştirilmiş Quality Score – kullanıcı profilinden gelen ağırlıklar
  3. Hybrid Ranker             – ikisini birleştirir, beğenilenleri çıkarır
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


SCORE_COLS = [
    "Rating score", "Accuracy score", "Cleanliness score",
    "Checkin score", "Communication score", "Location score",
    "Value for money score",
]

FEATURE_COLS = [
    "Price", "Beds number", "Bedrooms number", "Bathrooms number",
    "Maximum allowed guests", "Total reviews", "Reviews per month",
] + SCORE_COLS

DEFAULT_WEIGHTS = {
    "Rating score":          0.25,
    "Accuracy score":        0.10,
    "Cleanliness score":     0.20,
    "Checkin score":         0.05,
    "Communication score":   0.05,
    "Location score":        0.20,
    "Value for money score": 0.15,
}


def load_data(path: str, sample_size: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    coords = df["Coordinates"].str.split(", ", expand=True).astype(float)
    df["lat"] = coords[0]
    df["lon"] = coords[1]
    price_cap = df["Price"].quantile(0.99)
    df["Price"] = df["Price"].clip(upper=price_cap)
    df["is_superhost"] = (df["Host is superhost"] == "Superhost").astype(int)
    required = FEATURE_COLS + ["City", "Season", "Neighbourhood", "Property type"]
    df = df.dropna(subset=required)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def compute_quality_score(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """Kullanıcı profiline göre ağırlıklandırılmış kalite skoru."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    cols = list(weights.keys())
    scaler = MinMaxScaler()
    norm = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=cols,
        index=df.index,
    )
    w = np.array([weights[c] for c in cols])
    return (norm * w).sum(axis=1)


class ContentBasedRecommender:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.similarity_df = None
        self.item_ids = None
        self._fitted_df = None

    def fit(self, df: pd.DataFrame):
        feature_cols = [
            "Price", "Beds number", "Bedrooms number", "Bathrooms number",
            "Maximum allowed guests", "Total reviews",
        ]
        cat_dummies = pd.get_dummies(df[["Property type", "Neighbourhood"]], drop_first=True)
        features = pd.concat(
            [df[feature_cols].reset_index(drop=True), cat_dummies.reset_index(drop=True)],
            axis=1,
        )
        scaled = self.scaler.fit_transform(features)
        sim_matrix = cosine_similarity(scaled)
        self.item_ids = df["Listings id"].reset_index(drop=True)
        self.similarity_df = pd.DataFrame(
            sim_matrix, index=self.item_ids.values, columns=self.item_ids.values
        )
        self._fitted_df = df.reset_index(drop=True)
        return self

    def recommend(self, listing_id: int, n: int = 50) -> pd.DataFrame:
        if listing_id not in self.similarity_df.index:
            return pd.DataFrame()
        scores = self.similarity_df[listing_id].drop(listing_id).sort_values(ascending=False).head(n)
        result = self._fitted_df[self._fitted_df["Listings id"].isin(scores.index)].copy()
        result["content_score"] = result["Listings id"].map(scores)
        return result


@dataclass
class UserPreferences:
    city: str
    season: str
    property_type: Optional[str] = None
    min_beds: int = 1
    max_price: float = 500.0
    superhost_only: bool = False
    content_weight: float = 0.5
    quality_weight: float = 0.5
    # Profil bilgisi
    score_weights: Optional[dict] = None   # kullanıcı profilinden
    exclude_ids: Optional[list] = None     # daha önce beğenilmeyenler
    companion: str = "solo"
    interests: list = field(default_factory=list)


class HybridRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cb = ContentBasedRecommender()

    def recommend(self, prefs: UserPreferences, n_results: int = 8):
        # Filtrele
        pool = self.df[self.df["City"] == prefs.city].copy()
        pool = pool[pool["Season"] == prefs.season]
        pool = pool[pool["Price"] <= prefs.max_price]
        pool = pool[pool["Beds number"] >= prefs.min_beds]
        if prefs.property_type:
            pool = pool[pool["Property type"] == prefs.property_type]
        if prefs.superhost_only:
            pool = pool[pool["is_superhost"] == 1]
        # Beğenilmeyenleri çıkar
        if prefs.exclude_ids:
            pool = pool[~pool["Listings id"].isin(prefs.exclude_ids)]

        stats = {"pool_size": len(pool)}
        if len(pool) < 5:
            return pd.DataFrame(), stats

        # Kullanıcı profilinden gelen ağırlıklar (yoksa default)
        weights = prefs.score_weights or DEFAULT_WEIGHTS
        pool["quality_score"] = compute_quality_score(pool, weights)

        # Companion/Interest based boosting
        if hasattr(prefs, 'companion'):
            if prefs.companion == 'family':
                pool.loc[pool['Beds number'] >= 3, 'quality_score'] *= 1.15
            elif prefs.companion == 'couple':
                pool.loc[pool['Property type'].isin(['Entire home', 'Private room']), 'quality_score'] *= 1.05

        if hasattr(prefs, 'interests') and prefs.interests:
            if 'nature' in prefs.interests:
                pool.loc[pool['Property type'] == 'Entire home', 'quality_score'] *= 1.05
            if 'culture' in prefs.interests:
                pool.loc[pool['Location score'] >= 4.8, 'quality_score'] *= 1.05

        # Fiyat tercihi varsa (avg_liked_price) seed'i buna göre seç
        seed_id = pool.loc[pool["quality_score"].idxmax(), "Listings id"]

        self._cb.fit(pool)
        cb_recs = self._cb.recommend(seed_id, n=min(60, len(pool) - 1))

        if cb_recs.empty:
            result = pool.nlargest(n_results, "quality_score").copy()
            result["content_score"] = 0.0
        else:
            quality_map = pool.set_index("Listings id")["quality_score"]
            cb_recs["quality_score"] = cb_recs["Listings id"].map(quality_map).fillna(0)
            for col in ("content_score", "quality_score"):
                mn, mx = cb_recs[col].min(), cb_recs[col].max()
                if mx > mn:
                    cb_recs[col] = (cb_recs[col] - mn) / (mx - mn)
            w_c, w_q = prefs.content_weight, prefs.quality_weight
            total = w_c + w_q
            cb_recs["hybrid_score"] = (
                (w_c / total) * cb_recs["content_score"]
                + (w_q / total) * cb_recs["quality_score"]
            )
            result = cb_recs.nlargest(n_results, "hybrid_score").copy()

        stats["returned"] = len(result)
        return result, stats
