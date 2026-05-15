"""
Tarih Bazlı Popülerlik Motoru
================================
Kullanıcı tarih seçince:
  1. Tarihin hangi sezona denk geldiğini bulur
  2. O sezonda en popüler şehirleri sıralar
  3. Her şehir için "bu dönemde ortalama X gece kalınıyor, Y€ harcanıyor" der
  4. Sosyal kanıt skoru hesaplar
"""

import pandas as pd
import numpy as np
from datetime import date, datetime


# ── Tarih → Sezon Eşlemesi ────────────────────────────────────────────────

SEASON_MAP = {
    # (başlangıç_ay, bitiş_ay): sezon_adı
    (3,  5):  "Early Spring",
    (6,  8):  "Early Summer",
    (9,  11): "Early Autumn",
    (12, 2):  "Early Winter",
}

SEASON_DESCRIPTIONS = {
    "Early Spring": "🌸 İlkbahar — Kalabalık öncesi sakin dönem, fiyatlar uygun",
    "Early Summer": "☀️ Yaz — En yoğun dönem, erken rezervasyon şart",
    "Early Autumn": "🍂 Sonbahar — Harika hava, azalan kalabalık, iyi fiyatlar",
    "Early Winter": "❄️ Kış — Düşük sezon, en uygun fiyatlar, özgün atmosfer",
}


def date_to_season(travel_date: date) -> str:
    """Tarihi sezona çevir."""
    month = travel_date.month
    if 3 <= month <= 5:
        return "Early Spring"
    elif 6 <= month <= 8:
        return "Early Summer"
    elif 9 <= month <= 11:
        return "Early Autumn"
    else:
        return "Early Winter"


# ── Popülerlik Analizi ────────────────────────────────────────────────────

def get_destination_insights(
    df: pd.DataFrame,
    season: str,
    nights: int = 3,
) -> pd.DataFrame:
    """
    Sezona göre şehirlerin detaylı istatistikleri.
    Her satır bir şehir, sütunlar:
      - listing_count   : ilan sayısı (popülerlik göstergesi)
      - avg_price       : ortalama gecelik fiyat
      - total_budget    : avg_price × nights (tahmini toplam)
      - avg_rating      : ortalama puan
      - superhost_ratio : superhostların oranı (güven göstergesi)
      - popularity_score: normalize edilmiş popülerlik skoru
    """
    season_df = df[df["Season"] == season].copy()

    insights = (
        season_df.groupby("City")
        .agg(
            listing_count   =("Listings id",        "count"),
            avg_price       =("Price",               "mean"),
            median_price    =("Price",               "median"),
            avg_rating      =("Rating score",        "mean"),
            avg_cleanliness =("Cleanliness score",   "mean"),
            avg_location    =("Location score",      "mean"),
            superhost_count =("is_superhost",        "sum"),
            total_listings  =("is_superhost",        "count"),
        )
        .reset_index()
    )

    insights["superhost_ratio"] = (
        insights["superhost_count"] / insights["total_listings"]
    ).round(2)

    insights["total_budget"] = (insights["avg_price"] * nights).round(0)

    # Popülerlik skoru: ilan sayısı + puan + superhost oranı
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    pop_features = insights[["listing_count", "avg_rating", "superhost_ratio"]].copy()
    pop_norm = scaler.fit_transform(pop_features)
    insights["popularity_score"] = (
        0.5 * pop_norm[:, 0] +   # ilan sayısı ağırlığı
        0.3 * pop_norm[:, 1] +   # puan ağırlığı
        0.2 * pop_norm[:, 2]     # superhost ağırlığı
    ).round(3)

    return insights.sort_values("popularity_score", ascending=False).reset_index(drop=True)


def get_neighbourhood_insights(
    df: pd.DataFrame,
    city: str,
    season: str,
) -> pd.DataFrame:
    """Seçilen şehirde hangi mahalleler bu sezonda öne çıkıyor?"""
    subset = df[(df["City"] == city) & (df["Season"] == season)]
    nbh = (
        subset.groupby("Neighbourhood")
        .agg(
            listing_count=("Listings id",      "count"),
            avg_price    =("Price",             "mean"),
            avg_rating   =("Rating score",      "mean"),
            avg_location =("Location score",    "mean"),
        )
        .reset_index()
    )
    # En az 3 ilanı olan mahalleleri göster
    nbh = nbh[nbh["listing_count"] >= 3]
    return nbh.sort_values("avg_rating", ascending=False).head(8).reset_index(drop=True)


def get_price_calendar(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Şehrin sezonlara göre fiyat değişimi — 'ne zaman gitsem?' sorusu için."""
    city_df = df[df["City"] == city]
    calendar = (
        city_df.groupby("Season")
        .agg(
            avg_price   =("Price",        "mean"),
            median_price=("Price",        "median"),
            avg_rating  =("Rating score", "mean"),
            count       =("Listings id",  "count"),
        )
        .reset_index()
    )
    # Sezon sırasını düzelt
    season_order = ["Early Spring", "Early Summer", "Early Autumn", "Early Winter"]
    calendar["season_order"] = calendar["Season"].map(
        {s: i for i, s in enumerate(season_order)}
    )
    return calendar.sort_values("season_order").reset_index(drop=True)


def build_social_proof(
    df: pd.DataFrame,
    city: str,
    season: str,
    nights: int,
) -> dict:
    """
    'Bu tarihlerde bu şehre gidenler...' tarzı sosyal kanıt mesajları.
    Arayüzde kullanıcıya güven ve bağlam verir.
    """
    subset = df[(df["City"] == city) & (df["Season"] == season)]
    if subset.empty:
        return {}

    total = len(subset)
    entire_home_pct = (subset["Property type"] == "Entire home").mean()
    superhost_pct   = subset["is_superhost"].mean()
    avg_price       = subset["Price"].mean()
    median_price    = subset["Price"].median()
    top_nbh         = (
        subset.groupby("Neighbourhood")["Listings id"]
        .count().idxmax()
    )

    return {
        "total_listings":    total,
        "entire_home_pct":   round(entire_home_pct * 100),
        "superhost_pct":     round(superhost_pct * 100),
        "avg_price":         round(avg_price),
        "median_price":      round(median_price),
        "total_budget":      round(median_price * nights),
        "top_neighbourhood": top_nbh,
        "season_desc":       SEASON_DESCRIPTIONS.get(season, ""),
    }
