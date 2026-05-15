"""
Kullanıcı Profili & Öğrenme Sistemi
=====================================
Üç kaynaktan profil oluşturur:
  1. İlk giriş anketi (hızlı tercih belirleme)
  2. Beğen/beğenme geçmişi (👍👎)
  3. Geçmiş aramalar (fiyat, konum, özellik örüntüleri)

Profil JSON olarak kaydedilir → oturumlar arası hafıza
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np

PROFILE_PATH = "data/user_profile.json"

# Profil boyutları — her biri 0.0 ile 1.0 arasında
# 0 = hiç önemli değil, 1 = çok önemli
DEFAULT_PROFILE = {
    # Anket'ten gelen tercihler
    "budget_sensitivity":    0.5,   # Fiyata ne kadar duyarlı?
    "comfort_priority":      0.5,   # Konfor mu (entire home) yoksa ekonomi mi?
    "location_priority":     0.5,   # Konum puanı ne kadar önemli?
    "cleanliness_priority":  0.5,   # Temizlik ne kadar önemli?
    "superhost_preference":  0.0,   # Superhost tercih ediyor mu?

    # Öğrenilen tercihler (beğeni/arama geçmişinden)
    "avg_liked_price":       None,  # Beğenilen ilanların ortalama fiyatı
    "avg_liked_beds":        None,  # Beğenilen ilanların ortalama yatak sayısı
    "preferred_property_type": None,  # En çok beğenilen ilan türü
    "preferred_city":        None,  # En çok beğenilen şehir

    # Geçmiş aramalar
    "search_history": [],   # Son 10 arama parametresi
    "liked_ids":      [],   # 👍 verilen ilan ID'leri
    "disliked_ids":   [],   # 👎 verilen ilan ID'leri

    # Meta
    "total_interactions": 0,
    "onboarding_done":    False,

    # Yeni onboarding alanları
    "companion":          "solo",
    "interests":          [],
}


class UserProfile:
    """Kullanıcı profilini yükler, günceller, kaydeder."""

    def __init__(self, path: str = PROFILE_PATH):
        self.path = path
        self.data = self._load()

    # ── Yükleme / Kaydetme ────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Yeni anahtarları varsayılanlarla tamamla
            profile = DEFAULT_PROFILE.copy()
            profile.update(saved)
            return profile
        return DEFAULT_PROFILE.copy()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def reset(self):
        self.data = DEFAULT_PROFILE.copy()
        self.save()

    # ── Anket ─────────────────────────────────────────────────────────────

    def apply_onboarding(self, answers: dict):
        """
        Anket cevaplarını profil ağırlıklarına dönüştür.
        """
        style = answers.get("travel_style", "comfort")
        if style == "budget":
            self.data["budget_sensitivity"]  = 0.9
            self.data["comfort_priority"]    = 0.2
        elif style == "comfort":
            self.data["budget_sensitivity"]  = 0.5
            self.data["comfort_priority"]    = 0.7
        elif style == "luxury":
            self.data["budget_sensitivity"]  = 0.1
            self.data["comfort_priority"]    = 1.0

        self.data["companion"] = answers.get("companion", "solo")
        self.data["interests"] = answers.get("interests", [])

        priorities = answers.get("priorities", [])
        if "location"    in priorities: self.data["location_priority"]    = 0.9
        if "cleanliness" in priorities: self.data["cleanliness_priority"] = 0.9
        if "superhost"   in priorities: self.data["superhost_preference"] = 1.0
        if "communication" in priorities: self.data["communication_priority"] = 0.9 # Dynamic priority if we add it

        self.data["onboarding_done"] = True
        self.save()

    # ── Beğeni / Beğenmeme ────────────────────────────────────────────────

    def like(self, listing: pd.Series):
        """👍 — İlanı beğen, profilden öğren."""
        lid = int(listing["Listings id"])
        if lid in self.data["liked_ids"]:
            return  # Zaten beğenilmiş

        self.data["liked_ids"].append(lid)
        self.data["total_interactions"] += 1

        # Fiyat ortalamasını güncelle (hareketli ortalama)
        n = len(self.data["liked_ids"])
        prev_avg_price = self.data["avg_liked_price"] or listing["Price"]
        self.data["avg_liked_price"] = (
            (prev_avg_price * (n - 1) + listing["Price"]) / n
        )

        prev_avg_beds = self.data["avg_liked_beds"] or listing["Beds number"]
        self.data["avg_liked_beds"] = (
            (prev_avg_beds * (n - 1) + listing["Beds number"]) / n
        )

        # Bütçe hassasiyetini fiyata göre ayarla
        # Düşük fiyatlı ilanlar beğenilince budget_sensitivity artar
        if listing["Price"] < 100:
            self.data["budget_sensitivity"] = min(
                1.0, self.data["budget_sensitivity"] + 0.05
            )
        elif listing["Price"] > 300:
            self.data["budget_sensitivity"] = max(
                0.0, self.data["budget_sensitivity"] - 0.05
            )

        # Konum önceliğini güncelle
        if listing["Location score"] >= 4.8:
            self.data["location_priority"] = min(
                1.0, self.data["location_priority"] + 0.03
            )

        self.save()

    def dislike(self, listing: pd.Series):
        """👎 — İlanı beğenme, tersini öğren."""
        lid = int(listing["Listings id"])
        if lid not in self.data["disliked_ids"]:
            self.data["disliked_ids"].append(lid)
            self.data["total_interactions"] += 1

        # Pahalıysa ve beğenilmediyse bütçe hassasiyeti artar
        if listing["Price"] > 200:
            self.data["budget_sensitivity"] = min(
                1.0, self.data["budget_sensitivity"] + 0.04
            )
        self.save()

    # ── Arama Geçmişi ────────────────────────────────────────────────────

    def record_search(self, params: dict):
        """Her aramayı kaydet, son 10'u tut."""
        self.data["search_history"].append(params)
        self.data["search_history"] = self.data["search_history"][-10:]
        self.save()

    # ── Profil Özeti ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """UI'da gösterilecek profil özeti."""
        liked_count = len(self.data["liked_ids"])
        style = (
            "Bütçe gezgini 💸" if self.data["budget_sensitivity"] > 0.7
            else "Konfor odaklı 🛋️" if self.data["comfort_priority"] > 0.7
            else "Denge arayan ⚖️"
        )
        return {
            "style":       style,
            "liked":       liked_count,
            "searches":    len(self.data["search_history"]),
            "interactions": self.data["total_interactions"],
            "avg_price":   self.data["avg_liked_price"],
            "avg_beds":    self.data["avg_liked_beds"],
        }

    # ── Profil → Öneri Ağırlıkları ───────────────────────────────────────

    def to_score_weights(self) -> dict:
        """
        Profili kalite skoru ağırlıklarına dönüştür.
        Kullanıcının önem verdiği boyutlar daha fazla etki eder.
        """
        loc  = self.data["location_priority"]
        cln  = self.data["cleanliness_priority"]
        comm = self.data.get("communication_priority", 0.5)
        val  = 1.0 - self.data["budget_sensitivity"] * 0.5  # bütçe odaklıysa value önemli
        base = 0.25  # rating her zaman önemli

        # Normalize et → toplam 1.0 olsun
        raw = {
            "Rating score":          base,
            "Cleanliness score":     cln * 0.25,
            "Location score":        loc * 0.25,
            "Value for money score": val * 0.20,
            "Accuracy score":        0.10,
            "Checkin score":         0.05,
            "Communication score":   comm * 0.05,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}
