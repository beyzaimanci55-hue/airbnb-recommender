"""
Airbnb Kişiselleştirilmiş Öneri Sistemi
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from src.recommender  import load_data, HybridRecommender, UserPreferences
from src.user_profile import UserProfile
from src.date_insights import (
    date_to_season, get_destination_insights,
    get_neighbourhood_insights, get_price_calendar,
    build_social_proof, SEASON_DESCRIPTIONS,
)

# ── Sayfa Ayarları ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tatil Öneri Sistemi",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.card {
    background:white; border-radius:16px; padding:20px;
    box-shadow:0 2px 12px rgba(0,0,0,.08); margin-bottom:16px;
    border-left:5px solid #FF5A5F;
}
.card-neutral { border-left-color:#ccc; opacity:.7; }
.badge {
    display:inline-block; background:#FF5A5F; color:white;
    border-radius:20px; padding:3px 12px; font-size:.8em; font-weight:700;
}
.tag {
    display:inline-block; background:#f5f5f5; border-radius:8px;
    padding:2px 9px; font-size:.78em; margin:2px; color:#444;
}
.proof-box {
    background:#fff8f0; border-radius:12px; padding:16px;
    border:1px solid #ffe0c0; margin-bottom:16px;
}
.profile-box {
    background:#f0f7ff; border-radius:12px; padding:14px;
    border:1px solid #c0d8ff; margin-bottom:10px; font-size:.9em;
}
</style>
""", unsafe_allow_html=True)


# ── Veri & Model ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Veri yükleniyor…")
def get_data():
    path = os.path.join(os.path.dirname(__file__), "data", "airbnb_sample.csv")
    return load_data(path, sample_size=None)  # zaten küçük, hepsini kullan

df          = get_data()
recommender = HybridRecommender(df)

# ── Session state ──────────────────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = UserProfile()
profile = st.session_state.profile

for key, val in [
    ("recs", None), ("stats", {}), ("city", None),
    ("season", None), ("nights", 3),
    ("show_onboarding", not profile.data["onboarding_done"]),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# 1. ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.show_onboarding:
    st.title("👋 Merhaba! Seni tanıyalım")
    st.markdown("Sadece birkaç soru — daha iyi öneriler için profilini oluşturalım.")
    st.markdown("---")

    with st.form("onboarding"):
        travel_style = st.radio(
            "✈️ Seyahat tarzın nedir?",
            ["budget", "comfort", "luxury"],
            format_func=lambda x: {
                "budget":  "💸 Bütçe odaklı — ucuz ama işlevsel",
                "comfort": "🛋️  Konfor odaklı — kaliteli, makul fiyat",
                "luxury":  "✨ Lüks — fiyat ikinci planda",
            }[x],
            horizontal=True,
        )
        priorities = st.multiselect(
            "⭐ En çok önem verdiğin şeyler:",
            ["location", "cleanliness", "superhost", "value"],
            default=["location", "cleanliness"],
            format_func=lambda x: {
                "location":    "📍 Merkezi konum",
                "cleanliness": "🧹 Temizlik",
                "superhost":   "⭐ Superhost güvencesi",
                "value":       "💰 Para'nın karşılığı",
            }[x],
        )
        submitted = st.form_submit_button("🚀 Başla", use_container_width=True, type="primary")

    if submitted:
        profile.apply_onboarding({"travel_style": travel_style, "priorities": priorities})
        st.session_state.show_onboarding = False
        st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Tatil Öneri Sistemi")
    st.markdown("---")

    summary = profile.summary()
    st.markdown(f"""
<div class="profile-box">
<b>Profilin:</b> {summary['style']}<br>
👍 {summary['liked']} beğeni &nbsp;|&nbsp; 🔍 {summary['searches']} arama
{f"<br>💶 Ort. bütçen: €{summary['avg_price']:.0f}/gece" if summary['avg_price'] else ""}
</div>
""", unsafe_allow_html=True)

    if st.button("🔄 Profili Sıfırla", use_container_width=True):
        profile.reset()
        st.session_state.show_onboarding = True
        st.rerun()

    st.markdown("---")
    st.markdown("### 📅 Ne Zaman Gidiyorsun?")
    travel_date = st.date_input(
        "Gidiş Tarihi",
        value=date.today() + timedelta(days=30),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=365),
    )
    nights = st.slider("🌙 Kaç Gece?", 1, 14, 3)
    st.session_state.nights = nights

    season = date_to_season(travel_date)
    st.info(f"{SEASON_DESCRIPTIONS[season]}")
    st.session_state.season = season

    st.markdown("---")
    st.markdown("### 🔍 Tercihler")
    city = st.selectbox("📍 Şehir", sorted(df["City"].unique()))
    st.session_state.city = city

    prop_type = st.selectbox("🏠 İlan Türü", ["Herhangi"] + sorted(df["Property type"].unique()))
    max_price = st.slider("💶 Maks. Fiyat (€/gece)", 20, 900, 300, step=10)
    min_beds  = st.slider("🛏 Min. Yatak", 1, 8, 1)
    superhost = st.checkbox("⭐ Sadece Superhost")

    st.markdown("---")
    st.markdown("### ⚖️ Öneri Ağırlığı")
    content_w = st.slider(
        "Benzerlik ←→ Kalite", 0.0, 1.0, 0.5, 0.05,
        help="Sol: sana benzer ilanlar | Sağ: en kaliteli ilanlar",
    )
    quality_w = 1.0 - content_w

    run_btn = st.button("🚀 Öneri Getir", use_container_width=True, type="primary")


# ── Ana İçerik ─────────────────────────────────────────────────────────────
st.title("🏡 Sana Özel Tatil Önerileri")

# ── A: Tarih bazlı destinasyon analizi ────────────────────────────────────
st.markdown(f"## 📅 {travel_date.strftime('%d %B %Y')} — {season}")

dest_df = get_destination_insights(df, season, nights)
col1, col2 = st.columns([1.4, 1])

with col1:
    fig = px.bar(
        dest_df, x="City", y="popularity_score",
        color="avg_price", color_continuous_scale="RdYlGn_r",
        hover_data={"avg_rating": True, "avg_price": True, "total_budget": True},
        labels={"popularity_score": "Popülerlik", "avg_price": "Ort. Fiyat (€)"},
        title="Bu Tarihlerde Popüler Destinasyonlar",
    )
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**📊 Şehir Karşılaştırması**")
    disp = dest_df[["City","avg_price","total_budget","avg_rating","superhost_ratio"]].copy()
    disp.columns = ["Şehir", "Ort. Fiyat", f"~{nights}g Bütçe", "Puan", "Superhost%"]
    disp["Ort. Fiyat"]       = disp["Ort. Fiyat"].apply(lambda x: f"€{x:.0f}")
    disp[f"~{nights}g Bütçe"] = disp[f"~{nights}g Bütçe"].apply(lambda x: f"€{x:.0f}")
    disp["Puan"]             = disp["Puan"].apply(lambda x: f"⭐{x:.2f}")
    disp["Superhost%"]       = disp["Superhost%"].apply(lambda x: f"%{x*100:.0f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)


# ── B: Öneri çalıştır ─────────────────────────────────────────────────────
if run_btn:
    profile.record_search({
        "city": city, "season": season,
        "max_price": max_price, "min_beds": min_beds,
        "date": str(travel_date),
    })
    prefs = UserPreferences(
        city=city, season=season,
        property_type=None if prop_type == "Herhangi" else prop_type,
        min_beds=min_beds, max_price=max_price,
        superhost_only=superhost,
        content_weight=content_w, quality_weight=quality_w,
        score_weights=profile.to_score_weights(),
        exclude_ids=profile.data["disliked_ids"],
    )
    with st.spinner("🧠 Senin için en iyiler seçiliyor…"):
        recs, stats = recommender.recommend(prefs, n_results=8)
    st.session_state.recs  = recs
    st.session_state.stats = stats

# ── C: Sosyal kanıt ───────────────────────────────────────────────────────
if st.session_state.city and st.session_state.season:
    proof = build_social_proof(
        df, st.session_state.city,
        st.session_state.season,
        st.session_state.nights,
    )
    if proof:
        st.markdown("---")
        st.markdown(f"### 📍 {st.session_state.city} — Bu Dönemde Ne Oluyor?")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Toplam İlan",        f"{proof['total_listings']:,}")
        p2.metric("Medyan Gecelik",     f"€{proof['median_price']}")
        p3.metric(f"~{st.session_state.nights}g Bütçe", f"€{proof['total_budget']}")
        p4.metric("Superhost Oranı",    f"%{proof['superhost_pct']}")
        st.markdown(f"""
<div class="proof-box">
🏘️ Bu sezonda <b>{st.session_state.city}</b>'de rezervasyonların
<b>%{proof['entire_home_pct']}'i</b> tüm ev kiralıyor.
En popüler mahalle: <b>{proof['top_neighbourhood']}</b>.<br>
{proof['season_desc']}
</div>
""", unsafe_allow_html=True)

# ── D: Sonuçlar ───────────────────────────────────────────────────────────
recs = st.session_state.recs
if recs is not None and not recs.empty:
    st.markdown("---")
    st.subheader(f"🏠 Sana Özel {len(recs)} Öneri — {st.session_state.city}")

    tab1, tab2, tab3 = st.tabs(["📋 İlanlar", "🗺 Harita", "📈 Analiz"])

    with tab1:
        for idx, row in recs.iterrows():
            hybrid  = row.get("hybrid_score", 0)
            quality = row.get("quality_score", 0)
            content = row.get("content_score", 0)
            lid     = int(row["Listings id"])

            already_liked    = lid in profile.data["liked_ids"]
            already_disliked = lid in profile.data["disliked_ids"]
            card_class = "card card-neutral" if already_disliked else "card"
            superhost_tag = "⭐ Superhost" if row.get("is_superhost") == 1 else ""

            st.markdown(f"""
<div class="{card_class}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h4 style="margin:0">🏠 {row['Property type']} · {row['Neighbourhood']}</h4>
    <span class="badge">🏆 {hybrid:.2f}</span>
  </div>
  <div style="margin:8px 0">
    <span class="tag">💶 €{row['Price']:.0f}/gece</span>
    <span class="tag">🛏 {int(row['Beds number'])} yatak</span>
    <span class="tag">🚪 {int(row['Bedrooms number'])} oda</span>
    <span class="tag">👥 max {int(row['Maximum allowed guests'])} kişi</span>
    {f'<span class="tag">{superhost_tag}</span>' if superhost_tag else ""}
  </div>
  <div style="display:flex;gap:16px;font-size:.85em;color:#555">
    <span>⭐ {row['Rating score']:.1f}</span>
    <span>🧹 {row['Cleanliness score']:.1f}</span>
    <span>📍 {row['Location score']:.1f}</span>
    <span>💰 {row['Value for money score']:.1f}</span>
  </div>
  <div style="margin-top:6px;font-size:.75em;color:#aaa">
    Benzerlik: {content:.2f} | Kalite: {quality:.2f}
  </div>
</div>
""", unsafe_allow_html=True)

            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button(
                "✅ Beğenildi" if already_liked else "👍 Beğen",
                key=f"like_{lid}_{idx}", disabled=already_liked,
            ):
                profile.like(row)
                st.rerun()
            if c2.button(
                "❌ Çıkarıldı" if already_disliked else "👎 İstemiyorum",
                key=f"dis_{lid}_{idx}", disabled=already_disliked,
            ):
                profile.dislike(row)
                st.rerun()

    with tab2:
        if "lat" in recs.columns and "lon" in recs.columns:
            map_df = recs.dropna(subset=["lat","lon"]).copy()
            map_df["size"]  = (map_df.get("hybrid_score", 0.5) * 20 + 5).clip(5, 25)
            map_df["label"] = (
                map_df["Property type"] + " | €"
                + map_df["Price"].astype(int).astype(str)
                + " | ⭐" + map_df["Rating score"].round(1).astype(str)
            )
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon",
                color="hybrid_score", size="size",
                hover_name="label",
                hover_data={"Price": True, "Rating score": True,
                            "Neighbourhood": True, "lat": False, "lon": False, "size": False},
                color_continuous_scale="RdYlGn",
                zoom=11, height=500, mapbox_style="open-street-map",
            )
            fig_map.update_layout(margin=dict(t=10,b=0,l=0,r=0))
            st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            city_pool  = df[df["City"] == st.session_state.city]
            score_cols = list(profile.to_score_weights().keys())
            rec_avg    = recs[score_cols].mean()
            city_avg   = city_pool[score_cols].mean()
            labels     = [c.replace(" score","") for c in score_cols]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=rec_avg.tolist()+[rec_avg.iloc[0]],
                theta=labels+[labels[0]], fill="toself",
                name="Öneriler", line_color="#FF5A5F"))
            fig_r.add_trace(go.Scatterpolar(
                r=city_avg.tolist()+[city_avg.iloc[0]],
                theta=labels+[labels[0]], fill="toself",
                name="Şehir Ort.", line_color="#aaa", opacity=0.5))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(range=[4,5])),
                title="Öneriler vs Şehir Ortalaması",
                height=380, legend=dict(orientation="h"))
            st.plotly_chart(fig_r, use_container_width=True)

        with c2:
            cal = get_price_calendar(df, st.session_state.city)
            fig_cal = px.bar(
                cal, x="Season", y="median_price",
                color="avg_rating", color_continuous_scale="RdYlGn",
                labels={"median_price":"Medyan Fiyat (€)","avg_rating":"Ort. Puan"},
                title=f"{st.session_state.city} — Sezona Göre Fiyat",
            )
            fig_cal.update_layout(height=380, margin=dict(t=40,b=10))
            st.plotly_chart(fig_cal, use_container_width=True)

        st.markdown("#### 🧠 Profilinden Öğrenilen Ağırlıklar")
        w   = profile.to_score_weights()
        wdf = pd.DataFrame({"Boyut": list(w.keys()), "Ağırlık": list(w.values())})
        wdf = wdf.sort_values("Ağırlık", ascending=True)
        fig_w = px.bar(
            wdf, x="Ağırlık", y="Boyut", orientation="h",
            color="Ağırlık", color_continuous_scale="Reds",
            title="Profilinden Öğrenilen Tercihler",
        )
        fig_w.update_layout(height=280, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_w, use_container_width=True)

elif recs is not None and recs.empty:
    st.warning("Bu kriterlere uygun ilan bulunamadı. Filtreleri gevşetmeyi dene.")
else:
    st.info("👈 Soldaki panelden tarih ve tercihlerini seç, **Öneri Getir**'e tıkla.")

st.markdown("---")
st.caption("Kişiselleştirilmiş Hibrit Öneri Sistemi · CENG468 · Beyza İmancı")
