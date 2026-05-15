"""
StayFinder — Kişiselleştirilmiş Tatil Öneri Sistemi
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
from src.recommender   import load_data, HybridRecommender, UserPreferences
from src.user_profile  import UserProfile
from src.date_insights import (
    date_to_season, get_destination_insights,
    get_price_calendar, build_social_proof,
    SEASON_DESCRIPTIONS,
)

# ── Sayfa config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StayFinder · Tatil Önerileri",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fraunces:ital,wght@0,700;1,400&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section:first-child {
    background: #0A0A0F !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F0F18 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] section { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #D0CCE0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

/* ── All text visible in dark mode ── */
p, span, div, label, li { color: #D0CCE0; font-family: 'Plus Jakarta Sans', sans-serif; }
h1, h2, h3, h4 { color: #FFFFFF !important; font-family: 'Fraunces', Georgia, serif !important; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #13103A 0%, #0A0A0F 50%, #0D1A2E 100%);
    border-radius: 24px;
    border: 1px solid rgba(120, 100, 255, 0.2);
    padding: 52px 48px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(245,166,35,0.15) 0%, rgba(120,100,255,0.1) 40%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(80,200,180,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72em;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #F5A623 !important;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block; width: 28px; height: 2px;
    background: #F5A623; border-radius: 2px;
}
.hero-title {
    font-family: 'Fraunces', serif !important;
    font-size: 3em !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    line-height: 1.1 !important;
    margin: 0 0 6px !important;
}
.hero-title em {
    font-style: italic;
    background: linear-gradient(90deg, #F5A623, #FF6B9D, #7B64FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-meta {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.88em;
    color: #6B6880 !important;
    margin-top: 12px;
}

/* ── Cards ── */
.lcard {
    background: #13131C;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px 24px 18px;
    margin-bottom: 16px;
    position: relative;
    transition: all 0.2s ease;
}
.lcard:hover {
    border-color: rgba(245,166,35,0.4);
    background: #16161F;
    transform: translateY(-1px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.lcard-liked  { border-color: rgba(46,213,115,0.5) !important; background: #0D1A12 !important; }
.lcard-disliked { opacity: 0.35 !important; }

.lcard-title {
    font-family: 'Fraunces', serif;
    font-size: 1.1em;
    font-weight: 700;
    color: #FFFFFF !important;
    margin: 0 0 12px;
    padding-right: 80px;
}
.score-pill {
    position: absolute; top: 20px; right: 22px;
    background: linear-gradient(135deg, #F5A623 0%, #FF8C42 100%);
    color: #0A0A0F;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.78em; font-weight: 800;
    padding: 5px 14px; border-radius: 50px;
    letter-spacing: 0.02em;
}
.tag {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    color: #B0ACBF !important;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 4px 11px;
    font-size: 0.78em;
    margin: 2px 3px 2px 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.score-row {
    display: flex; flex-wrap: wrap; gap: 18px;
    margin-top: 14px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.82em;
    color: #6B6880 !important;
}
.score-row b { color: #E0DCF0 !important; }
.meta-row {
    font-size: 0.7em; color: #3A3850 !important;
    margin-top: 8px; font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Stat cards ── */
.stat-card {
    background: #13131C;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
}
.stat-val {
    font-family: 'Fraunces', serif;
    font-size: 2em; font-weight: 700;
    color: #F5A623 !important;
    display: block; line-height: 1;
}
.stat-lbl {
    font-size: 0.72em; color: #5A5870 !important;
    text-transform: uppercase; letter-spacing: 0.08em;
    font-family: 'Plus Jakarta Sans', sans-serif;
    margin-top: 6px; display: block;
}

/* ── Section header ── */
.sec-head {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 20px;
}
.sec-head-title {
    font-family: 'Fraunces', serif;
    font-size: 1.25em; font-weight: 700;
    color: #FFFFFF !important;
    white-space: nowrap;
}
.sec-head-line {
    flex: 1; height: 1px;
    background: linear-gradient(to right, rgba(255,255,255,0.08), transparent);
}

/* ── Proof box ── */
.proof-box {
    background: linear-gradient(135deg, #16100A, #0A0A0F);
    border: 1px solid rgba(245,166,35,0.2);
    border-left: 3px solid #F5A623;
    border-radius: 14px;
    padding: 16px 20px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.9em;
    color: #A09CB0 !important;
    line-height: 1.7;
    margin: 12px 0 24px;
}
.proof-box b { color: #F5A623 !important; }

/* ── Profile card ── */
.profile-card {
    background: linear-gradient(135deg, #13103A, #0F0F18);
    border: 1px solid rgba(120,100,255,0.2);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.profile-style {
    font-family: 'Fraunces', serif;
    font-size: 1.05em; font-weight: 700;
    color: #F5A623 !important;
    margin-bottom: 6px;
}
.profile-meta { font-size: 0.8em; color: #6B6880 !important; line-height: 1.6; }

/* ── Sidebar branding ── */
.brand-logo {
    font-family: 'Fraunces', serif;
    font-size: 1.6em; font-weight: 700;
    color: #FFFFFF !important;
}
.brand-sub {
    font-size: 0.7em; color: #4A4860 !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.sidebar-section-label {
    font-size: 0.68em !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #4A4860 !important;
    margin: 20px 0 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Season badge ── */
.season-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.25);
    color: #F5A623 !important;
    border-radius: 50px;
    padding: 5px 14px;
    font-size: 0.78em; font-weight: 600;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Streamlit overrides ── */
[data-testid="stMetric"] {
    background: #13131C !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    color: #F5A623 !important;
    font-size: 1.8em !important;
}
[data-testid="stMetricLabel"] {
    color: #5A5870 !important;
    font-size: 0.75em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Primary button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #F5A623 0%, #FF8C42 100%) !important;
    color: #0A0A0F !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(245,166,35,0.3) !important;
}

/* Secondary buttons */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    color: #8080A0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82em !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
    color: #5A5870 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #F5A623 !important;
    border-bottom: 2px solid #F5A623 !important;
    background: rgba(245,166,35,0.05) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* Alert */
[data-testid="stAlert"] {
    background: #13131C !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #A09CB0 !important;
}

/* Selectbox, slider labels */
[data-testid="stSelectbox"] > label,
[data-testid="stSlider"] > label,
[data-testid="stDateInput"] > label,
[data-testid="stCheckbox"] > label {
    color: #8080A0 !important;
    font-size: 0.82em !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Form */
[data-testid="stForm"] {
    background: #13131C !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
    padding: 24px !important;
}

/* Radio */
[data-testid="stRadio"] label { color: #D0CCE0 !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: #F5A623 !important; }
</style>
""", unsafe_allow_html=True)


# ── Plotly dark layout template ───────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#A09CB0", family="Plus Jakarta Sans"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
)
GOLD_SCALE = [[0,"#1a1530"],[0.5,"#F5A623"],[1,"#FF6B35"]]


# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Veri yükleniyor…")
def get_data():
    path = os.path.join(os.path.dirname(__file__), "data", "airbnb_sample.csv")
    return load_data(path, sample_size=None)

df          = get_data()
recommender = HybridRecommender(df)


# ── Session State ─────────────────────────────────────────────────────────────
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
# ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.show_onboarding:
    st.markdown("""
    <div class="hero-wrap" style="text-align:center; padding: 72px 48px;">
        <div class="hero-eyebrow" style="justify-content:center">Hoş Geldin</div>
        <h1 class="hero-title">Seyahat tarzını <em>anlayalım</em></h1>
        <p style="color:#5A5870; font-family:'Plus Jakarta Sans',sans-serif; margin-top:14px;">
            3 kısa soru · Tamamen sana özel öneriler için
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("onboarding"):
        st.markdown("#### ✈️ Seyahat tarzın nedir?")
        travel_style = st.radio("", ["budget", "comfort", "luxury"],
            format_func=lambda x: {
                "budget":  "💸  Bütçe odaklı — ucuz ama işlevsel",
                "comfort": "🛋️   Konfor odaklı — kaliteli, makul fiyat",
                "luxury":  "✨  Lüks — fiyat ikinci planda",
            }[x], horizontal=True, label_visibility="collapsed")

        st.markdown("#### ⭐ En çok önem verdiğin şeyler")
        priorities = st.multiselect("",
            ["location","cleanliness","superhost","value"],
            default=["location","cleanliness"],
            format_func=lambda x: {
                "location":    "📍 Merkezi konum",
                "cleanliness": "🧹 Temizlik",
                "superhost":   "⭐ Superhost güvencesi",
                "value":       "💰 Para'nın karşılığı",
            }[x], label_visibility="collapsed")

        if st.form_submit_button("🚀  Profilimi Oluştur", use_container_width=True, type="primary"):
            profile.apply_onboarding({"travel_style": travel_style, "priorities": priorities})
            st.session_state.show_onboarding = False
            st.rerun()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 20px">
        <div class="brand-logo">🏡 StayFinder</div>
        <div class="brand-sub">Kişisel Tatil Önerileri</div>
    </div>
    """, unsafe_allow_html=True)

    summary = profile.summary()
    st.markdown(f"""
<div class="profile-card">
    <div class="profile-style">{summary['style']}</div>
    <div class="profile-meta">
        👍 {summary['liked']} beğeni &nbsp;·&nbsp; 🔍 {summary['searches']} arama
        {f"<br>💶 Ort. bütçen: €{summary['avg_price']:.0f}/gece" if summary['avg_price'] else ""}
    </div>
</div>
""", unsafe_allow_html=True)

    if st.button("↺ Profili Sıfırla"):
        profile.reset(); st.session_state.show_onboarding = True; st.rerun()

    st.markdown('<p class="sidebar-section-label">📅 Ne Zaman?</p>', unsafe_allow_html=True)
    travel_date = st.date_input("Gidiş Tarihi",
        value=date.today()+timedelta(days=30),
        min_value=date.today(), max_value=date.today()+timedelta(days=365))
    nights = st.slider("🌙 Kaç Gece?", 1, 14, 3)
    st.session_state.nights = nights

    season = date_to_season(travel_date)
    st.session_state.season = season
    desc_parts = SEASON_DESCRIPTIONS[season].split("—")
    st.markdown(f'<span class="season-badge">{desc_parts[0].strip()}</span>', unsafe_allow_html=True)
    if len(desc_parts) > 1:
        st.markdown(f'<p style="color:#4A4860;font-size:0.78em;margin-top:6px;">{desc_parts[1].strip()}</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">🔍 Tercihler</p>', unsafe_allow_html=True)
    city = st.selectbox("📍 Şehir", sorted(df["City"].unique()))
    st.session_state.city = city
    prop_type = st.selectbox("🏠 İlan Türü", ["Herhangi"]+sorted(df["Property type"].unique()))
    max_price = st.slider("💶 Maks. Fiyat (€/gece)", 20, 900, 300, step=10)
    min_beds  = st.slider("🛏 Min. Yatak", 1, 8, 1)
    superhost = st.checkbox("⭐ Sadece Superhost")

    st.markdown('<p class="sidebar-section-label">⚖️ Öneri Ağırlığı</p>', unsafe_allow_html=True)
    content_w = st.slider("Benzerlik ↔ Kalite", 0.0, 1.0, 0.5, 0.05)
    quality_w = 1.0 - content_w

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✦ Öneri Getir", use_container_width=True, type="primary")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">StayFinder · Öneri Motoru</div>
    <h1 class="hero-title"><em>{city}</em> için<br>sana özel seçimler</h1>
    <p class="hero-meta">
        📅 {travel_date.strftime('%d %B %Y')} &nbsp;·&nbsp;
        🌙 {nights} gece &nbsp;·&nbsp;
        🗓 {season}
    </p>
</div>
""", unsafe_allow_html=True)


# ── A: Destinasyon Trendi ─────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <span class="sec-head-title">📅 Bu Dönemde Nereye Gidiliyor?</span>
    <div class="sec-head-line"></div>
</div>
""", unsafe_allow_html=True)

dest_df = get_destination_insights(df, season, nights)
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    fig = px.bar(dest_df, x="City", y="popularity_score",
        color="avg_price", color_continuous_scale=GOLD_SCALE,
        hover_data={"avg_rating":":.2f","avg_price":":.0f","total_budget":":.0f"},
        labels={"popularity_score":"Popülerlik","avg_price":"Ort. Fiyat (€)"})
    fig.update_layout(**PLOTLY_DARK, height=270, margin=dict(t=10,b=10,l=0,r=0),
        coloraxis_colorbar=dict(tickfont=dict(color="#A09CB0")))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    disp = dest_df[["City","avg_price","total_budget","avg_rating","superhost_ratio"]].copy()
    disp.columns = ["Şehir","Fiyat",f"{nights}g Bütçe","Puan","S.Host%"]
    disp["Fiyat"]         = disp["Fiyat"].apply(lambda x: f"€{x:.0f}")
    disp[f"{nights}g Bütçe"] = disp[f"{nights}g Bütçe"].apply(lambda x: f"€{x:.0f}")
    disp["Puan"]          = disp["Puan"].apply(lambda x: f"⭐{x:.2f}")
    disp["S.Host%"]       = disp["S.Host%"].apply(lambda x: f"%{x*100:.0f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)


# ── Öneri çalıştır ─────────────────────────────────────────────────────────────
if run_btn:
    profile.record_search({"city":city,"season":season,
        "max_price":max_price,"min_beds":min_beds,"date":str(travel_date)})
    prefs = UserPreferences(city=city, season=season,
        property_type=None if prop_type=="Herhangi" else prop_type,
        min_beds=min_beds, max_price=max_price, superhost_only=superhost,
        content_weight=content_w, quality_weight=quality_w,
        score_weights=profile.to_score_weights(),
        exclude_ids=profile.data["disliked_ids"])
    with st.spinner("En iyi seçenekler hazırlanıyor…"):
        recs, stats = recommender.recommend(prefs, n_results=8)
    st.session_state.recs  = recs
    st.session_state.stats = stats


# ── B: Sosyal Kanıt ───────────────────────────────────────────────────────────
if st.session_state.city and st.session_state.season:
    proof = build_social_proof(df, st.session_state.city,
                               st.session_state.season, st.session_state.nights)
    if proof:
        st.markdown("""
        <div class="sec-head">
            <span class="sec-head-title">📍 Bu Şehirde Ne Oluyor?</span>
            <div class="sec-head-line"></div>
        </div>""", unsafe_allow_html=True)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Toplam İlan",      f"{proof['total_listings']:,}")
        m2.metric("Medyan Gecelik",   f"€{proof['median_price']}")
        m3.metric(f"~{st.session_state.nights}g Bütçe", f"€{proof['total_budget']}")
        m4.metric("Superhost Oranı",  f"%{proof['superhost_pct']}")

        st.markdown(f"""
<div class="proof-box">
🏘️ <b>{st.session_state.city}</b>'de bu sezonda rezervasyonların <b>%{proof['entire_home_pct']}'i</b> tüm ev kiralıyor.
En popüler mahalle: <b>{proof['top_neighbourhood']}</b>. &nbsp;·&nbsp; {proof['season_desc']}
</div>""", unsafe_allow_html=True)


# ── C: Sonuçlar ───────────────────────────────────────────────────────────────
recs = st.session_state.recs

if recs is not None and not recs.empty:
    st.markdown(f"""
    <div class="sec-head">
        <span class="sec-head-title">✦ Sana Özel {len(recs)} Öneri</span>
        <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  📋  İlanlar  ","  🗺  Harita  ","  📈  Analiz  "])

    # ── Tab 1 ────────────────────────────────────────────────────────────────
    with tab1:
        for idx, row in recs.iterrows():
            hybrid  = row.get("hybrid_score", 0)
            quality = row.get("quality_score", 0)
            content = row.get("content_score", 0)
            lid     = int(row["Listings id"])
            liked   = lid in profile.data["liked_ids"]
            disliked= lid in profile.data["disliked_ids"]
            cls     = "lcard lcard-liked" if liked else ("lcard lcard-disliked" if disliked else "lcard")
            sh      = '<span class="tag">⭐ Superhost</span>' if row.get("is_superhost")==1 else ""

            st.markdown(f"""
<div class="{cls}">
  <span class="score-pill">🏆 {hybrid:.2f}</span>
  <div class="lcard-title">🏠 {row['Property type']} &nbsp;·&nbsp; {row['Neighbourhood']}</div>
  <div>
    <span class="tag">💶 €{row['Price']:.0f}/gece</span>
    <span class="tag">🛏 {int(row['Beds number'])} yatak</span>
    <span class="tag">🚪 {int(row['Bedrooms number'])} oda</span>
    <span class="tag">👥 max {int(row['Maximum allowed guests'])}</span>
    {sh}
  </div>
  <div class="score-row">
    <span>⭐ <b>{row['Rating score']:.1f}</b></span>
    <span>🧹 <b>{row['Cleanliness score']:.1f}</b></span>
    <span>📍 <b>{row['Location score']:.1f}</b></span>
    <span>💰 <b>{row['Value for money score']:.1f}</b></span>
  </div>
  <div class="meta-row">Benzerlik: {content:.2f} &nbsp;|&nbsp; Kalite: {quality:.2f}</div>
</div>""", unsafe_allow_html=True)

            c1,c2,_ = st.columns([1,1,5])
            with c1:
                if st.button("✓ Beğendim" if not liked else "✓ Beğenildi",
                             key=f"l_{lid}_{idx}", disabled=liked):
                    profile.like(row); st.rerun()
            with c2:
                if st.button("✕ İstemiyorum" if not disliked else "✕ Çıkarıldı",
                             key=f"d_{lid}_{idx}", disabled=disliked):
                    profile.dislike(row); st.rerun()

    # ── Tab 2: Harita ─────────────────────────────────────────────────────────
    with tab2:
        if "lat" in recs.columns and "lon" in recs.columns:
            mdf = recs.dropna(subset=["lat","lon"]).copy()
            mdf["size"]  = (mdf.get("hybrid_score",0.5)*20+8).clip(8,28)
            mdf["label"] = (mdf["Property type"]+" · €"+mdf["Price"].astype(int).astype(str)
                            +" · ⭐"+mdf["Rating score"].round(1).astype(str))
            fig_m = px.scatter_mapbox(mdf, lat="lat", lon="lon",
                color="hybrid_score", size="size", hover_name="label",
                hover_data={"Price":True,"Rating score":True,"Neighbourhood":True,
                            "lat":False,"lon":False,"size":False},
                color_continuous_scale=GOLD_SCALE,
                zoom=11, height=520, mapbox_style="carto-darkmatter")
            fig_m.update_layout(margin=dict(t=0,b=0,l=0,r=0),
                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_m, use_container_width=True)

    # ── Tab 3: Analiz ─────────────────────────────────────────────────────────
    with tab3:
        c1,c2 = st.columns(2, gap="large")

        with c1:
            city_pool  = df[df["City"]==st.session_state.city]
            score_cols = list(profile.to_score_weights().keys())
            rec_avg    = recs[score_cols].mean()
            city_avg   = city_pool[score_cols].mean()
            labels     = [c.replace(" score","") for c in score_cols]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=rec_avg.tolist()+[rec_avg.iloc[0]], theta=labels+[labels[0]],
                fill="toself", name="Öneriler",
                line_color="#F5A623", fillcolor="rgba(245,166,35,0.1)"))
            fig_r.add_trace(go.Scatterpolar(
                r=city_avg.tolist()+[city_avg.iloc[0]], theta=labels+[labels[0]],
                fill="toself", name="Şehir Ort.",
                line_color="#7B64FF", fillcolor="rgba(123,100,255,0.06)"))
            fig_r.update_layout(**PLOTLY_DARK, height=380,
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[4,5],gridcolor="rgba(255,255,255,0.06)",
                        linecolor="rgba(255,255,255,0.08)",tickfont=dict(color="#4A4860")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                        linecolor="rgba(255,255,255,0.08)")),
                title=dict(text="Öneriler vs Şehir Ortalaması",
                    font=dict(color="#F0EDE8",size=13)),
                legend=dict(orientation="h",font=dict(color="#A09CB0")))
            st.plotly_chart(fig_r, use_container_width=True)

        with c2:
            cal = get_price_calendar(df, st.session_state.city)
            fig_c = px.bar(cal, x="Season", y="median_price",
                color="avg_rating", color_continuous_scale=GOLD_SCALE,
                labels={"median_price":"Medyan Fiyat (€)","avg_rating":"Ort. Puan"},
                title=f"{st.session_state.city} — Sezona Göre Fiyat")
            fig_c.update_layout(**PLOTLY_DARK, height=380,
                title=dict(font=dict(color="#F0EDE8",size=13)),
                margin=dict(t=40,b=10,l=0,r=0),
                coloraxis_colorbar=dict(tickfont=dict(color="#A09CB0")))
            fig_c.update_traces(marker_line_width=0)
            st.plotly_chart(fig_c, use_container_width=True)

        w   = profile.to_score_weights()
        wdf = pd.DataFrame({"Boyut":list(w.keys()),"Ağırlık":list(w.values())}).sort_values("Ağırlık")
        fig_w = px.bar(wdf, x="Ağırlık", y="Boyut", orientation="h",
            color="Ağırlık", color_continuous_scale=GOLD_SCALE,
            title="Profilinden Öğrenilen Ağırlıklar")
        fig_w.update_layout(**PLOTLY_DARK, height=270,
            title=dict(font=dict(color="#F0EDE8",size=13)),
            margin=dict(t=40,b=0,l=0,r=0), showlegend=False,
            coloraxis_showscale=False)
        fig_w.update_traces(marker_line_width=0)
        st.plotly_chart(fig_w, use_container_width=True)

elif recs is not None and recs.empty:
    st.markdown("""
    <div style="background:#13131C;border:1px solid rgba(255,255,255,0.07);border-radius:18px;
                padding:40px;text-align:center;margin-top:24px;">
        <div style="font-size:2.5em;margin-bottom:12px;">🔍</div>
        <div style="font-family:'Fraunces',serif;font-size:1.3em;color:#FFFFFF;margin-bottom:8px;">
            Sonuç bulunamadı
        </div>
        <div style="color:#4A4860;font-size:0.88em;">
            Filtreleri biraz gevşet — fiyat aralığını artır veya min. yatak sayısını azalt.
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#13131C;border:1px dashed rgba(255,255,255,0.06);border-radius:18px;
                padding:48px;text-align:center;margin-top:24px;">
        <div style="font-size:2em;margin-bottom:12px;color:#2A2840;">✦</div>
        <div style="font-family:'Fraunces',serif;font-size:1.1em;color:#3A3850;">
            Soldaki panelden tercihlerini seç ve öneri getir.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:56px;padding-top:20px;border-top:1px solid rgba(255,255,255,0.04);
            text-align:center;color:#2A2840;font-size:0.75em;font-family:'Plus Jakarta Sans',sans-serif;">
    ✦ StayFinder · Hibrit Öneri Motoru · CENG468 · Beyza İmancı
</div>""", unsafe_allow_html=True)
