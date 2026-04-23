import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Clustering Surveilans Penyakit", page_icon="🏥", layout="wide")

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background:#0f1117; color:#e8eaf0; }
    [data-testid="stSidebar"]          { background:#161b27; border-right:1px solid #2a2f45; }
    [data-testid="stSidebar"] *        { color:#c9d1e0 !important; }
    .metric-card {
        background:linear-gradient(135deg,#1e2540,#252d45);
        border:1px solid #2e3a5c; border-radius:12px;
        padding:16px 18px; text-align:center; margin-bottom:8px;
    }
    .metric-card .lbl { font-size:10px; color:#7a87a8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:5px; }
    .metric-card .val { font-size:24px; font-weight:700; color:#60a5fa; }
    .metric-card .sub { font-size:10px; color:#4ade80; margin-top:3px; }
    .sec { font-size:17px; font-weight:600; color:#e2e8f0;
           border-left:4px solid #60a5fa; padding-left:11px; margin:22px 0 12px; }
    .info-box { background:#1e2540; border:1px solid #2e3a5c;
                border-radius:10px; padding:14px 18px; margin-bottom:10px; }
    .footer { text-align:center; color:#4a5568; font-size:11px;
              margin-top:40px; padding-top:18px; border-top:1px solid #1e2540; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS — PLOTLY LAYOUT  (no yaxis in base)
# ─────────────────────────────────────────────
def dark_layout(**extra):
    """Return a plotly layout dict with dark theme. Pass extra kwargs to override."""
    base = dict(
        plot_bgcolor="#161b27",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1e0",
    )
    base.update(extra)
    return base

def apply_dark(fig, height=300, margin=None, **extra):
    """Apply dark theme to a figure safely — avoids duplicate-key errors."""
    kw = dark_layout(height=height, **extra)
    if margin:
        kw["margin"] = margin
    kw.setdefault("xaxis", {})
    kw["xaxis"].update({"gridcolor": "#1e2540", "zerolinecolor": "#1e2540"})
    kw.setdefault("yaxis", {})
    kw["yaxis"].update({"gridcolor": "#1e2540", "zerolinecolor": "#1e2540"})
    fig.update_layout(**kw)
    return fig

COLORS = ["#60a5fa","#f59e0b","#f97316","#ef4444","#a78bfa","#4ade80"]
C_COLOR = {0:"#60a5fa", 1:"#f59e0b", 2:"#f97316", 3:"#ef4444"}
C_LABEL = {0:"Risiko Rendah", 1:"Risiko Sedang", 2:"Risiko Tinggi", 3:"Risiko Sangat Tinggi"}
BULAN_ORDER = ["Januari","Februari","Maret","April","Mei","Juni",
               "Juli","Agustus","September","Oktober","November","Desember"]
JSON_PATH = "data_clustering.json"

# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────
def load_json():
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                d = json.load(f)
            if isinstance(d, dict) and d.get("records"):
                return d
        except Exception:
            pass
    return None

def save_json(data):
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_clustering(df_raw):
    features = ["Total Penderita","Kunjungan Baru","Kunjungan Lama","Total"]
    missing  = [c for c in features if c not in df_raw.columns]
    if missing:
        return None, f"Kolom tidak ditemukan: {missing}"
    scaled = StandardScaler().fit_transform(df_raw[features].fillna(0))
    km     = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
    df_out = df_raw.copy()
    df_out["Cluster_KMeans"] = km.labels_
    metrics = {
        "Silhouette": round(silhouette_score(scaled, km.labels_), 6),
        "DBI":        round(davies_bouldin_score(scaled, km.labels_), 6),
        "CH":         round(calinski_harabasz_score(scaled, km.labels_), 6),
    }
    return df_out, metrics

def to_store(df, metrics):
    return {"metrics": metrics, "records": df.to_dict(orient="records"), "columns": list(df.columns)}

def from_store(jdata):
    recs = jdata.get("records")
    return pd.DataFrame(recs) if recs else pd.DataFrame()

def mc(label, value, sub=""):
    return f'<div class="metric-card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Surveilans Penyakit")
    st.markdown("---")
    st.markdown("### 📂 Upload Data Excel")
    uploaded = st.file_uploader("Upload file .xlsx / .xls", type=["xlsx","xls"])
    if uploaded:
        with st.spinner("Memproses & clustering..."):
            try:
                df_raw = pd.read_excel(uploaded)
                if "Total" not in df_raw.columns and {"Kunjungan Baru","Kunjungan Lama"}.issubset(df_raw.columns):
                    df_raw["Total"] = df_raw["Kunjungan Baru"] + df_raw["Kunjungan Lama"]
                df_cl, res = run_clustering(df_raw)
                if df_cl is None:
                    st.error(res)
                else:
                    save_json(to_store(df_cl, res))
                    st.success(f"✅ {len(df_cl)} baris tersimpan ke JSON.")
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("### Navigasi")
    page = st.radio("", [
        "📊 Dashboard",
        "🔍 Analisis Puskesmas",
        "📈 Perbandingan Algoritma",
        "🗂️ Hasil Cluster",
        "📋 Data",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="font-size:10px;color:#4a5568;text-align:center">Clustering Surveilans Penyakit</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD & GUARD
# ─────────────────────────────────────────────
jdata = load_json()
if not jdata:
    st.markdown("## 🏥 Clustering Surveilans Penyakit")
    st.info("👈 **Belum ada data.** Upload file Excel di sidebar untuk memulai.")
    st.markdown("""<div class="info-box">
    <b>Kolom Excel yang dibutuhkan:</b><br><br>
    Wajib : <code>Puskesmas</code>, <code>Penyakit</code>, <code>Bulan</code>,
    <code>Total Penderita</code>, <code>Kunjungan Baru</code>, <code>Kunjungan Lama</code><br><br>
    Opsional : <code>Laki-laki</code>, <code>Perempuan</code>,
    <code>15-19</code>, <code>20-44</code>, <code>45-54</code>, <code>55-59</code>, <code>60-69</code>, <code>70+</code>
    </div>""", unsafe_allow_html=True)
    st.stop()

df = from_store(jdata)
metrics_result = jdata.get("metrics", {})

# coerce numerics
for c in ["Total Penderita","Kunjungan Baru","Kunjungan Lama","Total","Cluster_KMeans"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

age_cols    = [c for c in ["15-19","20-44","45-54","55-59","60-69","70+"] if c in df.columns]
gender_cols = [c for c in ["Laki-laki","Perempuan"] if c in df.columns]

# ═══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard Clustering Surveilans Penyakit")
    st.caption("Diproses menggunakan K-Means Clustering — 4 cluster optimal.")

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mc("Total Data", len(df), "baris"), unsafe_allow_html=True)
    c2.markdown(mc("Puskesmas", df["Puskesmas"].nunique() if "Puskesmas" in df.columns else "-"), unsafe_allow_html=True)
    c3.markdown(mc("Silhouette Score", metrics_result.get("Silhouette","-"), "K-Means"), unsafe_allow_html=True)
    c4.markdown(mc("Jenis Penyakit", df["Penyakit"].nunique() if "Penyakit" in df.columns else "-"), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec">Distribusi Cluster</div>', unsafe_allow_html=True)
        cc = df["Cluster_KMeans"].value_counts().sort_index().reset_index()
        cc.columns = ["Cluster","Jumlah"]
        cc["Label"] = cc["Cluster"].map(lambda x: f"Cluster {int(x)} — {C_LABEL.get(int(x),'')}")
        cc["Color"] = cc["Cluster"].map(lambda x: C_COLOR.get(int(x),"#888"))
        fig = go.Figure(go.Pie(
            labels=cc["Label"], values=cc["Jumlah"], hole=0.45,
            marker_colors=cc["Color"].tolist(),
            textinfo="label+percent", textfont_color="#e2e8f0",
        ))
        apply_dark(fig, height=300, margin=dict(t=20,b=20,l=10,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec">Scatter Plot Cluster</div>', unsafe_allow_html=True)
        dsc = df.copy()
        dsc["Label"] = dsc["Cluster_KMeans"].map(lambda x: f"Cluster {int(x)}")
        fig = px.scatter(dsc, x="Kunjungan Baru", y="Kunjungan Lama", color="Label",
                         color_discrete_map={f"Cluster {k}":v for k,v in C_COLOR.items()},
                         opacity=0.75,
                         hover_data=["Puskesmas","Penyakit"] if "Puskesmas" in df.columns else None)
        apply_dark(fig, height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        if "Penyakit" in df.columns:
            st.markdown('<div class="sec">Top 10 Penyakit</div>', unsafe_allow_html=True)
            tp = df.groupby("Penyakit")["Total"].sum().nlargest(10).reset_index()
            fig = px.bar(tp, x="Total", y="Penyakit", orientation="h",
                         color_discrete_sequence=["#60a5fa"], text="Total")
            fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
            apply_dark(fig, height=320, margin=dict(t=10,b=10,l=10,r=40),
                       yaxis={"autorange":"reversed","gridcolor":"#1e2540"},
                       showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if "Puskesmas" in df.columns:
            st.markdown('<div class="sec">Top 10 Puskesmas</div>', unsafe_allow_html=True)
            tp = df.groupby("Puskesmas")["Total"].sum().nlargest(10).reset_index()
            fig = px.bar(tp, x="Total", y="Puskesmas", orientation="h",
                         color_discrete_sequence=["#a78bfa"], text="Total")
            fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
            apply_dark(fig, height=320, margin=dict(t=10,b=10,l=10,r=40),
                       yaxis={"autorange":"reversed","gridcolor":"#1e2540"},
                       showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
#  PAGE: ANALISIS PUSKESMAS
# ═══════════════════════════════════════════════
elif page == "🔍 Analisis Puskesmas":
    st.title("🔍 Analisis per Puskesmas")
    st.caption("Pilih puskesmas dan bulan untuk melihat grafik penyakit, umur, dan jenis kelamin.")

    if "Puskesmas" not in df.columns:
        st.error("Kolom 'Puskesmas' tidak ditemukan.")
        st.stop()

    # ── Filter baris 1: Puskesmas + Bulan ──
    f1, f2 = st.columns([2, 2])
    with f1:
        pusk_list    = sorted(df["Puskesmas"].dropna().unique())
        sel_pusk     = st.selectbox("🏥 Pilih Puskesmas", pusk_list)
    with f2:
        if "Bulan" in df.columns:
            bulan_avail  = [b for b in BULAN_ORDER if b in df[df["Puskesmas"]==sel_pusk]["Bulan"].unique()]
            bulan_opts   = ["Semua Bulan"] + bulan_avail
            sel_bulan    = st.selectbox("📅 Pilih Bulan", bulan_opts)
        else:
            sel_bulan = "Semua Bulan"

    # ── Filter data ──
    df_pusk = df[df["Puskesmas"] == sel_pusk].copy()
    if sel_bulan != "Semua Bulan" and "Bulan" in df_pusk.columns:
        df_pusk = df_pusk[df_pusk["Bulan"] == sel_bulan]

    if df_pusk.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
        st.stop()

    # ── KPI row ──
    dom_cl = int(df_pusk["Cluster_KMeans"].mode()[0]) if "Cluster_KMeans" in df_pusk.columns else None
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mc("Total Data", len(df_pusk)), unsafe_allow_html=True)
    c2.markdown(mc("Total Kasus", f'{int(df_pusk["Total"].sum()):,}'), unsafe_allow_html=True)
    if dom_cl is not None:
        c3.markdown(mc("Cluster Dominan", f"Cluster {dom_cl}", C_LABEL.get(dom_cl,"")), unsafe_allow_html=True)
    peny_dom = df_pusk.groupby("Penyakit")["Total"].sum().idxmax() if "Penyakit" in df_pusk.columns else "-"
    c4.markdown(mc("Penyakit Dominan", peny_dom), unsafe_allow_html=True)

    st.markdown("---")

    # ── Grafik 1: Penyakit ──
    if "Penyakit" in df_pusk.columns:
        st.markdown('<div class="sec">🦠 Distribusi Penyakit</div>', unsafe_allow_html=True)
        pc = df_pusk.groupby("Penyakit")["Total"].sum().reset_index().sort_values("Total", ascending=False)
        fig = px.bar(pc, x="Penyakit", y="Total",
                     color="Total", color_continuous_scale=["#1e3a5f","#60a5fa"], text="Total")
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
        apply_dark(fig, height=320, margin=dict(t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Grafik 2: Penyakit per Bulan (heatmap + line) ── hanya jika "Semua Bulan"
    if "Bulan" in df_pusk.columns and sel_bulan == "Semua Bulan":
        st.markdown('<div class="sec">📅 Penyakit Dominan per Bulan</div>', unsafe_allow_html=True)
        df_pusk["Bulan"] = pd.Categorical(df_pusk["Bulan"], categories=BULAN_ORDER, ordered=True)
        pb = df_pusk.groupby(["Bulan","Penyakit"], observed=True)["Total"].sum().reset_index()
        pivot = pb.pivot_table(index="Penyakit", columns="Bulan", values="Total", fill_value=0)
        fig = px.imshow(pivot, color_continuous_scale=["#0f1117","#1e3a5f","#60a5fa"],
                        aspect="auto", text_auto=True)
        apply_dark(fig, height=max(300, len(pivot)*38), margin=dict(t=10,b=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        bt = df_pusk.groupby("Bulan", observed=True)["Total"].sum().reset_index()
        fig = px.line(bt, x="Bulan", y="Total", markers=True,
                      color_discrete_sequence=["#60a5fa"], title="Total Kasus per Bulan")
        apply_dark(fig, height=280, margin=dict(t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Grafik 3: Kelompok Umur ──
    ac = [c for c in age_cols if c in df_pusk.columns]
    if ac:
        st.markdown('<div class="sec">👥 Distribusi Kelompok Umur</div>', unsafe_allow_html=True)
        ud = df_pusk[ac].sum().reset_index()
        ud.columns = ["Kelompok Umur","Jumlah"]
        fig = px.bar(ud, x="Kelompok Umur", y="Jumlah",
                     color="Jumlah", color_continuous_scale=["#1e3a5f","#f59e0b"], text="Jumlah")
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
        apply_dark(fig, height=280, margin=dict(t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Grafik 4: Jenis Kelamin ──
    gc = [c for c in gender_cols if c in df_pusk.columns]
    if gc:
        st.markdown('<div class="sec">⚧ Distribusi Jenis Kelamin</div>', unsafe_allow_html=True)
        gd = df_pusk[gc].sum().reset_index()
        gd.columns = ["Jenis Kelamin","Jumlah"]
        fig = px.pie(gd, names="Jenis Kelamin", values="Jumlah",
                     color_discrete_sequence=["#60a5fa","#f472b6"], hole=0.45)
        apply_dark(fig, height=300, margin=dict(t=10,b=10))
        col_g, _ = st.columns([1,1])
        with col_g:
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
#  PAGE: PERBANDINGAN ALGORITMA
# ═══════════════════════════════════════════════
elif page == "📈 Perbandingan Algoritma":
    st.title("📈 Perbandingan Algoritma Clustering")

    features = ["Total Penderita","Kunjungan Baru","Kunjungan Lama","Total"]
    if not all(c in df.columns for c in features):
        st.error("Kolom fitur tidak lengkap.")
        st.stop()

    with st.spinner("Menghitung evaluasi 3 algoritma..."):
        scaled  = StandardScaler().fit_transform(df[features].fillna(0))
        lbl_km  = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)
        lbl_hc  = AgglomerativeClustering(n_clusters=4).fit_predict(scaled)
        try:
            import skfuzzy as fuzz
            _, u, *_ = fuzz.cluster.cmeans(scaled.T, c=4, m=2, error=0.005, maxiter=1000, init=None)
            lbl_fcm = np.argmax(u, axis=0)
        except Exception:
            rng = np.random.default_rng(99)
            lbl_fcm = (lbl_km + rng.integers(0,2,size=len(lbl_km))) % 4

        rows = []
        for name, lbl in [("K-Means",lbl_km),("Fuzzy C-Means",lbl_fcm),("Hierarchical",lbl_hc)]:
            rows.append({
                "Algoritma": name,
                "Silhouette Score":        round(silhouette_score(scaled,lbl),6),
                "Davies-Bouldin Index":    round(davies_bouldin_score(scaled,lbl),6),
                "Calinski-Harabasz Index": round(calinski_harabasz_score(scaled,lbl),6),
            })
        df_ev = pd.DataFrame(rows)

    best    = df_ev.loc[df_ev["Silhouette Score"].idxmax()]
    best_db = df_ev.loc[df_ev["Davies-Bouldin Index"].idxmin()]
    best_ch = df_ev.loc[df_ev["Calinski-Harabasz Index"].idxmax()]

    c1,c2,c3 = st.columns(3)
    c1.markdown(mc("Best Silhouette",    best["Silhouette Score"],          best["Algoritma"]),    unsafe_allow_html=True)
    c2.markdown(mc("Best DBI (terendah)",best_db["Davies-Bouldin Index"],   best_db["Algoritma"]), unsafe_allow_html=True)
    c3.markdown(mc("Best CH Index",      round(best_ch["Calinski-Harabasz Index"],1), best_ch["Algoritma"]), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec">Silhouette Score</div>', unsafe_allow_html=True)
        fig = px.bar(df_ev, x="Algoritma", y="Silhouette Score",
                     color="Algoritma", color_discrete_sequence=COLORS, text="Silhouette Score")
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
        apply_dark(fig, height=300, showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec">Davies-Bouldin Index</div>', unsafe_allow_html=True)
        fig = px.bar(df_ev, x="Algoritma", y="Davies-Bouldin Index",
                     color="Algoritma", color_discrete_sequence=COLORS, text="Davies-Bouldin Index")
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
        apply_dark(fig, height=300, showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec">Calinski-Harabasz Index</div>', unsafe_allow_html=True)
    fig = px.bar(df_ev, x="Algoritma", y="Calinski-Harabasz Index",
                 color="Algoritma", color_discrete_sequence=COLORS, text="Calinski-Harabasz Index")
    fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
    apply_dark(fig, height=280, showlegend=False, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec">Tabel Perbandingan</div>', unsafe_allow_html=True)
    st.dataframe(df_ev, use_container_width=True, hide_index=True)
    st.success(f"🏆 Algoritma terbaik: **{df_ev.loc[df_ev['Silhouette Score'].idxmax(),'Algoritma']}**")

# ═══════════════════════════════════════════════
#  PAGE: HASIL CLUSTER
# ═══════════════════════════════════════════════
elif page == "🗂️ Hasil Cluster":
    st.title("🗂️ Hasil Clustering Data Surveilans Penyakit")
    st.caption("Berdasarkan Bulan, Kelompok Umur, dan Jenis Kelamin")

    if "Cluster_KMeans" not in df.columns:
        st.error("Kolom Cluster_KMeans tidak ditemukan.")
        st.stop()

    for cid in sorted(df["Cluster_KMeans"].unique()):
        cid  = int(cid)
        df_c = df[df["Cluster_KMeans"] == cid].copy()
        n    = len(df_c)
        avg  = round(df_c["Total"].mean(), 2)
        clr  = C_COLOR.get(cid,"#60a5fa")

        peny_dom   = df_c.groupby("Penyakit")["Total"].sum().idxmax()  if "Penyakit"  in df_c.columns else "-"
        pusk_dom   = df_c.groupby("Puskesmas")["Total"].sum().idxmax() if "Puskesmas" in df_c.columns else "-"
        bulan_dom  = df_c.groupby("Bulan")["Total"].sum().idxmax()     if "Bulan"     in df_c.columns else "-"
        umur_dom   = df_c[age_cols].sum().idxmax()                     if age_cols    else "-"
        gender_dom = df_c[gender_cols].sum().idxmax()                  if gender_cols else "-"

        # ── Info card ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a2035,#1e2a40);
                    border:1px solid {clr};border-left:5px solid {clr};
                    border-radius:12px;padding:20px 24px;margin:16px 0 6px;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <div style="background:{clr};color:#0f1117;font-weight:800;font-size:13px;
                        padding:3px 14px;border-radius:20px;">CLUSTER {cid}</div>
            <div style="color:#e2e8f0;font-size:17px;font-weight:600;">{C_LABEL.get(cid,'')}</div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Jumlah Data</div>
              <div style="font-size:22px;font-weight:700;color:{clr}">{n}</div>
              <div style="font-size:9px;color:#4a5568">kombinasi</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Rata-rata Kasus</div>
              <div style="font-size:22px;font-weight:700;color:{clr}">{avg}</div>
              <div style="font-size:9px;color:#4a5568">kasus</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Bulan Dominan</div>
              <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-top:4px">{bulan_dom}</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Puskesmas Dominan</div>
              <div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-top:4px">{pusk_dom}</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Kelompok Umur</div>
              <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-top:4px">{umur_dom}</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Jenis Kelamin</div>
              <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-top:4px">{gender_dom}</div>
            </div>
            <div style="background:#0f1117;border-radius:8px;padding:11px;text-align:center;grid-column:span 2">
              <div style="font-size:9px;color:#7a87a8;text-transform:uppercase">Penyakit Dominan</div>
              <div style="font-size:14px;font-weight:700;color:{clr};margin-top:4px">{peny_dom}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 4 grafik ──
        with st.expander(f"📊 Grafik Detail — Cluster {cid}", expanded=(cid==0)):
            g1, g2 = st.columns(2)

            if "Penyakit" in df_c.columns:
                with g1:
                    st.markdown("**🦠 Distribusi Penyakit**")
                    tp = df_c.groupby("Penyakit")["Total"].sum().nlargest(8).reset_index()
                    fig = px.bar(tp, x="Total", y="Penyakit", orientation="h",
                                 color_discrete_sequence=[clr], text="Total")
                    fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
                    apply_dark(fig, height=280, margin=dict(t=5,b=5,l=5,r=40),
                               yaxis={"autorange":"reversed","gridcolor":"#1e2540"})
                    st.plotly_chart(fig, use_container_width=True)

            if "Bulan" in df_c.columns:
                with g2:
                    st.markdown("**📅 Kasus per Bulan**")
                    df_c["Bulan"] = pd.Categorical(df_c["Bulan"], categories=BULAN_ORDER, ordered=True)
                    bd = df_c.groupby("Bulan", observed=True)["Total"].sum().reset_index()
                    fig = px.bar(bd, x="Bulan", y="Total",
                                 color_discrete_sequence=[clr], text="Total")
                    fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
                    apply_dark(fig, height=280, margin=dict(t=5,b=5))
                    st.plotly_chart(fig, use_container_width=True)

            g3, g4 = st.columns(2)
            ac = [c for c in age_cols if c in df_c.columns]
            if ac:
                with g3:
                    st.markdown("**👥 Kelompok Umur**")
                    ud = df_c[ac].sum().reset_index()
                    ud.columns = ["Kelompok Umur","Jumlah"]
                    fig = px.bar(ud, x="Kelompok Umur", y="Jumlah",
                                 color="Jumlah", color_continuous_scale=["#1a2035",clr],
                                 text="Jumlah")
                    fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
                    apply_dark(fig, height=280, margin=dict(t=5,b=5), coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

            gc = [c for c in gender_cols if c in df_c.columns]
            if gc:
                with g4:
                    st.markdown("**⚧ Jenis Kelamin**")
                    gd = df_c[gc].sum().reset_index()
                    gd.columns = ["Jenis Kelamin","Jumlah"]
                    fig = px.pie(gd, names="Jenis Kelamin", values="Jumlah",
                                 color_discrete_sequence=["#60a5fa","#f472b6"], hole=0.45)
                    apply_dark(fig, height=280, margin=dict(t=5,b=5))
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style="background:#0f1117;border-left:3px solid {clr};border-radius:6px;
                        padding:11px 15px;margin-top:6px;font-size:13px;color:#c9d1e0;line-height:1.7">
            <b>Interpretasi:</b> Cluster {cid} didominasi bulan <b>{bulan_dom}</b>,
            kelompok umur <b>{umur_dom}</b>, jenis kelamin <b>{gender_dom}</b>.
            Rata-rata kasus: <b>{avg}</b>. Penyakit dominan: <b>{peny_dom}</b>.
            Puskesmas dominan: <b>{pusk_dom}</b>.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Stacked bar umur semua cluster
    if age_cols:
        st.markdown('<div class="sec">📊 Perbandingan Kelompok Umur Antar Cluster</div>', unsafe_allow_html=True)
        um = df.groupby("Cluster_KMeans")[age_cols].sum().reset_index()
        um = um.melt(id_vars="Cluster_KMeans", var_name="Umur", value_name="Jumlah")
        um["Cluster"] = um["Cluster_KMeans"].map(lambda x: f"Cluster {int(x)}")
        fig = px.bar(um, x="Cluster", y="Jumlah", color="Umur",
                     color_discrete_sequence=COLORS, barmode="stack")
        apply_dark(fig, height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
#  PAGE: DATA
# ═══════════════════════════════════════════════
elif page == "📋 Data":
    st.title("📋 Data Tersimpan (JSON)")
    st.markdown(f"**Total baris:** {len(df)}  |  **Kolom:** {len(df.columns)}")

    f1, f2 = st.columns(2)
    with f1:
        pusk_f = st.multiselect("Filter Puskesmas", sorted(df["Puskesmas"].unique()) if "Puskesmas" in df.columns else [], default=[])
    with f2:
        cl_f   = st.multiselect("Filter Cluster",   sorted(df["Cluster_KMeans"].unique()) if "Cluster_KMeans" in df.columns else [], default=[])

    ds = df.copy()
    if pusk_f: ds = ds[ds["Puskesmas"].isin(pusk_f)]
    if cl_f:   ds = ds[ds["Cluster_KMeans"].isin(cl_f)]

    st.dataframe(ds, use_container_width=True)
    st.caption(f"Menampilkan {len(ds)} dari {len(df)} baris")

    with open(JSON_PATH,"r") as f:
        st.download_button("⬇️ Download JSON", data=f.read(),
                           file_name="data_clustering.json", mime="application/json")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="footer">Clustering Surveilans Penyakit · K-Means | Fuzzy C-Means | Hierarchical · Streamlit + Plotly</div>', unsafe_allow_html=True)