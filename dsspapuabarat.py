# app_streamlit_dss_iklim_papuabarat.py
import os
import io
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="DSS Iklim - Papua Barat", layout="wide")
st.title("🌦️ Decision Support System Iklim — Papua Barat")

# Local data paths (ubah sesuai kebutuhan)
LOCAL_XLSX_PATH = "PAPUABARAT2.xlsx"
LOCAL_CSV_PATH = "data_papuabarat.csv"

# ---------- DSS helper functions ----------

def klasifikasi_cuaca(ch, matahari):
    if ch > 20:
        return "Hujan"
    elif ch > 5:
        return "Berawan"
    elif matahari > 4:
        return "Cerah"
    else:
        return "Berawan"


def risiko_kekeringan_score(ch, matahari):
    # score 0..1, higher = higher drought risk
    ch_clamped = np.clip(ch, 0, 200)
    matahari_clamped = np.clip(matahari, 0, 16)
    score = (1 - (ch_clamped / 200)) * 0.7 + (matahari_clamped / 16) * 0.3
    return float(np.clip(score, 0, 1))


def risiko_kekeringan_label(score, thresholds=(0.6, 0.3)):
    high, med = thresholds
    if score >= high:
        return "Risiko Tinggi"
    elif score >= med:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"


def hujan_ekstrem_flag(ch, threshold=50):
    return int(ch > threshold)


def compute_weather_index(df):
    # Composite index 0..1 from rainfall, temp stress, humidity stress, wind
    eps = 1e-6
    r = df['curah_hujan'].fillna(0).astype(float).values
    r_norm = (r - r.min()) / (r.max() - r.min() + eps)

    t = df['Tavg'].fillna(df.get('Tn', df['Tavg'])).astype(float).values
    comfy_low, comfy_high = 24, 28
    t_dist = np.maximum(0, np.maximum(comfy_low - t, t - comfy_high))
    t_norm = (t_dist - t_dist.min()) / (t_dist.max() - t_dist.min() + eps)

    h = df['kelembaban'].fillna(0).astype(float).values
    hum_dist = np.maximum(0, np.maximum(40 - h, h - 70))
    h_norm = (hum_dist - hum_dist.min()) / (hum_dist.max() - hum_dist.min() + eps)

    w = df['kecepatan_angin'].fillna(0).astype(float).values
    w_norm = (w - w.min()) / (w.max() - w.min() + eps)

    composite = 0.35 * r_norm + 0.25 * t_norm + 0.2 * h_norm + 0.2 * w_norm
    return np.clip(composite, 0, 1)


# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data(local_xlsx=LOCAL_XLSX_PATH, local_csv=LOCAL_CSV_PATH):
    # Prioritize XLSX named PAPUABARAT2.xlsx (user request), fallback to CSV, else generate sample
    try:
        if os.path.exists(local_xlsx):
            df = pd.read_excel(local_xlsx, parse_dates=['Tanggal'])
            st.sidebar.success(f"Loaded local XLSX: {local_xlsx}")
        elif os.path.exists(local_csv):
            df = pd.read_csv(local_csv, parse_dates=['Tanggal'])
            st.sidebar.success(f"Loaded local CSV: {local_csv}")
        else:
            raise FileNotFoundError("No local data file found")
    except Exception:
        # Generate sample 2-year daily data as fallback
        st.sidebar.info("File data tidak ditemukan — membuat data contoh 2 tahun.")
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=730)
        rng = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'Tanggal': rng,
            'curah_hujan': np.random.gamma(1.5, 8, len(rng)).round(1),
            'Tn': np.random.normal(22, 2, len(rng)).round(1),
            'Tx': np.random.normal(31, 2.5, len(rng)).round(1),
            'Tavg': np.random.normal(26.5, 1.8, len(rng)).round(1),
            'kelembaban': np.random.randint(50, 95, len(rng)),
            'matahari': np.clip(np.random.normal(5, 2, len(rng)), 0, 12).round(1),
            'kecepatan_angin': np.random.uniform(0, 20, len(rng)).round(1),
            'Wilayah': np.random.choice(['Kaimana', 'Manokwari', 'Sorong', 'Fakfak'], len(rng))
        })
    # Ensure types & fill NAs
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    for col in ['curah_hujan', 'Tn', 'Tx', 'Tavg', 'kelembaban', 'matahari', 'kecepatan_angin']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.sort_values('Tanggal').reset_index(drop=True)


# Load
data = load_data()

# Sidebar controls
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Pengaturan</div>", unsafe_allow_html=True)

with st.sidebar:
    extreme_threshold = st.number_input("Ambang Hujan Ekstrem (mm/hari)", value=50, min_value=1)
    risk_high = st.slider("Ambang Risiko Tinggi (0–1)", 0.0, 1.0, 0.6, 0.01)
    risk_med = st.slider("Ambang Risiko Sedang (0–1)", 0.0, 1.0, 0.3, 0.01)
    ma_window = st.slider("Moving Average (hari)", 1, 60, 7)

# Filters
st.sidebar.markdown("<div class='sidebar-title'>📅 Filter Data</div>", unsafe_allow_html=True)

with st.sidebar:
    min_date = data['Tanggal'].min().date()
    max_date = data['Tanggal'].max().date()
    date_range = st.date_input(
        "Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


region = None
if 'Wilayah' in data.columns:
    st.sidebar.markdown("<div class='sidebar-title'>🗺️ Pilih Wilayah</div>", unsafe_allow_html=True)
    regions = ['All'] + sorted(data['Wilayah'].unique().tolist())
    region = st.sidebar.selectbox("Wilayah", regions)


start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (data['Tanggal'] >= start_date) & (data['Tanggal'] <= end_date)
if region and region != 'All':
    mask &= (data['Wilayah'] == region)
df = data.loc[mask].copy()
if df.empty:
    st.warning("Tidak ada data pada rentang/wilayah yang dipilih — menampilkan seluruh dataset.")
    df = data.copy()

# Derived fields
df['Prediksi Cuaca'] = df.apply(lambda r: klasifikasi_cuaca(r['curah_hujan'], r['matahari']), axis=1)
df['Hujan Ekstrem'] = df['curah_hujan'].apply(lambda x: "Ya" if x > extreme_threshold else "Tidak")
df['extreme_flag'] = df['curah_hujan'].apply(lambda x: hujan_ekstrem_flag(x, threshold=extreme_threshold))
df['RiskScore'] = df.apply(lambda r: risiko_kekeringan_score(r['curah_hujan'], r['matahari']), axis=1)
df['RiskLabel'] = df['RiskScore'].apply(lambda s: risiko_kekeringan_label(s, thresholds=(risk_high, risk_med)))
df['WeatherIndex'] = compute_weather_index(df)
df['Year'] = df['Tanggal'].dt.year
df['Month'] = df['Tanggal'].dt.month

# Top metrics
st.markdown("---")
st.subheader("Ringkasan Cepat")

# helper card renderer
def render_stat_card(col, title, value, subtitle=None):
    subtitle_html = (
        f"<div style='font-size:12px; color:#6b7280; margin-top:4px;'>{subtitle}</div>"
        if subtitle else ""
    )

    html = f"""
    <div class='summary-card'>
        <div style='font-size:16px; font-weight:600; color:#475569; margin-bottom:6px;'>{title}</div>
        <div style='font-size:28px; font-weight:700; white-space:nowrap; color:#0f172a;'>{value}</div>
        {subtitle_html}
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

# formatted values
start = df['Tanggal'].min().strftime("%Y-%m-%d")
end = df['Tanggal'].max().strftime("%Y-%m-%d")
period_text = f"{start} — {end}"

avg_rain = f"{df['curah_hujan'].mean():.2f}"
avg_temp = f"{df['Tavg'].mean():.2f}"
avg_risk = f"{df['RiskScore'].mean():.2f}"

# layout: periode lebih lebar
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

render_stat_card(c1, "Periode", period_text)
render_stat_card(c2, "Avg Rain (mm)", avg_rain)
render_stat_card(c3, "Avg Temp (°C)", avg_temp)
render_stat_card(c4, "Avg RiskScore", avg_risk)
# ---------------- 1. Rainfall Forecast ----------------
st.markdown("---")
st.header("1. Prediksi Curah Hujan (Rainfall Forecast)")
r1, r2 = st.columns([2, 1])
with r1:
    fig_rain_line = px.line(df, x='Tanggal', y='curah_hujan', title="Rainfall (Line Chart)")
    fig_rain_line.update_layout(yaxis_title="mm", xaxis_title="Tanggal")
    st.plotly_chart(fig_rain_line, use_container_width=True)

    fig_rain_area = px.area(df, x='Tanggal', y='curah_hujan', title="Rainfall (Area Chart)")
    st.plotly_chart(fig_rain_area, use_container_width=True)

with r2:
    st.write("**Kegunaan:**")
    st.write("- Identifikasi musim hujan & potensi banjir.")
    monthly_sum = df.set_index('Tanggal').resample('M')['curah_hujan'].sum().reset_index()
    fig_month_bar = px.bar(monthly_sum, x='Tanggal', y='curah_hujan', title="Monthly Rainfall Sum")
    st.plotly_chart(fig_month_bar, use_container_width=True)

# ---------------- 2. Temperature Forecast ----------------
st.markdown("---")
st.header("2. Prediksi Hari Panas / Temperatur (Temperature Forecast)")
t1, t2 = st.columns([2, 1])
with t1:
    fig_temp = px.line(df, x='Tanggal', y=['Tn', 'Tavg', 'Tx'], labels={'value': 'Temperature (°C)'},
                       title="Temperature Trends (Tn, Tavg, Tx)")
    st.plotly_chart(fig_temp, use_container_width=True)

    # Heatmap month x day of average Tavg (if range long enough)
    heat_df = df.copy()
    heat_df['Day'] = heat_df['Tanggal'].dt.day
    heat_df['MonthName'] = heat_df['Tanggal'].dt.strftime('%b')
    pivot = heat_df.pivot_table(index='MonthName', columns='Day', values='Tavg', aggfunc='mean')
    if pivot.shape[0] > 0 and pivot.shape[1] > 0:
        fig_heat = px.imshow(pivot, labels=dict(x='Day', y='Month', color='Tavg'),
                             title="Temperature Heatmap (Month x Day)")
        st.plotly_chart(fig_heat, use_container_width=True)

with t2:
    st.write("**Kegunaan:**")
    st.write("- Deteksi gelombang panas, perbandingan suhu rata-rata per periode.")

# ---------------- 3. Prediksi Risiko Kekeringan ----------------
st.markdown("---")
st.header("3. Prediksi Risiko Kekeringan")
d1, d2 = st.columns([2, 1])
with d1:
    latest_score = df['RiskScore'].mean()
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_score,
        number={'valueformat': ".2f"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, risk_med], 'color': "lightgreen"},
                {'range': [risk_med, risk_high], 'color': "yellow"},
                {'range': [risk_high, 1], 'color': "red"}
            ]
        },
        title={'text': "Average Drought Risk (0-1)"}
    ))
    gauge.update_layout(height=300)
    st.plotly_chart(gauge, use_container_width=True)

    if 'Wilayah' in df.columns:
        by_region = df.groupby('Wilayah')['RiskScore'].mean().reset_index().sort_values('RiskScore', ascending=False)
        fig_bar_region = px.bar(by_region, x='Wilayah', y='RiskScore', title="Average RiskScore per Wilayah")
        st.plotly_chart(fig_bar_region, use_container_width=True)

    df['Risk_MA'] = df['RiskScore'].rolling(window=ma_window).mean()
    fig_risk_line = px.line(df, x='Tanggal', y='RiskScore', title="Risk Score Over Time")
    fig_risk_line.add_scatter(x=df['Tanggal'], y=df['Risk_MA'], mode='lines', name=f'MA({ma_window})')
    st.plotly_chart(fig_risk_line, use_container_width=True)

with d2:
    st.write("**Kegunaan:**")
    st.write("- Visual mudah memahami ambang risiko; ubah threshold di sidebar untuk sensitivitas.")

# ---------------- 4. Prediksi Hujan Ekstrem ----------------
st.markdown("---")
st.header("4. Prediksi Hujan Ekstrem")
e1, e2 = st.columns([2, 1])
with e1:
    freq = df[df['Hujan Ekstrem'] == 'Ya'].groupby(df['Tanggal'].dt.to_period('M')).size().reset_index(name='count')
    if not freq.empty:
        freq['Tanggal'] = freq['Tanggal'].dt.to_timestamp()
        fig_freq = px.bar(freq, x='Tanggal', y='count', title="Frekuensi Hujan Ekstrem per Bulan")
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Tidak ada kejadian hujan ekstrem pada rentang ini.")

    fig_scatter = px.scatter(df, x='Tanggal', y='curah_hujan', color='Hujan Ekstrem',
                             title="Curah Hujan vs Waktu (Scatter)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    df['extreme_prob_30d'] = df['extreme_flag'].rolling(window=30, min_periods=1).mean()
    fig_prob = px.line(df, x='Tanggal', y='extreme_prob_30d', title="Rolling 30-day Probability of Extreme Rain")
    st.plotly_chart(fig_prob, use_container_width=True)

with e2:
    st.write("**Kegunaan:**")
    st.write("- Deteksi (> threshold) untuk peringatan dini banjir. Ubah threshold di sidebar.")

# ---------------- 5. Weather Index (composite) ----------------
st.markdown("---")
st.header("5. Prediksi Indeks Cuaca Gabungan (Weather Index Prediction)")
w1, w2 = st.columns([2, 1])
with w1:
    comp = pd.DataFrame({
        'rain': (df['curah_hujan'] - df['curah_hujan'].min()) / (df['curah_hujan'].max() - df['curah_hujan'].min() + 1e-6),
        'temp_stress': (np.maximum(0, np.maximum(24 - df['Tavg'], df['Tavg'] - 28))),
        'hum_stress': (np.maximum(0, np.maximum(40 - df['kelembaban'], df['kelembaban'] - 70))),
        'wind': (df['kecepatan_angin'] - df['kecepatan_angin'].min()) / (df['kecepatan_angin'].max() - df['kecepatan_angin'].min() + 1e-6)
    })
    for col in ['temp_stress', 'hum_stress']:
        comp[col] = (comp[col] - comp[col].min()) / (comp[col].max() - comp[col].min() + 1e-6)
    avg_comp = comp.mean()
    cats = ['Rain', 'TempStress', 'HumStress', 'Wind']
    vals = avg_comp.values.tolist()
    vals += vals[:1]
    fig_radar = go.Figure(data=go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill='toself', name='AvgComponents'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False,
                            title="Weather Index Components (Radar)")
    st.plotly_chart(fig_radar, use_container_width=True)

    fig_index = px.line(df, x='Tanggal', y='WeatherIndex', title="Composite Weather Index Over Time")
    st.plotly_chart(fig_index, use_container_width=True)

with w2:
    st.write("**Kegunaan:**")
    st.write("- Skor 0–1 untuk kondisi iklim keseluruhan; komponen membantu interpretasi.")

# ---------------- 6. Tren Bulanan/Tahunan ----------------
st.markdown("---")
st.header("6. Tren Kualitas Iklim Bulanan/Tahunan")
m1, m2 = st.columns(2)
with m1:
    years = sorted(df['Year'].unique())
    fig_multi = go.Figure()
    for y in years:
        tmp = df[df['Year'] == y].copy()
        monthly = tmp.groupby(tmp['Tanggal'].dt.month)['curah_hujan'].mean().reset_index(name='curah_hujan')
        # month numbers as x
        fig_multi.add_trace(go.Scatter(x=monthly['Tanggal'], y=monthly['curah_hujan'], mode='lines+markers', name=str(y)))
    fig_multi.update_layout(title="Monthly Average Rainfall by Year", xaxis_title="Month", yaxis_title="Rain (mm)")
    st.plotly_chart(fig_multi, use_container_width=True)

with m2:
    df['Rain_MA'] = df['curah_hujan'].rolling(window=ma_window).mean()
    fig_ma = px.line(df, x='Tanggal', y=['curah_hujan', 'Rain_MA'], title=f"Moving Average Rainfall (window={ma_window})")
    st.plotly_chart(fig_ma, use_container_width=True)

# ---------------- 7. Prediksi Anomali Iklim ----------------
st.markdown("---")
st.header("7. Prediksi Anomali Iklim")
a1, a2 = st.columns([2, 1])
with a1:
    baseline_temp = data.groupby(data['Tanggal'].dt.month)['Tavg'].mean()
    df['baseline_Tavg'] = df['Tanggal'].dt.month.map(baseline_temp)
    df['Tavg_anom'] = df['Tavg'] - df['baseline_Tavg']

    anom_pivot = df.pivot_table(index=df['Tanggal'].dt.year, columns=df['Tanggal'].dt.month, values='Tavg_anom', aggfunc='mean')
    if anom_pivot.shape[0] > 0 and anom_pivot.shape[1] > 0:
        fig_anom = px.imshow(anom_pivot, labels=dict(x='Month', y='Year', color='Tavg anomaly'),
                             title="Temperature Anomaly (Year x Month)")
        st.plotly_chart(fig_anom, use_container_width=True)

    fig_anom_line = go.Figure()
    fig_anom_line.add_trace(go.Scatter(x=df['Tanggal'], y=df['Tavg'], mode='lines', name='Tavg'))
    fig_anom_line.add_trace(go.Scatter(x=df['Tanggal'], y=df['baseline_Tavg'], mode='lines', name='Baseline (monthly mean)'))
    fig_anom_line.update_layout(title="Temperature with Baseline Comparison")
    st.plotly_chart(fig_anom_line, use_container_width=True)

with a2:
    st.write("**Kegunaan:**")
    st.write("- Deteksi pergeseran iklim terhadap baseline musiman.")

# ---------------- Data viewer & Export ----------------
st.markdown("---")
with st.expander("📁 Lihat dan Unduh Data Lengkap"):
    st.dataframe(df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Hasil_DSS', index=False)

    # move buffer pointer to start
    buffer.seek(0)

    st.download_button(
        "Unduh Excel Hasil Analisis",
        data=buffer.getvalue(),
        file_name="PAPUABARAT2_hasil_dss.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )








