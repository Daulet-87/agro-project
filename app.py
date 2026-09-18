import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.linear_model import Ridge
import plotly.graph_objects as go

# Настройка интерфейса
st.set_page_config(page_title="ИИ-Aqmola CropIQ", layout="wide")

st.title("🌾 Aqmola CropIQ")
st.write("Моделирование рисков и продуктивности яровой пшеницы в Зерендинском районе.")

# ==========================================
# 1. ФУНКЦИЯ ДЛЯ ЧИСТКИ ДАННЫХ
# ==========================================
def parse_interval(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("+", "").replace("%", "").strip()
    for sep in ["–", "-", "…"]:
        if sep in s:
            parts = s.split(sep)
            try:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            except:
                pass
    try:
        return float(s)
    except:
        return np.nan

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ ИЗ EXCEL
# ==========================================
excel_filename = "zerenda_data.xlsx"

@st.cache_data
def load_data(file_path):
    raw_df = pd.read_excel(file_path)
    clean_df = pd.DataFrame()
    clean_df["Год"] = raw_df["Год"]
    clean_df["Урожайность"] = raw_df["Урожайность пшеницы (ц/га)"].apply(parse_interval)
    clean_df["Осадки_Май"] = raw_df["Осадки Май (мм)"].apply(parse_interval)
    clean_df["Осадки_Июнь"] = raw_df["Осадки Июнь (мм)" if "Осадки Июнь (мм)" in raw_df.columns else raw_df.columns[3]].apply(parse_interval)
    clean_df["Осадки_Июль"] = raw_df["Осадки Июль (мм)"].apply(parse_interval)
    clean_df["Темп_Июль"] = raw_df["Средняя темп. июля (°C)"].apply(parse_interval)
    clean_df["NDVI"] = raw_df["Макс. NDVI (июль)"].apply(parse_interval)
    return clean_df

df = load_data(excel_filename)

# ==========================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
features = ["Осадки_Май", "Осадки_Июнь", "Осадки_Июль", "Темп_Июль", "NDVI"]
X = df[features]
y = df["Урожайность"]
model = Ridge(alpha=1.0)
model.fit(X, y)

# ==========================================
# 4. ЛЕВАЯ ПАНЕЛЬ (САЙДБАР)
# ==========================================
st.sidebar.header("🎛️ Параметры симуляции")

st.sidebar.subheader("🌦️ Прогноз погоды")
input_osadki_may = st.sidebar.slider("Осадки в Мае (мм)", 10.0, 60.0, float(df["Осадки_Май"].mean()), step=0.1)
input_osadki_june = st.sidebar.slider("Осадки в Июне (мм)", 10.0, 80.0, float(df["Осадки_Июнь"].mean()), step=0.1)
input_osadki_july = st.sidebar.slider("Осадки в Июле (мм)", 15.0, 120.0, float(df["Осадки_Июль"].mean()), step=0.1)
input_temp_july = st.sidebar.slider("Средняя темп. июля (°C)", 18.0, 26.0, float(round(df["Темп_Июль"].mean(), 1)), step=0.1)

st.sidebar.subheader("🛰️ Спутниковый мониторинг")
input_ndvi = st.sidebar.slider("Стартовый индекс NDVI (Июль)", 0.20, 0.85, float(round(df["NDVI"].mean(), 2)), step=0.01)

st.sidebar.subheader("🛡️ Меры защиты полей")
tech_drought = st.sidebar.checkbox("Защита от засухи (No-Till)")
tech_flood = st.sidebar.checkbox("Защита от переувлажнения (Фунгициды)")
tech_pest = st.sidebar.checkbox("Защита от вредителей (Инсектициды)")

# Поле для ввода гектаров пашни в левую панель
st.sidebar.subheader("📐 Масштаб хозяйства")
area = st.sidebar.number_input("Площадь посева пшеницы (га):", value=1000, step=100, min_value=1)

# КНОПКА РАСЧЕТА
st.sidebar.markdown("---")
btn_calculate = st.sidebar.button("🚀 РАССЧИТАТЬ ПРОГНОЗ ИИ", type="primary", use_container_width=True)

# ==========================================
# 5. ГЛАВНОЕ ОКНО РАСЧЕТА
# ==========================================
if "calculated" not in st.session_state:
    st.session_state.calculated = False

if btn_calculate:
    st.session_state.calculated = True
    
    # 1. Определение рисков
    is_drought = input_osadki_july < 32.0 or input_temp_july > 22.5
    is_flooded = input_osadki_july > 80.0
    is_pest = input_temp_july > 21.0 and input_osadki_june > 45.0

    # Сценарий БЕЗ защиты (Штрафуем NDVI)
    base_ndvi = input_ndvi
    if is_drought and not tech_drought: base_ndvi -= 0.12
    if is_flooded and not tech_flood: base_ndvi -= 0.10
    if is_pest and not tech_pest: base_ndvi -= 0.15
    base_ndvi = max(0.20, base_ndvi)

    d_base = pd.DataFrame([[input_osadki_may, input_osadki_june, input_osadki_july, input_temp_july, base_ndvi]], columns=features)
    y_base = float(model.predict(d_base)[0])

    # Сценарий С защитой
    adj_july = input_osadki_july + (15.0 if (is_drought and tech_drought) else 0.0)
    adj_temp = input_temp_july - (0.5 if (is_drought and tech_drought) else 0.0)
    
    boosted_ndvi = input_ndvi
    d_boost = pd.DataFrame([[input_osadki_may, input_osadki_june, adj_july, adj_temp, boosted_ndvi]], columns=features)
    y_boost = float(model.predict(d_boost)[0])

    # 🌟 ЖЕЛЕЗНЫЙ МАТЕМАТИЧЕСКИЙ ЭФФЕКТ ДЛЯ КАЖДОЙ ГАЛОЧКИ ПРИ НАЛИЧИИ УГРОЗЫ
    if is_flooded and tech_flood:
        y_boost += 2.10  # Прямая прибавка за спасение от грибка
    if is_pest and tech_pest:
        y_boost += 2.45  # Прямая прибавка за уничтожение вредителей

    # Рамки урожайности
    st.session_state.y_base = max(4.0, min(y_base, 22.0))
    st.session_state.y_boost = max(4.0, min(y_boost, 22.0))
    st.session_state.is_drought = is_drought
    st.session_state.is_flooded = is_flooded
    st.session_state.is_pest = is_pest
    st.session_state.area = area  # Запоминаем площадь для вывода на экран

# Вывод результатов
if st.session_state.calculated:
    yb = st.session_state.y_base
    ybt = st.session_state.y_boost
    current_area = st.session_state.area

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌾 ИИ-Прогноз урожайности пшеницы")
        st.metric(label="С мерами защиты", value=f"{ybt:.2f} ц/га", delta=f"+{ybt - yb:.2f} ц/га")
    with col2:
        st.subheader("📋 Метрика в тоннах с гектара")
        st.metric(label="Эффективность", value=f"{ybt / 10:.3f} тонн / га")

    st.markdown("---")
    st.subheader("🤖 Отчет ИИ-Советника по рискам")
    
    r_cnt = 0
    if st.session_state.is_drought:
        st.error("🔥 **Обнаружен маркер засухи!**")
        if not tech_drought:
            st.warning("👉 *Рекомендация:* Активируйте галочку **'Защита от засухи (No-Till)'** слева.")
            r_cnt += 1
            
    if st.session_state.is_flooded:
        st.info("🌊 **Обнаружен избыток осадков в июле!**")
        if not tech_flood:
            st.warning("👉 *Рекомендация:* Активируйте галочку **'Защита от переувлажнения (Фунгициды)'** слева.")
            r_cnt += 1
            
    if st.session_state.is_pest:
        st.error("🐛 **Высокий риск размножения вредителей!**")
        if not tech_pest:
            st.warning("👉 *Рекомендация:* Активируйте галочку **'Защита от вредителей (Инсектициды)'** слева.")
            r_cnt += 1

    if r_cnt == 0:
        st.success("🎉 Все технологические риски успешно нивелированы защитными мерами! Урожай в безопасности.")

    # КРУГОВАЯ ГРАФИКА PLOTLY
    st.markdown("### 📊 Реализация потенциала урожайности региона")
    max_potential = 20.0
    pct_base = min(100.0, (yb / max_potential) * 100)
    pct_boost = min(100.0, (ybt / max_potential) * 100)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.write("<center><b>Базовый сценарий (Без защиты)</b></center>", unsafe_allow_html=True)
        fig_base = go.Figure(go.Pie(values=[pct_base, 100 - pct_base], hole=.6, marker_colors=['#EF5350', '#E0E0E0'], textinfo='none'))
        fig_base.update_layout(annotations=[dict(text=f'{yb:.2f}<br>ц/га', x=0.5, y=0.5, font_size=20, showarrow=False)], showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_base, use_container_width=True)

    with fig_col2:
        st.write("<center><b>Интенсивный сценарий (С ИИ-Защитой)</b></center>", unsafe_allow_html=True)
        fig_boost = go.Figure(go.Pie(values=[pct_boost, max(0.0, 100 - pct_boost)], hole=.6, marker_colors=['#2E7D32', '#E0E0E0'], textinfo='none'))
        fig_boost.update_layout(annotations=[dict(text=f'{ybt:.2f}<br>ц/га', x=0.5, y=0.5, font_size=20, showarrow=False)], showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_boost, use_container_width=True)

    # ==========================================
    # 📐 НАТУРАЛЬНЫЙ БАЛАНС ХОЗЯЙСТВА (В НИЖНЕЙ ЧАСТИ ЭКРАНА)
    # ==========================================
    st.markdown("---")
    st.subheader("📐 Натуральный баланс производства ТОО")
    
    # 150 кг семян на гектар = 0.15 тонн / га
    total_seeds_tons = current_area * 0.15
    
    # Расчет валового сбора (ц/га переводим в тонны/га и умножаем на гектары)
    total_harvest_tons_base = (yb / 10) * current_area
    total_harvest_tons_boosted = (ybt / 10) * current_area
    saved_tons = total_harvest_tons_boosted - total_harvest_tons_base

    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        st.info(f"🚜 **Посевные площади:**\n\n Введено: **{current_area:,.0f} гектаров**")
    with col_b2:
        st.warning(f"📦 **Расход семенного материала:**\n\n Требуется засыпать: **{total_seeds_tons:,.1f} тонн семян** *(норма 150 кг/га)*")
    with col_b3:
        st.success(f"🌾 **Ожидаемый валовый сбор зерна:**\n\n Будет собрано: **{total_harvest_tons_boosted:,.1f} тонн** *(сохранено от рисков: +{saved_tons:,.1f} тонн)*")

else:
    st.info("👈 Выставите параметры погоды, NDVI и масштаб полей, выберите технологии защиты и нажмите кнопку **«РАССЧИТАТЬ ПРОГНОЗ ИИ»** в левой панели.")

st.markdown("---")
st.subheader("📜 База данных Зерендинского района")
st.dataframe(df.style.format("{:.2f}", subset=features + ["Урожайность"]))
