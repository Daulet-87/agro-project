import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.linear_model import Ridge

# Настройка интерфейса Streamlit
st.set_page_config(
    page_title="АгроПрогноз & ROI Симулятор", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 ИИ-Система прогнозирования урожайности и окупаемости инвестиций")
st.write("**Регион анализа:** Акмолинская область, Зерендинский район. Модель обучена на исторических данных (2015-2025 гг.).")

# ==========================================
# 1. ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОЙ ОЧИСТКИ ДАННЫХ
# ==========================================
def parse_interval(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("+", "").replace("%", "").strip()
    parts = re.split(r"–|-|…", val_str)
    try:
        parts = [float(p.strip()) for p in parts if p.strip()]
        if len(parts) == 2:
            return sum(parts) / 2
        elif len(parts) == 1:
            return parts[0]
    except ValueError:
        pass
    return np.nan

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ ИЗ EXCEL
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        raw_df = pd.read_excel(file_path)
        clean_df = pd.DataFrame()
        clean_df["Год"] = raw_df["Год"]
        clean_df["Урожайность"] = raw_df["Урожайность пшеницы (ц/га)"].apply(parse_interval)
        clean_df["Осадки_Май"] = raw_df["Осадки Май (мм)"].apply(parse_interval)
        clean_df["Осадки_Июнь"] = raw_df["Осадки Июнь (мм)"].apply(parse_interval)
        clean_df["Осадки_Июль"] = raw_df["Осадки Июль (мм)"].apply(parse_interval)
        clean_df["Темп_Июль"] = raw_df["Средняя темп. июля (°C)"].apply(parse_interval)
        clean_df["NDVI"] = raw_df["Макс. NDVI (июль)"].apply(parse_interval)
        return clean_df, None
    except Exception as e:
        return None, str(e)

# Чтение файла
excel_filename = "zerenda_data.xlsx"
df, error_msg = load_and_preprocess_data(excel_filename)

if error_msg:
    st.error(f"❌ Ошибка загрузки '{excel_filename}'. Проверьте файл. Техническая ошибка: {error_msg}")
    st.stop()

# ==========================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
features = ["Осадки_Май", "Осадки_Июнь", "Осадки_Июль", "Темп_Июль", "NDVI"]
X = df[features]
y = df["Урожайность"]

model = Ridge(alpha=1.0)
model.fit(X, y)

st.success(f"📊 База данных успешно обновлена! Модель обучена на истории Зерендинского района за {len(df)} лет.")

# ==========================================
# 4. БОКОВАЯ ПАНЕЛЬ (САЙДБАР) — ПОЛЗУНКИ УПРАВЛЕНИЯ
# ==========================================
st.sidebar.header("🎛️ Элементы управления")

st.sidebar.subheader("🌦️ Прогноз погоды на сезон")
input_osadki_may = st.sidebar.slider("Осадки в Мае (мм)", 10, 60, int(df["Осадки_Май"].mean()))
input_osadki_june = st.sidebar.slider("Осадки в Июне (мм)", 10, 80, int(df["Осадки_Июнь"].mean()))
input_osadki_july = st.sidebar.slider("Осадки в Июле (мм)", 15, 100, int(df["Осадки_Июль"].mean()))
input_temp_july = st.sidebar.slider("Средняя темп. июля (°C)", 18.0, 26.0, float(round(df["Темп_Июль"].mean(), 1)), 0.1)

st.sidebar.subheader("💰 Бизнес-параметры хозяйства")
area = st.sidebar.number_input("Площадь пашни хозяйства (га):", value=1000, step=100)
price_per_ton = st.sidebar.number_input("Рыночная цена пшеницы (₸ / тонна):", value=90000, step=5000)

# ==========================================
# 5. ГЛАВНЫЙ ЭКРАН: ИНТЕРАКТИВНЫЙ БЛОК ИНВЕСТИЦИЙ (ROI)
# ==========================================
st.markdown("---")
st.header("💵 Симулятор инвестиций в урожайность (Минеральные удобрения)")
st.write(
    "Здесь бизнесмен может рассчитать, выгодно ли вносить удобрения. "
    "Внесение фосфорно-азотных удобрений увеличивает густоту и здоровье посевов, что напрямую поднимает индекс **NDVI**."
)

# Интерактивный выбор инвестиционного плана
col_inv1, col_inv2 = st.columns(2)

with col_inv1:
    cost_per_hectare = st.slider(
        "Затраты на покупку и внесение удобрений (₸ на 1 гектар):", 
        min_value=0, 
        max_value=40000, 
        value=15000, 
        step=1000
    )

with col_inv2:
    # Конвертируем затраты в потенциальный прирост NDVI (базовое агрономическое допущение)
    # Например, каждые 10 000 тенге на га увеличивают NDVI на ~0.08
    expected_ndvi_boost = (cost_per_hectare / 10000) * 0.08
    st.info(f"📈 Ожидаемый прирост индекса здоровья поля (NDVI) составит: **+{expected_ndvi_boost:.3f}** к базовому значению.")

# Расчет двух сценариев: БЕЗ ТЕХНОЛОГИЙ (Базовый NDVI) и С УДОБРЕНИЯМИ (Повышенный NDVI)
base_ndvi = float(round(df["NDVI"].mean(), 2))
boosted_ndvi = min(0.85, base_ndvi + expected_ndvi_boost)

# Базовый прогноз
data_base = pd.DataFrame([[input_osadki_may, input_osadki_june, input_osadki_july, input_temp_july, base_ndvi]], columns=features)
yield_base = max(4.0, model.predict(data_base)[0])

# Прогноз с удобрениями
data_boosted = pd.DataFrame([[input_osadki_may, input_osadki_june, input_osadki_july, input_temp_july, boosted_ndvi]], columns=features)
yield_boosted = max(4.0, model.predict(data_boosted)[0])

# Экономические расчеты
total_cost = cost_per_hectare * area

tons_base = (yield_base * area) / 10
revenue_base = tons_base * price_per_ton

tons_boosted = (yield_boosted * area) / 10
revenue_boosted = tons_boosted * price_per_ton

additional_revenue = max(0.0, revenue_boosted - revenue_base)
net_profit = additional_revenue - total_cost

# Определение ROI (окупаемости)
roi = (net_profit / total_cost * 100) if total_cost > 0 else 0.0

# Вывод экономических метрик
st.markdown("### 📊 Финансовый результат апгрейда технологии")
metric_1, metric_2, metric_3 = st.columns(3)

with metric_1:
    st.metric(
        label="Прирост урожайности", 
        value=f"+{yield_boosted - yield_base:.2f} ц/га",
        delta=f"Итого: {yield_boosted:.2f} ц/га"
    )
with metric_2:
    st.metric(
        label="Общие затраты на удобрения", 
        value=f"{total_cost:,.0f} ₸",
        delta=f"{cost_per_hectare:,.0f} ₸ / га",
        delta_color="inverse"
    )
with metric_3:
    if net_profit > 0:
        st.metric(
            label="Чистая доп. прибыль (ROI)", 
            value=f"+{net_profit:,.0f} ₸", 
            delta=f"Окупаемость: {roi:.1f}%"
        )
    else:
        st.metric(
            label="Чистая доп. прибыль (ROI)", 
            value=f"{net_profit:,.0f} ₸", 
            delta=f"Убыток! ROI: {roi:.1f}%",
            delta_color="inverse"
        )

# ==========================================
# 6. СТРОИМ ИНТЕРАКТИВНЫЙ БИЗНЕС-ГРАФИК
# ==========================================
st.markdown("### 📈 Сравнение финансовых показателей")

# Подготовка данных для столбчатой диаграммы
chart_data = pd.DataFrame({
    'Категория': ['Без удобрений (Базовый)', 'С удобрениями (Интенсивный)'],
    'Выручка с урожая (₸)': [revenue_base, revenue_boosted],
    'Расходы на удобрения (₸)': [0, total_cost]
})

st.bar_chart(chart_data.set_index('Категория'))

# ==========================================
# 7. ТЕХНИЧЕСКИЙ АНАЛИЗ (ДЛЯ ЖЮРИ)
# ==========================================
st.markdown("---")
with st.expander("🛠️ Посмотреть исторические данные и структуру моделей (Инструмент дата-аналитика)"):
    col_table, col_trend = st.columns(2)
    with col_table:
        st.write("**Данные из Excel после автоматической предобработки:**")
        st.dataframe(df.style.format("{:.2f}", subset=features + ["Урожайность"]))
    with col_trend:
        st.write("**Исторический тренд урожайности в Зеренде (ц/га):**")
        st.line_chart(df.set_index("Год")["Урожайность"])
