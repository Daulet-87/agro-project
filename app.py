import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.linear_model import Ridge

# Настройка интерфейса Streamlit
st.set_page_config(
    page_title="ИИ-АгроЩит Зеренда", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 ИИ-Система «АгроЩит» — Прогнозирование баланса урожая")
st.write("Моделирование физических объемов производства яровой пшеницы в Зерендинском районе.")

# ==========================================
# 1. ФУНКЦИЯ ДЛЯ АВТОМАТИЧЕСКОЙ ОЧИСТКИ ДАННЫХ
# ==========================================
def parse_interval(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("+", "").replace("%", "").strip()
    parts = re.split(r"–|-|…", val_str)
    try:
        nums = []
        for p in parts:
            if p.strip():
                nums.append(float(p.strip()))
        if len(nums) == 2:
            return sum(nums) / 2
        elif len(nums) == 1:
            return nums[0]
    except Exception:
        pass
    return np.nan

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ ИЗ EXCEL
# ==========================================
excel_filename = "zerenda_data.xlsx"

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

df, error_msg = load_and_preprocess_data(excel_filename)

if error_msg:
    st.error(f"❌ Ошибка загрузки '{excel_filename}'. Проверьте файл. Техническая ошибка: {error_msg}")
    st.stop()

# ==========================================
# 3. ОБУЧЕНИЕ ИИ-МОДЕЛИ (Ridge)
# ==========================================
features = ["Осадки_Май", "Осадки_Июнь", "Осадки_Июль", "Темп_Июль", "NDVI"]

def train_model(dataframe):
    X = dataframe[features]
    y = dataframe["Урожайность"]
    clf = Ridge(alpha=1.0)
    clf.fit(X, y)
    return clf

model = train_model(df)

# ==========================================
# 4. ЛЕВАЯ ПАНЕЛЬ (САЙДБАР) — ВСЕ НАСТРОЙКИ И КНОПКА ТУТ
# ==========================================
st.sidebar.header("🎛️ Параметры симуляции")

st.sidebar.subheader("🌦️ Прогноз погоды")
input_osadki_may = st.sidebar.slider("Осадки в Мае (мм)", 10, 60, int(df["Осадки_Май"].mean()))
input_osadki_june = st.sidebar.slider("Осадки в Июне (мм)", 10, 80, int(df["Осадки_Июнь"].mean()))
input_osadki_july = st.sidebar.slider("Осадки в Июле (мм)", 15, 120, int(df["Осадки_Июль"].mean()))
input_temp_july = st.sidebar.slider("Средняя темп. июля (°C)", 18.0, 26.0, float(round(df["Темп_Июль"].mean(), 1)), 0.1)

st.sidebar.subheader("🛡️ Меры защиты полей")
tech_drought = st.sidebar.checkbox("Защита от засухи (No-Till + Антистрессанты)")
tech_flood = st.sidebar.checkbox("Защита от переувлажнения (Фунгициды + Десикация)")
tech_pest = st.sidebar.checkbox("Защита от вредителей (Инсектициды)")

st.sidebar.subheader("📐 Масштаб ТОО")
area = st.sidebar.number_input("Площадь пашни хозяйства (га):", value=1000, step=100)

# ГЛАВНАЯ КНОПКА В САЙДБАРЕ
st.sidebar.markdown("---")
btn_calculate = st.sidebar.button("🚀 РАССЧИТАТЬ ПРОГНОЗ ИИ", type="primary", use_container_width=True)

# ==========================================
# 5. ИНТЕРФЕЙС ГЛАВНОГО ОКНА (ВКЛАДКИ)
# ==========================================
tab1, tab2 = st.tabs(["📊 ИИ-Анализ и Баланс Урожая", "🗃️ Управление исторической базой"])

# СОСТОЯНИЕ ДЛЯ ХРАНЕНИЯ РЕЗУЛЬТАТОВ РАСЧЕТА В СЕССИИ
if "calculated" not in st.session_state:
    st.session_state.calculated = False

if btn_calculate:
    st.session_state.calculated = True
    
    # Логика моделирования рисков
    is_drought = input_osadki_july < 32 or input_temp_july > 22.5
    is_flooded = input_osadki_july > 80
    is_pest_danger = input_temp_july > 21.0 and input_osadki_june > 45

    base_ndvi = float(round(df["NDVI"].mean(), 2))

    # Снижение NDVI без защиты
    if is_drought and not tech_drought: base_ndvi -= 0.12
    if is_flooded and not tech_flood: base_ndvi -= 0.10
    if is_pest_danger and not tech_pest: base_ndvi -= 0.15
    base_ndvi = max(0.20, base_ndvi)

    # Расчет базового сценария (без защиты)
    data_base = pd.DataFrame([[input_osadki_may, input_osadki_june, input_osadki_july, input_temp_july, base_ndvi]], columns=features)
    yield_base = max(4.0, min(float(model.predict(data_base)[0]), 22.0))

    # Расчет адаптивного сценария (с защитой)
    adj_osadki_july = input_osadki_july + (15.0 if tech_drought else 0.0)
    adj_temp_july = input_temp_july - (0.5 if tech_drought else 0.0)
    boosted_ndvi = float(round(df["NDVI"].mean(), 2)) + (0.05 if (tech_flood or tech_pest) else 0.0)
    boosted_ndvi = min(0.85, boosted_ndvi)

    data_boosted = pd.DataFrame([[input_osadki_may, input_osadki_june, adj_osadki_july, adj_temp_july, boosted_ndvi]], columns=features)
    yield_boosted = max(4.0, min(float(model.predict(data_boosted)[0]), 22.0))

    # Сохраняем расчеты в сессию
    st.session_state.yield_base = yield_base
    st.session_state.yield_boosted = yield_boosted
    st.session_state.is_drought = is_drought
    st.session_state.is_flooded = is_flooded
    st.session_state.is_pest_danger = is_pest_danger

# ВКЛАДКА 1: ВЫВОД РЕЗУЛЬТАТОВ РАСЧЕТА
with tab1:
    if st.session_state.calculated:
        y_base = st.session_state.yield_base
        y_boosted = st.session_state.yield_boosted
        
        # Расчет посевного и уборочного баланса
        seed_rate = 0.15  # 150 кг на га = 0.15 тонн на га
        total_seeds = area * seed_rate
        
        yield_ton_per_ha_base = y_base / 10
        yield_ton_per_ha_boosted = y_boosted / 10
        
        total_harvest_base = yield_ton_per_ha_base * area
        total_harvest_boosted = yield_ton_per_ha_boosted * area

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("🌾 Прогноз продуктивности с 1 гектара")
            st.metric(label="Урожайность (С мерами защиты)", value=f"{y_boosted:.2f} ц/га", delta=f"+{y_boosted - y_base:.2f} ц/га")
            st.info(f"📋 В весовом эквиваленте: **{yield_ton_per_ha_boosted:.3f} тонн / га**")

        with col_m2:
            st.subheader("📐 Натуральный баланс производства ТОО")
            st.warning(f"🚜 **Необходимый объем семян для посева:** {total_seeds:,.1f} тонн семян")
            st.success(f"🌾 **Ожидаемый валовый сбор урожая:** {total_harvest_boosted:,.1f} тонн чистого зерна")

        # Информационные агро-рекомендации ИИ
        st.markdown("---")
        st.subheader("🤖 Отчет ИИ-Советника об угрозах")
        
        recs_count = 0
        if st.session_state.is_drought:
            st.error("🔥 **Зафиксирован критический маркер засухи!**")
            if not tech_drought: 
                st.warning("⚠️ Поле не защищено. Требуется активировать галочку 'Защита от засухи' слева для сохранения влаги.")
                recs_count += 1
        if st.session_state.is_flooded:
            st.info("🌊 **Зафиксирован избыток осадков в июле!**")
            if not tech_flood: 
                st.warning("⚠️ Высокий риск грибка и неравномерного созревания. Активируйте галочку 'Защита от переувлажнения'.")
                recs_count += 1
        if st.session_state.is_pest_danger:
            st.error("🐛 **Погода благоприятна для размножения вредителей!**")
            if not tech_pest: 
                st.warning("⚠️ Обнаружен риск потери биомассы. Активируйте галочку 'Защита от вредителей'.")
                recs_count += 1

        if recs_count == 0:
            st.success("🎉 Все технологические риски успешно нивелированы защитными мерами. Потенциал сбора максимальный!")
            
        # Натуральный график сравнения сбора
        st.markdown("### 📊 Сравнение сбора зерна холдинга (Тонн)")
        chart_df = pd.DataFrame({
            'Сценарий': ['Без защиты рисков', 'С ИИ-Агрозащитой'],
            'Валовый сбор (Тонн)': [total_harvest_base, total_harvest_boosted]
        })
        st.bar_chart(chart_df.set_index('Сценарий'))
        
    else:
        st.info("👈 Выставите параметры погоды, выберите технологии защиты и нажмите кнопку **«РАССЧИТАТЬ ПРОГНОЗ ИИ»** в левой панели для получения аналитики.")

# ВКЛАДКА 2: ВНЕСЕНИЕ НОВЫХ ДАННЫХ И ИСТОРИЯ
with tab2:
    st.subheader("➕ Добавление результатов прошедшего года")
    st.write("Заполните показатели ушедшего сезона, чтобы занести их в Excel и обновить ИИ-модель:")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        new_year = st.number_input("Год:", min_value=2026, max_value=2040, value=2026)
        new_yield = st.text_input("Урожайность пшеницы (ц/га):", value="12.0 – 13.5")
    with col_in2:
        new_may = st.text_input("Осадки Май (мм):", value="25 – 30")
        new_june = st.text_input("Осадки Июнь (мм):", value="45 – 50")
    with col_in3:
        new_july = st.text_input("Осадки Июль (мм):", value="60 – 65")
        new_temp = st.text_input("Средняя темп. июля (°C):", value="+20.0…+20.5")
        
    col_in4, col_in5 = st.columns(2)
    with col_in4:
        new_ndvi = st.text_input("Макс. NDVI (июль):", value="0.55 – 0.60")
    with col_in5:
        st.write("###")
        btn_add_data = st.button("💾 Записать в Excel и переобучить модель", use_container_width=True)

    if btn_add_data:
        try:
            current_excel = pd.read_excel(excel_filename)
            
            new_row_dict = {
