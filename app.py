import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Настройка интерфейса Streamlit
st.set_page_config(
    page_title="ИИ-АгроЩит Акмолинской области", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 ИИ-Система «АгроЩит» — Прогнозирование урожайности и комплексное управление рисками")
st.write("**Регион анализа:** Акмолинская область, Зерендинский район. Моделирование рисков на следующий (2026) год.")

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
            return parts
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
    st.error(f"❌ Ошибка загрузки '{excel_filename}'. Техническая ошибка: {error_msg}")
    st.stop()

# ==========================================
# 3. ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
features = ["Осадки_Май", "Осадки_Июнь", "Осадки_Июль", "Темп_Июль", "NDVI"]
X = df[features]
y = df["Урожайность"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.success(f"📊 База данных обновлена! Обучено на реальной истории за {len(df)} лет.")

# ==========================================
# 4. БОКОВАЯ ПАНЕЛЬ (САЙДБАР) — ВСЕ РИСКИ И ГАЛОЧКИ
# ==========================================
st.sidebar.header("🎛️ Параметры сезона")

st.sidebar.subheader("🌦️ Прогноз погоды")
input_osadki_may = st.sidebar.slider("Осадки в Мае (мм)", 10, 60, int(df["Осадки_Май"].mean()))
input_osadki_june = st.sidebar.slider("Осадки в Июне (мм)", 10, 80, int(df["Осадки_Июнь"].mean()))
input_osadki_july = st.sidebar.slider("Осадки в Июле (мм)", 15, 120, int(df["Осадки_Июль"].mean()))
input_temp_july = st.sidebar.slider("Средняя темп. июля (°C)", 18.0, 26.0, float(round(df["Темп_Июль"].mean(), 1)), 0.1)

st.sidebar.subheader("💰 Бизнес-параметры")
area = st.sidebar.number_input("Площадь хозяйства (га):", value=1000, step=100)
price_per_ton = st.sidebar.number_input("Цена за тонну пшеницы (₸):", value=90000, step=5000)

# ДИНАМИЧЕСКИЕ ТЕХНОЛОГИЧЕСКИЕ ГАЛОЧКИ
st.sidebar.subheader("🛡️ Управление агроугрозами")

# Определяем типы рисков на лету для удобства пользователя
drought_risk = input_osadki_july < 30 or input_temp_july > 22.0
flood_risk = input_osadki_july > 80  # Избыток влаги
pest_risk = input_temp_july > 21.0 and input_osadki_june > 45  # Идеально для размножения вредителей

# Показываем галочки в зависимости от того, какую погоду выставил фермер
tech_notill = False
tech_antistress = False
tech_fungicide = False
tech_desiccation = False
tech_insecticide = False

if drought_risk:
    st.sidebar.warning("🚨 Прогноз указывает на ЗАСУХУ")
    tech_notill = st.sidebar.checkbox("Внедрить No-Till технологию")
    tech_antistress = st.sidebar.checkbox("Внести листовые антистрессанты")

if flood_risk:
    st.sidebar.warning("🌊 Прогноз указывает на ИЗБЫТОК ВЛАГИ")
    tech_fungicide = st.sidebar.checkbox("Внести фунгициды (защита от грибка/ржавчины)")
    tech_desiccation = st.sidebar.checkbox("Запланировать десикацию (сушку колоса на корню перед уборкой)")

if pest_risk:
    st.sidebar.warning("🐛 Высокий риск НАШЕСТВИЯ ВРЕДИТЕЛЕЙ")
    tech_insecticide = st.sidebar.checkbox("Провести инсектицидную обработку полей")

# Подсчет дополнительных затрат на гектар исходя из включенных мер защиты
protection_cost_per_ha = 0
if tech_notill: protection_cost_per_ha += 8000
if tech_antistress: protection_cost_per_ha += 6000
if tech_fungicide: protection_cost_per_ha += 12000
if tech_desiccation: protection_cost_per_ha += 9000
if tech_insecticide: protection_cost_per_ha += 7000

st.markdown("---")
st.header("💵 Симулятор затрат и ИИ-Аналитика рисков")
base_cost_per_ha = st.slider("Базовые затраты на стандартные удобрения (₸ на 1 гектар):", 0, 40000, 15000, 1000)

total_cost_per_ha = base_cost_per_ha + protection_cost_per_ha
st.write(f"💵 **Итоговые операционные затраты составят:** {total_cost_per_ha:,.0f} ₸ / га (из них на защиту: {protection_cost_per_ha:,.0f} ₸)")

# ==========================================
# 5. ВЫЧИСЛЕНИЯ ПРИ НАЖАТИИ КНОПКИ
# ==========================================
st.markdown("###")
if st.button("🚀 ЗАПУСТИТЬ КОМПЛЕКСНЫЙ ИИ-АНАЛИЗ СЕЗОНА", type="primary", use_container_width=True):
    with st.spinner("ИИ просчитывает фитоклиматические сценарии..."):
        
        # 1. Базовый сценарий (без мер защиты)
        base_ndvi = float(round(df["NDVI"].mean(), 2))
        
        # Если фермер ничего не делает в плохую погоду, NDVI падает из-за болезней/засухи/вредителей
        if drought_risk and not (tech_notill or tech_antistress): base_ndvi -= 0.15
        if flood_risk and not tech_fungicide: base_ndvi -= 0.12  # Ржавчина съедает биомассу
        if pest_risk and not tech_insecticide: base_ndvi -= 0.18  # Вредители съедают листья
        base_ndvi = max(0.20, base_ndvi)

        data_base = pd.DataFrame([[input_osadki_may, input_osadki_june, input_osadki_july, input_temp_july, base_ndvi]], columns=features)
        yield_base = float(model.predict(data_base)[0])

        # 2. Адаптивный сценарий (с защитой)
        boosted_ndvi = float(round(df["NDVI"].mean(), 2)) + (base_cost_per_ha / 10000) * 0.08
        
        # Корректируем виртуальные маркеры для модели, если галочки активированы
        adj_osadki_july = input_osadki_july
        adj_temp_july = input_temp_july
        
        if tech_notill: adj_osadki_july += 15.0
        if tech_antistress: adj_temp_july -= 0.5
        if tech_fungicide or tech_insecticide: boosted_ndvi += 0.05
        boosted_ndvi = min(0.85, boosted_ndvi)

        data_boosted = pd.DataFrame([[input_osadki_may, input_osadki_june, adj_osadki_july, adj_temp_july, boosted_ndvi]], columns=features)
        yield_boosted = float(model.predict(data_boosted)[0])

        # Финансовые расчеты
        total_investment = total_cost_per_ha * area
        revenue_base = ((yield_base * area) / 10) * price_per_ton
        revenue_boosted = ((yield_boosted * area) / 10) * price_per_ton
        net_profit = (revenue_boosted - revenue_base) - (protection_cost_per_ha * area)
        roi = (net_profit / total_investment * 100) if total_investment > 0 else 0.0

        # Корректировка цены из-за потери качества зерна, если была влага, но нет фунгицидов
        quality_loss_text = ""
        if flood_risk and not tech_fungicide:
            revenue_base *= 0.8  # Минус 20% цены из-за падения класса до фуража
            quality_loss_text = "⚠️ *Внимание: Цена базового сценария снижена на 20%, так как без фунгицидов зерно прорастет и заразится грибком (фураж).* "

        # Вывод экономических метрик
        st.markdown("### 📊 Аналитический отчет ИИ")
        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.metric(
                label="Прогноз урожайности (С защитой)", 
                value=f"{yield_boosted:.2f} ц/га",
                delta=f"+{yield_boosted - yield_base:.2f} ц/га к базовому сценарию"
            )
        with metric_2:
            st.metric(label="Общий бюджет сезона", value=f"{total_investment:,.0f} ₸", delta=f"Затраты на защиту: {protection_cost_per_ha * area:,.0f} ₸", delta_color="inverse")
        with metric_3:
            if net_profit > 0:
                st.metric(label="Чистая прибыль от защитных мер", value=f"+{net_profit:,.0f} ₸", delta=f"Общий ROI: {roi:.1f}%")
            else:
                st.metric(label="Чистая прибыль от защитных мер", value=f"{net_profit:,.0f} ₸", delta=f"Общий ROI: {roi:.1f}%", delta_color="inverse")

        if quality_loss_text:
            st.warning(quality_loss_text)

        # ==========================================
        # МОДУЛЬ ЭКСПЕРТНЫХ СОВЕТОВ (РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА)
        # ==========================================
        st.markdown("---")
        st.subheader("🤖 Персонализированные рекомендации ИИ-Советника")
        
        recs = []
        if drought_risk:
            st.error("🔥 **КРИТИЧЕСКИЙ РИСК ЗАСУХИ!**")
            if not tech_notill: recs.append("💡 Внедрите технологию **No-Till** для снижения испарения влаги из почвы.")
            if not tech_antistress: recs.append("💡 Внесите **аминокислотные антистрессанты** по листу, чтобы защитить растение от температурного шока.")
        
        if flood_risk:
            st.info("🌊 **РИСК ПЕРЕУВЛАЖНЕНИЯ И ПОТЕРИ КАЧЕСТВА УРОЖАЯ!**")
            if not tech_fungicide: recs.append("💡 **Срочно внесите фунгициды** (Триазолы или Стробилурины). При избытке влаги пшеница мгновенно заболеет ржавчиной, что снизит класс зерна до некондиционного.")
