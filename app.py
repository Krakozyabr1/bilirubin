import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Оптика шкіри: TcB")

# ==========================================
# 1. ФІЗИЧНА БАЗА ТА МОДЕЛЮВАННЯ ШЛЯХІВ
# ==========================================
@st.cache_data
def generate_photon_paths(n_photons=10000):
    """
    Генерація часткових шляхів (в мм) для 3-х довжин хвиль.
    Використовуємо Гамма-розподіл для імітації "хвостів" Монте-Карло.
    - Синій (460): малий шлях в дермі, застрягає в епідермісі.
    - Зелений (550): середнє проникнення.
    - Червоний (650): глибоке проникнення (великий шлях).
    """
    np.random.seed(42)
    paths = {
        'blue':  {'L_epi': np.random.gamma(2, 0.05, n_photons), 'L_der': np.random.gamma(2, 0.05, n_photons)},
        'green': {'L_epi': np.random.gamma(2, 0.06, n_photons), 'L_der': np.random.gamma(2, 0.10, n_photons)},
        'red':   {'L_epi': np.random.gamma(2, 0.08, n_photons), 'L_der': np.random.gamma(2, 0.30, n_photons)}
    }
    return paths

paths = generate_photon_paths()

# Наближені відносні коефіцієнти екстинкції (eps) з OMLC
# Порядок: [Bilirubin, Hemoglobin, Melanin]
eps = {
    'blue':  np.array([0.01241, 0.48458, 0.09210]),  # Білірубін сильно поглинає синій
    'green': np.array([0.00001, 0.68850, 0.04700]),  # Гемоглобін домінує
    'red':   np.array([0.00002, 0.02940, 0.02640])   # Білірубін прозорий, впливає в основному меланін
}

def calculate_absorbance(C_bil, C_hb, C_mel, color):
    """Розрахунок абсорбції за методом White Monte Carlo"""
    # mu_a для епідермісу (тільки меланін) і дерми (білірубін + гемоглобін)
    mu_a_epi = C_mel * eps[color][2]
    mu_a_der = C_bil * eps[color][0] + C_hb * eps[color][1]
    
    # Виживаність фотонів: I_out = Сума( exp(-mu_a * L) )
    survival_probs = np.exp(-(mu_a_epi * paths[color]['L_epi'] + mu_a_der * paths[color]['L_der']))
    I_out = np.sum(survival_probs)
    I_in = len(paths[color]['L_epi'])
    
    # Оптична щільність (Абсорбція)
    return -np.log10(I_out / I_in)

def non_linear_model(X, w0, w1, w2, w3, w4, w5):
    Ab = X[:, 0]
    Ag = X[:, 1]

    return w0 + w1 * Ab + w2 * Ag + w3 * (Ab**2) + w4 * (Ag**2) + w5 * (Ab * Ag)

# ==========================================
# 2. КАЛІБРУВАННЯ ПРИЛАДУ
# ==========================================
@st.cache_data
def calibrate_device(calibration_mode):
    """
    Створюємо 500 'віртуальних немовлят' з випадковими концентраціями
    для розрахунку матриць деконволюції (вагових коефіцієнтів).
    """
    np.random.seed(101)
    n_samples = 500
    C_bils = np.random.uniform(0.1, 2.0, n_samples)
    C_hbs  = np.random.uniform(0.5, 1.5, n_samples)

    if calibration_mode == "Світла шкіра (Тип I-II)":
        C_mels = np.random.uniform(0.1, 0.3, n_samples)
    elif calibration_mode == "Середня шкіра (Тип III-IV)":
        C_mels = np.random.uniform(0.4, 0.7, n_samples)
    elif calibration_mode == "Темна шкіра (Тип V-VI)":
        C_mels = np.random.uniform(0.8, 1.2, n_samples)
    else:
        C_mels = np.random.uniform(0.1, 1.2, n_samples)
    
    A_B, A_G, A_R = [], [], []
    for b, h, m in zip(C_bils, C_hbs, C_mels):
        A_B.append(calculate_absorbance(b, h, m, 'blue'))
        A_G.append(calculate_absorbance(b, h, m, 'green'))
        A_R.append(calculate_absorbance(b, h, m, 'red'))
        
    A_matrix_3w = np.column_stack((A_B, A_G, A_R, np.ones(n_samples)))
    A_matrix_2w = np.column_stack((A_B, A_G, np.ones(n_samples)))
    
    # Метод найменших квадратів (Псевдообернена матриця)
    weights_3w, _, _, _ = np.linalg.lstsq(A_matrix_3w, C_bils, rcond=None)
    weights_2w, _, _, _ = np.linalg.lstsq(A_matrix_2w, C_bils, rcond=None)
    
    return weights_2w, weights_3w


# ==========================================
# 3. ІНТЕРФЕЙС ТА ЛОГІКА
# ==========================================
st.title("Візуалізація алгоритмів транскутанної білірубінометрії")

st.sidebar.header("Калібрування")
calib_mode = st.sidebar.selectbox(
    "Вибірка пацієнтів для калібрування 2-хвильового приладу:",
    [
        "Всі раси (Універсально)", 
        "Світла шкіра (Тип I-II)", 
        "Середня шкіра (Тип III-IV)", 
        "Темна шкіра (Тип V-VI)"
    ]
)

w2, w3 = calibrate_device(calib_mode)

st.sidebar.header("Параметри пацієнта (In-vivo)")
c_bil_ui = st.sidebar.slider("Концентрація Білірубіну (мкмоль/л)", 0, 300, 100, 5)
c_hb_ui  = st.sidebar.slider("Гемоглобін у тканині (г/л)", 1.0, 10.0, 2.5, 0.1)
c_mel_ui = st.sidebar.slider("Меланін (%, об'ємна частка)", 1.0, 20.0, 3.0, 0.5)

# ==========================================
# МАСШТАБУВАННЯ ДЛЯ МАТЕМАТИКИ
# ==========================================
c_bil_real = c_bil_ui / 120.0  # 120 мкмоль/л = 1.0 абстрактної одиниці
c_hb_real  = c_hb_ui / 2.5     # 2.5 г/л = 1.0 абстрактної одиниці
c_mel_real = c_mel_ui / 10.0   # 10% = 1.0 абстрактної одиниці (тому 3% стане 0.3)

# Розрахунок поточних абсорбцій
A_blue = calculate_absorbance(c_bil_real, c_hb_real, c_mel_real, 'blue')
A_green = calculate_absorbance(c_bil_real, c_hb_real, c_mel_real, 'green')
A_red = calculate_absorbance(c_bil_real, c_hb_real, c_mel_real, 'red')

# Розрахунок приладом
c_bil_2w = w2[0]*A_blue + w2[1]*A_green + w2[2]
c_bil_3w = w3[0]*A_blue + w3[1]*A_green + w3[2]*A_red + w3[3]

err_2w = abs(c_bil_2w - c_bil_real) / (c_bil_real + 1e-9) * 100
err_3w = abs(c_bil_3w - c_bil_real) / (c_bil_real + 1e-9) * 100

# Виведення метрик
col1, col2, col3 = st.columns(3)
col1.metric("Реальна концентрація", f"{c_bil_real*120.0:.2f}")
col2.metric("2-хвильовий метод (Синій+Зелений)", f"{c_bil_2w*120.0:.2f}", f"Похибка: {err_2w:.1f}%", delta_color="inverse")
col3.metric("3-хвильовий метод (+Червоний)", f"{c_bil_3w*120.0:.2f}", f"Похибка: {err_3w:.1f}%", delta_color="inverse")

# ==========================================
# 4. ГРАФІКИ
# ==========================================
st.markdown("### Аналіз точності в залежності від рівня білірубіну")

bil_range = np.linspace(0, 2.5, 20)
est_2w_list, est_3w_list = [], []

for b_val in bil_range:
    ab = calculate_absorbance(b_val, c_hb_real, c_mel_real, 'blue')
    ag = calculate_absorbance(b_val, c_hb_real, c_mel_real, 'green')
    ar = calculate_absorbance(b_val, c_hb_real, c_mel_real, 'red')
    
    est_2w_list.append(w2[0]*ab + w2[1]*ag + w2[2])

    est_3w_list.append(w3[0]*ab + w3[1]*ag + w3[2]*ar + w3[3])

est_2w_list = np.array(est_2w_list)
est_3w_list = np.array(est_3w_list)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=bil_range*120.0, y=bil_range*120.0, mode='lines', name='Ідеал (Реальність)', line=dict(dash='dash', color='gray')))
fig1.add_trace(go.Scatter(x=bil_range*120.0, y=est_2w_list*120.0, mode='lines+markers', name='2-хвильовий алгоритм', marker=dict(color='red')))
fig1.add_trace(go.Scatter(x=bil_range*120.0, y=est_3w_list*120.0, mode='lines+markers', name='3-хвильовий алгоритм (Матричний)', marker=dict(color='green')))

fig1.update_layout(xaxis_title="Реальний Білірубін", yaxis_title="Виміряний Білірубін", height=500, template="plotly_dark")
st.plotly_chart(fig1, use_container_width=True)