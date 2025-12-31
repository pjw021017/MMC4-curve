# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline

from train_export_model import train_bundle

# ===== 모델: 매 시행 랜덤 학습 =====
st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")

# “매 시행 랜덤 학습”을 만족시키려면 cache를 기본적으로 끄거나,
# 버튼으로 명시적으로 재학습을 트리거해야 함.
# 아래는 버튼 누를 때마다 seed 랜덤 재학습 방식(권장: UI 조작 시마다 재학습되는 것 방지).
if "train_seed" not in st.session_state:
    st.session_state["train_seed"] = int(np.random.randint(0, 2**31 - 1))

def train_now(seed: int):
    return train_bundle(seed=seed)

# 사이드바: 재학습 버튼
st.sidebar.header("학습")
st.sidebar.write(f"현재 seed: {st.session_state['train_seed']}")
if st.sidebar.button("모델 랜덤 재학습"):
    st.session_state["train_seed"] = int(np.random.randint(0, 2**31 - 1))

bundle = train_now(st.session_state["train_seed"])
model = bundle["model"]
meta  = bundle["meta"]

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]
MEDIANS = meta.get("num_medians", {})

# ===== 특징공학 (원본 그대로) =====
def build_enhanced_features(df_, input_cols, cat_inputs):
    X = df_[input_cols + cat_inputs].copy()

    def _as_series(col):
        if col not in X.columns:
            return None
        s = X[col]
        if isinstance(s, pd.DataFrame):
            s = s.bfill(axis=1).iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    ys  = _as_series('Yield Stress(MPa)')
    uts = _as_series('Ultimate Tensile Stress(MPa)')
    tel = _as_series('Total Elongation')
    rv  = _as_series('r-value')
    etaT= _as_series('Triaxiality(Tension)')
    efT = _as_series('Fracture strain(Tension)')

    if uts is not None and ys is not None:
        X['UTS_to_Yield']   = uts / (ys + 1e-8)
        X['Strength_Range'] = uts - ys
    if tel is not None:
        X['Elong_sq'] = tel**2
        X['Elong_sqrt'] = np.sqrt(tel + 1e-8)
        if rv is not None:
            X['Elong_x_r']   = tel * rv
            X['Elong_div_r'] = tel / (rv + 1e-8)
    if etaT is not None:
        X['etaT_sq']   = etaT**2
        X['etaT_cube'] = etaT**3
        if tel is not None:
            X['Total Elongation_x_Triaxiality(Tension)'] = tel * etaT
        if rv is not None:
            X['r-value_x_Triaxiality(Tension)'] = rv * etaT
    if rv is not None:
        X['r_sq']  = rv**2
        X['r_log'] = np.log(rv + 1e-8)
    if efT is not None:
        X['efT_sq'] = efT**2
        if etaT is not None:
            X['efT_x_etaT'] = efT * etaT
    return X

# ===== MMC4 수식/피팅 (원본 그대로) =====
def theta_bar(eta):
    arg = -(27.0/2.0)*eta*(eta**2 - 1.0/3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0/np.pi)*np.arccos(arg)

def c6_effective(eta, C6):
    t = 1.0/np.sqrt(3.0)
    return np.where((eta<=-t)|((eta>=0)&(eta<=t)),1.0,float(C6))

def mmc4_eps(eta, C):
    C1,C2,C3,C4,C5,C6=C
    tb=theta_bar(eta)
    c6e=c6_effective(eta,C6)
    k=np.sqrt(3.0)/(2.0-np.sqrt(3.0))
    term1=(C1/C4)*( C5 + k*(c6e-C5)*(1.0/np.cos(tb*np.pi/6.0) - 1.0) )
    base=1.0 + (C3**2)/3.0*np.cos(tb*np.pi/6.0) + C3*(eta + (1.0/3.0)*np.sin(tb*np.pi/6.0))
    base=np.maximum(base,1e-6)
    return term1 * ( base ** (-1.0/C2) )

def fit_mmc4(etas, epss):
    def resid(p): return mmc4_eps(etas,p)-epss
    x0=np.array([1.0,1.0,0.2,1.0,0.6,0.8])
    lb=np.array([0.001,0.10,-2.0,0.10,0.0,0.0])
    ub=np.array([10.0,5.0,2.0,5.0,2.0,2.0])
    res=least_squares(resid,x0,bounds=(lb,ub),max_nfev=5000,verbose=0)
    return res.x

# ===== 입력 UI =====
st.header("입력값(6개)")
c1, c2 = st.columns(2)
with c1:
    ys  = st.number_input("Yield Stress(MPa)", value=float(MEDIANS.get("Yield Stress(MPa)", 200.0)))
    uts = st.number_input("Ultimate Tensile Stress(MPa)", value=float(MEDIANS.get("Ultimate Tensile Stress(MPa)", 350.0)))
    tel = st.number_input("Total Elongation", value=float(MEDIANS.get("Total Elongation", 0.20)), format="%.5f")
with c2:
    rv   = st.number_input("r-value", value=float(MEDIANS.get("r-value", 1.00)), format="%.5f")
    etaT = st.number_input("Triaxiality(Tension)", value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)),
                           min_value=-0.5, max_value=0.7, step=0.01, format="%.2f")
    efT  = st.number_input("Fracture strain(Tension)", value=0.20, min_value=0.0, step=0.001, format="%.5f")

extra_inputs = [c for c in INPUTS if c not in [
    "Yield Stress(MPa)","Ultimate Tensile Stress(MPa)","Total Elongation","r-value",
    "Triaxiality(Tension)","Fracture strain(Tension)"
]]
adv_vals = {}
if len(extra_inputs) or len(CAT_INPUTS):
    with st.expander("고급 옵션", expanded=False):
        for c in CAT_INPUTS:
            adv_vals[c] = st.text_input(c, value="")
        for c in extra_inputs:
            adv_vals[c] = st.number_input(c, value=float(MEDIANS.get(c, 0.0)))

# ===== 예측 및 플롯 =====
if st.button("예측 및 MMC4 플롯"):
    # 입력 DF
    x_dict = {
        "Yield Stress(MPa)": ys,
        "Ultimate Tensile Stress(MPa)": uts,
        "Total Elongation": tel,
        "r-value": rv,
        "Triaxiality(Tension)": etaT,
        "Fracture strain(Tension)": efT,
    }
    for c in extra_inputs:
        x_dict[c] = adv_vals.get(c, np.nan)
    for c in CAT_INPUTS:
        x_dict[c] = adv_vals.get(c, "")

    x_one = pd.DataFrame([x_dict])

    for col in INPUTS:
        if col not in x_one.columns:
            x_one[col] = np.nan
    for col in CAT_INPUTS:
        if col not in x_one.columns:
            x_one[col] = ""

    X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

    # 스키마 정렬
    pre = model.named_steps['pre']
    num_cols_model = list(next(t[2] for t in pre.transformers if t[0] == 'num'))
    cat_cols_model = []
    for name, trans, cols in pre.transformers:
        if name == 'cat' and cols is not None and cols != 'drop':
            cat_cols_model = list(cols)

    for c in num_cols_model:
        if c not in X_tmp.columns:
            X_tmp[c] = np.nan
    for c in cat_cols_model:
        if c not in x_one.columns:
            x_one[c] = ""

    X_feed = pd.DataFrame()
    X_feed[num_cols_model] = X_tmp[num_cols_model]
    if cat_cols_model:
        X_feed[cat_cols_model] = x_one[cat_cols_model].astype(str)

    # 타깃 예측
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # 4점 구성
    eta_shear = y_hat.get("Triaxiality(Shear0)")
    ef_shear  = y_hat.get("Fracture strain(Shear0)")
    eta_notch = y_hat.get("Triaxiality(Notch R05)")
    ef_notch  = y_hat.get("Fracture strain(Notch R05)")
    ef_bulge  = y_hat.get("Fracture strain(Punch Bulge)")

    eta_tens  = float(etaT)
    ef_tens   = float(efT)
    eta_bulge = 2.0/3.0

    pts = [
        ("Shear",   float(eta_shear), float(ef_shear)),
        ("Tension", float(eta_tens),  float(ef_tens)),
        ("Notch",   float(eta_notch), float(ef_notch)),
        ("Bulge",   float(eta_bulge), float(ef_bulge)),
    ]
    pts = sorted(pts, key=lambda t: t[1])

    etas = np.array([p[1] for p in pts], dtype=float)
    epss = np.array([p[2] for p in pts], dtype=float)
    C_hat = fit_mmc4(etas, epss)

    # ===== 플롯 범위(요구사항 고정) =====
    eta_lo, eta_hi = -0.1, 0.7
    eta_grid = np.linspace(eta_lo, eta_hi, 240)

    # 기본은 MMC4 그대로(= Shear 이전 포함 전 구간 보간 없음)
    eps_curve = mmc4_eps(eta_grid, C_hat)

    # y 상한: Bulge + 0.3
    bulge_y = float(next(y for name, _, y in pts if name == "Bulge"))
    y_cap = bulge_y + 0.3

    # ===== Bulge 이후(η>2/3)만 C¹ 보간 + 단조 증가 보장 =====
    bulge_eta = 2.0/3.0
    h = 1e-4
    def d_mmc4(e):
        return float((mmc4_eps(e + h, C_hat) - mmc4_eps(e - h, C_hat)) / (2.0*h))

    # join은 "mmc4 자체"에서 연속되게(값/미분 모두 mmc4 기준)
    y0 = float(mmc4_eps(bulge_eta, C_hat))
    d0_raw = d_mmc4(bulge_eta)

    # 종점은 y_cap에 맞춰서 시각화 상한까지 매끈하게 상승
    x1 = float(eta_hi)
    y1 = float(y_cap)

    dx = (x1 - bulge_eta)
    m = (y1 - y0) / (dx + 1e-12)

    # 단조 증가 조건: d0>=0, 또한 Hermite 단조성(overshoot 방지) 위해 d0 <= 3m
    # (두 점 Hermite에서 대표적인 충분조건)
    d0 = max(0.0, d0_raw)
    d0 = min(d0, 3.0*m) if m > 0 else 0.0

    # 기울기 변화가 이상해 보인다고 했으니, 끝점도 동일 기울기로 설정
    d1 = d0

    right_conn = CubicHermiteSpline([bulge_eta, x1], [y0, y1], [d0, d1])

    mask_right = eta_grid > bulge_eta
    if np.any(mask_right):
        eps_curve[mask_right] = right_conn(eta_grid[mask_right])

    # 혹시 수치 튐 방지: y_cap 이하로만 클리핑(시각화 범위 유지)
    eps_curve = np.minimum(eps_curve, y_cap)

    # ===== 시각화 =====
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta_grid, eps_curve, lw=2, label="MMC4 curve (fit)")
    for name, x, y in pts:
        ax.scatter([x], [y], s=70, edgecolor="k", zorder=5, label=name)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_xlim(eta_lo, eta_hi)
    ax.set_ylim(0.0, y_cap)
    ax.set_title("MMC4 Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.write("4점:", pts)
    st.write("C =", [float(v) for v in C_hat])
    st.write(f"Bulge join: y0=mmc4(2/3)={y0:.6f}, d0(mm4)={d0_raw:.6f} → used d0={d0:.6f}")
