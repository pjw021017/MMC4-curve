# -*- coding: utf-8 -*-
# app_mmc4_compact.py — Tension η·εf 입력, Notch/Shear (η,εf)+Bulge εf 예측, Bulge η=2/3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ✅ pkl 로드 대신: 학습 스크립트에서 바로 모델 번들 생성
from train_export_model import train_bundle

# ===== 모델 로드(=실행 시 학습) =====
st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")

# Streamlit은 rerun이 잦아서, 캐시 없으면 매 상호작용마다 재학습됩니다.
@st.cache_resource(show_spinner=True)
def get_trained_bundle():
    return train_bundle()

try:
    bundle = get_trained_bundle()
    model = bundle["model"]
    meta  = bundle["meta"]
except Exception as e:
    st.error(f"모델 학습/로드 실패: {e}")
    st.stop()

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]      # 5 타깃
MEDIANS = meta.get("num_medians", {})
FEATURES = meta.get("feature_columns", [])

# (선택) 재학습 버튼
if st.sidebar.button("모델 재학습(캐시 초기화)"):
    get_trained_bundle.clear()
    st.rerun()

# ===== 특징공학 =====
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

# ===== MMC4 수식/적합 (버튼 밖으로 빼서 session_state 재출력에도 동일 적용) =====
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
    from scipy.optimize import least_squares
    def resid(p): return mmc4_eps(etas,p)-epss
    x0=np.array([1.0,1.0,0.2,1.0,0.6,0.8])
    lb=np.array([0.001,0.10,-2.0,0.10,0.0,0.0])
    ub=np.array([10.0,5.0,2.0,5.0,2.0,2.0])
    res=least_squares(resid,x0,bounds=(lb,ub),max_nfev=5000,verbose=0)
    return res.x

# ===== 입력 폼 =====
st.header("입력값(노란색)")
c1, c2 = st.columns(2)
with c1:
    ys  = st.number_input("Yield Stress(MPa)", value=float(MEDIANS.get("Yield Stress(MPa)", 200.0)))
    uts = st.number_input("Ultimate Tensile Stress(MPa)", value=float(MEDIANS.get("Ultimate Tensile Stress(MPa)", 350.0)))
    tel = st.number_input("Total Elongation", value=float(MEDIANS.get("Total Elongation", 0.20)), format="%.5f")
with c2:
    rv   = st.number_input("r-value", value=float(MEDIANS.get("r-value", 1.00)), format="%.5f")
    etaT = st.number_input("Triaxiality(Tension)", value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)), min_value=-0.5, max_value=0.7, step=0.01, format="%.2f")
    efT  = st.number_input("Fracture strain(Tension)", value=0.20, min_value=0.0, step=0.001, format="%.5f")

extra_inputs = [c for c in INPUTS if c not in ["Yield Stress(MPa)","Ultimate Tensile Stress(MPa)","Total Elongation","r-value","Triaxiality(Tension)","Fracture strain(Tension)"]]
adv_vals = {}
if len(extra_inputs) or len(CAT_INPUTS):
    with st.expander("고급 옵션", expanded=False):
        for c in CAT_INPUTS:
            adv_vals[c] = st.text_input(c, value="")
        for c in extra_inputs:
            adv_vals[c] = st.number_input(c, value=float(MEDIANS.get(c, 0.0)))

# ===== 예측 =====
if st.button("예측 및 MMC4 플롯"):
    # 1) 입력 dict
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

    # 2) 누락 컬럼 보강
    for col in INPUTS:
        if col not in x_one.columns:
            x_one[col] = np.nan
    for col in CAT_INPUTS:
        if col not in x_one.columns:
            x_one[col] = ""

    # 3) 특징공학
    X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

    # 4) 스키마 정렬
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

    # 5) 예측 수행(타깃 5개)
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # 6) 4점 확정: Tension은 입력값 사용, Bulge η=2/3 고정
    eta_shear = y_hat.get("Triaxiality(Shear0)")
    ef_shear  = y_hat.get("Fracture strain(Shear0)")
    eta_notch = y_hat.get("Triaxiality(Notch R05)")
    ef_notch  = y_hat.get("Fracture strain(Notch R05)")
    ef_bulge  = y_hat.get("Fracture strain(Punch Bulge)")

    eta_tens  = float(etaT)
    ef_tens   = float(efT)
    eta_bulge = 2.0/3.0  # 요구사항 반영: Bulge η는 타깃 제외, 상수 사용

    # 7) 누락 검사(정확 표기)
    missing = []
    if eta_shear is None: missing.append("η(Shear)")
    if ef_shear  is None: missing.append("εf(Shear)")
    if eta_notch is None: missing.append("η(Notch R05)")
    if ef_notch  is None: missing.append("εf(Notch R05)")
    if ef_bulge  is None: missing.append("εf(Bulge)")
    if missing:
        st.error("부족한 값: " + ", ".join(missing) + " — 학습/스키마를 점검하십시오.")
        st.stop()

    pts = [
        ("Shear",   float(eta_shear), float(ef_shear)),
        ("Tension", float(eta_tens),  float(ef_tens)),
        ("Notch",   float(eta_notch), float(ef_notch)),
        ("Bulge",   float(eta_bulge), float(ef_bulge)),
    ]
    pts = sorted(pts, key=lambda t: t[1])

    etas=np.array([p[1] for p in pts])
    epss=np.array([p[2] for p in pts])
    C_hat=fit_mmc4(etas,epss)

    # ===== [추가] 결과를 session_state에 저장해서, η_query 입력 시에도 결과가 유지되도록 처리 =====
    st.session_state["mmc4_pts"] = pts
    st.session_state["mmc4_C_hat"] = C_hat

# ===== [추가] 저장된 결과가 있으면(한 번이라도 예측했으면) 계속 출력 + 특정 η 조회 기능 제공 =====
if "mmc4_pts" in st.session_state and "mmc4_C_hat" in st.session_state:
    pts = st.session_state["mmc4_pts"]
    C_hat = st.session_state["mmc4_C_hat"]

    # ===== 플롯 =====
    eta_lo, eta_hi = -0.1, 0.7
    eta_grid = np.linspace(eta_lo, eta_hi, 200)

    eps_curve = mmc4_eps(eta_grid, C_hat)

    etas = np.array([p[1] for p in pts])
    eta_min = float(etas.min())
    eta_max = float(etas.max())

    e_min = float(mmc4_eps(eta_min, C_hat))
    delta = eta_min - eta_lo
    curv = abs(e_min) / (delta**2 + 1e-6) * 0.6

    def left_curve(eta):
        return e_min + curv * (eta - eta_min)**2

    mask_left = eta_grid < eta_min
    if np.any(mask_left):
        eps_curve[mask_left] = left_curve(eta_grid[mask_left])

    from scipy.interpolate import CubicHermiteSpline

    h = 1e-4
    def d_mmc4(e):
        return float((mmc4_eps(e + h, C_hat) - mmc4_eps(e - h, C_hat)) / (2.0 * h))

    e_max = float(mmc4_eps(eta_max, C_hat))
    d_max = d_mmc4(eta_max)

    y_hi = e_max + d_max * (eta_hi - eta_max)
    right_conn = CubicHermiteSpline([eta_max, eta_hi], [e_max, y_hi], [d_max, d_max])

    mask_right = eta_grid > eta_max
    if np.any(mask_right):
        eps_curve[mask_right] = right_conn(eta_grid[mask_right])

    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(eta_grid,eps_curve,lw=2,label="MMC4 curve (fit)")
    for name,x,y in pts:
        ax.scatter([x],[y],s=60,edgecolor="k",zorder=5,label=name)
        ax.annotate(name,(x,y),xytext=(5,5),textcoords="offset points",fontsize=9)
    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_xlim(eta_lo, eta_hi)
    ax.set_ylim(0.0, float(next(y for name, _, y in pts if name == "Bulge")) + 0.3)
    ax.set_title("MMC4 Curve")
    ax.grid(True,alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**4점 요약**")
    for name,x,y in pts:
        st.text(f"{name:8s} η={x:.4f}  εf={y:.5f}")
    st.markdown("**적합 파라미터 C1~C6**")
    st.text("C = [" + ", ".join(f"{v:.6f}" for v in C_hat) + "]")

    st.markdown("**특정 η 입력 → εf 출력**")
    eta_query = st.number_input(
        "조회할 Triaxiality (η_query)",
        value=float(np.clip((eta_lo + eta_hi) / 2.0, eta_lo, eta_hi)),
        min_value=float(eta_lo),
        max_value=float(eta_hi),
        step=0.01,
        format="%.5f",
        key="mmc4_eta_query"
    )

    if eta_query < eta_min:
        eps_query = float(left_curve(eta_query))
    elif eta_query > eta_max:
        eps_query = float(right_conn(eta_query))
    else:
        eps_query = float(mmc4_eps(eta_query, C_hat))

    st.write(f"**결과:** η = {eta_query:.5f} 에서 **εf = {eps_query:.6f}** 입니다.")
