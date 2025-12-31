# -*- coding: utf-8 -*-
# app_mmc4_compact.py — Tension η·εf 입력, Notch/Shear (η,εf)+Bulge εf 예측, Bulge η=2/3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline

from train_export_model import (
    DEFAULT_DATA_PATH,
    train_bundle_from_bytes,
    train_bundle_from_path,
)

# ===== 데이터/모델(실행 시 학습) =====
st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")

# --- 사이드바: 데이터 소스/학습 옵션 ---
st.sidebar.header("데이터 & 학습 설정")

use_default = st.sidebar.checkbox("리포지토리 기본 데이터 사용", value=True)
uploaded = None
if not use_default:
    uploaded = st.sidebar.file_uploader(
        "학습 데이터 업로드 (CSV/XLSX)", type=["csv", "xlsx", "xls", "xlsm", "txt"]
    )

test_size = st.sidebar.slider("검증(Test) 비율", min_value=0.1, max_value=0.4, value=0.25, step=0.05)
n_estimators = st.sidebar.slider("n_estimators (ExtraTrees)", min_value=50, max_value=600, value=300, step=50)
random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

@st.cache_resource(show_spinner=True)
def get_bundle_cached(data_bytes: bytes, filename: str, test_size: float, n_estimators: int, random_state: int):
    return train_bundle_from_bytes(
        data_bytes,
        filename,
        test_size=test_size,
        n_estimators=n_estimators,
        random_state=random_state,
    )

@st.cache_resource(show_spinner=True)
def get_bundle_default_cached(path: str, test_size: float, n_estimators: int, random_state: int):
    return train_bundle_from_path(
        path,
        test_size=test_size,
        n_estimators=n_estimators,
        random_state=random_state,
    )

# --- 학습 수행(캐시로 rerun 시 재학습 방지) ---
bundle = None
data_label = None

if use_default:
    try:
        bundle = get_bundle_default_cached(DEFAULT_DATA_PATH, test_size, n_estimators, random_state)
        data_label = DEFAULT_DATA_PATH
    except Exception as e:
        st.sidebar.error(f"기본 데이터 로드/학습 실패: {e}")
        st.sidebar.info("기본 데이터 파일이 repo에 없으면 업로드 모드로 전환하십시오.")
else:
    if uploaded is not None:
        try:
            data_bytes = uploaded.getvalue()
            bundle = get_bundle_cached(data_bytes, uploaded.name, test_size, n_estimators, random_state)
            data_label = uploaded.name
        except Exception as e:
            st.sidebar.error(f"업로드 데이터 학습 실패: {e}")

# 재학습 버튼(캐시 초기화)
if st.sidebar.button("모델 재학습(캐시 초기화)"):
    get_bundle_cached.clear()
    get_bundle_default_cached.clear()
    st.rerun()

if bundle is None:
    st.error("학습할 데이터를 불러오지 못했습니다. 사이드바에서 데이터 업로드 또는 기본 데이터 사용을 확인하십시오.")
    st.stop()

model = bundle["model"]
meta  = bundle["meta"]

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]      # 5 타깃
MEDIANS = meta.get("num_medians", {})
FEATURES = meta.get("feature_columns", [])

with st.expander("학습 요약(데이터/성능)", expanded=False):
    st.write(f"- 데이터: `{data_label}`")
    st.write(f"- 학습 행 수: {meta.get('train_rows', 'NA')}")
    st.write(f"- Test 비율: {meta.get('test_size', test_size)}")
    st.write(f"- ExtraTrees n_estimators: {meta.get('n_estimators', n_estimators)}")
    st.write(f"- 전체 평탄화 R²: {meta.get('r2_flat', bundle.get('r2_flat')):.4f}")
    try:
        metrics_df = pd.DataFrame(bundle.get("metrics", {})).T
        st.dataframe(metrics_df, use_container_width=True)
    except Exception:
        pass

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

    ys  = _as_series("Yield Stress(MPa)")
    uts = _as_series("Ultimate Tensile Stress(MPa)")
    tel = _as_series("Total Elongation")
    rv  = _as_series("r-value")
    etaT= _as_series("Triaxiality(Tension)")
    efT = _as_series("Fracture strain(Tension)")

    if uts is not None and ys is not None:
        X["UTS_to_Yield"]   = uts / (ys + 1e-8)
        X["Strength_Range"] = uts - ys
    if tel is not None:
        X["Elong_sq"] = tel**2
        X["Elong_sqrt"] = np.sqrt(tel + 1e-8)
        if rv is not None:
            X["Elong_x_r"]   = tel * rv
            X["Elong_div_r"] = tel / (rv + 1e-8)
    if etaT is not None:
        X["etaT_sq"]   = etaT**2
        X["etaT_abs"]  = np.abs(etaT)
    if efT is not None:
        X["efT_sq"]  = efT**2
        X["log_efT"] = np.log(efT + 1e-8)
    if etaT is not None and efT is not None:
        X["etaT_x_efT"] = etaT * efT
        X["efT_div_etaT"] = efT / (np.abs(etaT) + 1e-8)

    return X

# ===== MMC4 Curve =====
def mmc4_eta_epsf(eta, C):
    # C = [C1..C6] 6개
    C1, C2, C3, C4, C5, C6 = C
    return (
        C1
        + C2 * eta
        + C3 * (eta**2)
        + C4 * (eta**3)
        + C5 * np.exp(-eta)
        + C6 * np.exp(eta)
    )

def fit_mmc4_curve(points, init_C=None):
    # points: list of (name, eta, epsf)
    etas = np.array([p[1] for p in points], dtype=float)
    epsf = np.array([p[2] for p in points], dtype=float)

    if init_C is None:
        # 단순 초기값: 상수항=평균, 나머지=작게
        init_C = np.array([float(np.mean(epsf)), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    def residual(C):
        return mmc4_eta_epsf(etas, C) - epsf

    res = least_squares(residual, init_C, max_nfev=10000)
    return res.x, res

def smooth_curve_cubic(points, n=200):
    # points: (name, eta, epsf) 정렬된 상태 가정
    pts_sorted = sorted(points, key=lambda x: x[1])
    xs = np.array([p[1] for p in pts_sorted], dtype=float)
    ys = np.array([p[2] for p in pts_sorted], dtype=float)

    # 기울기(단순 finite difference)
    dydx = np.zeros_like(ys)
    for i in range(len(xs)):
        if i == 0:
            dydx[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i] + 1e-8)
        elif i == len(xs)-1:
            dydx[i] = (ys[i] - ys[i-1]) / (xs[i] - xs[i-1] + 1e-8)
        else:
            dydx[i] = (ys[i+1] - ys[i-1]) / (xs[i+1] - xs[i-1] + 1e-8)

    spline = CubicHermiteSpline(xs, ys, dydx)
    x_new = np.linspace(xs.min(), xs.max(), n)
    y_new = spline(x_new)
    return x_new, y_new

# ===== 입력 UI =====
st.subheader("1) 입력 (Tension 기준 + 선택 입력)")

c1, c2 = st.columns(2)
with c1:
    ys  = st.number_input("Yield Stress(MPa)", value=float(MEDIANS.get("Yield Stress(MPa)", 300.0)), format="%.5f")
    uts = st.number_input("Ultimate Tensile Stress(MPa)", value=float(MEDIANS.get("Ultimate Tensile Stress(MPa)", 500.0)), format="%.5f")
    tel = st.number_input("Total Elongation", value=float(MEDIANS.get("Total Elongation", 0.20)), format="%.5f")
with c2:
    rv   = st.number_input("r-value", value=float(MEDIANS.get("r-value", 1.00)), format="%.5f")
    etaT = st.number_input(
        "Triaxiality(Tension)",
        value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)),
        min_value=-0.5, max_value=0.7, step=0.01, format="%.2f"
    )
    efT  = st.number_input("Fracture strain(Tension)", value=float(MEDIANS.get("Fracture strain(Tension)", 0.20)), min_value=0.0, step=0.001, format="%.5f")

extra_inputs = [
    c for c in INPUTS
    if c not in [
        "Yield Stress(MPa)", "Ultimate Tensile Stress(MPa)", "Total Elongation", "r-value",
        "Triaxiality(Tension)", "Fracture strain(Tension)"
    ]
]
adv_vals = {}
if len(extra_inputs) or len(CAT_INPUTS):
    with st.expander("고급 옵션", expanded=False):
        for c in CAT_INPUTS:
            adv_vals[c] = st.text_input(c, value="")
        for c in extra_inputs:
            adv_vals[c] = st.number_input(c, value=float(MEDIANS.get(c, 0.0)))

# ===== 예측 =====
st.subheader("2) 예측 및 곡선 생성")

if st.button("예측 실행"):
    # 1) 입력 DF 생성
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

    # 4) 스키마 정렬(모델 preprocessor가 기대하는 컬럼 확보)
    pre = model.named_steps["pre"]
    num_cols_model = list(next(t[2] for t in pre.transformers if t[0] == "num"))
    cat_cols_model = []
    for name, trans, cols in pre.transformers:
        if name == "cat" and cols is not None and cols != "drop":
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

    eta_tens  = etaT
    ef_tens   = efT
    eta_bulge = 2.0 / 3.0

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

    st.write("예측 결과(타깃 5개):")
    st.dataframe(y_hat.to_frame("pred").T, use_container_width=True)

    st.write("곡선 피팅에 사용되는 4점:")
    st.dataframe(pd.DataFrame(pts, columns=["Case", "η", "εf"]), use_container_width=True)

    # 8) MMC4 파라미터 피팅
    C_opt, res = fit_mmc4_curve(pts)
    st.write("최적화된 C (C1~C6):", C_opt)

    # 9) 곡선 생성(η 범위는 4점 기준)
    eta_min = min(p[1] for p in pts)
    eta_max = max(p[1] for p in pts)
    eta_grid = np.linspace(eta_min, eta_max, 300)
    eps_grid = mmc4_eta_epsf(eta_grid, C_opt)

    # 10) 스무딩 보조(시각화)
    x_smooth, y_smooth = smooth_curve_cubic(pts, n=250)

    # 11) Plot
    fig = plt.figure(figsize=(7, 4))
    plt.plot(eta_grid, eps_grid, label="MMC4 fit")
    plt.plot(x_smooth, y_smooth, label="CubicHermite (guide)", linestyle="--")
    plt.scatter([p[1] for p in pts], [p[2] for p in pts], zorder=5, label="Points")
    for name, x, y in pts:
        plt.text(x, y, f"  {name}", fontsize=9)
    plt.xlabel("η (Triaxiality)")
    plt.ylabel("εf (Fracture strain)")
    plt.title("MMC4 Curve (fit by 4 points)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig, clear_figure=True)
