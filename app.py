# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from openpyxl import load_workbook

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline


# =========================
# 고정 설정
# =========================
TARGETS_FIXED = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}

DEFAULT_XLSX = "250811_산학프로젝트_포스코의 워크시트.xlsx"
DEFAULT_PKL  = "mmc4_model.pkl"


# =========================
# Streamlit Page
# =========================
st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")


# =========================
# 유틸
# =========================
def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def read_excel_from_bytes(excel_bytes: bytes, sheet_name=None) -> pd.DataFrame:
    bio = BytesIO(excel_bytes)
    return pd.read_excel(bio, sheet_name=sheet_name)

def rgb_from_cell(cell):
    try:
        fill = cell.fill
        if fill is None or fill.fill_type is None:
            return None
        rgb = None
        if hasattr(fill, "fgColor") and getattr(fill.fgColor, "type", None) == "rgb":
            rgb = fill.fgColor.rgb
        if (rgb is None or rgb == "00000000") and hasattr(fill, "start_color"):
            rgb = getattr(fill.start_color, "rgb", None)
        if rgb is None:
            return None
        rgb = rgb.replace("0x", "").replace("#", "")
        if len(rgb) == 8:
            rgb = rgb[2:]
        return rgb.upper() if len(rgb) == 6 else None
    except Exception:
        return None

def classify_rgb(rgb):
    if not rgb:
        return None
    r = int(rgb[0:2], 16)
    g = int(rgb[2:4], 16)
    b = int(rgb[4:6], 16)
    # 노란색(입력), 하늘색(출력) 판정(기존 로직 유지)
    if r >= 200 and g >= 200 and b <= 120:
        return "yellow"
    if r <= 180 and g >= 170 and b >= 190:
        return "skyblue"
    return None

def _as_series_single(X: pd.DataFrame, colname: str):
    """
    중복 컬럼이 있어도 단일 Series로 축약(왼쪽→오른쪽 첫 유효값)
    """
    if colname not in X.columns:
        return None
    cols = [c for c in X.columns if c == colname]
    sub = X[cols]
    if isinstance(sub, pd.Series):
        return pd.to_numeric(sub, errors="coerce")
    sub_num = sub.apply(pd.to_numeric, errors="coerce")
    ser = sub_num.bfill(axis=1).iloc[:, 0]
    return ser

def build_enhanced_features(df_, input_cols, cat_inputs):
    X = df_[input_cols + cat_inputs].copy()

    ys  = _as_series_single(X, "Yield Stress(MPa)")
    uts = _as_series_single(X, "Ultimate Tensile Stress(MPa)")
    tel = _as_series_single(X, "Total Elongation")
    rv  = _as_series_single(X, "r-value")
    etaT= _as_series_single(X, "Triaxiality(Tension)")
    efT = _as_series_single(X, "Fracture strain(Tension)")

    if uts is not None and ys is not None:
        X["UTS_to_Yield"]   = uts / (ys + 1e-8)
        X["Strength_Range"] = uts - ys
    if tel is not None:
        X["Elong_sq"]   = tel**2
        X["Elong_sqrt"] = np.sqrt(tel + 1e-8)
        if rv is not None:
            X["Elong_x_r"]   = tel * rv
            X["Elong_div_r"] = tel / (rv + 1e-8)
    if etaT is not None:
        X["etaT_sq"]   = etaT**2
        X["etaT_cube"] = etaT**3
        if tel is not None:
            X["Total Elongation_x_Triaxiality(Tension)"] = tel * etaT
        if rv is not None:
            X["r-value_x_Triaxiality(Tension)"] = rv * etaT
    if rv is not None:
        X["r_sq"]  = rv**2
        X["r_log"] = np.log(rv + 1e-8)
    if efT is not None:
        X["efT_sq"] = efT**2
        if etaT is not None:
            X["efT_x_etaT"] = efT * etaT

    # 상호작용(기존 학습 스크립트와 동일 계열)
    key = [("Total Elongation", tel), ("r-value", rv), ("Triaxiality(Tension)", etaT)]
    for i in range(len(key)):
        for j in range(i + 1, len(key)):
            n1, s1 = key[i]
            n2, s2 = key[j]
            if s1 is not None and s2 is not None:
                X[f"{n1}_x_{n2}"] = s1 * s2

    return X


# =========================
# MMC4 수식/적합
# =========================
def theta_bar(eta):
    arg = -(27.0/2.0)*eta*(eta**2 - 1.0/3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0/np.pi)*np.arccos(arg)

def c6_effective(eta, C6):
    t = 1.0/np.sqrt(3.0)
    return np.where((eta <= -t) | ((eta >= 0) & (eta <= t)), 1.0, float(C6))

def mmc4_eps(eta, C):
    C1, C2, C3, C4, C5, C6 = C
    tb = theta_bar(eta)
    c6e = c6_effective(eta, C6)
    k = np.sqrt(3.0) / (2.0 - np.sqrt(3.0))
    term1 = (C1 / C4) * (C5 + k * (c6e - C5) * (1.0 / np.cos(tb * np.pi / 6.0) - 1.0))
    base = 1.0 + (C3**2) / 3.0 * np.cos(tb * np.pi / 6.0) + C3 * (eta + (1.0/3.0) * np.sin(tb * np.pi / 6.0))
    base = np.maximum(base, 1e-6)
    return term1 * (base ** (-1.0 / C2))

def fit_mmc4(etas, epss):
    def resid(p):
        return mmc4_eps(etas, p) - epss
    x0 = np.array([1.0, 1.0, 0.2, 1.0, 0.6, 0.8])
    lb = np.array([0.001, 0.10, -2.0, 0.10, 0.0, 0.0])
    ub = np.array([10.0, 5.0,  2.0, 5.0,  2.0, 2.0])
    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=5000, verbose=0)
    return res.x


# =========================
# 학습(엑셀) - 캐시
# =========================
@st.cache_resource(show_spinner=True)
def train_from_excel_bytes(excel_bytes: bytes, sheet_name: str | None, n_estimators: int, random_state: int):
    # 1) 워크북으로 헤더 색상 읽기(입/출력 컬럼 자동 판정)
    bio = BytesIO(excel_bytes)
    wb = load_workbook(bio, data_only=True)
    ws = wb[sheet_name] if (sheet_name and sheet_name in wb.sheetnames) else wb[wb.sheetnames[0]]

    df = read_excel_from_bytes(excel_bytes, sheet_name=sheet_name)
    df = df.dropna(axis=1, how="all")

    headers, classes = [], []
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        headers.append(cell.value)
        classes.append(classify_rgb(rgb_from_cell(cell)))

    input_cols, output_cols = [], []
    for name, cls in zip(headers, classes):
        if name in df.columns:
            if cls == "yellow":
                input_cols.append(name)
            elif cls == "skyblue":
                output_cols.append(name)

    # 2) 강제 입력 반영(노란값)
    for c in FORCE_INPUT:
        if c in df.columns:
            if c in output_cols:
                output_cols.remove(c)
            if c not in input_cols:
                input_cols.append(c)

    # 3) 타깃은 5개로 고정(존재하는 것만)
    output_cols = [c for c in TARGETS_FIXED if c in df.columns]
    if len(output_cols) != len(TARGETS_FIXED):
        missing = [c for c in TARGETS_FIXED if c not in df.columns]
        raise ValueError(f"엑셀에 타깃 컬럼이 부족합니다: {missing}")

    # 범주 입력(있으면)
    cat_inputs = ["Material"] if "Material" in df.columns else []

    # 4) 숫자 변환(학습 관련 컬럼만)
    for col in set(input_cols + output_cols):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = list(dict.fromkeys(input_cols + cat_inputs + output_cols))
    df_model = df[needed].copy()
    df_model = df_model.dropna(subset=output_cols, how="any")

    # 5) 특징공학
    X_all = build_enhanced_features(df_model, input_cols, cat_inputs)
    y_all = df_model[output_cols].copy()

    # 6) split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=random_state, shuffle=True
    )

    # 7) 파이프라인
    num_cols = [c for c in X_all.columns if c not in cat_inputs]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", ohe, cat_inputs),
        ],
        remainder="drop"
    )

    def create_et():
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

    pipe = Pipeline([
        ("pre", pre),
        ("reg", MultiOutputRegressor(create_et()))
    ])

    pipe.fit(X_train, y_train)

    # 8) metrics
    y_pred = pd.DataFrame(pipe.predict(X_test), columns=y_test.columns, index=y_test.index)
    r2_flat = float(r2_score(y_test.values.flatten(), y_pred.values.flatten()))
    mae_flat = float(mean_absolute_error(y_test.values.flatten(), y_pred.values.flatten()))

    meta = {
        "input_cols": input_cols,
        "cat_inputs": cat_inputs,
        "output_cols": list(y_all.columns),
        "num_medians": df[input_cols].median(numeric_only=True).to_dict(),
        "feature_columns": list(X_all.columns),
        "metrics": {"r2_flat": r2_flat, "mae_flat": mae_flat},
        "sheet_used": ws.title,
        "rows_used": int(df_model.shape[0]),
        "n_estimators": int(n_estimators),
        "random_state": int(random_state)
    }

    return {"model": pipe, "meta": meta}


# =========================
# 모델 준비(로드 시도 → 실패 시 학습)
# =========================
st.sidebar.header("실행 모드")

mode = st.sidebar.radio(
    "모델 소스",
    options=["pkl 로드(가능하면)", "엑셀로 재학습(권장)"],
    index=1
)

# 엑셀 소스(업로드 or repo 파일)
st.sidebar.subheader("학습 데이터(.xlsx)")
uploaded_xlsx = st.sidebar.file_uploader("엑셀 업로드(없으면 repo 파일 사용)", type=["xlsx"])
sheet_name_in = st.sidebar.text_input("시트 이름(비우면 첫 시트)", value="").strip()
sheet_name = None if sheet_name_in == "" else sheet_name_in

n_estimators = st.sidebar.slider("n_estimators", 50, 800, 300, 50)
random_state = st.sidebar.number_input("random_state", value=20250821, step=1)

bundle = None
load_error = None

# 1) pkl 로드 시도
if mode == "pkl 로드(가능하면)":
    try:
        if not Path(DEFAULT_PKL).exists():
            raise FileNotFoundError(DEFAULT_PKL)
        bundle = joblib.load(DEFAULT_PKL)
        if not isinstance(bundle, dict) or "model" not in bundle or "meta" not in bundle:
            raise ValueError("pkl bundle 포맷이 dict(model, meta) 형태가 아닙니다.")
    except Exception as e:
        load_error = str(e)
        bundle = None

# 2) 실패하거나 재학습 모드면 학습 수행
if bundle is None:
    if mode == "pkl 로드(가능하면)":
        st.sidebar.warning("pkl 로드 실패 → 엑셀 재학습으로 전환합니다.")
        st.sidebar.caption(f"pkl 로드 에러: {load_error}")

    # 엑셀 bytes 확보
    if uploaded_xlsx is not None:
        xlsx_bytes = uploaded_xlsx.getvalue()
        xlsx_name = uploaded_xlsx.name
    else:
        # repo 기본 파일 사용
        if not Path(DEFAULT_XLSX).exists():
            st.error(f"학습 엑셀을 찾을 수 없습니다. 업로드하거나 repo에 {DEFAULT_XLSX}를 두세요.")
            st.stop()
        xlsx_bytes = Path(DEFAULT_XLSX).read_bytes()
        xlsx_name = DEFAULT_XLSX

    xlsx_md5 = md5_bytes(xlsx_bytes)
    st.sidebar.caption(f"학습 엑셀: {xlsx_name}")
    st.sidebar.caption(f"MD5: {xlsx_md5[:10]}...")

    # 학습 실행(캐시)
    bundle = train_from_excel_bytes(
        excel_bytes=xlsx_bytes,
        sheet_name=sheet_name,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )
    st.sidebar.success(
        f"학습 완료 | R2(flat)={bundle['meta']['metrics']['r2_flat']:.4f}, "
        f"MAE(flat)={bundle['meta']['metrics']['mae_flat']:.6f} | rows={bundle['meta']['rows_used']}"
    )

model = bundle["model"]
meta  = bundle["meta"]

INPUTS   = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS  = meta["output_cols"]
MEDIANS  = meta.get("num_medians", {})

with st.expander("현재 모델 메타/지표", expanded=False):
    st.write(meta)


# =========================
# 입력 UI
# =========================
st.header("입력값(노란색)")
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
    "Yield Stress(MPa)","Ultimate Tensile Stress(MPa)","Total Elongation",
    "r-value","Triaxiality(Tension)","Fracture strain(Tension)"
]]
adv_vals = {}
if len(extra_inputs) or len(CAT_INPUTS):
    with st.expander("고급 옵션", expanded=False):
        for c in CAT_INPUTS:
            adv_vals[c] = st.text_input(c, value="")
        for c in extra_inputs:
            adv_vals[c] = st.number_input(c, value=float(MEDIANS.get(c, 0.0)))


# =========================
# 예측 및 MMC4 피팅
# =========================
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

    # 4) 스키마 정렬 (파이프라인 pre 기준)
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

    # 5) 예측
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # 6) 4점 확정: Tension은 입력값, Bulge η=2/3 고정
    eta_shear = y_hat.get("Triaxiality(Shear0)")
    ef_shear  = y_hat.get("Fracture strain(Shear0)")
    eta_notch = y_hat.get("Triaxiality(Notch R05)")
    ef_notch  = y_hat.get("Fracture strain(Notch R05)")
    ef_bulge  = y_hat.get("Fracture strain(Punch Bulge)")

    eta_tens  = float(etaT)
    ef_tens   = float(efT)
    eta_bulge = 2.0/3.0

    missing = []
    if eta_shear is None: missing.append("η(Shear)")
    if ef_shear  is None: missing.append("εf(Shear)")
    if eta_notch is None: missing.append("η(Notch R05)")
    if ef_notch  is None: missing.append("εf(Notch R05)")
    if ef_bulge  is None: missing.append("εf(Bulge)")
    if missing:
        st.error("부족한 값: " + ", ".join(missing))
        st.stop()

    pts = [
        ("Shear",   float(eta_shear), float(ef_shear)),
        ("Tension", float(eta_tens),  float(ef_tens)),
        ("Notch",   float(eta_notch), float(ef_notch)),
        ("Bulge",   float(eta_bulge), float(ef_bulge)),
    ]
    pts = sorted(pts, key=lambda t: t[1])

    etas = np.array([p[1] for p in pts])
    epss = np.array([p[2] for p in pts])
    C_hat = fit_mmc4(etas, epss)

    st.session_state["mmc4_pts"] = pts
    st.session_state["mmc4_C_hat"] = C_hat


# =========================
# 결과 출력(상태 유지)
# =========================
if "mmc4_pts" in st.session_state and "mmc4_C_hat" in st.session_state:
    pts = st.session_state["mmc4_pts"]
    C_hat = st.session_state["mmc4_C_hat"]

    eta_lo, eta_hi = -0.1, 0.7
    eta_grid = np.linspace(eta_lo, eta_hi, 200)
    eps_curve = mmc4_eps(eta_grid, C_hat)

    etas = np.array([p[1] for p in pts])
    eta_min = float(etas.min())
    eta_max = float(etas.max())

    # 좌측: 포물선 상승
    e_min = float(mmc4_eps(eta_min, C_hat))
    delta = eta_min - eta_lo
    curv = abs(e_min) / (delta**2 + 1e-6) * 0.6

    def left_curve(eta):
        return e_min + curv * (eta - eta_min)**2

    mask_left = eta_grid < eta_min
    if np.any(mask_left):
        eps_curve[mask_left] = left_curve(eta_grid[mask_left])

    # 우측: C1 연속 연결
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta_grid, eps_curve, lw=2, label="MMC4 curve (fit)")
    for name, x, y in pts:
        ax.scatter([x], [y], s=60, edgecolor="k", zorder=5, label=name)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_xlim(eta_lo, eta_hi)
    ax.set_ylim(0.0, float(next(y for name, _, y in pts if name == "Bulge")) + 0.3)
    ax.set_title("MMC4 Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**4점 요약**")
    for name, x, y in pts:
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
