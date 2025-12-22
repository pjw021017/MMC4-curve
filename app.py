# -*- coding: utf-8 -*-
"""
app.py — 학습 DB에서 MMC4 C1~C6를 먼저 피팅해서 학습하고,
Streamlit 앱에서 입력값 → C1~C6 예측 → MMC4 이론식으로 곡선을 그려주는 앱.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from openpyxl import load_workbook

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

from scipy.optimize import least_squares

# ===== 기본 설정 =====
DATA_XLSX = "250811_산학프로젝트_포스코의 워크시트.xlsx"  # 같은 폴더에 두기
SHEET_FALLBACK = "학습DB"

# 랜덤 시드 고정 안 함 → 실행마다 트리 구조가 조금씩 달라질 수 있음
RSEED = None

st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화 (C1~C6 예측 버전)")

# ===== 문자열 정규화 & 컬럼 매핑 유틸 =====
def _norm(s: str) -> str:
    s = str(s).lower()
    return re.sub(r"[\s\-_]", "", s)

def resolve_col(df: pd.DataFrame, inc, alt=None):
    """
    inc, alt: 포함되어야 할 키워드 리스트
    가장 점수가 높은 컬럼을 하나 골라서 리턴
    """
    inc = [_norm(x) for x in inc]
    alt = [_norm(x) for x in (alt or [])]
    best, score = None, -1
    for c in df.columns:
        nc = _norm(c)
        sc = sum(k in nc for k in inc) + 0.5 * sum(a in nc for a in alt)
        if sc > score:
            score, best = sc, c
    return best

# 엑셀 헤더를 "정규 이름"으로 맞추기 위한 매핑
CANONICAL_INPUTS = {
    "Yield Stress(MPa)":        (["yield", "stress"], ["ys", "yieldstrength"]),
    "Ultimate Tensile Stress(MPa)": (["ultimate", "tensile", "stress"], ["uts", "tensilestrength"]),
    "Total Elongation":         (["total", "elongation"], ["elongation", "totalelong"]),
    "r-value":                  (["rvalue"], ["r-value", "r value", "lankford", "rvalueavg"]),
    "Triaxiality(Tension)":     (["triaxiality", "tension"], ["eta", "tension"]),
    "Fracture strain(Tension)": (["fracture", "strain", "tension"], ["fracturestrain", "tension"]),
}

CANONICAL_MEAS = {
    # Tension (입력에도 포함)
    "Triaxiality(Tension)":     (["triaxiality", "tension"], ["eta", "tension"]),
    "Fracture strain(Tension)": (["fracture", "strain", "tension"], ["fracturestrain", "tension"]),
    # Notch R0.5
    "Triaxiality(Notch R05)":       (["triaxiality", "notch", "r05"], ["eta", "notch", "r05", "0.5"]),
    "Fracture strain(Notch R05)":   (["fracture", "strain", "notch", "r05"], ["fracturestrain", "notch", "r05", "0.5"]),
    # Shear0
    "Triaxiality(Shear0)":          (["triaxiality", "shear0"], ["eta", "shear"]),
    "Fracture strain(Shear0)":      (["fracture", "strain", "shear0"], ["fracturestrain", "shear"]),
    # Punch Bulge
    "Triaxiality(Punch Bulge)":     (["triaxiality", "punch", "bulge"], ["eta", "bulge", "punch"]),
    "Fracture strain(Punch Bulge)": (["fracture", "strain", "punch", "bulge"], ["fracturestrain", "bulge", "punch"]),
}

GEOM_ETA_FALLBACK = {
    "Shear": -0.05,         # 대략적인 전단 상태
    "Notch": 0.55,          # 노치 인장 상태
    "Bulge": 2.0 / 3.0,     # 펀치 벌지
}

def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    CANONICAL_INPUTS + CANONICAL_MEAS를 참고해서
    실제 엑셀 헤더를 정규 이름으로 rename
    """
    all_map = {}
    all_map.update(CANONICAL_INPUTS)
    all_map.update(CANONICAL_MEAS)

    df = df.copy()
    for canon, (inc, alt) in all_map.items():
        if canon in df.columns:
            continue
        c = resolve_col(df, inc, alt)
        if c is not None and c in df.columns:
            df.rename(columns={c: canon}, inplace=True)
    return df

# ===== 엑셀 유틸 =====
def read_excel_safe(path):
    ext = Path(path).suffix.lower()
    try:
        if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
            return pd.read_excel(path, engine="openpyxl")
        elif ext == ".xls":
            return pd.read_excel(path, engine="xlrd")
        elif ext == ".xlsb":
            return pd.read_excel(path, engine="pyxlsb")
    except Exception:
        pass
    return pd.read_excel(path)

# ===== 특징공학 =====
def _as_series_single(X: pd.DataFrame, colname: str):
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
    """
    기존에 쓰던 파생특성 생성. input_cols에는 정규 이름이 들어 있다고 가정.
    """
    X = df_[input_cols + cat_inputs].copy()

    ys   = _as_series_single(X, 'Yield Stress(MPa)')
    uts  = _as_series_single(X, 'Ultimate Tensile Stress(MPa)')
    tel  = _as_series_single(X, 'Total Elongation')
    rv   = _as_series_single(X, 'r-value')
    etaT = _as_series_single(X, 'Triaxiality(Tension)')
    efT  = _as_series_single(X, 'Fracture strain(Tension)')

    if uts is not None and ys is not None:
        X['UTS_to_Yield'] = uts / (ys + 1e-8)
        X['Strength_Range'] = uts - ys
    if tel is not None:
        X['Elong_sq'] = tel ** 2
        X['Elong_sqrt'] = np.sqrt(tel + 1e-8)
        if rv is not None:
            X['Elong_x_r'] = tel * rv
            X['Elong_div_r'] = tel / (rv + 1e-8)
    if etaT is not None:
        X['etaT_sq'] = etaT ** 2
        X['etaT_cube'] = etaT ** 3
        if tel is not None:
            X['Total Elongation_x_Triaxiality(Tension)'] = tel * etaT
        if rv is not None:
            X['r-value_x_Triaxiality(Tension)'] = rv * etaT
    if rv is not None:
        X['r_sq'] = rv ** 2
        X['r_log'] = np.log(rv + 1e-8)
    if efT is not None:
        X['efT_sq'] = efT ** 2
        if etaT is not None:
            X['efT_x_etaT'] = efT * etaT

    key = [('Total Elongation', tel),
           ('r-value', rv),
           ('Triaxiality(Tension)', etaT)]
    for i in range(len(key)):
        for j in range(i + 1, len(key)):
            n1, s1 = key[i]
            n2, s2 = key[j]
            if s1 is not None and s2 is not None:
                X[f'{n1}_x_{n2}'] = s1 * s2
    return X

# ===== MMC4 이론식 =====
def theta_bar(eta):
    eta = np.asarray(eta, dtype=float)
    arg = -(27.0 / 2.0) * eta * (eta ** 2 - 1.0 / 3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0 / np.pi) * np.arccos(arg)

def c6_effective(eta, C6):
    t = 1.0 / np.sqrt(3.0)  # ≈ 0.577
    eta = np.asarray(eta, dtype=float)
    return np.where(
        (eta <= -t) | ((eta >= 0.0) & (eta <= t)),
        1.0,
        float(C6)
    )

def mmc4_eps(eta, C):
    """
    ε_f(η) = [ C1/C4 (C5 + k (C6eff - C5)(1/cos(θ̄π/6)-1)) ] *
             [ √(1 + C3²/3) cos(θ̄π/6) + C3(η + 1/3 sin(θ̄π/6)) ]^(-1/C2)
    """
    C1, C2, C3, C4, C5, C6 = C
    eta = np.asarray(eta, dtype=float)

    tb = theta_bar(eta)
    c6e = c6_effective(eta, C6)

    cos_term = np.cos(tb * np.pi / 6.0)
    cos_term = np.clip(cos_term, 1e-6, None)  # cos가 0 근처에서 폭주 방지

    k = np.sqrt(3.0) / (2.0 - np.sqrt(3.0))

    term1 = (C1 / C4) * (
        C5 + k * (c6e - C5) * (1.0 / cos_term - 1.0)
    )

    base = np.sqrt(1.0 + (C3 ** 2) / 3.0) * cos_term \
           + C3 * (eta + (1.0 / 3.0) * np.sin(tb * np.pi / 6.0))

    base = np.maximum(base, 1e-6)
    return term1 * (base ** (-1.0 / C2))

def fit_mmc4(etas, epss):
    """
    4개 점(η, εf)에 대해 C1~C6 피팅
    """
    etas = np.asarray(etas, dtype=float)
    epss = np.asarray(epss, dtype=float)

    def resid(p):
        return mmc4_eps(etas, p) - epss

    x0 = np.array([1.0, 1.0, 0.2, 1.0, 0.6, 0.8])
    lb = np.array([0.001, 0.10, -2.0, 0.10, 0.0, 0.0])
    ub = np.array([10.0,  5.0,  2.0,  5.0, 2.0, 2.0])

    res = least_squares(
        resid, x0, bounds=(lb, ub),
        max_nfev=5000, verbose=0
    )
    return res.x

# ===== 모델 학습 (엑셀 → C1~C6 회귀모델) =====
@st.cache_resource
def load_and_train():
    if not os.path.exists(DATA_XLSX):
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없음: {DATA_XLSX}")

    # 1) 엑셀 로드 & 헤더 정규화
    wb = load_workbook(DATA_XLSX, data_only=True)
    sheet_name = wb.sheetnames[0] if wb.sheetnames else SHEET_FALLBACK

    df_raw = read_excel_safe(DATA_XLSX)
    df_raw = df_raw.dropna(axis=1, how="all")
    df = rename_to_canonical(df_raw)

    # Bulge η 없으면 2/3 고정
    if "Triaxiality(Punch Bulge)" not in df.columns:
        df["Triaxiality(Punch Bulge)"] = 2.0 / 3.0

    # 필요한 8개 컬럼이 있는지 확인
    eta_cols = [
        "Triaxiality(Shear0)",
        "Triaxiality(Tension)",
        "Triaxiality(Notch R05)",
        "Triaxiality(Punch Bulge)"
    ]
    ef_cols = [
        "Fracture strain(Shear0)",
        "Fracture strain(Tension)",
        "Fracture strain(Notch R05)",
        "Fracture strain(Punch Bulge)"
    ]
    for c in eta_cols + ef_cols:
        if c not in df.columns:
            raise RuntimeError(f"필수 컬럼이 엑셀에 없습니다: {c}")

    # 숫자 변환
    for c in eta_cols + ef_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) 각 재료(row)에 대해 C1~C6 피팅
    C_cols = [f"MMC4_C{i}" for i in range(1, 7)]
    C_list = []
    idx_list = []

    for idx, row in df.iterrows():
        etas = row[eta_cols].values.astype(float)
        epss = row[ef_cols].values.astype(float)

        if not (np.all(np.isfinite(etas)) and np.all(np.isfinite(epss))):
            continue
        try:
            C = fit_mmc4(etas, epss)
        except Exception:
            continue
        C_list.append(C)
        idx_list.append(idx)

    if len(idx_list) == 0:
        raise RuntimeError("C1~C6를 피팅할 수 있는 유효한 행이 없습니다.")

    df_C = df.loc[idx_list].copy()
    C_array = np.vstack(C_list)
    for i, cname in enumerate(C_cols):
        df_C[cname] = C_array[:, i]

    # 3) 입력 특성/범주형 컬럼 정의
    input_cols = [
        "Yield Stress(MPa)",
        "Ultimate Tensile Stress(MPa)",
        "Total Elongation",
        "r-value",
        "Triaxiality(Tension)",
        "Fracture strain(Tension)",
    ]
    missing_inputs = [c for c in input_cols if c not in df_C.columns]
    if missing_inputs:
        raise RuntimeError(f"입력 컬럼이 엑셀에 없습니다: {missing_inputs}")

    for c in input_cols:
        df_C[c] = pd.to_numeric(df_C[c], errors="coerce")

    cat_inputs = []
    if "Material" in df_C.columns:
        cat_inputs.append("Material")

    df_C = df_C.dropna(subset=input_cols + C_cols, how="any")

    # 4) 특징공학 + 회귀모델 학습 (X → C1~C6)
    X_all = build_enhanced_features(df_C, input_cols, cat_inputs)
    y_all = df_C[C_cols].copy()

    num_cols = [c for c in X_all.columns if c not in cat_inputs]

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', ohe, cat_inputs)
        ],
        remainder='drop'
    )

    def create_et():
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            max_features='sqrt',
            bootstrap=True,
            random_state=RSEED,
            n_jobs=-1
        )

    pipe = Pipeline([
        ('pre', pre),
        ('reg', MultiOutputRegressor(create_et()))
    ])

    pipe.fit(X_all, y_all)

    # 5) 표식용 geometry별 η (데이터가 있으면 중앙값, 없으면 fallback)
    geom_etas = {}
    if "Triaxiality(Shear0)" in df_C.columns:
        geom_etas["Shear"] = float(pd.to_numeric(df_C["Triaxiality(Shear0)"], errors="coerce").median())
    if "Triaxiality(Notch R05)" in df_C.columns:
        geom_etas["Notch"] = float(pd.to_numeric(df_C["Triaxiality(Notch R05)"], errors="coerce").median())
    if "Triaxiality(Punch Bulge)" in df_C.columns:
        geom_etas["Bulge"] = float(pd.to_numeric(df_C["Triaxiality(Punch Bulge)"], errors="coerce").median())

    for k, v in GEOM_ETA_FALLBACK.items():
        geom_etas.setdefault(k, v)

    num_medians = df_C[input_cols].median(numeric_only=True).to_dict()

    meta = {
        "input_cols": input_cols,
        "cat_inputs": cat_inputs,
        "C_cols": C_cols,
        "geom_etas": geom_etas,
        "num_medians": num_medians,
    }
    return pipe, meta

# ===== 모델 로드 =====
try:
    model, meta = load_and_train()
except Exception as e:
    st.error(f"모델 학습/로드 실패: {e}")
    st.stop()

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
C_COLS = meta["C_cols"]
GEOM_ETAS = meta["geom_etas"]
MEDIANS = meta.get("num_medians", {})

# ===== 입력 폼 =====
st.header("입력값")

c1, c2 = st.columns(2)
with c1:
    ys = st.number_input(
        "Yield Stress(MPa)",
        value=float(MEDIANS.get("Yield Stress(MPa)", 200.0))
    )
    uts = st.number_input(
        "Ultimate Tensile Stress(MPa)",
        value=float(MEDIANS.get("Ultimate Tensile Stress(MPa)", 350.0))
    )
    tel = st.number_input(
        "Total Elongation",
        value=float(MEDIANS.get("Total Elongation", 0.20)),
        format="%.5f"
    )
with c2:
    rv = st.number_input(
        "r-value",
        value=float(MEDIANS.get("r-value", 1.00)),
        format="%.5f"
    )
    etaT = st.number_input(
        "Triaxiality(Tension)",
        value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)),
        min_value=-0.5, max_value=0.7, step=0.01, format="%.2f"
    )
    efT = st.number_input(
        "Fracture strain(Tension)",
        value=float(MEDIANS.get("Fracture strain(Tension)", 0.20)),
        min_value=0.0, step=0.001, format="%.5f"
    )

extra_inputs = [
    c for c in INPUTS
    if c not in [
        "Yield Stress(MPa)", "Ultimate Tensile Stress(MPa)",
        "Total Elongation", "r-value",
        "Triaxiality(Tension)", "Fracture strain(Tension)"
    ]
]

adv_vals = {}
if extra_inputs or CAT_INPUTS:
    with st.expander("고급 옵션", expanded=False):
        for c in CAT_INPUTS:
            adv_vals[c] = st.text_input(c, value="")
        for c in extra_inputs:
            adv_vals[c] = st.number_input(
                c, value=float(MEDIANS.get(c, 0.0))
            )

# ===== 예측 및 MMC4 플롯 =====
if st.button("C1~C6 예측 및 MMC4 곡선 그리기"):
    # (1) 입력 한 줄 구성
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

    # (2) 특징공학
    X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

    # (3) 파이프라인 스키마에 맞추기
    pre = model.named_steps["pre"]
    num_cols_model = []
    cat_cols_model = []

    for name, trans, cols in pre.transformers:
        if name == "num":
            num_cols_model = list(cols)
        elif name == "cat" and cols is not None and cols != "drop":
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

    # (4) C1~C6 예측
    C_hat = model.predict(X_feed)[0]
    C_hat = np.asarray(C_hat, dtype=float)

    # (5) 곡선 및 4점(이론점) 계산
    eta_grid = np.linspace(-0.1, 0.7, 400)
    eps_curve = mmc4_eps(eta_grid, C_hat)

    # 표식을 찍기 위한 대표 η
    eta_shear = GEOM_ETAS["Shear"]
    eta_notch = GEOM_ETAS["Notch"]
    eta_bulge = GEOM_ETAS["Bulge"]

    ef_shear = mmc4_eps(eta_shear, C_hat)
    ef_notch = mmc4_eps(eta_notch, C_hat)
    ef_bulge = mmc4_eps(eta_bulge, C_hat)

    pts = [
        ("Shear",   eta_shear, ef_shear),
        ("Tension", float(etaT), mmc4_eps(float(etaT), C_hat)),  # 곡선상의 Tension 값
        ("Notch",   eta_notch, ef_notch),
        ("Bulge",   eta_bulge, ef_bulge),
    ]

    # (6) 플롯
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta_grid, eps_curve, lw=2, label="MMC4 curve (from C1~C6)")

    colors = {
        "Shear": "C0",
        "Tension": "C1",
        "Notch": "C2",
        "Bulge": "C3",
    }
    for name, x, y in pts:
        ax.scatter([x], [y], s=60, color=colors.get(name, "k"),
                   edgecolor="k", zorder=5, label=name)
        ax.annotate(
            name, (x, y),
            xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_title("MMC4 Curve (C1~C6 predicted)")
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(0.0, float(ef_bulge) + 0.3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**대표 4점 (MMC4 곡선 상)**")
    for name, x, y in pts:
        st.text(f"{name:8s} η={x:.4f}  εf={y:.5f}")

    st.markdown("**예측된 MMC4 파라미터 C1~C6**")
    st.text("C = [" + ", ".join(f"{v:.6f}" for v in C_hat) + "]")
