# -*- coding: utf-8 -*-
"""
app.py — 엑셀에서 바로 학습해서 MMC4 curve를 그려주는 Streamlit 앱
- train_export_model.py + app_mmc4_compact.py 통합
- mmc4_model.pkl / pickle 전혀 사용하지 않음
- 앱 실행 시 엑셀에서 바로 학습하고, 그 모델로 MMC4 곡선 도식화

추가 요구사항 반영
1) 원본 데이터(약 1000행) 전부에 대해 MMC4 Curve를 계산/플롯 (배경, 연하게)
2) Mild/HSS/AHSS 별 색상: 주황/초록/파랑 (배경/대표곡선 동일)
3) 배경 곡선에는 예측 점(4점) 표시하지 않음
4) MMC4 Curve 공간에서 Mild/HSS/AHSS 대표 곡선을 “구분선”으로 표시(각 그룹 median 곡선)
5) 1~4를 배경으로 연하게 삽입
6) 기존처럼 사용자 6개 입력 → 해당 소재 curve 출력
7) curve 위치 기반으로 Mild/HSS/AHSS 분류 결과 출력 (C 파라미터 출력과 η_query 출력 사이)
"""

import warnings
warnings.filterwarnings("ignore")

import os
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
from scipy.interpolate import CubicHermiteSpline


# =========================
# 기본 설정
# =========================
DATA_XLSX = "250811_산학프로젝트_포스코의 워크시트.xlsx"  # 같은 폴더에 두기
SHEET_FALLBACK = "학습DB"

TARGETS_FIXED = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}

ETA_BULGE = 2.0 / 3.0
ETA_LO, ETA_HI = -0.1, 0.7

# 색상(요구사항)
CLR = {
    "Mild":  "#FF7F0E",  # 주황
    "HSS":   "#2CA02C",  # 초록
    "AHSS":  "#1F77B4",  # 파랑
    "USER":  "#111111",  # 사용자 곡선(검정 계열)
}

# 강종(A~J) → 구분(표 기반)
GRADE_TO_FAMILY = {
    "A": "Mild",
    "B": "HSS",
    "C": "HSS",
    "D": "AHSS",
    "E": "HSS",
    "F": "AHSS",
    "G": "AHSS",
    "H": "AHSS",
    "I": "AHSS",
    "J": "AHSS",
}

st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 MMC4 곡선 시각화")


# =========================
# 세션 상태 초기화
# =========================
if "mmc4_ready" not in st.session_state:
    st.session_state["mmc4_ready"] = False

# =========================
# Sidebar: 재학습 트리거
# =========================
st.sidebar.header("학습/표시")
st.sidebar.caption("random_state=None이더라도 Streamlit cache가 있으면 같은 모델이 유지될 수 있습니다.")
do_retrain = st.sidebar.button("모델 다시 학습(랜덤)")
show_background = st.sidebar.checkbox("배경(1000개 MMC4 Curve) 표시", value=True)
bg_stride = st.sidebar.number_input("배경 곡선 간격(stride, 1=전부)", min_value=1, max_value=50, value=1, step=1)
bg_alpha = st.sidebar.slider("배경 곡선 투명도", 0.01, 0.30, 0.08, 0.01)


# =========================
# 엑셀/색상 기반 유틸
# =========================
def read_excel_safe(path: str) -> pd.DataFrame:
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
    if r >= 200 and g >= 200 and b <= 120:
        return "yellow"
    if r <= 180 and g >= 170 and b >= 190:
        return "skyblue"
    return None


def infer_by_name(columns):
    in_pats = ["yield", "ultimate", "elongation", "r-value", "tension", "입력", "input"]
    out_pats = ["notch", "shear", "bulge", "punch", "fracture", "triaxiality", "예측", "output"]
    ins, outs = [], []
    for c in columns:
        cl = str(c).lower()
        if any(p in cl for p in in_pats):
            ins.append(c)
        if any(p in cl for p in out_pats):
            outs.append(c)
    return ins, outs


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


def grade_to_family_from_material(material_value) -> str:
    """
    Material 컬럼의 첫 글자를 A~J 강종으로 가정.
    예: 'A...', 'B...', 'C...' 형태.
    """
    if material_value is None:
        return "UNKNOWN"
    s = str(material_value).strip()
    if len(s) == 0:
        return "UNKNOWN"
    g = s[0].upper()
    return GRADE_TO_FAMILY.get(g, "UNKNOWN")


# =========================
# 특징공학
# =========================
def build_enhanced_features(df_, input_cols, cat_inputs):
    X = df_[input_cols + cat_inputs].copy()

    ys = _as_series_single(X, 'Yield Stress(MPa)')
    uts = _as_series_single(X, 'Ultimate Tensile Stress(MPa)')
    tel = _as_series_single(X, 'Total Elongation')
    rv = _as_series_single(X, 'r-value')
    etaT = _as_series_single(X, 'Triaxiality(Tension)')
    efT = _as_series_single(X, 'Fracture strain(Tension)')

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


# =========================
# MMC4 수식
# =========================
def theta_bar(eta):
    eta = np.asarray(eta, dtype=float)
    arg = -(27.0 / 2.0) * eta * (eta ** 2 - 1.0 / 3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0 / np.pi) * np.arccos(arg)


def c6_effective(eta, C6):
    t = 1.0 / np.sqrt(3.0)
    eta = np.asarray(eta, dtype=float)
    return np.where((eta <= -t) | ((eta >= 0) & (eta <= t)), 1.0, float(C6))


def mmc4_eps(eta, C):
    C1, C2, C3, C4, C5, C6 = C
    eta = np.asarray(eta, dtype=float)

    tb = theta_bar(eta)
    c6e = c6_effective(eta, C6)
    k = np.sqrt(3.0) / (2.0 - np.sqrt(3.0))

    term1 = (C1 / C4) * (
        C5 + k * (c6e - C5) * (1.0 / np.cos(tb * np.pi / 6.0) - 1.0)
    )

    base = np.sqrt(1.0 + (C3 ** 2) / 3.0) * np.cos(tb * np.pi / 6.0) \
           + C3 * (eta + (1.0 / 3.0) * np.sin(tb * np.pi / 6.0))

    base = np.maximum(base, 1e-6)
    return term1 * (base ** (-1.0 / C2))


def fit_mmc4(etas, epss, x0=None):
    """
    4점(etas, epss)에 대해 C1~C6 최적화.
    배경 1000곡선 속도를 위해 max_nfev는 과도하게 키우지 않음.
    """
    etas = np.asarray(etas, dtype=float)
    epss = np.asarray(epss, dtype=float)

    def resid(p):
        return mmc4_eps(etas, p) - epss

    if x0 is None:
        x0 = np.array([1.0, 1.0, 0.2, 1.0, 0.6, 0.8])

    lb = np.array([0.001, 0.10, -2.0, 0.10, 0.0, 0.0])
    ub = np.array([10.0, 5.0,  2.0, 5.0,  2.0, 2.0])

    res = least_squares(
        resid, x0, bounds=(lb, ub),
        max_nfev=1800,  # 배경 계산 고려
        verbose=0
    )
    return res.x


def build_curve_from_C(C_hat, eta_grid):
    """
    요구사항:
    - η <= 2/3: MMC4 식 그대로
    - η > 2/3: 원본(app_mmc4_compact.py) 우측 Hermite 외삽 그대로
    """
    eta_grid = np.asarray(eta_grid, dtype=float)
    eps_curve = mmc4_eps(eta_grid, C_hat)

    # Bulge 기준 우측 Hermite 외삽
    h = 1e-4
    def d_mmc4(e):
        return float((mmc4_eps(e + h, C_hat) - mmc4_eps(e - h, C_hat)) / (2.0 * h))

    e_max = float(mmc4_eps(ETA_BULGE, C_hat))
    d_max = d_mmc4(ETA_BULGE)

    y_hi = e_max + d_max * (float(eta_grid.max()) - ETA_BULGE)
    right_conn = CubicHermiteSpline([ETA_BULGE, float(eta_grid.max())], [e_max, y_hi], [d_max, d_max])

    mask_right = eta_grid > ETA_BULGE
    if np.any(mask_right):
        eps_curve[mask_right] = right_conn(eta_grid[mask_right])

    return eps_curve


# =========================
# 모델 학습 (엑셀 → 파이프라인)
# =========================
@st.cache_resource
def load_and_train():
    if not os.path.exists(DATA_XLSX):
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없음: {DATA_XLSX}")

    wb = load_workbook(DATA_XLSX, data_only=True)
    sheet_name = wb.sheetnames[0] if wb.sheetnames else SHEET_FALLBACK
    ws = wb[sheet_name]

    df = read_excel_safe(DATA_XLSX)
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

    if len(output_cols) == 0:
        _, inf_out = infer_by_name(list(df.columns))
        output_cols = [c for c in inf_out if pd.api.types.is_numeric_dtype(df[c])]
        input_cols = [c for c in df.columns if c not in output_cols and pd.api.types.is_numeric_dtype(df[c])]

    cat_inputs = ['Material'] if 'Material' in df.columns else []

    for c in list(df.columns):
        if c in FORCE_INPUT:
            if c in output_cols:
                output_cols.remove(c)
            if pd.api.types.is_numeric_dtype(df[c]) and c not in input_cols:
                input_cols.append(c)

    output_cols = [c for c in TARGETS_FIXED if c in df.columns]

    for col in set(input_cols + output_cols):
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    needed = list(dict.fromkeys(input_cols + cat_inputs + output_cols))
    df_model = df[needed].copy()
    df_model = df_model.dropna(subset=[c for c in output_cols if c in df_model.columns], how="any")

    X_all = build_enhanced_features(df_model, input_cols, cat_inputs)
    y_all = df_model[[c for c in output_cols if c in df_model.columns]].copy()

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
            random_state=None,   # 고정하지 않음
            n_jobs=-1
        )

    pipe = Pipeline([
        ('pre', pre),
        ('reg', MultiOutputRegressor(create_et()))
    ])

    pipe.fit(X_all, y_all)

    meta = {
        "input_cols": input_cols,
        "cat_inputs": cat_inputs,
        "output_cols": list(y_all.columns),
        "num_medians": df[input_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True).to_dict(),
        "feature_columns": list(X_all.columns),
        "has_material": ("Material" in df.columns),
    }
    return pipe, meta


# 재학습 버튼 처리 (cache clear + 결과 초기화)
if do_retrain:
    load_and_train.clear()
    st.session_state["mmc4_ready"] = False
    for k in ["pts", "C_hat", "ef_bulge", "e_max", "d_max", "user_curve", "family_pred", "family_scores"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# =========================
# 모델 로드/학습
# =========================
try:
    model, meta = load_and_train()
except Exception as e:
    st.error(f"모델 학습 실패: {e}")
    st.stop()

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]
MEDIANS = meta.get("num_medians", {})


# =========================
# 배경용 데이터 로드
# =========================
@st.cache_data(show_spinner=False)
def load_dataset_for_background():
    df = read_excel_safe(DATA_XLSX)
    df = df.dropna(axis=1, how="all").copy()

    # 숫자 변환(가능한 컬럼만)
    for c in df.columns:
        if c == "Material":
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass

    # 최소 필요: 6 입력 + Material(있으면) + tension 2개
    must = [
        "Yield Stress(MPa)",
        "Ultimate Tensile Stress(MPa)",
        "Total Elongation",
        "r-value",
        "Triaxiality(Tension)",
        "Fracture strain(Tension)",
    ]
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise ValueError(f"배경 곡선 계산에 필요한 컬럼이 엑셀에 없습니다: {missing}")

    # 배경은 “행 단위”로 동작: tension η, εf가 NaN이면 불가
    df = df.dropna(subset=must, how="any").reset_index(drop=True)

    # 강종 패밀리
    if "Material" in df.columns:
        df["_FAMILY_"] = df["Material"].apply(grade_to_family_from_material)
    else:
        df["_FAMILY_"] = "UNKNOWN"

    return df


@st.cache_data(show_spinner=False)
def compute_background_curves(stride: int = 1):
    """
    원본 데이터셋 전체(또는 stride 간격) 곡선을 계산하여 저장.
    - 계산량이 커서 cache_data로 묶음.
    """
    df = load_dataset_for_background()
    eta_grid = np.linspace(ETA_LO, ETA_HI, 250)

    curves = {
        "Mild": [],
        "HSS": [],
        "AHSS": [],
    }

    # preprocessor 스키마 준비
    pre = model.named_steps["pre"]
    num_cols_model = list(next(t[2] for t in pre.transformers if t[0] == "num"))
    cat_cols_model = []
    for name, trans, cols in pre.transformers:
        if name == "cat" and cols is not None and cols != "drop":
            cat_cols_model = list(cols)

    last_x0 = None
    n = len(df)
    idxs = list(range(0, n, int(stride)))

    for i in idxs:
        row = df.iloc[i]

        family = row["_FAMILY_"]
        if family not in curves:
            continue

        # 입력 6개 + (Material 있으면 포함)
        x_dict = {
            "Yield Stress(MPa)": row["Yield Stress(MPa)"],
            "Ultimate Tensile Stress(MPa)": row["Ultimate Tensile Stress(MPa)"],
            "Total Elongation": row["Total Elongation"],
            "r-value": row["r-value"],
            "Triaxiality(Tension)": row["Triaxiality(Tension)"],
            "Fracture strain(Tension)": row["Fracture strain(Tension)"],
        }
        if "Material" in df.columns:
            x_dict["Material"] = row["Material"]

        x_one = pd.DataFrame([x_dict])

        # 누락 입력 보완(모델 input_cols 기준)
        for col in INPUTS:
            if col not in x_one.columns:
                x_one[col] = np.nan
        for col in CAT_INPUTS:
            if col not in x_one.columns:
                x_one[col] = ""

        # 특징공학
        X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

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

        # 예측
        try:
            y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)
        except Exception:
            continue

        eta_shear = y_hat.get("Triaxiality(Shear0)")
        ef_shear  = y_hat.get("Fracture strain(Shear0)")
        eta_notch = y_hat.get("Triaxiality(Notch R05)")
        ef_notch  = y_hat.get("Fracture strain(Notch R05)")
        ef_bulge  = y_hat.get("Fracture strain(Punch Bulge)")
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [eta_shear, ef_shear, eta_notch, ef_notch, ef_bulge]):
            continue

        eta_tens = float(row["Triaxiality(Tension)"])
        ef_tens  = float(row["Fracture strain(Tension)"])

        pts = [
            (float(eta_shear), float(ef_shear)),
            (float(eta_tens),  float(ef_tens)),
            (float(eta_notch), float(ef_notch)),
            (float(ETA_BULGE), float(ef_bulge)),
        ]
        pts = sorted(pts, key=lambda t: t[0])
        etas = np.array([p[0] for p in pts], dtype=float)
        epss = np.array([p[1] for p in pts], dtype=float)

        try:
            C_hat = fit_mmc4(etas, epss, x0=last_x0)
            last_x0 = C_hat.copy()
            curve = build_curve_from_C(C_hat, eta_grid)
        except Exception:
            continue

        curves[family].append(curve.astype(float))

    # 그룹 대표곡선(median)
    med = {}
    for fam, arr in curves.items():
        if len(arr) == 0:
            med[fam] = None
        else:
            M = np.vstack(arr)
            med[fam] = np.median(M, axis=0)

    return eta_grid, curves, med


# =========================
# 입력 폼 (사용자 6개)
# =========================
st.header("입력값(노란색)")
c1, c2 = st.columns(2)

with c1:
    ys = st.number_input("Yield Stress(MPa)", value=float(MEDIANS.get("Yield Stress(MPa)", 200.0)))
    uts = st.number_input("Ultimate Tensile Stress(MPa)", value=float(MEDIANS.get("Ultimate Tensile Stress(MPa)", 350.0)))
    tel = st.number_input("Total Elongation", value=float(MEDIANS.get("Total Elongation", 0.20)), format="%.5f")

with c2:
    rv = st.number_input("r-value", value=float(MEDIANS.get("r-value", 1.00)), format="%.5f")
    etaT = st.number_input("Triaxiality(Tension)", value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)),
                           min_value=-0.5, max_value=0.7, step=0.01, format="%.2f")
    efT = st.number_input("Fracture strain(Tension)", value=0.20, min_value=0.0, step=0.001, format="%.5f")

extra_inputs = [
    c for c in INPUTS
    if c not in [
        "Yield Stress(MPa)", "Ultimate Tensile Stress(MPa)",
        "Total Elongation", "r-value",
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


# =========================
# 버튼: 사용자 예측 + 곡선 생성
# =========================
if st.button("예측 및 MMC4 플롯"):
    # (1) 입력 한 줄 만들기
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

    # (2) 누락 입력 보완
    for col in INPUTS:
        if col not in x_one.columns:
            x_one[col] = np.nan
    for col in CAT_INPUTS:
        if col not in x_one.columns:
            x_one[col] = ""

    # (3) 특징공학
    X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

    # (4) 파이프라인 입력 스키마 맞추기
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

    # (5) 예측
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # (6) 4점 구성 — Tension은 입력값, Bulge η=2/3 상수
    eta_shear = y_hat.get("Triaxiality(Shear0)")
    ef_shear  = y_hat.get("Fracture strain(Shear0)")
    eta_notch = y_hat.get("Triaxiality(Notch R05)")
    ef_notch  = y_hat.get("Fracture strain(Notch R05)")
    ef_bulge  = y_hat.get("Fracture strain(Punch Bulge)")

    missing = []
    if eta_shear is None: missing.append("η(Shear)")
    if ef_shear is None:  missing.append("εf(Shear)")
    if eta_notch is None: missing.append("η(Notch)")
    if ef_notch is None:  missing.append("εf(Notch)")
    if ef_bulge is None:  missing.append("εf(Bulge)")
    if missing:
        st.error("부족한 값: " + ", ".join(missing) + " — 엑셀/학습 스키마를 확인하세요.")
        st.stop()

    pts = [
        ("Shear",   float(eta_shear), float(ef_shear)),
        ("Tension", float(etaT),      float(efT)),
        ("Notch",   float(eta_notch), float(ef_notch)),
        ("Bulge",   float(ETA_BULGE), float(ef_bulge)),
    ]
    pts = sorted(pts, key=lambda t: t[1])

    etas = np.array([p[1] for p in pts], dtype=float)
    epss = np.array([p[2] for p in pts], dtype=float)
    C_hat = fit_mmc4(etas, epss)

    # 사용자 곡선(그리드 고정)
    eta_grid_user = np.linspace(ETA_LO, ETA_HI, 250)
    user_curve = build_curve_from_C(C_hat, eta_grid_user)

    # 세션 저장
    st.session_state["mmc4_ready"] = True
    st.session_state["pts"] = pts
    st.session_state["C_hat"] = C_hat
    st.session_state["ef_bulge"] = float(ef_bulge)
    st.session_state["eta_grid_user"] = eta_grid_user
    st.session_state["user_curve"] = user_curve


# =========================
# 결과 표시
# =========================
if st.session_state.get("mmc4_ready", False):
    pts = st.session_state["pts"]
    C_hat = st.session_state["C_hat"]
    ef_bulge = st.session_state["ef_bulge"]
    eta_grid_user = st.session_state["eta_grid_user"]
    user_curve = st.session_state["user_curve"]

    # ---- 배경 곡선 계산(캐시) ----
    bg_eta_grid, bg_curves, bg_median = None, None, None
    if show_background:
        with st.spinner("배경(전체 데이터) MMC4 Curve 계산/로드 중... (최초 1회만 느릴 수 있음)"):
            bg_eta_grid, bg_curves, bg_median = compute_background_curves(stride=int(bg_stride))

    # ---- 사용자 강종 분류(대표곡선과의 거리) ----
    family_pred = None
    family_scores = {}
    if show_background and (bg_median is not None):
        # 사용자 곡선과 배경 대표곡선의 그리드가 같도록 보정(필요 시 선형 보간)
        # (현재 둘 다 250 포인트, 범위 동일이므로 대부분 동일하지만 안전 장치)
        if bg_eta_grid is not None and not np.allclose(bg_eta_grid, eta_grid_user):
            uc = np.interp(bg_eta_grid, eta_grid_user, user_curve)
        else:
            uc = user_curve.copy()

        for fam in ["Mild", "HSS", "AHSS"]:
            ref = bg_median.get(fam, None) if bg_median else None
            if ref is None:
                continue
            rmse = float(np.sqrt(np.mean((uc - ref) ** 2)))
            family_scores[fam] = rmse

        if len(family_scores) > 0:
            family_pred = min(family_scores.items(), key=lambda kv: kv[1])[0]

        st.session_state["family_pred"] = family_pred
        st.session_state["family_scores"] = family_scores

    # ---- 플롯(배경 + 구분선 + 사용자곡선) ----
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # 1) 배경(연하게)
    if show_background and bg_curves is not None:
        for fam in ["Mild", "HSS", "AHSS"]:
            arr = bg_curves.get(fam, [])
            if len(arr) == 0:
                continue
            for curve in arr:
                ax.plot(bg_eta_grid, curve, color=CLR[fam], alpha=float(bg_alpha), lw=1)

    # 2) 구분선(대표 median curve)
    if show_background and bg_median is not None:
        for fam in ["Mild", "HSS", "AHSS"]:
            med = bg_median.get(fam, None)
            if med is None:
                continue
            ax.plot(
                bg_eta_grid, med,
                color=CLR[fam], lw=2.6, alpha=1.0,
                label=f"{fam} 대표 곡선(중앙값)"
            )

    # 3) 사용자 곡선(진하게)
    ax.plot(eta_grid_user, user_curve, color=CLR["USER"], lw=2.8, label="사용자 MMC4 Curve")

    # (요구사항) 배경에는 예측 점 안 찍음. 사용자 점도 기본은 안 찍음.
    # 필요하면 아래 주석 해제:
    # for name, x, y in pts:
    #     ax.scatter([x], [y], s=50, edgecolor="k", zorder=5)

    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_title("MMC4 Curve (Background: Full Dataset / Grouped by Mild-HSS-AHSS)")
    ax.set_xlim(ETA_LO, ETA_HI)
    ax.set_ylim(0.0, float(ef_bulge) + 0.3)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    st.pyplot(fig)

    # ---- 4점/파라미터 출력 ----
    st.markdown("**4점 요약(사용자)**")
    for name, x, y in pts:
        st.text(f"{name:8s} η={x:.4f}  εf={y:.5f}")

    st.markdown("**적합 파라미터 C1~C6(사용자)**")
    st.text("C = [" + ", ".join(f"{v:.6f}" for v in C_hat) + "]")

    # ---- (추가) 강종 계열 분류 결과 ----
    st.markdown("**강종 계열 분류(곡선 위치 기반)**")
    if not show_background:
        st.write("배경 곡선 표시가 꺼져 있어 분류를 수행하지 않았습니다. (사이드바에서 배경 표시를 켜세요.)")
    else:
        family_pred = st.session_state.get("family_pred", None)
        scores = st.session_state.get("family_scores", {})
        if family_pred is None or len(scores) == 0:
            st.write("분류에 필요한 대표 곡선이 충분하지 않아 분류를 수행할 수 없습니다.")
        else:
            st.write(f"결론: 해당 강종은 **{family_pred}** 강으로 분류됩니다.")
            # 점수(낮을수록 가까움)
            st.caption("각 그룹 대표 곡선과의 거리(RMSE, 낮을수록 유사)")
            st.write({k: float(f"{v:.6f}") for k, v in scores.items()})

    # ---- 특정 η 입력 → εf 출력 ----
    st.markdown("**특정 η 입력 → εf 출력(사용자 곡선)**")
    eta_query = st.number_input(
        "조회할 Triaxiality (η_query)",
        value=float(st.session_state.get("eta_query", 0.30000)),
        min_value=float(ETA_LO),
        max_value=float(ETA_HI),
        step=0.01,
        format="%.5f",
        key="eta_query",
    )

    # 사용자 곡선에서 보간(구간별 로직을 그대로 쓰려면 C로 직접 계산해도 되지만,
    # 이미 user_curve가 해당 로직을 반영한 값이므로 여기서는 안전하게 보간)
    eps_query = float(np.interp(eta_query, eta_grid_user, user_curve))
    st.write(f"결과: η = {eta_query:.5f} 에서 εf = {eps_query:.6f} 입니다.")

else:
    st.info("입력값을 설정한 뒤, '예측 및 MMC4 플롯' 버튼을 눌러 결과를 생성하세요.")
