# -*- coding: utf-8 -*-
"""
app.py — 엑셀에서 바로 학습해서 MMC4 curve를 그려주는 Streamlit 앱
- train_export_model.py + app_mmc4_compact.py 합본 버전
- pkl 없이 실행 시점에 모델을 학습하므로, scikit-learn 버전 차이로 인한 pickle 오류 없음.
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

# ===== 기본 설정 =====
DATA_XLSX = "250811_산학프로젝트_포스코의 워크시트.xlsx"  # 같은 폴더에 두기
SHEET_FALLBACK = "학습DB"
RSEED = 20250821
np.random.seed(RSEED)

# 타깃(5개): Notch 2 + Shear 2 + Bulge εf
TARGETS_FIXED = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]

# Tension η, εf 는 항상 "입력"으로 사용
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}

st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")


# ===== 엑셀/색상 기반 유틸 =====
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
        return "yellow"      # 입력
    if r <= 180 and g >= 170 and b >= 190:
        return "skyblue"     # 출력
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
    """중복 컬럼이 있어도 단일 Series로 축약(왼쪽→오른쪽 첫 유효값)."""
    if colname not in X.columns:
        return None
    cols = [c for c in X.columns if c == colname]
    sub = X[cols]
    if isinstance(sub, pd.Series):
        return pd.to_numeric(sub, errors="coerce")
    sub_num = sub.apply(pd.to_numeric, errors="coerce")
    ser = sub_num.bfill(axis=1).iloc[:, 0]
    return ser


# ===== 특징공학 =====
def build_enhanced_features(df_, input_cols, cat_inputs):
    """train_export_model.py와 동일한 특징공학을 단일 함수로 분리"""
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

    # 추가 상호작용
    key = [('Total Elongation', tel), ('r-value', rv), ('Triaxiality(Tension)', etaT)]
    for i in range(len(key)):
        for j in range(i + 1, len(key)):
            n1, s1 = key[i]
            n2, s2 = key[j]
            if s1 is not None and s2 is not None:
                X[f'{n1}_x_{n2}'] = s1 * s2
    return X


# ===== 모델 학습 (엑셀 → 파이프라인) =====
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

    # 색상 정보가 거의 없을 때 이름으로 추론
    if len(output_cols) == 0:
        inf_in, inf_out = infer_by_name(list(df.columns))
        output_cols = [c for c in inf_out if pd.api.types.is_numeric_dtype(df[c])]
        input_cols = [c for c in df.columns
                      if c not in output_cols and pd.api.types.is_numeric_dtype(df[c])]

    cat_inputs = ['Material'] if 'Material' in df.columns else []

    # Tension η, εf 강제로 입력에 포함
    for c in list(df.columns):
        if c in FORCE_INPUT:
            if c in output_cols:
                output_cols.remove(c)
            if pd.api.types.is_numeric_dtype(df[c]) and c not in input_cols:
                input_cols.append(c)

    # 타깃은 고정 리스트 중 실제 존재하는 것만 사용
    output_cols = [c for c in TARGETS_FIXED if c in df.columns]

    # 숫자 변환
    for col in set(input_cols + output_cols):
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    needed = list(dict.fromkeys(input_cols + cat_inputs + output_cols))
    df_model = df[needed].copy()
    df_model = df_model.dropna(
        subset=[c for c in output_cols if c in df_model.columns],
        how="any"
    )

    X_all = build_enhanced_features(df_model, input_cols, cat_inputs)
    y_all = df_model[[c for c in output_cols if c in df_model.columns]].copy()

    # 파이프라인 구성
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

    meta = {
        "input_cols": input_cols,
        "cat_inputs": cat_inputs,
        "output_cols": list(y_all.columns),
        "num_medians": df[input_cols].apply(
            pd.to_numeric, errors="coerce"
        ).median(numeric_only=True).to_dict(),
        "feature_columns": list(X_all.columns)
    }
    return pipe, meta


# ===== 모델 로드/학습 호출 =====
try:
    model, meta = load_and_train()
except Exception as e:
    st.error(f"모델 학습/로드 실패: {e}")
    st.stop()

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]
MEDIANS = meta.get("num_medians", {})


# ===== 입력 폼 =====
st.header("입력값(노란색)")
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
        value=0.20, min_value=0.0, step=0.001, format="%.5f"
    )

# INPUTS 중에서 위 6개를 제외한 나머지 + 범주형 입력(있다면)을 고급 옵션으로
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
            adv_vals[c] = st.number_input(
                c, value=float(MEDIANS.get(c, 0.0))
            )


# ===== MMC4 수식 관련 함수 =====
def theta_bar(eta):
    """
    θ̄ = 1 - (2/π) arccos{ -(27/2) η (η² - 1/3) }
    """
    arg = -(27.0 / 2.0) * eta * (eta ** 2 - 1.0 / 3.0)
    # 수치 오차 방지용 클리핑
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0 / np.pi) * np.arccos(arg)


def c6_effective(eta, C6):
    """
    C6(η) =
      1   ,  η ≤ -1/√3  or  0 ≤ η ≤ 1/√3
      C6  ,  -1/√3 < η < 0  or  η > 1/√3
    """
    t = 1.0 / np.sqrt(3.0)
    return np.where(
        (eta <= -t) | ((eta >= 0.0) & (eta <= t)),
        1.0,
        float(C6)
    )


def mmc4_eps(eta, C):
    C1, C2, C3, C4, C5, C6 = C
    tb = theta_bar(eta)
    c6e = c6_effective(eta, C6)

    # k = √3 / (2 - √3)
    k = np.sqrt(3.0) / (2.0 - np.sqrt(3.0))

    term1 = (C1 / C4) * (
        C5 + k * (c6e - C5) * (1.0 / np.cos(tb * np.pi / 6.0) - 1.0)
    )

    base = np.sqrt((1.0 + C3**2) / 3.0) * np.cos(tb * np.pi / 6.0) \
           + C3 * (eta + (1.0 / 3.0) * np.sin(tb * np.pi / 6.0))

    base = np.maximum(base, 1e-6)

    return term1 * (base ** (-1.0 / C2))



def fit_mmc4(etas, epss):
    """
    주어진 4점(또는 N점)의 (η, εf)에 대해 MMC4 파라미터 C1~C6 피팅
    """
    def resid(p):
        return mmc4_eps(etas, p) - epss

    # 초기값과 bounds는 논문/경험 기반의 적당한 범위
    x0 = np.array([1.0, 1.0, 0.2, 1.0, 0.6, 0.8])
    lb = np.array([0.001, 0.10, -2.0, 0.10, 0.0, 0.0])
    ub = np.array([10.0, 5.0,  2.0, 5.0, 2.0, 2.0])

    res = least_squares(
        resid, x0, bounds=(lb, ub),
        max_nfev=5000, verbose=0
    )
    return res.x



# ===== 예측 및 플롯 =====
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

    # (2) 누락 입력 컬럼 보완
    for col in INPUTS:
        if col not in x_one.columns:
            x_one[col] = np.nan
    for col in CAT_INPUTS:
        if col not in x_one.columns:
            x_one[col] = ""

    # (3) 특징공학
    X_tmp = build_enhanced_features(x_one, INPUTS, CAT_INPUTS)

    # (4) 파이프라인 스키마에 맞춰 정렬
    pre = model.named_steps['pre']
    num_cols_model = list(
        next(t[2] for t in pre.transformers if t[0] == 'num')
    )
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

    # (5) 예측 (5 타깃)
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # (6) 4점 구성 — Tension은 입력값 사용, Bulge η=2/3
    eta_shear = y_hat.get("Triaxiality(Shear0)")
    ef_shear = y_hat.get("Fracture strain(Shear0)")
    eta_notch = y_hat.get("Triaxiality(Notch R05)")
    ef_notch = y_hat.get("Fracture strain(Notch R05)")
    ef_bulge = y_hat.get("Fracture strain(Punch Bulge)")

    eta_tens = float(etaT)
    ef_tens = float(efT)
    eta_bulge = 2.0 / 3.0

    missing = []
    if eta_shear is None:
        missing.append("η(Shear)")
    if ef_shear is None:
        missing.append("εf(Shear)")
    if eta_notch is None:
        missing.append("η(Notch R05)")
    if ef_notch is None:
        missing.append("εf(Notch R05)")
    if ef_bulge is None:
        missing.append("εf(Bulge)")

    if missing:
        st.error("부족한 값: " + ", ".join(missing) + " — 엑셀/학습 스키마를 확인하세요.")
        st.stop()

    pts = [
        ("Shear", float(eta_shear), float(ef_shear)),
        ("Tension", float(eta_tens), float(ef_tens)),
        ("Notch", float(eta_notch), float(ef_notch)),
        ("Bulge", float(eta_bulge), float(ef_bulge)),
    ]
    pts = sorted(pts, key=lambda t: t[1])

    etas = np.array([p[1] for p in pts])
    epss = np.array([p[2] for p in pts])
    C_hat = fit_mmc4(etas, epss)

    # (7) 플롯
    # η 그리드 범위: -0.1 ~ 0.7
    eta_grid = np.linspace(-0.1, 0.7, 200)
    eps_curve = mmc4_eps(eta_grid, C_hat)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta_grid, eps_curve, lw=2, label="MMC4 curve (fit)")

    for name, x, y in pts:
        ax.scatter([x], [y], s=60, edgecolor="k", zorder=5, label=name)
        ax.annotate(
            name, (x, y),
            xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    ax.set_xlabel("Triaxiality (η)")
    ax.set_ylabel("Fracture strain (εf)")
    ax.set_title("MMC4 Curve")

    # 축 범위 고정: x축 -0.1~0.7, y축 0~Bulge점 εf + 0.3
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(0.0, float(ef_bulge) + 0.3)

    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.markdown("**4점 요약**")
    for name, x, y in pts:
        st.text(f"{name:8s} η={x:.4f}  εf={y:.5f}")

    st.markdown("**적합 파라미터 C1~C6**")
    st.text("C = [" + ", ".join(f"{v:.6f}" for v in C_hat) + "]")
