# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from scipy.optimize import least_squares
from scipy.interpolate import CubicHermiteSpline


# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="MMC4 Predictor", layout="wide")
st.title("POSCO MMC4 — 모델 예측 기반 곡선 시각화")


# =========================
# 모델 로드 (pkl 파일 또는 업로드)
# =========================
@st.cache_resource(show_spinner=False)
def load_bundle_from_path(pkl_path: str):
    bundle = joblib.load(pkl_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "meta" not in bundle:
        raise ValueError("mmc4_model.pkl 포맷이 올바르지 않습니다. (bundle['model'], bundle['meta'] 필요)")
    return bundle

@st.cache_resource(show_spinner=False)
def load_bundle_from_bytes(pkl_bytes: bytes):
    bio = BytesIO(pkl_bytes)
    bundle = joblib.load(bio)
    if not isinstance(bundle, dict) or "model" not in bundle or "meta" not in bundle:
        raise ValueError("업로드된 pkl 포맷이 올바르지 않습니다. (bundle['model'], bundle['meta'] 필요)")
    return bundle


st.sidebar.header("모델(pkl) 설정")
default_pkl = st.sidebar.text_input("pkl 파일명/경로", value="mmc4_model.pkl")
uploaded_pkl = st.sidebar.file_uploader("pkl 업로드(옵션)", type=["pkl"])

bundle = None
load_err = None


def resolve_pkl_path(name_or_path: str) -> Path | None:
    """
    Streamlit Cloud에서 cwd가 달라져도, app.py 위치 기준으로 pkl을 찾도록 보강.
    """
    here = Path(__file__).resolve().parent

    candidates = [
        Path(name_or_path),                 # 현재 작업 디렉토리 기준
        here / name_or_path,                # app.py 위치 기준
        here / "mmc4_model.pkl",             # 표준 파일명
    ]

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass

    # 혹시 폴더 구조가 꼬였을 때를 대비해 repo 내 전체 검색(비용 낮음)
    try:
        hits = list(here.rglob("mmc4_model.pkl"))
        if hits:
            return hits[0]
    except Exception:
        pass

    return None


if uploaded_pkl is not None:
    try:
        bundle = load_bundle_from_bytes(uploaded_pkl.getvalue())
    except Exception as e:
        load_err = f"(업로드 pkl) {e}"
else:
    try:
        pkl_path = resolve_pkl_path(default_pkl)
        if pkl_path is None:
            raise FileNotFoundError(f"{default_pkl} (candidates not found)")
        bundle = load_bundle_from_path(str(pkl_path))
        st.sidebar.caption(f"로드된 pkl: {pkl_path}")
    except Exception as e:
        load_err = f"(경로 pkl) {e}"

if bundle is None:
    st.error(f"모델 로드 실패: {load_err}")

    # ---- 디버그 정보 출력 (원인 즉시 확인용) ----
    here = Path(__file__).resolve().parent
    st.write("### 디버그 정보")
    st.write("현재 작업 디렉토리(CWD):", os.getcwd())
    st.write("app.py 폴더:", str(here))

    try:
        st.write("CWD 파일 목록:", sorted(os.listdir(os.getcwd()))[:200])
    except Exception as e:
        st.write("CWD 목록 조회 실패:", e)

    try:
        st.write("app.py 폴더 파일 목록:", sorted(os.listdir(here))[:200])
    except Exception as e:
        st.write("app.py 폴더 목록 조회 실패:", e)

    try:
        hits = [str(p) for p in here.rglob("mmc4_model.pkl")]
        st.write("repo 내 mmc4_model.pkl 검색 결과:", hits)
    except Exception as e:
        st.write("검색 실패:", e)

    st.info("해결: 1) Streamlit Cloud가 최신 커밋을 보고 있는지(브랜치/커밋) 확인  2) Main file path가 루트 app.py인지 확인  3) 임시로 좌측 'pkl 업로드'로 실행")
    st.stop()

model = bundle["model"]
meta  = bundle["meta"]

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]
MEDIANS = meta.get("num_medians", {})


# =========================
# 특징공학
# =========================
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


# =========================
# MMC4 수식/적합
# =========================
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
    etaT = st.number_input(
        "Triaxiality(Tension)",
        value=float(MEDIANS.get("Triaxiality(Tension)", 0.33)),
        min_value=-0.5, max_value=0.7, step=0.01, format="%.2f"
    )
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

    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

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

    etas=np.array([p[1] for p in pts])
    epss=np.array([p[2] for p in pts])
    C_hat=fit_mmc4(etas,epss)

    st.session_state["mmc4_pts"] = pts
    st.session_state["mmc4_C_hat"] = C_hat


# =========================
# 플롯/결과 표시
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

    e_min = float(mmc4_eps(eta_min, C_hat))
    delta = eta_min - eta_lo
    curv = abs(e_min) / (delta**2 + 1e-6) * 0.6

    def left_curve(eta):
        return e_min + curv * (eta - eta_min)**2

    mask_left = eta_grid < eta_min
    if np.any(mask_left):
        eps_curve[mask_left] = left_curve(eta_grid[mask_left])

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
