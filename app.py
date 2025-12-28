# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from mmc4_train_lib import train_bundle_from_excel_bytes

# scipy는 MMC4 피팅/연결에 필요
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares


# ===== 페이지 =====
st.set_page_config(page_title="MMC4 Predictor (Cloud Train)", layout="wide")
st.title("POSCO MMC4 — Cloud에서 학습 + 예측 + 곡선 시각화 (PKL 없음)")


# ===== MMC4 수식/적합 =====
def theta_bar(eta):
    arg = -(27.0/2.0)*eta*(eta**2 - 1.0/3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0/np.pi)*np.arccos(arg)

def c6_effective(eta, C6):
    t = 1.0/np.sqrt(3.0)
    return np.where((eta<=-t)|((eta>=0)&(eta<=t)),1.0,float(C6))

def mmc4_eps(eta, C):
    C1,C2,C3,C4,C5,C6 = C
    tb = theta_bar(eta)
    c6e = c6_effective(eta, C6)
    k = np.sqrt(3.0)/(2.0-np.sqrt(3.0))
    term1 = (C1/C4)*( C5 + k*(c6e-C5)*(1.0/np.cos(tb*np.pi/6.0) - 1.0) )
    base = 1.0 + (C3**2)/3.0*np.cos(tb*np.pi/6.0) + C3*(eta + (1.0/3.0)*np.sin(tb*np.pi/6.0))
    base = np.maximum(base, 1e-6)
    return term1 * ( base ** (-1.0/C2) )

def fit_mmc4(etas, epss):
    def resid(p): return mmc4_eps(etas, p) - epss
    x0 = np.array([1.0,1.0,0.2,1.0,0.6,0.8])
    lb = np.array([0.001,0.10,-2.0,0.10,0.0,0.0])
    ub = np.array([10.0,5.0,2.0,5.0,2.0,2.0])
    res = least_squares(resid, x0, bounds=(lb,ub), max_nfev=5000, verbose=0)
    return res.x


# ===== 학습 캐시 =====
@st.cache_resource(show_spinner=True)
def cached_train(excel_bytes: bytes, sheet_name, n_estimators: int, retrain_nonce: int, bytes_md5: str):
    # bytes_md5는 캐시 키 안정화를 위해 포함(함수 내부에서는 사용 안 함)
    return train_bundle_from_excel_bytes(
        excel_bytes=excel_bytes,
        sheet_name=sheet_name,
        n_estimators=n_estimators,
    )


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# ===== 사이드바: 학습 =====
st.sidebar.header("학습 (PKL 없음)")
uploaded = st.sidebar.file_uploader("학습용 엑셀 업로드 (.xlsx)", type=["xlsx"])

sheet_name_in = st.sidebar.text_input("시트 이름(없으면 비움)", value="").strip()
sheet_name = None if sheet_name_in == "" else sheet_name_in

n_estimators = st.sidebar.slider("n_estimators (클수록 느림)", 50, 1000, 300, 50)

if "retrain_nonce" not in st.session_state:
    st.session_state["retrain_nonce"] = 0

c1, c2, c3 = st.sidebar.columns(3)
with c1:
    train_btn = st.button("학습", use_container_width=True)
with c2:
    bump_btn = st.button("재학습+1", use_container_width=True)
with c3:
    clear_btn = st.button("캐시삭제", use_container_width=True)

if bump_btn:
    st.session_state["retrain_nonce"] += 1
    st.sidebar.success(f"retrain_nonce={st.session_state['retrain_nonce']}")

if clear_btn:
    st.cache_resource.clear()
    st.cache_data.clear()
    st.sidebar.success("캐시를 삭제했습니다.")


# ===== 번들 확보 =====
bundle = None
if uploaded is None:
    st.info("왼쪽에서 엑셀(.xlsx)을 업로드한 뒤 학습을 누르세요.")
    st.stop()

excel_bytes = uploaded.getvalue()
excel_md5 = md5_bytes(excel_bytes)

st.sidebar.caption(f"파일 MD5: {excel_md5[:10]}...")

if train_btn:
    with st.spinner("학습 중... (Cloud 환경에서는 시간이 걸릴 수 있음)"):
        bundle = cached_train(
            excel_bytes=excel_bytes,
            sheet_name=sheet_name,
            n_estimators=n_estimators,
            retrain_nonce=st.session_state["retrain_nonce"],
            bytes_md5=excel_md5,
        )
    st.sidebar.success(
        f"학습 완료 | R2={bundle['meta']['metrics']['r2']:.4f}, MAE={bundle['meta']['metrics']['mae']:.6f}"
    )
else:
    # 학습 버튼을 누르지 않았으면, 캐시에 이미 학습된 동일 조합이 있는지 시도
    try:
        bundle = cached_train(
            excel_bytes=excel_bytes,
            sheet_name=sheet_name,
            n_estimators=n_estimators,
            retrain_nonce=st.session_state["retrain_nonce"],
            bytes_md5=excel_md5,
        )
    except Exception:
        st.warning("아직 학습된 모델이 없습니다. 왼쪽에서 '학습'을 눌러 진행하세요.")
        st.stop()

model = bundle["model"]
meta = bundle["meta"]

INPUTS = meta["input_cols"]
CAT_INPUTS = meta["cat_inputs"]
OUTPUTS = meta["output_cols"]
MEDIANS = meta.get("num_medians", {})

with st.expander("학습 정보", expanded=False):
    st.write({
        "rows": meta.get("rows"),
        "cols": meta.get("cols"),
        "sheet_name": meta.get("sheet_name"),
        "n_estimators": meta.get("n_estimators"),
        "metrics": meta.get("metrics"),
        "num_inputs": len(INPUTS),
        "cat_inputs": len(CAT_INPUTS),
        "targets": OUTPUTS,
    })


# ===== 특징공학 (추론용: 학습과 동일한 로직) =====
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


# ===== 입력 폼 =====
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


# ===== 예측 =====
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

    # 누락 컬럼 보강
    for col in INPUTS:
        if col not in x_one.columns:
            x_one[col] = np.nan
    for col in CAT_INPUTS:
        if col not in x_one.columns:
            x_one[col] = ""

    # 특징공학
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

    # 예측 (타깃 5개)
    y_hat = pd.Series(model.predict(X_feed)[0], index=OUTPUTS)

    # 4점 구성: Tension은 입력값 사용, Bulge η=2/3 고정
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
        st.error("부족한 값: " + ", ".join(missing) + " — 학습/스키마를 점검하십시오.")
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


# ===== 결과 표시 =====
if "mmc4_pts" in st.session_state and "mmc4_C_hat" in st.session_state:
    pts = st.session_state["mmc4_pts"]
    C_hat = st.session_state["mmc4_C_hat"]

    eta_lo, eta_hi = -0.1, 0.7
    eta_grid = np.linspace(eta_lo, eta_hi, 200)

    # 내부(4점 범위)는 MMC4 그대로
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

    # 우측: 끝점 기울기 유지(C1 연속) 연결
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
