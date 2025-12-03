# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mmc4_core as core  # 방금 만든 모듈

# --------- 데이터 & 컬럼 매핑 ---------
@st.cache_resource
def load_base():
    # 학습 DB 로드
    try:
        df = pd.read_excel(core.XLSX_PATH, sheet_name=core.SHEET)
    except Exception:
        df = pd.read_excel(core.XLSX_PATH)

    # 입력 컬럼과 표시이름 매핑
    input_cols = []
    label_to_col = {}
    for label, (inc, alt) in core.INPUT_CANDS.items():
        c = core.resolve_col(df, inc, alt)
        if c is not None:
            input_cols.append(c)
            label_to_col[label] = c

    # 출력 컬럼
    out_cols = {
        lab: core.resolve_col(df, *core.OUTPUT_CANDS[lab])
        for lab in core.OUTPUT_CANDS
    }

    # Punch bulge η 고정 처리 (네 main()과 같게)
    pb_src = out_cols.get(core.PB_ETA_LABEL)
    if (pb_src is None) or (pb_src not in df.columns):
        fixed_col = "Triaxiality(Punch Bulge) (assumed 2/3, fixed)"
        df[fixed_col] = 2.0 / 3.0
        out_cols[core.PB_ETA_LABEL] = fixed_col
    else:
        med = pd.to_numeric(df[pb_src], errors="coerce").median()
        fixed_col = f"{pb_src} (fixed median)"
        df[fixed_col] = med
        out_cols[core.PB_ETA_LABEL] = fixed_col

    if "__source__" not in df.columns:
        df["__source__"] = "orig"

    return df, input_cols, out_cols, label_to_col

# --------- 한 세트 입력에 대한 예측 + MMC4 피팅 ---------
def predict_points(df, input_cols, out_cols, label_to_col, inputs):
    # inputs: 표시이름 → 값
    row_dict = {label_to_col[k]: v for k, v in inputs.items()}
    row_X = pd.DataFrame([row_dict])[input_cols]

    # Tension 입력값
    tens_eta_col = label_to_col["Triaxiality(Tension)"]
    tens_eps_col = label_to_col["Fracture strain(Tension)"]
    tens_eta = float(row_X[tens_eta_col].iloc[0])
    tens_eps = float(row_X[tens_eps_col].iloc[0])

    # Bulge η (2/3 또는 중앙값으로 고정)
    bulge_eta_col = out_cols[core.PB_ETA_LABEL]
    eta_bulge = float(df[bulge_eta_col].iloc[0])

    # 모델 기반 예측 (Shear / Notch / Bulge εf)
    shear_eta = core.predict_target_for_row(
        df, input_cols, out_cols, "Triaxiality(Shear0)", row_X
    )
    shear_eps = core.predict_target_for_row(
        df, input_cols, out_cols, "Fracture strain(Shear0)", row_X
    )
    notch_eta = core.predict_target_for_row(
        df, input_cols, out_cols, "Triaxiality(Notch R05)", row_X
    )
    notch_eps = core.predict_target_for_row(
        df, input_cols, out_cols, "Fracture strain(Notch R05)", row_X
    )
    bulge_eps = core.predict_target_for_row(
        df, input_cols, out_cols, "Fracture strain(Punch Bulge)", row_X
    )

    etas = np.array([shear_eta, tens_eta, notch_eta, eta_bulge], dtype=float)
    epss = np.array([shear_eps, tens_eps, notch_eps, bulge_eps], dtype=float)

    # MMC4 파라미터 피팅 및 곡선
    C = core.fit_mmc4(etas, epss)
    eta_grid = np.linspace(etas.min(), etas.max(), 50)
    curve = core.mmc4_eps(eta_grid, C)

    points = {
        "Shear": (shear_eta, shear_eps),
        "Tension": (tens_eta, tens_eps),
        "Notch": (notch_eta, notch_eps),
        "Bulge": (eta_bulge, bulge_eps),
    }
    return eta_grid, curve, points

# --------- Streamlit UI ---------
def main():
    st.title("POSCO MMC4 – 모델 예측 기반 곡선 시각화")

    df, input_cols, out_cols, label_to_col = load_base()

    st.subheader("입력값 (노란색)")
    with st.form("mmc4_form"):
        ys = st.number_input("Yield Stress(MPa)", value=150.0)
        uts = st.number_input("Ultimate Tensile Stress(MPa)", value=280.0)
        elong = st.number_input("Total Elongation(%)", value=40.0)
        rvalue = st.number_input("r-value", value=1.0)
        eta_t = st.number_input("Triaxiality(Tension)", value=0.30)
        eps_t = st.number_input("Fracture strain(Tension)", value=1.0)

        submitted = st.form_submit_button("예측 및 MMC4 곡선 보기")

    if submitted:
        inputs = {
            "Yield Stress": ys,
            "Ultimate Tensile Stress": uts,
            "Total Elongation": elong,
            "r-value": rvalue,
            "Triaxiality(Tension)": eta_t,
            "Fracture strain(Tension)": eps_t,
        }
        eta_grid, curve, points = predict_points(
            df, input_cols, out_cols, label_to_col, inputs
        )

        fig, ax = plt.subplots()
        ax.plot(eta_grid, curve, label="MMC4 curve (fit)")
        for name, (x, y) in points.items():
            ax.scatter(x, y, label=name)
        ax.set_xlabel("Triaxiality η")
        ax.set_ylabel("Fracture strain εf")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
