# -*- coding: utf-8 -*-
# train_export_model.py — Bulge η 제외, Tension η·εf은 입력, 5타깃 학습
import os, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

# ===== 설정 =====
EXCEL_PATH = "250811_산학프로젝트_포스코의 워크시트.xlsx"   # 데이터 파일명 고정
SHEET_NAME = None  # None이면 첫 시트
OUT_PKL = "mmc4_model.pkl"

# ★ 타깃(총 5개): Notch 2 + Shear 2 + Bulge εf (Bulge η 제외)
TARGETS_FIXED = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]

# ★ 입력 화이트리스트: Tension의 η, εf은 항상 입력(노란값)
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}

# ===== 유틸 =====
def read_excel_safe(path):
    ext = Path(path).suffix.lower()
    try:
        if ext in [".xlsx", ".xlsm", ".xls"]:
            return pd.read_excel(path, sheet_name=SHEET_NAME)
        return pd.read_csv(path)
    except Exception:
        # openpyxl fallback (xlsx)
        wb = load_workbook(path, data_only=True)
        ws = wb.active if SHEET_NAME is None else wb[SHEET_NAME]
        rows = list(ws.values)
        header = rows[0]
        data = rows[1:]
        return pd.DataFrame(data, columns=header)

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ===== 특징공학(학습/추론 동일하게 유지) =====
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

def main():
    # 1) 데이터 로드
    if not Path(EXCEL_PATH).exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {EXCEL_PATH}")

    df = read_excel_safe(EXCEL_PATH)

    # 2) 타깃/입력 컬럼 추출
    #    - 타깃은 TARGETS_FIXED
    #    - 입력은 (타깃 제외) + FORCE_INPUT 포함
    targets = [c for c in TARGETS_FIXED if c in df.columns]
    if len(targets) != len(TARGETS_FIXED):
        missing = [c for c in TARGETS_FIXED if c not in df.columns]
        raise ValueError(f"타깃 컬럼 누락: {missing}")

    # 후보 입력: 타깃 제외 + FORCE_INPUT 포함
    all_cols = list(df.columns)
    input_cols = [c for c in all_cols if c not in targets]
    for c in FORCE_INPUT:
        if c not in input_cols and c in df.columns:
            input_cols.append(c)

    # 3) 범주형 입력 추정(문자/카테고리)
    cat_inputs = []
    num_inputs = []
    for c in input_cols:
        if c not in df.columns:
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            cat_inputs.append(c)
        else:
            num_inputs.append(c)

    # 4) 숫자 변환
    df = coerce_numeric(df, num_inputs + targets)

    # 5) 결측치 중앙값 저장(앱 기본값)
    num_medians = {c: float(df[c].median()) if c in df.columns else 0.0 for c in num_inputs}

    # 6) 특징공학
    X_base = df[input_cols].copy()
    X_feat = build_enhanced_features(df, num_inputs, cat_inputs)

    # 7) 최종 입력 스키마
    #    - 수치: 원 num_inputs + 파생수치(특징공학 결과 중 새로 생긴 수치 컬럼)
    #    - 범주: cat_inputs
    feat_cols = [c for c in X_feat.columns if c not in (num_inputs + cat_inputs)]
    feature_columns = num_inputs + feat_cols + cat_inputs

    # 8) 학습 데이터 구성
    X = pd.DataFrame()
    for c in num_inputs + feat_cols:
        if c in X_feat.columns:
            X[c] = X_feat[c]
        else:
            X[c] = np.nan
    for c in cat_inputs:
        X[c] = df[c].astype(str)

    y = df[targets].copy()

    # 9) 전처리 + 모델
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_inputs + feat_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_inputs),
        ],
        remainder="drop"
    )

    reg = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=800,
            random_state=42,
            n_jobs=-1
        )
    )

    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])

    # 10) 학습/평가
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    r2 = r2_score(y_test, pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_test, pred)

    print(f"[OK] R2={r2:.4f}  MAE={mae:.6f}")

    # 11) 번들 저장
    meta = {
        "input_cols": num_inputs,            # 앱에서 노란입력 기본에 활용
        "cat_inputs": cat_inputs,
        "output_cols": targets,
        "num_medians": num_medians,
        "feature_columns": feature_columns,
    }
    bundle = {"model": pipe, "meta": meta}
    joblib.dump(bundle, OUT_PKL)
    print(f"[OK] saved: {OUT_PKL}")

if __name__ == "__main__":
    main()
