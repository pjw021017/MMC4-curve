# -*- coding: utf-8 -*-
# mmc4_train_lib.py
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor


# ===== 타깃(총 5개): Notch 2 + Shear 2 + Bulge εf (Bulge η 제외) =====
TARGETS_FIXED = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]

# ★ 입력 강제 포함(노란값)
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ===== 특징공학(학습/추론 동일) =====
def build_enhanced_features(df_: pd.DataFrame, input_cols: list[str], cat_inputs: list[str]) -> pd.DataFrame:
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


def read_excel_from_bytes(excel_bytes: bytes, sheet_name=None) -> pd.DataFrame:
    bio = BytesIO(excel_bytes)
    # 엔진은 pandas가 자동 선택; openpyxl은 requirements에 포함
    return pd.read_excel(bio, sheet_name=sheet_name)


def train_bundle_from_excel_bytes(
    excel_bytes: bytes,
    sheet_name=None,
    n_estimators: int = 300,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    반환: bundle={"model": pipe, "meta": meta}
    - pkl 저장 없음(사용자 요청)
    - Cloud에서는 이 bundle을 st.cache_resource로 캐시해 재사용
    """
    df = read_excel_from_bytes(excel_bytes, sheet_name=sheet_name)

    # 타깃 체크
    targets = [c for c in TARGETS_FIXED if c in df.columns]
    if len(targets) != len(TARGETS_FIXED):
        missing = [c for c in TARGETS_FIXED if c not in df.columns]
        raise ValueError(f"타깃 컬럼 누락: {missing}")

    # 입력 후보: 타깃 제외 + FORCE_INPUT 포함
    all_cols = list(df.columns)
    input_cols = [c for c in all_cols if c not in targets]
    for c in FORCE_INPUT:
        if c not in input_cols and c in df.columns:
            input_cols.append(c)

    # 범주/수치 분리
    cat_inputs, num_inputs = [], []
    for c in input_cols:
        if c not in df.columns:
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            cat_inputs.append(c)
        else:
            num_inputs.append(c)

    # 숫자 변환
    df = _coerce_numeric(df, num_inputs + targets)

    # 기본값(중앙값)
    num_medians = {c: float(df[c].median()) if c in df.columns else 0.0 for c in num_inputs}

    # 특징공학
    X_feat = build_enhanced_features(df, num_inputs, cat_inputs)
    feat_cols = [c for c in X_feat.columns if c not in (num_inputs + cat_inputs)]
    feature_columns = num_inputs + feat_cols + cat_inputs

    # 최종 X, y
    X = pd.DataFrame()
    for c in num_inputs + feat_cols:
        X[c] = X_feat[c] if c in X_feat.columns else np.nan
    for c in cat_inputs:
        X[c] = df[c].astype(str)

    y = df[targets].copy()

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
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    )

    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    r2 = r2_score(y_test, pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_test, pred)

    meta = {
        "input_cols": num_inputs,
        "cat_inputs": cat_inputs,
        "output_cols": targets,
        "num_medians": num_medians,
        "feature_columns": feature_columns,
        "metrics": {"r2": float(r2), "mae": float(mae)},
        "sheet_name": sheet_name,
        "n_estimators": int(n_estimators),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }

    return {"model": pipe, "meta": meta}
