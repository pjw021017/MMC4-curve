# -*- coding: utf-8 -*-
"""
train_export_model.py

목표:
- Streamlit 앱에서 실행 시점에 모델을 학습(또는 재학습)할 수 있도록,
  "학습 함수" 형태로 제공.
- pkl 저장은 기본적으로 하지 않음(옵션으로만 지원).

학습 개요:
- 입력: (기본) POSCO 워크시트 파일 또는 사용자가 업로드한 파일(CSV/XLSX)
- 타깃(5개): Notch 2 + Shear 2 + Bulge εf (Bulge η는 학습에서 제외)
- 특징공학: app_mmc4_compact.py와 동일한 파생피처 생성
- 모델: ExtraTrees 기반 MultiOutputRegressor + (수치)결측 중앙값 대체 + (범주)OHE
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# =========================
# 기본 설정
# =========================

# 기본 데이터 파일(리포지토리에 포함되어 있을 때 자동 사용)
DEFAULT_DATA_PATH = "250811_산학프로젝트_포스코의 워크시트.xlsx"

# 타깃(총 5개): Notch 2 + Shear 2 + Bulge εf (Bulge η 제외)
TARGETS_FIXED: List[str] = [
    "Triaxiality(Notch R05)",     "Fracture strain(Notch R05)",
    "Triaxiality(Shear0)",        "Fracture strain(Shear0)",
    "Fracture strain(Punch Bulge)"
]

# Tension η·εf는 "항상 입력으로 간주"
FORCE_INPUT = {"Triaxiality(Tension)", "Fracture strain(Tension)"}

# 기본적으로 우선 포함하고 싶은 입력(존재하면 우선 정렬)
PREFERRED_INPUTS_ORDER: List[str] = [
    "Yield Stress(MPa)",
    "Ultimate Tensile Stress(MPa)",
    "Total Elongation",
    "r-value",
    "Triaxiality(Tension)",
    "Fracture strain(Tension)",
]

DEFAULT_CAT_INPUTS: List[str] = ["Material"]  # 존재할 때만 사용


# =========================
# 데이터 로드 유틸
# =========================

def load_dataframe_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Streamlit 업로드 bytes를 DataFrame으로 변환(CSV/XLSX 지원)."""
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(data)

    if suffix in [".csv", ".txt"]:
        return pd.read_csv(bio)
    if suffix in [".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"]:
        return pd.read_excel(bio)

    # 확장자가 애매하면 excel → csv 순으로 시도
    try:
        return pd.read_excel(bio)
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio)


def load_dataframe_from_path(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    return pd.read_excel(path)


# =========================
# 컬럼 추정 / 정리
# =========================

def infer_input_and_targets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    df에서 input_cols(수치), cat_inputs(범주), output_cols(타깃)을 추정.

    정책:
    - 타깃은 TARGETS_FIXED 중 df에 존재하는 것만 사용
    - 입력은 "수치형 컬럼 - 타깃"을 기본으로 하되 FORCE_INPUT은 반드시 포함
    - 범주 입력은 DEFAULT_CAT_INPUTS 중 존재하는 것만 사용
    """
    cat_inputs = [c for c in DEFAULT_CAT_INPUTS if c in df.columns]

    output_cols = [c for c in TARGETS_FIXED if c in df.columns]
    if len(output_cols) == 0:
        raise ValueError(
            "타깃 컬럼을 찾지 못했습니다. 데이터에 다음 컬럼 중 일부가 있어야 합니다:\n"
            + "\n".join(TARGETS_FIXED)
        )

    # numeric 후보: to_numeric 후 유효값이 1개라도 있는 컬럼
    numeric_cols: List[str] = []
    for c in df.columns:
        if c in cat_inputs:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() > 0:
            numeric_cols.append(c)

    input_cols = [c for c in numeric_cols if c not in output_cols]

    # FORCE_INPUT 강제 반영
    for c in FORCE_INPUT:
        if c in df.columns and c not in input_cols and c not in output_cols:
            input_cols.append(c)
        if c in output_cols:
            output_cols = [t for t in output_cols if t != c]
            if c not in input_cols:
                input_cols.append(c)

    # 입력을 선호 순서로 정렬
    ordered = [c for c in PREFERRED_INPUTS_ORDER if c in input_cols]
    rest = [c for c in input_cols if c not in ordered]
    input_cols = ordered + sorted(rest)

    return input_cols, cat_inputs, output_cols


# =========================
# 특징공학 (앱과 동일 로직)
# =========================

def _as_series_single(X: pd.DataFrame, colname: str) -> Optional[pd.Series]:
    """동일 이름 중복 컬럼이 있어도 단일 Series로 축약(왼쪽→오른쪽 첫 유효값)."""
    if colname not in X.columns:
        return None
    cols = [c for c in X.columns if c == colname]
    sub = X[cols]
    if isinstance(sub, pd.Series):
        return pd.to_numeric(sub, errors="coerce")
    sub_num = sub.apply(pd.to_numeric, errors="coerce")
    ser = sub_num.bfill(axis=1).iloc[:, 0]
    return ser


def build_enhanced_features(df_: pd.DataFrame, input_cols: Sequence[str], cat_inputs: Sequence[str]) -> pd.DataFrame:
    """원본 입력(input_cols + cat_inputs)에서 파생 피처를 추가한 DataFrame 반환."""
    X = df_[list(input_cols) + list(cat_inputs)].copy()

    ys = _as_series_single(X, "Yield Stress(MPa)")
    uts = _as_series_single(X, "Ultimate Tensile Stress(MPa)")
    tel = _as_series_single(X, "Total Elongation")
    rv = _as_series_single(X, "r-value")
    etaT = _as_series_single(X, "Triaxiality(Tension)")
    efT = _as_series_single(X, "Fracture strain(Tension)")

    # 강도 파생
    if uts is not None and ys is not None:
        X["UTS_to_Yield"] = uts / (ys + 1e-8)
        X["Strength_Range"] = uts - ys

    # 연신 파생
    if tel is not None:
        X["Elong_sq"] = tel ** 2
        X["Elong_sqrt"] = np.sqrt(tel + 1e-8)
        if rv is not None:
            X["Elong_x_r"] = tel * rv
            X["Elong_div_r"] = tel / (rv + 1e-8)

    # Tension η, εf 파생
    if etaT is not None:
        X["etaT_sq"] = etaT ** 2
        X["etaT_abs"] = np.abs(etaT)
    if efT is not None:
        X["efT_sq"] = efT ** 2
        X["log_efT"] = np.log(efT + 1e-8)

    # 상호작용(있을 때만)
    if etaT is not None and efT is not None:
        X["etaT_x_efT"] = etaT * efT
        X["efT_div_etaT"] = efT / (np.abs(etaT) + 1e-8)

    return X


def align_features_for_model(X_raw: pd.DataFrame, feature_columns: Sequence[str], fill_value: float = np.nan) -> pd.DataFrame:
    """학습 시 feature_columns 스키마에 맞춰 컬럼 보강/정렬."""
    X = X_raw.copy()
    for c in feature_columns:
        if c not in X.columns:
            X[c] = fill_value
    return X[list(feature_columns)]


# =========================
# 학습
# =========================

@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    meta: Dict
    metrics: Dict[str, Dict[str, float]]
    r2_flat: float


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_model(
    df: pd.DataFrame,
    *,
    input_cols: Optional[Sequence[str]] = None,
    cat_inputs: Optional[Sequence[str]] = None,
    output_cols: Optional[Sequence[str]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> TrainResult:
    """DataFrame으로부터 모델 학습 후 (model, meta, metrics) 반환."""
    if input_cols is None or cat_inputs is None or output_cols is None:
        inf_in, inf_cat, inf_out = infer_input_and_targets(df)
        input_cols = list(inf_in) if input_cols is None else list(input_cols)
        cat_inputs = list(inf_cat) if cat_inputs is None else list(cat_inputs)
        output_cols = list(inf_out) if output_cols is None else list(output_cols)

    # 숫자 변환
    for col in set(list(input_cols) + list(output_cols)):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 타깃 결측 제거
    df_model = df.dropna(subset=list(output_cols)).reset_index(drop=True)
    if len(df_model) < 10:
        raise ValueError(f"학습 가능한 행이 너무 적습니다. (타깃 결측 제거 후 n={len(df_model)})")

    X_all = build_enhanced_features(df_model, input_cols, cat_inputs)
    y_all = df_model[list(output_cols)].copy()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=float(test_size), random_state=int(random_state), shuffle=True
    )

    num_cols = [c for c in X_all.columns if c not in cat_inputs]
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", _make_ohe(), list(cat_inputs)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    def _create_et():
        return ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=None,
            max_features="sqrt",
            bootstrap=True,
            random_state=int(random_state),
            n_jobs=-1,
        )

    pipe = Pipeline([
        ("pre", pre),
        ("reg", MultiOutputRegressor(_create_et())),
    ])

    pipe.fit(X_train, y_train)

    # 메트릭
    y_pred = pd.DataFrame(pipe.predict(X_test), columns=y_test.columns, index=y_test.index)

    def compute_metrics(y_true: pd.DataFrame, y_hat: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for c in y_true.columns:
            yt = y_true[c].astype(float).values
            yp = y_hat[c].astype(float).values
            out[c] = {
                "R2": float(r2_score(yt, yp)),
                "MAE": float(mean_absolute_error(yt, yp)),
                "MAPE": float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100.0),
            }
        return out

    r2_flat = float(r2_score(y_test.values.flatten(), y_pred.values.flatten()))
    metrics = compute_metrics(y_test, y_pred)

    meta = {
        "input_cols": list(input_cols),
        "cat_inputs": list(cat_inputs),
        "output_cols": list(output_cols),
        "num_medians": df_model[list(input_cols)].median(numeric_only=True).to_dict() if len(input_cols) else {},
        "feature_columns": list(X_all.columns),
        "train_rows": int(len(df_model)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
        "r2_flat": float(r2_flat),
    }

    return TrainResult(model=pipe, meta=meta, metrics=metrics, r2_flat=r2_flat)


def train_bundle_from_path(
    path: Union[str, Path],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Dict:
    df = load_dataframe_from_path(path)
    res = train_model(df, test_size=test_size, random_state=random_state, n_estimators=n_estimators)
    return {"model": res.model, "meta": res.meta, "metrics": res.metrics, "r2_flat": res.r2_flat}


def train_bundle_from_bytes(
    data: bytes,
    filename: str,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Dict:
    df = load_dataframe_from_bytes(data, filename)
    res = train_model(df, test_size=test_size, random_state=random_state, n_estimators=n_estimators)
    return {"model": res.model, "meta": res.meta, "metrics": res.metrics, "r2_flat": res.r2_flat}


# =========================
# (옵션) 로컬 CLI 실행
# =========================

def _print_metrics(metrics: Dict[str, Dict[str, float]], r2_flat: float) -> None:
    print("\n=== 학습 요약 ===")
    print(f"전체 평탄화 R² = {r2_flat:.6f}")
    for k, v in metrics.items():
        print(f"{k:32s}  R²={v['R2']:7.4f}  MAE={v['MAE']:10.6f}  MAPE={v['MAPE']:7.2f}%")


if __name__ == "__main__":
    data_path = os.environ.get("MMC4_DATA_PATH", DEFAULT_DATA_PATH)
    test_size = float(os.environ.get("MMC4_TEST_SIZE", "0.25"))
    n_estimators = int(os.environ.get("MMC4_N_ESTIMATORS", "300"))
    random_state = int(os.environ.get("MMC4_RANDOM_STATE", "42"))

    bundle = train_bundle_from_path(
        data_path, test_size=test_size, random_state=random_state, n_estimators=n_estimators
    )
    _print_metrics(bundle["metrics"], bundle["r2_flat"])

    # pkl 저장은 기본 OFF (원하면 환경변수로 켜기)
    if os.environ.get("MMC4_SAVE_PKL", "0") == "1":
        import joblib
        joblib.dump({"model": bundle["model"], "meta": bundle["meta"]}, "mmc4_model.pkl", compress=7)
        print("\nSaved: mmc4_model.pkl")
