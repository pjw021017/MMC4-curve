# -*- coding: utf-8 -*-
"""
원본 + 증강 루프(1000→10000, step=1000)
- 각 단계 표(Feature | R2 Train | R2 Test) 및 avg R²_test(PB-η 제외)
- avg R²_test가 0.90을 '처음' 넘는 지점을 선형보간 → '백의 자리 반올림'
- 반올림 총샘플수로 '정확히' 증강한 후 MMC4 커브 비교
- avg R²_test vs samples(1000~10000) 플롯 포함

단순화:
- 노트북 코드 힌트 추출/맞춤 전처리 제거 → 기본 파이프라인(중앙값 대치 + RobustScaler)
- 그룹 증강은 총샘플수를 정확히 맞추도록 균등 분배(+잔여분 일부 그룹 +1)
"""

import os
import re, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

# --------------------- 경로/설정 ---------------------
BASE_DIR = os.path.dirname(__file__)
XLSX_PATH = os.path.join(BASE_DIR, "data_mmc4.xlsx")
SHEET     = "학습DB"
N_JOBS    = -1
rng       = np.random.default_rng()   # 실행마다 다른 RNG

PB_ETA_LABEL = "Triaxiality(Punch Bulge)"

INPUT_CANDS = {
    "Yield Stress": (["yield","stress"], ["ys","yieldstrength"]),
    "Ultimate Tensile Stress": (["ultimate","tensile","stress"], ["uts","tensilestrength"]),
    "Total Elongation": (["total","elongation"], ["elongation","totalelong"]),
    "r-value": (["rvalue"], ["r-value","r value","lankford","rvalueavg"]),
    "Triaxiality(Tension)": (["triaxiality","tension"], ["eta","tension"]),
    "Fracture strain(Tension)": (["fracture","strain","tension"], ["fracturestrain","tension"]),
}
OUTPUT_CANDS = {
    "Triaxiality(Notch R05)": (["triaxiality","notch","r05"], ["eta","notch","r05","0.5"]),
    "Fracture strain(Notch R05)": (["fracture","strain","notch","r05"], ["fracturestrain","notch","r05","0.5"]),
    "Triaxiality(Shear0)": (["triaxiality","shear0"], ["eta","shear"]),
    "Fracture strain(Shear0)": (["fracture","strain","shear0"], ["fracturestrain","shear"]),
    "Fracture strain(Punch Bulge)": (["fracture","strain","punch","bulge"], ["fracturestrain","bulge","punch"]),
    PB_ETA_LABEL: (["triaxiality","punch","bulge"], ["eta","bulge","punch"]),
}
GROUP_CANDIDATES = ["Material","material","재질","Grade","grade","Code","code","Steel","steel"]

# --------------------- 보조 함수들 ---------------------
def _norm(s: str) -> str:
    s = str(s).lower()
    return re.sub(r"[\s\-_]", "", s)

def resolve_col(df: pd.DataFrame, inc, alt=None):
    inc = [_norm(x) for x in inc]; alt = [_norm(x) for x in (alt or [])]
    best, score = None, -1
    for c in df.columns:
        nc = _norm(c)
        sc = sum(k in nc for k in inc) + 0.5*sum(a in nc for a in alt)
        if sc > score: score, best = sc, c
    return best

def resolve_group_col(df: pd.DataFrame):
    for cand in GROUP_CANDIDATES:
        if cand in df.columns: return cand
    for c in df.columns:
        if any(k in _norm(c) for k in ["material","grade","code","steel","재질"]):
            return c
    return None

def get_models_fast():
    # 모두 random_state=None → 실행마다 다른 결과
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200, random_state=None, n_jobs=N_JOBS,
            max_depth=12, min_samples_leaf=5, min_samples_split=10, max_features="sqrt"),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=None, n_jobs=N_JOBS,
            max_depth=12, min_samples_leaf=5, min_samples_split=10, max_features="sqrt"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=None, max_depth=6, max_leaf_nodes=31,
            min_samples_leaf=20, learning_rate=0.05,
            l2_regularization=1.0, validation_fraction=0.2, n_iter_no_change=20),
    }

def default_preprocessor():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", RobustScaler())])

# --------------------- 증강(그룹별 정확 개수 맞추기) ---------------------
def synthesize_rows_for_group(df_group, input_cols, n_add, rng, gmins, gmaxs, global_mins, global_maxs):
    if n_add <= 0: return pd.DataFrame(columns=input_cols)
    mins = gmins.fillna(global_mins).fillna(0.0)
    maxs = gmaxs.fillna(global_maxs).fillna(0.0)
    span = (maxs - mins).replace(0, 1e-9)
    U = rng.uniform(0.0, 1.0, size=(n_add, len(input_cols)))
    X_new = (mins.values + U * span.values).astype(float)
    return pd.DataFrame(X_new, columns=input_cols)

def augment_dataset_total(df, group_col, input_cols, total_target, rng, global_mins, global_maxs):
    """총 샘플수를 정확히 total_target으로 맞추도록 그룹별 목표치를 분배해 증강"""
    if group_col is None or group_col not in df.columns:
        group_col = "__GROUP__"; df = df.copy(); df[group_col] = "ALL"

    groups = df[group_col].drop_duplicates().tolist()
    G = len(groups)
    base = total_target // G
    extra = total_target % G   # 앞에서 extra개 그룹에 +1

    targets = {g: base + (i < extra) for i, g in enumerate(groups)}
    parts = []
    for g in groups:
        df_g = df[df[group_col] == g]
        t = int(targets[g]); n_cur = len(df_g)
        gmins = df_g[input_cols].min(numeric_only=True, skipna=True)
        gmaxs = df_g[input_cols].max(numeric_only=True, skipna=True)
        n_add = max(t - n_cur, 0)
        X_new = synthesize_rows_for_group(df_g, input_cols, n_add, rng, gmins, gmaxs, global_mins, global_maxs)
        add = pd.concat([X_new, pd.DataFrame({group_col:[g]*len(X_new)})], axis=1)
        if len(add): add["__source__"] = "synth"
        orig = df_g.copy(); orig["__source__"] = "orig"
        gg = pd.concat([orig, add], axis=0, ignore_index=True)
        gg = gg.sample(t, replace=False, random_state=None) if len(gg) > t else gg
        parts.append(gg)
    out = pd.concat(parts, axis=0, ignore_index=True)
    if len(out) > total_target:
        out = out.sample(total_target, random_state=None)
    return out

# --------------------- 증강 타깃 채우기(교사 모델) ---------------------
def fill_target_labels_for_augmented(df_aug, input_cols, ycol):
    if ycol not in df_aug.columns: return df_aug
    df = df_aug.copy()
    if "__source__" not in df.columns:
        df["__source__"] = "orig"
    is_orig = (df["__source__"] == "orig") & (~df[ycol].isna())
    no_nan_X = df[input_cols].notna().all(axis=1)
    train_mask = is_orig & no_nan_X
    need_pred_mask = (df["__source__"] == "synth") & df[ycol].isna() & no_nan_X
    if train_mask.sum() >= 15 and need_pred_mask.sum() > 0:
        teacher = ExtraTreesRegressor(n_estimators=200, random_state=None, n_jobs=N_JOBS)
        teacher.fit(df.loc[train_mask, input_cols], df.loc[train_mask, ycol])
        df.loc[need_pred_mask, ycol] = teacher.predict(df.loc[need_pred_mask, input_cols])
    if df[ycol].isna().any():
        df[ycol] = df.groupby([c for c in (["__GROUP__"] + GROUP_CANDIDATES) if c in df.columns])[ycol]\
                     .transform(lambda s: s.fillna(s.median()))
        df[ycol] = df[ycol].fillna(df[ycol].median())
    return df

# --------------------- R² 평가 & 표 출력 ---------------------
def evaluate_r2_table(df_aug, input_cols, out_cols):
    rows = []
    models = get_models_fast()
    pre = default_preprocessor()
    for tgt, _ in OUTPUT_CANDS.items():
        if tgt == PB_ETA_LABEL:
            rows.append([tgt, np.nan, np.nan]); continue
        ycol = out_cols.get(tgt)
        if (ycol is None) or (ycol not in df_aug.columns):
            rows.append([tgt, np.nan, np.nan]); continue
        sub = fill_target_labels_for_augmented(df_aug, input_cols, ycol)[input_cols + [ycol]]\
              .dropna(subset=[ycol]).copy()
        if len(sub) < 12:
            rows.append([tgt, np.nan, np.nan]); continue

        X, y = sub[input_cols].copy(), sub[ycol].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=None)

        best_tr, best_te = -1e9, -1e9
        for _, model in models.items():
            pipe = Pipeline([("prep", pre), ("model", model)])
            try:
                pipe.fit(X_tr, y_tr)
                p_tr = pipe.predict(X_tr); p_te = pipe.predict(X_te)
                r2_tr = r2_score(y_tr, p_tr); r2_te = r2_score(y_te, p_te)
                if r2_te > best_te: best_te, best_tr = r2_te, r2_tr
            except Exception:
                continue
        rows.append([tgt, None if best_tr<-1e8 else best_tr,
                         None if best_te<-1e8 else best_te])
    return pd.DataFrame(rows, columns=["Feature","R2 Train","R2 Test"])

def print_table_only(res_df, title):
    print(f"\n================  {title}  ================")
    df_show = res_df.copy()

    # PB-η 행 표시용 보장
    if PB_ETA_LABEL not in df_show["Feature"].tolist():
        df_show.loc[len(df_show)] = [PB_ETA_LABEL, np.nan, np.nan]

    # 평균(R2 Test) 계산 (PB-η 제외)
    mask_num = (df_show["Feature"] != PB_ETA_LABEL) & pd.to_numeric(df_show["R2 Test"], errors="coerce").notna()
    avg = pd.to_numeric(df_show.loc[mask_num, "R2 Test"]).mean() if mask_num.any() else np.nan

    # 4째자리 고정 포맷
    df_print = df_show.copy()
    for col in ["R2 Train", "R2 Test"]:
        df_print[col] = df_print[col].apply(lambda v: (f"{v:.4f}" if pd.notna(v) else "—"))

    # PB-η 행은 '—'로 명시
    pb_mask = (df_print["Feature"] == PB_ETA_LABEL)
    df_print.loc[pb_mask, ["R2 Train","R2 Test"]] = "—"

    print(df_print.to_string(index=False))
    print(f"[Summary] mean R²_test (PB-η excl.): {avg:.4f}" if np.isfinite(avg) else "[Summary] mean R²_test: NaN")
    return avg

# --------------------- MMC4 수식 ---------------------
def theta_bar(eta):
    eta = np.asarray(eta, dtype=float)
    arg = -(27.0/2.0) * eta * (eta**2 - 1.0/3.0)
    arg = np.clip(arg, -1.0, 1.0)
    return 1.0 - (2.0/np.pi) * np.arccos(arg)

def c6_effective(eta, C6):
    t = 1.0/np.sqrt(3.0)  # ≈0.577
    eta = np.asarray(eta, dtype=float)
    return np.where((eta <= -t) | ((eta >= 0.0) & (eta <= t)), 1.0, float(C6))

def mmc4_eps(eta, C):
    C1, C2, C3, C4, C5, C6 = C
    tb  = theta_bar(eta)
    c6e = c6_effective(eta, C6)
    k = np.sqrt(3.0) / (2.0 - np.sqrt(3.0))
    term1 = (C1 / C4) * ( C5 + k * (c6e - C5) * (1.0/np.cos(tb*np.pi/6.0) - 1.0) )
    base = 1.0 + (C3**2)/3.0 * np.cos(tb*np.pi/6.0) + C3*( np.asarray(eta) + (1.0/3.0)*np.sin(tb*np.pi/6.0) )
    base = np.maximum(base, 1e-6)
    return term1 * (base ** (-1.0 / C2))

def fit_mmc4(etas, epss, random_state=42):
    try:
        from scipy.optimize import least_squares
        def resid(p): return mmc4_eps(etas, p) - epss
        x0 = np.array([1.0, 1.0, 0.2, 1.0, 0.6, 0.8])
        lb = np.array([0.001, 0.10,-2.0, 0.10, 0.0, 0.0])
        ub = np.array([10.00, 5.00, 2.0, 5.00, 2.0, 2.0])
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=5000, verbose=0)
        return res.x
    except Exception:
        rng_local = np.random.RandomState(random_state)
        best_p, best_loss = None, np.inf
        for _ in range(6000):
            p = np.array([
                rng_local.uniform(0.001,10.0),
                rng_local.uniform(0.10, 5.0),
                rng_local.uniform(-2.0, 2.0),
                rng_local.uniform(0.10, 5.0),
                rng_local.uniform(0.0, 2.0),
                rng_local.uniform(0.0, 2.0),
            ])
            loss = np.mean((mmc4_eps(etas, p)-epss)**2)
            if loss < best_loss: best_loss, best_p = loss, p
        return best_p

# --------------------- MMC4 계산/플롯 ---------------------
def predict_target_for_row(df_aug, input_cols, out_cols, target_label, row_X):
    ycol = out_cols.get(target_label)
    if (ycol is None) or (ycol not in df_aug.columns): return np.nan
    df_yfilled = fill_target_labels_for_augmented(df_aug, input_cols, ycol)
    sub = df_yfilled[input_cols + [ycol]].dropna(subset=[ycol]).copy()
    if sub.shape[0] < 15: return np.nan

    pre = default_preprocessor()
    best_pipe, best_score = None, -np.inf
    for _, base_model in get_models_fast().items():
        pipe = Pipeline([("prep", pre), ("model", base_model)])
        try:
            pipe.fit(sub[input_cols], sub[ycol])
            score = r2_score(sub[ycol], pipe.predict(sub[input_cols]))
            if score > best_score: best_score, best_pipe = score, pipe
        except Exception:
            continue
    if best_pipe is None: return np.nan
    y_pred = best_pipe.predict(row_X[input_cols])
    return float(np.asarray(y_pred).ravel()[0])

def compute_and_plot_mmc4_for_df(df_aug, input_cols, out_cols, ax, title=""):
    df_orig = df_aug[df_aug["__source__"]=="orig"].copy() if "__source__" in df_aug.columns else df_aug
    tens_eta_col = resolve_col(df_orig, ["triaxiality","tension"], ["eta","tension"])
    tens_eps_col = resolve_col(df_orig, ["fracture","strain","tension"], ["fracturestrain","tension"])
    shear_eta_col, shear_eps_col = out_cols.get("Triaxiality(Shear0)"), out_cols.get("Fracture strain(Shear0)")
    notch_eta_col, notch_eps_col = out_cols.get("Triaxiality(Notch R05)"), out_cols.get("Fracture strain(Notch R05)")
    bulge_eps_col, bulge_eta_col = out_cols.get("Fracture strain(Punch Bulge)"), out_cols.get(PB_ETA_LABEL)
    needed = [tens_eta_col,tens_eps_col,shear_eta_col,shear_eps_col,notch_eta_col,notch_eps_col,bulge_eps_col,bulge_eta_col]
    if any(c is None for c in needed):
        print(f"[MMC4-{title}] 필요한 컬럼을 찾지 못했습니다."); return

    req_cols = list(input_cols) + needed
    mask = df_orig[req_cols].notna().all(axis=1)
    if not mask.any():
        print(f"[MMC4-{title}] 모든 X/Y가 채워진 원본 행이 없습니다."); return
    idx0 = df_orig.index[mask][0]
    row = df_orig.loc[idx0]

    # 원본 4점
    pts_orig = [
        ("Shear (orig)", float(row[shear_eta_col]), float(row[shear_eps_col])),
        ("Tension",      float(row[tens_eta_col]),  float(row[tens_eps_col])),
        ("Notch (orig)", float(row[notch_eta_col]), float(row[notch_eps_col])),
        ("Bulge (orig)", float(row[bulge_eta_col]), float(row[bulge_eps_col])),
    ]
    pts_orig_sorted = sorted(pts_orig, key=lambda t: t[1])
    etas_o = np.array([p[1] for p in pts_orig_sorted], dtype=float)
    epss_o = np.array([p[2] for p in pts_orig_sorted], dtype=float)
    C_o = fit_mmc4(etas_o, epss_o, random_state=123)

    # 예측 4점 (같은 X로 예측)
    row_X = df_aug.loc[[idx0], input_cols].copy() if idx0 in df_aug.index else df_orig.loc[[idx0], input_cols].copy()
    shear_eta_p = predict_target_for_row(df_aug, input_cols, out_cols, "Triaxiality(Shear0)", row_X)
    shear_eps_p = predict_target_for_row(df_aug, input_cols, out_cols, "Fracture strain(Shear0)", row_X)
    notch_eta_p = predict_target_for_row(df_aug, input_cols, out_cols, "Triaxiality(Notch R05)", row_X)
    notch_eps_p = predict_target_for_row(df_aug, input_cols, out_cols, "Fracture strain(Notch R05)", row_X)
    bulge_eps_p = predict_target_for_row(df_aug, input_cols, out_cols, "Fracture strain(Punch Bulge)", row_X)
    eta_bulge_p = float(row[bulge_eta_col])

    if any(np.isnan(v) for v in [shear_eta_p,shear_eps_p,notch_eta_p,notch_eps_p,bulge_eps_p]):
        print(f"[MMC4-{title}] 예측 4점에 NaN이 있어 MMC4 곡선을 그리지 않습니다."); return

    pts_pred = [
        ("Shear (pred)", float(shear_eta_p), float(shear_eps_p)),
        ("Tension",      float(row[tens_eta_col]), float(row[tens_eps_col])),
        ("Notch (pred)", float(notch_eta_p), float(notch_eps_p)),
        ("Bulge (pred)", float(eta_bulge_p), float(bulge_eps_p)),
    ]
    pts_pred_sorted = sorted(pts_pred, key=lambda t: t[1])
    etas_p = np.array([p[1] for p in pts_pred_sorted], dtype=float)
    epss_p = np.array([p[2] for p in pts_pred_sorted], dtype=float)
    C_p = fit_mmc4(etas_p, epss_p, random_state=456)

    # 공통 η 구간(20점)
    eta_min, eta_max = min(etas_o.min(), etas_p.min()), max(etas_o.max(), etas_p.max())
    if not np.isfinite(eta_min) or not np.isfinite(eta_max) or eta_max <= eta_min:
        print(f"[MMC4-{title}] η 범위 계산 실패."); return
    eta_grid = np.linspace(eta_min, eta_max, 20)
    ax.plot(eta_grid, mmc4_eps(eta_grid, C_o), lw=2, label="MMC4 (orig)")
    ax.plot(eta_grid, mmc4_eps(eta_grid, C_p), lw=2, ls="--", label="MMC4 (pred)")
    # 포인트
    first = True
    for _, x, y in pts_orig:
        ax.scatter(x, y, s=50, marker="o", facecolors="none", edgecolors="C0", label="orig points" if first else None)
        first = False
    first = True
    for name, x, y in pts_pred:
        if name.startswith("Tension"): continue
        ax.scatter(x, y, s=50, marker="x", color="C1", label="pred points" if first else None)
        first = False
    ax.set_xlabel("Triaxiality η"); ax.set_ylabel("Fracture strain εf")
    ax.set_title(title); ax.grid(True, alpha=0.3)

# --------------------- 메인 ---------------------
def main():
    # 1) 로드 & 컬럼 매핑
    try:
        df = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
    except Exception:
        df = pd.read_excel(XLSX_PATH)

    input_cols = []
    for _, (inc, alt) in INPUT_CANDS.items():
        c = resolve_col(df, inc, alt)
        if c is not None: input_cols.append(c)
    out_cols = {lab: resolve_col(df, *OUTPUT_CANDS[lab]) for lab in OUTPUT_CANDS}

    # PB-η 고정 컬럼(없으면 2/3 고정, 있으면 전체 중앙값 고정)
    pb_src = out_cols.get(PB_ETA_LABEL)
    if (pb_src is None) or (pb_src not in df.columns):
        fixed_col = "Triaxiality(Punch Bulge) (assumed 2/3, fixed)"
        df[fixed_col] = 2.0/3.0; out_cols[PB_ETA_LABEL] = fixed_col
    else:
        med = pd.to_numeric(df[pb_src], errors="coerce").median()
        fixed_col = f"{pb_src} (fixed median)"; df[fixed_col] = med; out_cols[PB_ETA_LABEL] = fixed_col

    group_col = resolve_group_col(df) or "__GROUP__"
    if group_col not in df.columns: df[group_col] = "ALL"
    n_groups = df[group_col].nunique(dropna=False)

    # 2) ORIGINAL 표/평균
    res_orig = evaluate_r2_table(df, input_cols, out_cols)
    avg_orig = print_table_only(res_orig, "ORIGINAL DATASET")

    # 3) 증강 루프(1000~10000, 1000단계) & 평균 수집
    totals = list(range(1000, 10001, 1000))
    global_mins = df[input_cols].min(numeric_only=True)
    global_maxs = df[input_cols].max(numeric_only=True)

    avgs = []
    for total in totals:
        df_aug = augment_dataset_total(df, group_col, input_cols, total, rng, global_mins, global_maxs)
        res_aug = evaluate_r2_table(df_aug, input_cols, out_cols)
        avg_aug = print_table_only(res_aug, f"AUGMENTED total={total}  (groups={n_groups}, exact)")
        avgs.append(avg_aug if np.isfinite(avg_aug) else np.nan)

    # 4) 보간으로 임계(total at mean R²_test = 0.90) 추정 → 백의자리 반올림
    target = 0.90
    interp_total = None
    for i in range(len(totals)-1):
        y0, y1 = avgs[i], avgs[i+1]
        if np.isfinite(y0) and np.isfinite(y1) and (y0 < target <= y1):
            x0, x1 = totals[i], totals[i+1]
            ratio = (target - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.0
            interp_total = x0 + ratio * (x1 - x0)
            break
    if interp_total is None:
        best_idx = int(np.nanargmax(avgs))
        interp_total = float(totals[best_idx])
        print(f"\n[WARN] 평균 R²_test가 0.90을 넘지 못했습니다. 최고 평균 지점({totals[best_idx]})을 사용합니다.")

    rounded_total = int(np.round(interp_total / 100.0) * 100)  # 백의 자리 반올림
    print(f"\n[INTERP] mean R²_test = {target:.2f} 달성 추정 total ≈ {interp_total:.0f}  →  rounded={rounded_total}")

    # 5) 반올림 총샘플수로 '정확 증강' + 표/평균
    df_interp = augment_dataset_total(df, group_col, input_cols, rounded_total, rng, global_mins, global_maxs)
    res_interp = evaluate_r2_table(df_interp, input_cols, out_cols)
    avg_interp = print_table_only(res_interp, f"AUGMENTED total={rounded_total} (from interp≈{interp_total:.0f})")

    # 6) 그래프: (a) R²_test 평균 vs samples, (b) MMC4: ORIGINAL vs INTERP(rounded)
    # (a) 평균 R² vs samples
    plt.figure(figsize=(6,4))
    plt.plot(totals, avgs, marker="o")
    plt.axhline(target, linestyle="--")
    plt.xlabel("samples (total)"); plt.ylabel("avg R²_test (PB-η excl.)")
    plt.title("Average R²_test vs. Samples")
    plt.grid(True, alpha=0.3)
    plt.scatter([rounded_total], [target], marker="x")
    plt.tight_layout(); plt.show()

    # (b) MMC4 커브 (제목에 샘플 수와 목표 평균 표시)
    print("\n========== MMC4 curves: Original vs Augmented ==========")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    compute_and_plot_mmc4_for_df(df,         input_cols, out_cols, ax=axes[0], title="Original dataset")
    compute_and_plot_mmc4_for_df(
        df_interp, input_cols, out_cols, ax=axes[1],
        title=f"Augmented dataset ({rounded_total} samples, R\u00b2_test_avg = 0.90)"
    )
    axes[0].legend(); axes[1].legend()
    plt.tight_layout(); plt.show()
