# main.py
from __future__ import annotations
import os, time, logging, warnings, random, joblib, yaml
from pathlib import Path
from datetime import datetime 
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, make_scorer)
from utils import (remove_corr_vif, build_preprocessor, FeatureGroups,
                   save_confusion_matrix, save_pr_curve, save_roc_curve, save_learning_curve)
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib                

warnings.filterwarnings("ignore", category=FutureWarning)

# ==== GLOBAL SETTINGS =======================================================
BASE_DIR      = Path("/home/kanghosung/hw1_patt/pattern-recognition/model/catboost_pipeline")
RAW_CSV       = Path("/home/kanghosung/hw1_patt/pattern-recognition/data/train.csv")

USE_GPU       = True          # False → CPU
GPU_DEVICES   = "0"           # "0,1" 형태도 가능

SEED          = 42            # 절대 고정
N_ITER_SEARCH = 30
N_ITERATIONS  = 2000
EARLY_STOP    = 50
# =============================================================================


# -------------------- PIPELINE STEPS ----------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["id","shares","y"]).copy()
    X['content_title_ratio'] = X['n_tokens_content']/(X['n_tokens_title']+1)
    X['keyword_density'] = X['num_keywords']/(X['n_tokens_content']+1)
    X['img_video_ratio'] = X['num_imgs']/(X['num_videos']+1)
    X['total_links'] = X['num_hrefs']+X['num_self_hrefs']
    X['positive_to_negative'] = X['global_rate_positive_words']/(X['global_rate_negative_words']+1e-5)
    X['title_sentiment_sum'] = X['title_subjectivity']+X['title_sentiment_polarity']
    X['abs_title_sentiment_diff'] = np.abs(X['title_subjectivity']-X['title_sentiment_polarity'])
    X['kw_avg_range'] = X['kw_max_avg']-X['kw_min_avg']
    X['self_share_range'] = X['self_reference_max_shares']-X['self_reference_min_shares']
    lda = ['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']
    X['lda_entropy'] = -X[lda].apply(lambda r: np.sum(r*np.log(r+1e-6)), axis=1)
    X['feat_content_to_img_ratio'] = X['n_tokens_content']/(X['num_imgs']+1)
    X['feat_global_sentiment_strength'] = np.abs(X['global_sentiment_polarity'])*X['global_subjectivity']
    X['feat_content_to_href_ratio'] = X['n_tokens_content']/(X['num_hrefs']+1)
    return X

def filter_features(X: pd.DataFrame, num_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    removed = remove_corr_vif(X, num_cols)
    X_f = X.drop(columns=removed)
    return X_f, removed

def split_data(X: pd.DataFrame, y: np.ndarray, seed: int):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=seed)

    return X_tr, X_va, y_tr, y_va, X_test, y_test

def build_cb_kwargs(scale_pos_weight: float, cat_idx: list[int]) -> dict:
    return {
        "iterations": N_ITERATIONS,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "custom_metric": ["Logloss"],
        "early_stopping_rounds": EARLY_STOP,
        "use_best_model": True,
        "random_state": SEED,
        "verbose": 100,
        "task_type": "GPU" if USE_GPU else "CPU",
        "devices": GPU_DEVICES if USE_GPU else None,
        "scale_pos_weight": scale_pos_weight,
        "cat_features": cat_idx
    }

# -------------------- MAIN ---------------------------------------------------
def main():
    # reproducibility
    np.random.seed(SEED); random.seed(SEED)

    # GPU env
    if USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # ----- paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = BASE_DIR / "log" / f"result_{ts}"
    proc_dir   = result_dir / "processed_data"
    proc_dir.mkdir(parents=True, exist_ok=True)
    LOSS_CURVE_IMG = result_dir / "loss_curve.jpg"
    AUC_CURVE_IMG  = result_dir / "auc_curve.jpg"

    # ----- logging
    log_file = result_dir / "training_log.txt"
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()],
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()

    if USE_GPU:
        logger.info("GPU : CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES"))
    else:
        logger.info("running on CPU")

    start = time.time()
    logger.info("=== PIPELINE STARTED ===")

    # 1. load & feature eng
    df = load_data(RAW_CSV)
    X = engineer_features(df)
    y = df["y"].values

    # 2. groups definition 
    log1p_vars = ['num_hrefs','num_self_hrefs','num_imgs','num_videos',
                  'self_reference_max_shares','self_reference_avg_sharess',
                  'self_reference_min_shares','kw_max_max','kw_avg_max',
                  'lda_entropy','total_links','self_share_range']
    signed_log_vars = ['rate_positive_words','rate_negative_words',
                      'avg_positive_polarity','avg_negative_polarity',
                      'min_positive_polarity','max_positive_polarity',
                      'min_negative_polarity','max_negative_polarity',
                      'positive_to_negative']
    standard_vars = ['n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens',
                    'average_token_length','kw_avg_min','kw_avg_avg',
                    'LDA_00','LDA_01','LDA_02','LDA_03','LDA_04',
                    'global_subjectivity','global_sentiment_polarity',
                    'title_subjectivity','title_sentiment_polarity',
                    'kw_avg_range','global_rate_positive_words',
                    'global_rate_negative_words']
    minmax_vars = ['n_tokens_title','n_tokens_content','num_keywords',
                  'kw_min_min','kw_max_min','kw_min_avg','kw_max_avg',
                  'abs_title_subjectivity','abs_title_sentiment_polarity',
                  'content_title_ratio','keyword_density','img_video_ratio',
                  'title_sentiment_sum','abs_title_sentiment_diff',
                  'kw_min_max','feat_content_to_img_ratio',
                  'feat_global_sentiment_strength','feat_content_to_href_ratio']
    onehot_vars = ['data_channel','weekday']
    num_cols = log1p_vars+signed_log_vars+standard_vars+minmax_vars

    # 3. filter features
    X, removed = filter_features(X, num_cols)

    groups = FeatureGroups(
        log1p=[c for c in log1p_vars   if c in X.columns],
        signed=[c for c in signed_log_vars if c in X.columns],
        standard=[c for c in standard_vars if c in X.columns],
        minmax=[c for c in minmax_vars   if c in X.columns],
        cat=onehot_vars
    )
    preprocessor, cat_idx = build_preprocessor(groups)

    # 4. split
    X_tr, X_va, y_tr, y_va, X_te, y_te = split_data(X, y, SEED)
    pos, neg = (y_tr==1).sum(), (y_tr==0).sum()
    spw = neg/pos if pos>0 else 1.0

    # 5. hyper-param search
    param_dist = {
        'learning_rate':[0.01,0.03,0.05,0.07,0.1],
        'depth':[4,6,8,10],
        'l2_leaf_reg':[1,3,5,7,9,12],
        'border_count':[32,64,128,254],
        'bagging_temperature':[0,0.5,1.0,1.5,2.0],
        'random_strength':[0.1,0.5,1,2,5],
        'colsample_bylevel':[0.6,0.7,0.8,0.9,1.0]
    }
    pipe = SKPipeline([
        ("prep", preprocessor),
        ("clf", CatBoostClassifier(
            iterations=1000, random_state=SEED, verbose=0,
            task_type="GPU" if USE_GPU else "CPU",
            devices=GPU_DEVICES if USE_GPU else None,
            scale_pos_weight=spw, cat_features=cat_idx,
            train_dir=str(result_dir)))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    total_fits = N_ITER_SEARCH * cv.get_n_splits()  

    with tqdm_joblib(tqdm(desc="RandomizedSearchCV", total=total_fits)):

        search = RandomizedSearchCV(
            pipe,
            {f"clf__{k}": v for k, v in param_dist.items()},
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=cv,
            n_jobs=2,
            random_state=SEED,
            verbose=0,    
            error_score="raise"
        )
        search.fit(X_tr, y_tr)
    best_params = {k.split("__")[1]: v for k, v in search.best_params_.items()}  

    # 6. final train
    cb_kw = build_cb_kwargs(spw, cat_idx) | best_params | {"train_dir": str(result_dir)}
    model = CatBoostClassifier(**cb_kw)
    X_tr_p = preprocessor.fit_transform(X_tr); X_va_p = preprocessor.transform(X_va)
    model.fit(X_tr_p, y_tr, eval_set=(X_va_p, y_va))
    evals = model.get_evals_result()
    train_auc  = evals["learn"]["AUC"]
    valid_auc  = evals["validation"]["AUC"]
    train_loss = evals["learn"]["Logloss"]
    valid_loss = evals["validation"]["Logloss"]

    save_learning_curve(train_loss, valid_loss, "Logloss", LOSS_CURVE_IMG)
    save_learning_curve(train_auc,  valid_auc,  "AUC",     AUC_CURVE_IMG)
    model.save_model(result_dir / "final_model.cbm")

    # 7. evaluation
    X_te_p = preprocessor.transform(X_te)
    y_hat = model.predict(X_te_p); y_prob = model.predict_proba(X_te_p)[:,1]
    acc, f1 = accuracy_score(y_te,y_hat), f1_score(y_te,y_hat)
    auc_val = roc_auc_score(y_te,y_prob)

    # plots
    save_confusion_matrix(y_te, y_hat, result_dir/"confusion_matrix.jpg")
    save_pr_curve(y_te, y_prob, result_dir/"pr_curve.jpg")
    save_roc_curve(y_te, y_prob, result_dir/"roc_curve.jpg")

    # classification report
    with open(result_dir/"classification_report.txt","w") as f:
        f.write(classification_report(y_te, y_hat, digits=4))

    # shap
    expl = shap.TreeExplainer(model)
    shap_vals = expl.shap_values(X_te_p, check_additivity=False)
    shap.summary_plot(shap_vals, X_te_p, show=False)
    plt.tight_layout(); plt.savefig(result_dir/"shap_summary.jpg", dpi=300); plt.close()

    # 8. save preprocessing + data
    joblib.dump({"preprocessor":preprocessor,"cat_idx":cat_idx,"removed":removed},
                proc_dir/"preprocessor.joblib")
    np.savez_compressed(proc_dir/"train.npz", data=X_tr_p, label=y_tr)
    np.savez_compressed(proc_dir/"valid.npz", data=X_va_p, label=y_va)
    np.savez_compressed(proc_dir/"test.npz",  data=X_te_p, label=y_te)

    # 9. log summary
    elapsed = time.time()-start
    h,m,s = int(elapsed//3600),int((elapsed%3600)//60),int(elapsed%60)
    logger.info("FINAL  Acc=%.4f  F1=%.4f  AUC=%.4f  (%.0fh %.0fm %.0fs)",
                acc, f1, auc_val, h, m, s)
    logger.debug(
        "Settings → USE_GPU=%s  GPU_DEVICES=%s  SEED=%d",
        USE_GPU, GPU_DEVICES, SEED)

    print("Done. Artifacts @", result_dir)

# -------------------- ENTRY POINT -------------------------------------------
if __name__ == "__main__":
    main()
