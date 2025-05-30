# -*- coding: utf-8 -*-
"""
One-file CatBoost pipeline
• Full preprocessing, feature engineering, VIF/corr filtering
• RandomizedSearch + early stopping
• Complete evaluation suite (hold-out, CV, CM, PR, ROC, report)
• Feature importance, permutation importance, SHAP
• Saves all artifacts in ./log/result_<timestamp>/
"""
# --------------------------------------------------
# 1. Imports & global paths
# --------------------------------------------------
#from google.colab import drive              # ⇢ remove if not on Colab
#drive.mount('/content/drive')               # ⇢ remove if not on Colab

import os, shutil, logging, warnings, time, joblib, numpy as np, pandas as pd
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.model_selection import (RandomizedSearchCV, train_test_split,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   FunctionTransformer)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, auc,
                             make_scorer)
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt, seaborn as sns, shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= CONFIG ===========================================================
USE_GPU      = True        # True → GPU / False → CPU
GPU_DEVICES  = "0"         # 여러 장이면 "0,1" 같이 쉼표 구분
GLOBAL_SEED  = 42          # 재현용 시드

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
else:                               
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# ============================================================================

np.random.seed(GLOBAL_SEED)

# -------- base & result directories -----------------------------------------
BASE_DIR = '/home/kanghosung/hw1_patt/pattern-recognition/model/Catboost'
RAW_CSV  = '/home/kanghosung/hw1_patt/pattern-recognition/data/train.csv'

TRAIN_ITERATION = 200
FINAL_ITERATION = 2000

SEARCH_N_JOBS = 1
GPU_RAM_PART = 0.5

timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_DIR  = os.path.join(BASE_DIR, 'log', f'result_{timestamp}')
PROC_DIR    = os.path.join(RESULT_DIR, 'processed_data')
os.makedirs(PROC_DIR, exist_ok=True)
os.chdir(BASE_DIR)

# overwrite if collision (rare)
if len(os.listdir(PROC_DIR)):
    shutil.rmtree(RESULT_DIR)
    os.makedirs(PROC_DIR, exist_ok=True)

MODEL_FILE = os.path.join(RESULT_DIR, "final_model.cbm")
FI_CSV     = os.path.join(RESULT_DIR, "feature_importance.csv")
PI_CSV     = os.path.join(RESULT_DIR, "permutation_importance.csv")
LOG_FILE   = os.path.join(RESULT_DIR, "training_log.txt")

CM_IMG   = os.path.join(RESULT_DIR, "confusion_matrix.jpg")
PR_IMG   = os.path.join(RESULT_DIR, "pr_curve.jpg")
ROC_IMG  = os.path.join(RESULT_DIR, "roc_curve.jpg")
SHAP_IMG = os.path.join(RESULT_DIR, "shap_summary.jpg")
CLSRPT_TXT = os.path.join(RESULT_DIR, "classification_report.txt")


# -------- logging -----------------------------------------------------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(fh); logger.addHandler(sh)
if USE_GPU:
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES"))
else:
    logger.info("CPU")

# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------
def build_cb_kwargs(extra_params=None):
    base = {
        "task_type": "GPU" if USE_GPU else "CPU",
        "random_state": GLOBAL_SEED,
        "verbose": 0
    }
    if USE_GPU:
        base["devices"] = GPU_DEVICES   
    if extra_params:
        base.update(extra_params)
    return base

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    df_num = df_num.fillna(df_num.mean()).loc[:, df_num.std(ddof=0) > 0]
    if df_num.empty:
        return pd.DataFrame(columns=["feature","VIF"])
    vif = pd.DataFrame()
    vif["feature"] = df_num.columns
    vif["VIF"] = [variance_inflation_factor(df_num.values, i)
                  for i in range(df_num.shape[1])]
    return vif

def remove_corr_vif(df, cols, c_thr=0.95, v_thr=10):
    removed = []
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    drop = {col for col in upper.columns
            for row in upper.index
            if upper.loc[row, col] > c_thr}
    df_f = df.drop(columns=list(drop)); removed.extend(drop)
    X = df_f[[c for c in cols if c in df_f.columns]].copy()
    while True:
        vif_df = calculate_vif(X)
        if vif_df.empty: break
        worst = vif_df.sort_values("VIF", ascending=False).iloc[0]
        if worst["VIF"] <= v_thr: break
        X.drop(columns=[worst["feature"]], inplace=True)
        removed.append(worst["feature"])
    return removed

def save_npz(path, arr, lbl):
    np.savez_compressed(path, data=arr, label=lbl)

def composite_scorer(y_true, y_pred, **kwargs):
    y_prob = y_pred[:,1] if y_pred.ndim>1 else y_pred
    y_hat  = (y_prob>=0.5).astype(int)
    return (accuracy_score(y_true,y_hat)+f1_score(y_true,y_hat)+
            roc_auc_score(y_true,y_prob))/3

def log1p_clip(x):   return np.log1p(np.clip(x, 0, None))
def signed_log1p(x): return np.sign(x) * np.log1p(np.abs(x))
def to_str(x):       return x.astype(str)

# --------------------------------------------------
# 3. Load & feature engineering
# --------------------------------------------------
start = time.time()
logger.info("Step 1  ▶ Loading data")
df = pd.read_csv(RAW_CSV)
X, y = df.drop(columns=['id','shares','y']), df['y'].values

logger.info("Step 2  ▶ Feature engineering")
X['content_title_ratio']      = X['n_tokens_content'] / (X['n_tokens_title']+1)
X['keyword_density']          = X['num_keywords'] / (X['n_tokens_content']+1)
X['img_video_ratio']          = X['num_imgs'] / (X['num_videos']+1)
X['total_links']              = X['num_hrefs'] + X['num_self_hrefs']
X['positive_to_negative']     = X['global_rate_positive_words'] / (X['global_rate_negative_words']+1e-5)
X['title_sentiment_sum']      = X['title_subjectivity'] + X['title_sentiment_polarity']
X['abs_title_sentiment_diff'] = np.abs(X['title_subjectivity'] - X['title_sentiment_polarity'])
X['kw_avg_range']             = X['kw_max_avg'] - X['kw_min_avg']
X['self_share_range']         = X['self_reference_max_shares'] - X['self_reference_min_shares']
lda_cols = ['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']
X['lda_entropy']              = -X[lda_cols].apply(lambda r: np.sum(r*np.log(r+1e-6)), axis=1)
X['feat_content_to_img_ratio']= X['n_tokens_content'] / (X['num_imgs']+1)
X['feat_global_sentiment_strength'] = np.abs(X['global_sentiment_polarity']) * X['global_subjectivity']
X['feat_content_to_href_ratio']= X['n_tokens_content'] / (X['num_hrefs']+1)

# -------- column groups (원본 그대로) ---------------------------------------
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
num_cols    = log1p_vars+signed_log_vars+standard_vars+minmax_vars

# --------------------------------------------------
# 4. Feature filtering (corr + VIF)
# --------------------------------------------------
logger.info("Step 3  ▶ Removing highly-correlated / high-VIF vars")
removed = remove_corr_vif(X, num_cols)
X.drop(columns=removed, inplace=True)

# --------------------------------------------------
# 5. Preprocessing pipelines
# --------------------------------------------------
logger.info("Step 4  ▶ Building preprocessing pipeline")
log1p = [c for c in log1p_vars   if c in X.columns]
signed = [c for c in signed_log_vars if c in X.columns]
std    = [c for c in standard_vars  if c in X.columns]
mm     = [c for c in minmax_vars    if c in X.columns]
cat    = onehot_vars                # always kept

log1p_pipe  = SKPipeline([("imp", SimpleImputer(strategy="median")),
                          ("log1p", FunctionTransformer(log1p_clip))])
signed_pipe = SKPipeline([("imp", SimpleImputer(strategy="median")),
                          ("slog", FunctionTransformer(signed_log1p))])
std_pipe   = SKPipeline([("imp",SimpleImputer(strategy="median")),
                         ("std",StandardScaler())])
mm_pipe    = SKPipeline([("imp",SimpleImputer(strategy="median")),
                         ("mm",MinMaxScaler())])
cat_pipe    = SKPipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("str",  FunctionTransformer(to_str))])

preprocessor = ColumnTransformer([
    ("log1p", log1p_pipe, log1p),
    ("signed", signed_pipe, signed),
    ("std", std_pipe, std),
    ("mm", mm_pipe, mm),
    ("cat", cat_pipe, cat)
], remainder="passthrough")

pre_len = len(log1p)+len(signed)+len(std)+len(mm)
cat_idx = list(range(pre_len, pre_len+len(cat)))

# --------------------------------------------------
# 6. Train/valid/test split
# --------------------------------------------------
logger.info("Step 5  ▶ Train/valid/test split")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval,
    random_state=42)

# --------------------------------------------------
# 7. Class imbalance weight
# --------------------------------------------------
pos, neg = (y_trainval==1).sum(), (y_trainval==0).sum()
scale_pos_weight = neg/pos if pos>0 else 1.0
logger.info("Class 0=%d  Class 1=%d  scale_pos_weight=%.3f", neg, pos, scale_pos_weight)

# --------------------------------------------------
# 8. RandomizedSearchCV
# --------------------------------------------------
logger.info("Step 6  ▶ Hyper-param search")
param_dist = {
    'learning_rate':[0.01,0.03,0.05,0.07,0.1],
    'depth':[4,6,8,10],
    'l2_leaf_reg':[1,3,5,7,9,12],
    'border_count':[32,64,128,254],
    'bagging_temperature':[0,0.5,1.0,1.5,2.0],
    'random_strength':[0.1,0.5,1,2,5],
    #'colsample_bylevel':[1.0]
}


pipe_full = SKPipeline([
    ("prep", preprocessor),
    ("clf", CatBoostClassifier(
        iterations=TRAIN_ITERATION,
        scale_pos_weight=scale_pos_weight,
        cat_features=cat_idx,
        train_dir=RESULT_DIR,
        **build_cb_kwargs()
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=pipe_full, error_score='raise',
    param_distributions={f"clf__{k}":v for k,v in param_dist.items()},
    n_iter=30, scoring='roc_auc', cv=cv, n_jobs=SEARCH_N_JOBS, random_state=42, verbose=1)
search.fit(X_trainval, y_trainval)
best_params = search.best_params_; best_auc = search.best_score_
logger.info("Best params: %s  |  CV AUC=%.4f", best_params, best_auc)

# --------------------------------------------------
# 9. Final model training (early stopping)
# --------------------------------------------------
logger.info("Step 7  ▶ Final model training")
X_train_pre = preprocessor.fit_transform(X_train)
X_valid_pre = preprocessor.transform(X_valid)
X_test_pre  = preprocessor.transform(X_test)

final_model = CatBoostClassifier(
    iterations=FINAL_ITERATION,
    eval_metric='AUC',
    early_stopping_rounds=50,
    use_best_model=True,
    scale_pos_weight=scale_pos_weight,
    cat_features=cat_idx,
    train_dir=RESULT_DIR,
    gpu_ram_part=GPU_RAM_PART,
    **build_cb_kwargs({k.split('__')[1]: v for k, v in best_params.items()})
)

final_model.fit(X_train_pre, y_train, eval_set=(X_valid_pre, y_valid))
best_iter = final_model.get_best_iteration()
logger.info("Best iteration: %d", best_iter)

final_model.save_model(MODEL_FILE)

# --------------------------------------------------
# 10. Save processed data & preprocessor
# --------------------------------------------------
joblib.dump({"preprocessor":preprocessor, "cat_idx":cat_idx,
             "removed_vars":removed}, os.path.join(PROC_DIR,"preprocessor.joblib"))
save_npz(os.path.join(PROC_DIR,"train.npz"), X_train_pre, y_train)
save_npz(os.path.join(PROC_DIR,"valid.npz"), X_valid_pre, y_valid)
save_npz(os.path.join(PROC_DIR,"test.npz"),  X_test_pre,  y_test)

# --------------------------------------------------
# 11. Hold-out metrics
# --------------------------------------------------
logger.info("Step 8  ▶ Hold-out evaluation")
y_pred = final_model.predict(X_test_pre)
y_prob = final_model.predict_proba(X_test_pre)[:,1]
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc_val = roc_auc_score(y_test, y_prob)
comp = (acc+f1+auc_val)/3
logger.info("Hold-out  Acc=%.4f  F1=%.4f  AUC=%.4f  Comp=%.4f",
            acc, f1, auc_val, comp)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(CM_IMG, dpi=300); plt.close()

# Classification report
report = classification_report(y_test, y_pred, digits=4)
with open(CLSRPT_TXT,"w") as f: f.write(report)
logger.info("\n"+report)

# PR curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.tight_layout(); plt.savefig(PR_IMG, dpi=300); plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
plt.plot([0,1],[0,1],'--',lw=0.6)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.savefig(ROC_IMG, dpi=300); plt.close()

# --------------------------------------------------
# 12. Feature & permutation importance
# --------------------------------------------------
feat_imp = final_model.get_feature_importance(prettified=True)
feat_imp.to_csv(FI_CSV,index=False)

pi = permutation_importance(final_model, X_test_pre, y_test,
                            scoring=make_scorer(composite_scorer,
                                                needs_proba=True),
                            n_repeats=5,n_jobs=-1,random_state=42)
pi_df = pd.DataFrame({"feature":X.columns,
                      "importance_mean":pi.importances_mean,
                      "importance_std":pi.importances_std})
pi_df.to_csv(PI_CSV,index=False)

# --------------------------------------------------
# 13. SHAP values
# --------------------------------------------------
explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X_test_pre, check_additivity=False)
shap.summary_plot(shap_vals, X_test_pre, show=False)
plt.tight_layout(); plt.savefig(SHAP_IMG, dpi=300); plt.close()

# --------------------------------------------------
# 14. Final summary
# --------------------------------------------------
elapsed = time.time()-start
h,m,s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
logger.info("Total runtime: %dh %dm %ds", h, m, s)
with open(LOG_FILE,"a") as f:
    f.write("\n===== FINAL SUMMARY =====\n")
    f.write(f"Best params: {best_params}\n")
    f.write(f"CV AUC: {best_auc:.4f}\n")
    f.write(f"Hold-out Acc: {acc:.4f}  F1: {f1:.4f}  AUC: {auc_val:.4f}\n")
    f.write(f"Composite: {comp:.4f}\n")
    f.write(f"Best iteration: {best_iter}\n")
    f.write(f"Elapsed: {h}h {m}m {s}s\n")
print("done:", RESULT_DIR)
