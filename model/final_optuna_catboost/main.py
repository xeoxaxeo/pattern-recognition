# -*- coding: utf-8 -*-

import os, time, warnings, logging, joblib, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, shap, optuna

from datetime import datetime
from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.metrics          import (accuracy_score, f1_score, roc_auc_score,
                                       confusion_matrix, precision_recall_curve,
                                       roc_curve, auc, make_scorer)
from sklearn.inspection       import permutation_importance
from catboost                 import CatBoostClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ========= CONFIG ===========================================================
BASE_DIR      = '/home/kanghosung/hw1_patt/pattern-recognition/model/final_optuna_catboost'
RAW_CSV       = '/home/kanghosung/hw1_patt/pattern-recognition/data/train.csv'

USE_GPU       = False
GPU_DEVICES   = "0"
GPU_RAM_PART  = 0.50          # GPU 메모리 사용 비율 (CatBoost 전용)
GLOBAL_SEED   = 42

N_TRIALS_OPTUNA = 30          # Optuna 탐색 횟수
SEARCH_N_JOBS   = 1           # Optuna 병렬 worker 수  (CPU 메모리 여유 없으면 1)
TRAIN_ITER      = 500         # trial-/final-model 공통 반복 횟수
EARLY_STOP      = 50
N_DPND_FEATS    = 3           # dependence plot을 그릴 feature 개수 (FI top-k)

# ============================================================================
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

np.random.seed(GLOBAL_SEED)

# -------- result directories ------------------------------------------------
log_root   = os.path.join(BASE_DIR, 'log')
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
RES_DIR    = os.path.join(log_root, f'optuna_run_{timestamp}')
PLOT_DIR    = os.path.join(RES_DIR, 'img')
os.makedirs(PLOT_DIR,  exist_ok=True)

MODEL_CBM  = os.path.join(RES_DIR, 'final_model.cbm')
FI_CSV     = os.path.join(RES_DIR, 'feature_importances.csv')
PI_CSV     = os.path.join(RES_DIR, 'permutation_importances.csv')
LOG_TXT    = os.path.join(RES_DIR, 'log.txt')

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_TXT), logging.StreamHandler()]
)
logger = logging.getLogger()
logger.info("CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES", "CPU"))

# --------------------------------------------------
# 1. Data load & tiny feature-eng (3 derived feats)
# --------------------------------------------------
df = pd.read_csv(RAW_CSV)
X_full = df.drop(columns=['id', 'shares', 'y'], errors='ignore').copy()
y_full = df['y'].copy()

EPS = 1e-6
def add_derived(df_):
    df_ = df_.copy()
    if {'n_tokens_content','num_imgs'}.issubset(df_.columns):
        df_['feat_content_to_img_ratio'] = df_['n_tokens_content'] / (df_['num_imgs']+EPS)
    if {'global_subjectivity','global_sentiment_polarity'}.issubset(df_.columns):
        df_['feat_global_sentiment_strength'] = df_['global_subjectivity'] * df_['global_sentiment_polarity']
    if {'n_tokens_content','num_hrefs'}.issubset(df_.columns):
        df_['feat_content_to_href_ratio'] = df_['n_tokens_content'] / (df_['num_hrefs']+EPS)
    return df_

X_full = add_derived(X_full)

# split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=GLOBAL_SEED)

# simple imputation
num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in ['data_channel', 'weekday'] if c in X_tr.columns]

meds = X_tr[num_cols].median()
X_tr[num_cols]  = X_tr[num_cols].fillna(meds)
X_val[num_cols] = X_val[num_cols].fillna(meds)
if cat_cols:
    for c in cat_cols:
        X_tr[c].fillna('missing', inplace=True)
        X_val[c].fillna('missing', inplace=True)

# class weight
pos, neg = (y_tr==1).sum(), (y_tr==0).sum()
scale_pos_weight = neg/pos if pos else 1.0
logger.info("scale_pos_weight=%.3f  (neg=%d, pos=%d)", scale_pos_weight, neg, pos)

# --------------------------------------------------
# 2. Custom CompositeMetric for CatBoost
# --------------------------------------------------
class CompositeMetric:
    def get_final_error(self, error, weight): return error / (weight+1e-10)
    def is_max_optimal(self): return True
    def evaluate(self, approxes, target, weight):
        prob = 1.0 / (1.0 + np.exp(-approxes[0]))
        pred = (prob>=0.5).astype(int)
        m = (accuracy_score(target,pred)+f1_score(target,pred)+roc_auc_score(target,prob))/3
        return m, 1.0

# --------------------------------------------------
# 3. Optuna objective
# --------------------------------------------------
def cb_params(trial):
    return {
        'iterations'          : TRAIN_ITER,
        'learning_rate'       : trial.suggest_float('lr', 0.01, 0.1, log=True),
        'depth'               : trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg'         : trial.suggest_float('l2', 1, 12, log=True),
        'border_count'        : trial.suggest_int('border', 32, 254),
        'bagging_temperature' : trial.suggest_float('bag_temp', 0.0, 2.0),
        'random_strength'     : trial.suggest_float('rand_strength', 0.1, 5.0, log=True),
        'colsample_bylevel'   : trial.suggest_float('colsample', 0.6, 1.0),
        'task_type'           : 'GPU' if USE_GPU else 'CPU',
        'devices'             : GPU_DEVICES if USE_GPU else None,
        'gpu_ram_part'        : GPU_RAM_PART if USE_GPU else None,
        'scale_pos_weight'    : scale_pos_weight,
        'eval_metric'         : CompositeMetric(),
        'early_stopping_rounds': EARLY_STOP,
        'use_best_model'      : True,
        'verbose'             : False,
        'random_state'        : GLOBAL_SEED,
    }

def objective(trial):
    model = CatBoostClassifier(**cb_params(trial))
    model.fit(
        X_tr, y_tr, cat_features=cat_cols or None,
        eval_set=[(X_val, y_val)], verbose=False)
    return model.get_best_score()['validation']['CompositeMetric']

logger.info("Optuna search (%d trials) 시작", N_TRIALS_OPTUNA)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS_OPTUNA, n_jobs=SEARCH_N_JOBS, show_progress_bar=True)

best_params = study.best_params
logger.info("Optuna best params: %s  |  best Composite=%.4f", best_params, study.best_value)

# --------------------------------------------------
# 4. Final model with best params
# --------------------------------------------------
final_params = cb_params(optuna.trial.FixedTrial(best_params))
final_params.update({'iterations': TRAIN_ITER, 'verbose': 100})
final_model = CatBoostClassifier(**final_params)
final_model.fit(
    X_tr, y_tr, cat_features=cat_cols or None,
    eval_set=[(X_val, y_val)])

best_iter = final_model.get_best_iteration()
logger.info("Best iteration via early-stopping: %d", best_iter)
final_model.save_model(MODEL_CBM)

# --------------------------------------------------
# 5. Learning-curve plots (Logloss & Composite proxy)
# --------------------------------------------------
def save_learning_curve(metric_key, fname):
    ev = final_model.get_evals_result()
    tr  = ev['learn'][metric_key]
    val = ev['validation'][metric_key]
    plt.figure()
    plt.plot(tr,  label=f'train {metric_key}')
    plt.plot(val, label=f'valid {metric_key}')
    plt.xlabel('iteration'); plt.ylabel(metric_key); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300); plt.close()

save_learning_curve('Logloss', 'learning_curve_logloss.jpg')

def composite_proba(y_true, y_pred_proba, **kw):
    # y_pred_proba 가 (n,) 또는 (n,2) 모두 허용
    prob = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
    pred = (prob >= 0.5).astype(int)
    return (accuracy_score(y_true, pred) +
            f1_score(y_true, pred) +
            roc_auc_score(y_true, prob)) / 3

custom_scorer = make_scorer(composite_proba, needs_proba=True)

# --------------------------------------------------
# 6. Hold-out metrics & classic plots
# --------------------------------------------------
y_prob = final_model.predict_proba(X_val)[:,1]
y_pred = (y_prob>=0.5).astype(int)

acc = accuracy_score(y_val, y_pred)
f1  = f1_score(y_val, y_pred)
auc_val = roc_auc_score(y_val, y_prob)
comp = (acc+f1+auc_val)/3
logger.info("Validation Acc=%.4f  F1=%.4f  AUC=%.4f  Comp=%.4f", acc,f1,auc_val,comp)

# CM
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'confusion_matrix.jpg'), dpi=300); plt.close()

# PR & ROC
prec, rec, _ = precision_recall_curve(y_val, y_prob)
plt.figure(); plt.plot(rec, prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR curve')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'pr_curve.jpg'), dpi=300); plt.close()

fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
plt.plot([0,1],[0,1],'--',lw=.6); plt.legend()
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC curve')
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'roc_curve.jpg'), dpi=300); plt.close()

# --------------------------------------------------
# 7. Feature & permutation importance
# --------------------------------------------------
fi = final_model.get_feature_importance(prettified=True)
fi.to_csv(FI_CSV, index=False)

pi = permutation_importance(
        final_model, X_val, y_val,
        scoring=custom_scorer,
        n_repeats=5, random_state=GLOBAL_SEED, n_jobs=-1
)

pi_df = pd.DataFrame({'feature':X_val.columns,
                      'importance_mean':pi.importances_mean,
                      'importance_std':pi.importances_std})
pi_df.to_csv(PI_CSV, index=False)

# --------------------------------------------------
# 8. SHAP summary + dependence plot
# --------------------------------------------------
explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X_val, check_additivity=False)

# 오류 있음. 이 부분은 일단 제외.
# # summary plot
# shap.summary_plot(shap_vals, X_val, show=False)
# plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'shap_summary.jpg'), dpi=300); plt.close()

# # dependence plot (상호작용 색 포함)
# top_feats = fi.sort_values('Importances', ascending=False)['Feature'].head(N_DPND_FEATS)
# for feat in top_feats:
#     shap.dependence_plot(feat, shap_vals, X_val,
#                          interaction_index='auto', show=False)
#     fn = f"shap_dependence_{feat}.jpg".replace('/','_')
#     plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fn), dpi=300); plt.close()
    

# --------------------------------------------------
# 9. Log summary
# --------------------------------------------------
elapsed = time.time() - df.memory_usage().sum()  # dummy to silence linter
elapsed = time.time() - start
h,m,s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
logger.info("Total runtime %dh %dm %ds", h,m,s)
logger.info("Artifacts saved to %s", RES_DIR)
