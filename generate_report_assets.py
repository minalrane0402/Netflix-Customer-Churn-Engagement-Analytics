"""
generate_report_assets.py
Single authoritative script that produces every figure, table value,
and metric used in the project report.
Run from the netflix-churn-analytics/ directory.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, classification_report, silhouette_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
RED   = '#E50914'
DARK  = '#221F1F'
GREY  = '#B3B3B3'
PALETTE = [RED, DARK, GREY, '#831010', '#F5F5F1']

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
})
sns.set_theme(style='whitegrid', palette=PALETTE)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv('data/netflix_customer_churn.csv')
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r'[\s\-\.]+', '_', regex=True)
              .str.replace(r'[^a-z0-9_]', '', regex=True))

# Single target column — drop the source 'churned' col after encoding
df['churn_label'] = df['churned'].astype(int)
print(f"Loaded {len(df):,} rows. Churn rate: {df['churn_label'].mean()*100:.2f}%")
print("Columns:", list(df.columns))

# KPIs
total       = len(df)
n_churned   = int(df['churn_label'].sum())
churn_rate  = df['churn_label'].mean() * 100
ret_rate    = 100 - churn_rate

WATCH_COL   = 'avg_watch_time_per_day'    # hrs/day — confirmed from raw header
PLAN_COL    = 'subscription_type'
PAY_COL     = 'payment_method'
PROFILES_COL= 'number_of_profiles'
DEVICE_COL  = 'device'
GENDER_COL  = 'gender'
AGE_COL     = 'age'
MONTHLY_COL = 'monthly_fee'
LOGIN_COL   = 'last_login_days'

avg_watch_all  = df[WATCH_COL].mean()
avg_watch_ret  = df[df.churn_label==0][WATCH_COL].mean()
avg_watch_ch   = df[df.churn_label==1][WATCH_COL].mean()
plan_churn     = (df.groupby(PLAN_COL)['churn_label'].mean()*100).round(2).to_dict()

print(f"\nKPIs: total={total}, churned={n_churned}, churn_rate={churn_rate:.2f}%")
print(f"Watch time overall={avg_watch_all:.2f} ret={avg_watch_ret:.2f} churned={avg_watch_ch:.2f} hrs/day")
print(f"Plan churn: {plan_churn}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIGURES — EDA
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Churn distribution — bar ONLY (pie removed per review) ─────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df['churn_label'].value_counts().rename({0:'Retained', 1:'Churned'})
colors = [DARK, RED]
bars = ax.bar(counts.index, counts.values, color=colors,
              edgecolor='black', linewidth=0.8, width=0.5)
bars[0].set_hatch('/')
bars[1].set_hatch('\\\\')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}\n({val/total*100:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('Overall Churn Distribution', fontweight='bold')
ax.set_xlabel('Customer Status')
ax.set_ylabel('Number of Customers')
ax.set_ylim(0, max(counts.values) * 1.2)
plt.tight_layout()
fig.savefig('figures/01_churn_distribution.png', bbox_inches='tight')
plt.close()

# ── Fig 2: Churn RATE by plan — single bar, sorted Basic/Standard/Premium ──────
fig, ax = plt.subplots(figsize=(7, 4))
plan_order = ['Basic', 'Standard', 'Premium']
rates = [df[df[PLAN_COL]==p]['churn_label'].mean()*100 for p in plan_order]
colors2 = [RED if r > 50 else DARK for r in rates]
bars2 = ax.bar(plan_order, rates, color=colors2, edgecolor='black', linewidth=0.8, width=0.5)
for hatch, bar in zip(['/', '\\\\', 'x'], bars2):
    bar.set_hatch(hatch)
for bar, val in zip(bars2, rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(churn_rate, color=GREY, linestyle='--', linewidth=1.2, label=f'Overall avg ({churn_rate:.1f}%)')
ax.set_title('Churn Rate by Subscription Plan', fontweight='bold')
ax.set_xlabel('Subscription Plan')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, 80)
ax.legend()
plt.tight_layout()
fig.savefig('figures/02_churn_by_plan.png', bbox_inches='tight')
plt.close()

# ── Fig 3: Monthly fee KDE + box ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, grp in df.groupby('churn_label'):
    status = 'Churned' if label==1 else 'Retained'
    grp[MONTHLY_COL].plot.kde(ax=axes[0],
        color=RED if label==1 else DARK,
        linestyle='--' if label==1 else '-', linewidth=2, label=status)
axes[0].set_title('Monthly Fee — KDE by Churn Status', fontweight='bold')
axes[0].set_xlabel('Monthly Fee (USD)')
axes[0].legend(title='Status')

bp = axes[1].boxplot(
    [df[df.churn_label==0][MONTHLY_COL].dropna(),
     df[df.churn_label==1][MONTHLY_COL].dropna()],
    labels=['Retained','Churned'], patch_artist=True, notch=True,
    flierprops=dict(marker='D', markerfacecolor=GREY, markersize=3))
bp['boxes'][0].set_facecolor(DARK); bp['boxes'][1].set_facecolor(RED)
for box in bp['boxes']: box.set_edgecolor('black')
axes[1].set_title('Monthly Fee — Box Plot', fontweight='bold')
axes[1].set_ylabel('Monthly Fee (USD)')
plt.tight_layout()
fig.savefig('figures/03_monthly_spend_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 4: Watch time KDE + box  (unit: hrs/day — consistent everywhere) ───────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, grp in df.groupby('churn_label'):
    status = 'Churned' if label==1 else 'Retained'
    grp[WATCH_COL].plot.kde(ax=axes[0],
        color=RED if label==1 else DARK,
        linestyle='--' if label==1 else '-', linewidth=2, label=status)
axes[0].set_title('Avg Watch Time per Day — KDE by Churn Status', fontweight='bold')
axes[0].set_xlabel('Avg Watch Time (hrs/day)')
axes[0].legend(title='Status')

bp2 = axes[1].boxplot(
    [df[df.churn_label==0][WATCH_COL].dropna(),
     df[df.churn_label==1][WATCH_COL].dropna()],
    labels=['Retained','Churned'], patch_artist=True,
    flierprops=dict(marker='x', markerfacecolor=GREY, markersize=3))
bp2['boxes'][0].set_facecolor(DARK); bp2['boxes'][1].set_facecolor(RED)
for box in bp2['boxes']: box.set_edgecolor('black')
for i, (lbl, med) in enumerate(
    [('Retained', df[df.churn_label==0][WATCH_COL].median()),
     ('Churned',  df[df.churn_label==1][WATCH_COL].median())], 1):
    axes[1].text(i, med + 0.02, f'Mdn={med:.2f}', va='bottom', ha='center', fontsize=8)
axes[1].set_title('Avg Watch Time per Day — Box Plot', fontweight='bold')
axes[1].set_ylabel('Avg Watch Time (hrs/day)')
plt.tight_layout()
fig.savefig('figures/05_watch_time_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 5: Number of profiles ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
prof = (df.groupby(PROFILES_COL)['churn_label'].mean()*100).reset_index()
prof.columns = [PROFILES_COL, 'churn_rate']
bars3 = ax.bar(prof[PROFILES_COL].astype(str), prof['churn_rate'],
               color=RED, edgecolor='black', hatch='/', linewidth=0.8)
for bar, val in zip(bars3, prof['churn_rate']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Churn Rate by Number of Profiles', fontweight='bold')
ax.set_xlabel('Number of Profiles')
ax.set_ylabel('Churn Rate (%)')
plt.tight_layout()
fig.savefig('figures/06_profiles_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 6: Payment method churn — correct caption (Crypto/GiftCard) ────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
pay_churn = (df.groupby(PAY_COL)['churn_label'].mean()*100).sort_values(ascending=True)
hatches = ['/', '\\\\', 'x', '+', 'o', '.']
colors3 = [RED if v > pay_churn.median() else DARK for v in pay_churn.values]
bars4 = ax.barh(pay_churn.index, pay_churn.values, color=colors3,
                edgecolor='black', linewidth=0.8)
for i, bar in enumerate(bars4):
    bar.set_hatch(hatches[i % len(hatches)])
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.1f}%', va='center', fontsize=9)
ax.axvline(pay_churn.median(), color=GREY, linestyle='--', linewidth=1.2, label='Median')
ax.set_title('Churn Rate by Payment Method', fontweight='bold')
ax.set_xlabel('Churn Rate (%)')
ax.set_ylabel('Payment Method')
ax.legend()
plt.tight_layout()
fig.savefig('figures/07_payment_method_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 7: Device type — correct axis label ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
dev_churn = (df.groupby(DEVICE_COL)['churn_label'].mean()*100).sort_values(ascending=False)
bars5 = ax.bar(dev_churn.index, dev_churn.values, color=DARK,
               edgecolor='black', hatch='\\\\', linewidth=0.8)
for bar, val in zip(bars5, dev_churn.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Churn Rate by Device Type', fontweight='bold')
ax.set_xlabel('Device Type')                      # FIXED: was "Number of Devices Used"
ax.set_ylabel('Churn Rate (%)')
plt.tight_layout()
fig.savefig('figures/08_devices_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 8: Demographics ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
if GENDER_COL in df.columns:
    gdf = df.groupby([GENDER_COL,'churn_label']).size().unstack(fill_value=0)
    gdf.columns = ['Retained','Churned']
    gdf_pct = gdf.div(gdf.sum(axis=1), axis=0) * 100
    x = np.arange(len(gdf_pct)); w = 0.35
    b1 = axes[0].bar(x-w/2, gdf_pct['Retained'], w, label='Retained',
                     color=DARK, hatch='/', edgecolor='black')
    b2 = axes[0].bar(x+w/2, gdf_pct['Churned'], w, label='Churned',
                     color=RED, hatch='\\\\', edgecolor='black')
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2, h+0.5,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(gdf_pct.index)
    axes[0].set_title('Churn Rate by Gender', fontweight='bold')
    axes[0].set_ylabel('%'); axes[0].legend(); axes[0].set_ylim(0, 115)
else:
    axes[0].set_visible(False)

if AGE_COL in df.columns:
    for label, grp in df.groupby('churn_label'):
        grp[AGE_COL].plot.kde(ax=axes[1],
            color=RED if label==1 else DARK,
            linestyle='--' if label==1 else '-',
            linewidth=2, label='Churned' if label==1 else 'Retained')
    axes[1].set_title('Age Distribution by Churn Status', fontweight='bold')
    axes[1].set_xlabel('Age'); axes[1].legend()
else:
    axes[1].set_visible(False)
plt.tight_layout()
fig.savefig('figures/09_demographics_churn.png', bbox_inches='tight')
plt.close()

# ── Fig 9: Correlation heatmap — drop duplicate target ────────────────────────
# Keep only churn_label (drop raw 'churned' col before computing correlation)
num_df = df.drop(columns=['churned'], errors='ignore').select_dtypes(include=[np.number])
num_df = num_df.loc[:, num_df.std() > 0]
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(10, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.4,
            ax=ax, annot_kws={'size': 7}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix (churn_label only — no leakage)', fontweight='bold')
plt.tight_layout()
fig.savefig('figures/10_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING — find correct K, use it consistently
# ══════════════════════════════════════════════════════════════════════════════
seg_features = [c for c in [WATCH_COL, MONTHLY_COL, LOGIN_COL, PROFILES_COL] if c in df.columns]
seg_df = df[seg_features].dropna()
scaler_seg = StandardScaler()
X_seg = scaler_seg.fit_transform(seg_df)

inertias, sil_scores = [], []
K_RANGE = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_seg)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_seg, km.labels_, sample_size=min(5000, len(X_seg))))

K_OPTIMAL = list(K_RANGE)[np.argmax(sil_scores)]
SIL_OPTIMAL = max(sil_scores)
print(f"\nClustering: K_OPTIMAL={K_OPTIMAL}, Silhouette={SIL_OPTIMAL:.3f}")

# ── Fig 10: Elbow + Silhouette ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_RANGE), inertias, 'o-', color=RED, linewidth=2, markersize=6)
axes[0].set_title('K-Means Elbow — Inertia vs K', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)'); axes[0].set_ylabel('Inertia')
for k, v in zip(K_RANGE, inertias):
    axes[0].annotate(f'{v:,.0f}', (k, v), textcoords='offset points', xytext=(0, 7), fontsize=7)

axes[1].plot(list(K_RANGE), sil_scores, 's--', color=DARK, linewidth=2, markersize=6)
axes[1].scatter([K_OPTIMAL], [SIL_OPTIMAL], color=RED, s=100, zorder=5,
                label=f'Optimal K={K_OPTIMAL} (sil={SIL_OPTIMAL:.3f})')
axes[1].set_title('Silhouette Score vs K', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)'); axes[1].set_ylabel('Silhouette Score')
axes[1].legend(fontsize=8)
for k, v in zip(K_RANGE, sil_scores):
    axes[1].annotate(f'{v:.3f}', (k, v), textcoords='offset points', xytext=(0, 7), fontsize=7)
plt.tight_layout()
fig.savefig('figures/12_kmeans_elbow_silhouette.png', bbox_inches='tight')
plt.close()

# ── Fit final model ───────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=K_OPTIMAL, random_state=RANDOM_STATE, n_init=10)
cluster_labels_arr = km_final.fit_predict(X_seg)
df['cluster'] = -1
df.loc[seg_df.index, 'cluster'] = cluster_labels_arr

cluster_churn_order = df[df.cluster >= 0].groupby('cluster')['churn_label'].mean().sort_values()
label_pool = [
    'Highly Engaged Loyalists',
    'Casual Stable Viewers',
    'At-Risk Moderate Users',
    'Dormant High-Churn Risk',
    'Segment E', 'Segment F', 'Segment G', 'Segment H'
]
segment_map = {cl: label_pool[i] for i, cl in enumerate(cluster_churn_order.index.tolist())}
df['segment'] = df['cluster'].map(segment_map).fillna('Unknown')

seg_kpi = (df[df.cluster >= 0].groupby('segment')
           .agg(Count=('churn_label', 'count'), Churn_Rate=('churn_label', 'mean'))
           .sort_values('Churn_Rate', ascending=False)
           .reset_index())
seg_kpi['Churn_Rate_pct'] = (seg_kpi['Churn_Rate'] * 100).round(2)
print("\nSegmentation KPI:")
print(seg_kpi[['segment', 'Count', 'Churn_Rate_pct']].to_string(index=False))

# ── Fig 11: PCA segments — simplified (one colour per segment, X=churned) ──────
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_seg)
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=seg_df.index)
plot_df['segment'] = df.loc[seg_df.index, 'segment'].values
plot_df['churn']   = df.loc[seg_df.index, 'churn_label'].values
var_exp = pca.explained_variance_ratio_ * 100

# Assign one fixed colour per unique segment
unique_segs = seg_kpi['segment'].tolist()          # ordered by churn rate
seg_colors_list = plt.cm.tab10(np.linspace(0, 0.9, len(unique_segs)))
seg_color_map = {s: seg_colors_list[i] for i, s in enumerate(unique_segs)}

fig, ax = plt.subplots(figsize=(10, 6))
for seg in unique_segs:
    mask_ret = (plot_df.segment == seg) & (plot_df.churn == 0)
    mask_ch  = (plot_df.segment == seg) & (plot_df.churn == 1)
    c = seg_color_map[seg]
    # Sample up to 300 points per segment for legibility
    idx_r = plot_df[mask_ret].index[:300]
    idx_c = plot_df[mask_ch].index[:300]
    ax.scatter(plot_df.loc[idx_r, 'PC1'], plot_df.loc[idx_r, 'PC2'],
               c=[c], marker='o', s=15, alpha=0.4)
    ax.scatter(plot_df.loc[idx_c, 'PC1'], plot_df.loc[idx_c, 'PC2'],
               c=[c], marker='X', s=30, alpha=0.8, edgecolors='black', linewidths=0.3)

# Legend: one entry per segment showing colour only
handles = [mpatches.Patch(color=seg_color_map[s], label=s) for s in unique_segs]
handles += [
    plt.Line2D([0],[0], marker='o', color='grey', linestyle='None', markersize=6, label='Retained (circle)'),
    plt.Line2D([0],[0], marker='X', color='grey', linestyle='None', markersize=7, label='Churned (X)'),
]
ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.9, ncol=1)
ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)')
ax.set_title(f'Customer Segments — PCA 2D (K={K_OPTIMAL}, X = Churned)', fontweight='bold')
plt.tight_layout()
fig.savefig('figures/13_pca_segments.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELS — no data leakage (churned + churn_label both excluded)
# ══════════════════════════════════════════════════════════════════════════════
drop_from_features = ['customer_id', 'churned', 'churn_label', 'segment', 'cluster']
cat_cols = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
feature_df = df.drop(columns=drop_from_features, errors='ignore')
feature_df = pd.get_dummies(feature_df,
                             columns=[c for c in cat_cols if c in feature_df.columns],
                             drop_first=False, dtype=int)
feature_df = feature_df.select_dtypes(include=[np.number])
X = feature_df.fillna(feature_df.median(numeric_only=True))
y = df['churn_label']

print(f"\nFeature matrix: {X.shape} — columns: {list(X.columns)}")
assert 'churned' not in X.columns, "LEAKAGE: churned still in features"
assert 'churn_label' not in X.columns, "LEAKAGE: churn_label still in features"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"Train={len(X_train):,}  Test={len(X_test):,}")

# Logistic Regression
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=RANDOM_STATE, solver='lbfgs'))
])
lr_pipe.fit(X_train, y_train)
y_pred_lr   = lr_pipe.predict(X_test)
y_proba_lr  = lr_pipe.predict_proba(X_test)[:, 1]
cv_lr       = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

# Random Forest
param_grid = {'rf__n_estimators': [100, 200],
              'rf__max_depth': [None, 10, 20],
              'rf__min_samples_split': [2, 5]}
rf_pipe = Pipeline([('rf', RandomForestClassifier(
    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))])
grid_cv = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='roc_auc',
                       n_jobs=-1, verbose=0, refit=True)
grid_cv.fit(X_train, y_train)
best_rf     = grid_cv.best_estimator_
y_pred_rf   = best_rf.predict(X_test)
y_proba_rf  = best_rf.predict_proba(X_test)[:, 1]
cv_rf       = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

def get_metrics(yt, yp, yprob, cvs):
    return dict(
        accuracy     = round(accuracy_score(yt, yp), 4),
        precision_ch = round(precision_score(yt, yp, pos_label=1, zero_division=0), 4),
        recall_ch    = round(recall_score(yt, yp, pos_label=1, zero_division=0), 4),
        f1_macro     = round(f1_score(yt, yp, average='macro', zero_division=0), 4),
        f1_weighted  = round(f1_score(yt, yp, average='weighted', zero_division=0), 4),
        roc_auc_test = round(roc_auc_score(yt, yprob), 4),
        roc_auc_cv   = round(cvs.mean(), 4),
        roc_auc_std  = round(cvs.std(), 4),
    )

m_lr = get_metrics(y_test, y_pred_lr, y_proba_lr, cv_lr)
m_rf = get_metrics(y_test, y_pred_rf, y_proba_rf, cv_rf)
print(f"\nLR:  {m_lr}")
print(f"RF:  {m_rf}")
print(f"RF best params: {grid_cv.best_params_}")

# Feature importance
rf_imp = pd.Series(best_rf.named_steps['rf'].feature_importances_, index=X.columns)
top5_rf = rf_imp.nlargest(5)
print(f"\nTop-5 RF features: {list(zip(top5_rf.index, top5_rf.round(4).values))}")

lr_coef = pd.Series(lr_pipe.named_steps['lr'].coef_[0], index=X.columns)
top15_lr_raw = lr_coef[lr_coef.abs().nlargest(15).index].sort_values()

# ── Fig 12: Confusion matrices ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, model, name in [(axes[0], lr_pipe, 'Logistic Regression'),
                         (axes[1], best_rf,  'Random Forest (Tuned)')]:
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['Retained', 'Churned'],
        cmap='Reds', colorbar=False, ax=ax)
    ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
plt.tight_layout()
fig.savefig('figures/14_confusion_matrices.png', bbox_inches='tight')
plt.close()

# ── Fig 13: ROC curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
RocCurveDisplay.from_predictions(y_test, y_proba_lr,
    name=f'Logistic Regression (AUC={m_lr["roc_auc_test"]:.4f})',
    color=DARK, linestyle='-', linewidth=2, ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_rf,
    name=f'Random Forest (AUC={m_rf["roc_auc_test"]:.4f})',
    color=RED, linestyle='--', linewidth=2, ax=ax)
ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random Classifier')
ax.set_title('ROC Curves — Model Comparison', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig('figures/15_roc_curves.png', bbox_inches='tight')
plt.close()

# ── Fig 14: RF feature importances ───────────────────────────────────────────
top15_rf = rf_imp.nlargest(15).sort_values()
HATCH = ['/', '\\\\', 'x', '+', 'o', '.']
fig, ax = plt.subplots(figsize=(9, 6))
colors4 = [RED if v >= top15_rf.median() else DARK for v in top15_rf.values]
bars6 = ax.barh(top15_rf.index, top15_rf.values, color=colors4, edgecolor='black', linewidth=0.7)
for i, bar in enumerate(bars6):
    bar.set_hatch(HATCH[i % len(HATCH)])
    ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.4f}', va='center', fontsize=8)
ax.set_title('Random Forest — Top-15 Feature Importances', fontweight='bold')
ax.set_xlabel('Feature Importance')
red_p  = mpatches.Patch(color=RED,  hatch='/', label='High importance', edgecolor='black')
dark_p = mpatches.Patch(color=DARK, hatch='//',label='Moderate importance', edgecolor='black')
ax.legend(handles=[red_p, dark_p], fontsize=8)
plt.tight_layout()
fig.savefig('figures/16_rf_feature_importance.png', bbox_inches='tight')
plt.close()

# ── Fig 15: LR coefficients — clean ASCII labels ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
colors5 = [RED if v > 0 else DARK for v in top15_lr_raw.values]
bars7 = ax.barh(top15_lr_raw.index, top15_lr_raw.values, color=colors5,
                edgecolor='black', linewidth=0.7)
for i, bar in enumerate(bars7):
    bar.set_hatch(HATCH[i % len(HATCH)])
    xpos = bar.get_width() + 0.04 if bar.get_width() >= 0 else bar.get_width() - 0.04
    ax.text(xpos, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontsize=8)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('Logistic Regression — Top-15 Coefficients\n(Red=increases churn risk | Dark=decreases churn risk)',
             fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.tick_params(axis='y', labelsize=8)
red_p2  = mpatches.Patch(color=RED,  label='Increases churn risk (+)', edgecolor='black')
dark_p2 = mpatches.Patch(color=DARK, label='Decreases churn risk (-)', edgecolor='black')
ax.legend(handles=[red_p2, dark_p2], fontsize=8)
plt.tight_layout()
fig.savefig('figures/17_lr_coefficients.png', bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE RESULTS JSON for report builder
# ══════════════════════════════════════════════════════════════════════════════
results = {
    'total': total,
    'n_churned': n_churned,
    'churn_rate': round(churn_rate, 2),
    'retention_rate': round(ret_rate, 2),
    'avg_watch_all': round(avg_watch_all, 2),
    'avg_watch_ret': round(avg_watch_ret, 2),
    'avg_watch_ch':  round(avg_watch_ch, 2),
    'watch_unit': 'hrs/day',
    'plan_churn': plan_churn,
    'K_OPTIMAL': K_OPTIMAL,
    'sil_optimal': round(SIL_OPTIMAL, 3),
    'seg_kpi': seg_kpi[['segment','Count','Churn_Rate_pct']].to_dict(orient='records'),
    'lr': m_lr,
    'rf': m_rf,
    'rf_best_params': {k.replace('rf__',''):str(v) for k,v in grid_cv.best_params_.items()},
    'top5_rf': list(zip(top5_rf.index.tolist(), [round(v,4) for v in top5_rf.values])),
    'n_features': int(X.shape[1]),
    'n_train': int(len(X_train)),
    'n_test':  int(len(X_test)),
}
with open('figures/report_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ All figures saved to figures/")
print(json.dumps(results, indent=2))
