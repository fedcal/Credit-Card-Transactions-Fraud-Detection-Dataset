"""Genera i 4 notebook didattici in `notebooks/`.

Lo script e' la SORGENTE DI VERITA' dei notebook: per modificarli si
edita questo file e si rilancia (`python scripts/build_notebooks.py`).
Vantaggi:
    - Sorgente in formato testuale -> diff Git leggibili.
    - Riproducibilita' (chiunque rigenera notebook identici).
    - Niente metadati casuali (kernel locale, output cache) committati.

I notebook vengono SCRITTI ma NON ESEGUITI dallo script: l'esecuzione
end-to-end e' in `scripts/run_full.sh` e richiede i CSV Kaggle in data/raw/.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write_notebook(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    print(f"[OK] {out.relative_to(ROOT)}  ({len(cells)} celle)")


SETUP_DATASET_NOTE = (
    "!!! note \"Dataset richiesto\"\n"
    "    Il dataset Kaggle (~470MB) NON e' in repo per limiti di GitHub.\n"
    "    Scaricalo da <https://www.kaggle.com/datasets/kartik2112/fraud-detection>\n"
    "    e copia `fraudTrain.csv` e `fraudTest.csv` in `data/raw/`.\n"
)


# ============================================================================
# Notebook 01 — EDA + class imbalance
# ============================================================================
nb01 = [
    md(
        "# 01 — Esplorazione del dataset Kaggle Fraud Detection\n\n"
        "## Obiettivi didattici\n\n"
        "1. Comprendere la struttura del dataset (~1.5M transazioni, "
        "frodi ~0.5%).\n"
        "2. Visualizzare lo **sbilanciamento delle classi** e i suoi effetti "
        "sulle metriche standard.\n"
        "3. Analizzare la **distribuzione dell'importo** e il legame con "
        "l'etichetta di frode.\n"
        "4. Identificare **pattern temporali** (orario, giorno, mese) "
        "associati alle frodi.\n"
        "5. Esplorare la **distanza geografica** cliente-merchant.\n\n"
        + SETUP_DATASET_NOTE
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "\n"
        "from fraud_pipeline.data import load_train_test, class_distribution\n"
        "from fraud_pipeline.features import haversine_km\n"
    ),
    md(
        "## Caricamento\n\n"
        "Carichiamo i due CSV Kaggle. Il modulo `data.load_train_test` esegue "
        "automaticamente: validazione schema, parse del datetime, ordinamento "
        "cronologico."
    ),
    code(
        "df_train, df_test = load_train_test()\n"
        "print(f'Train: {df_train.shape}, frodi={df_train.is_fraud.sum()}')\n"
        "print(f'Test : {df_test.shape}, frodi={df_test.is_fraud.sum()}')\n"
        "df_train.head(3)\n"
    ),
    md(
        "## Distribuzione classi: sbilanciamento estremo\n\n"
        "La classe positiva (`is_fraud=1`) rappresenta una frazione molto "
        "piccola dei dati. Conseguenze pratiche:\n\n"
        "- **Accuracy non utile**: predire sempre 0 da' >99% accuracy ma 0% "
        "recall sulle frodi.\n"
        "- **Metriche da preferire**: AUC-PR (Average Precision), Recall, F1, "
        "F2 sulla classe positiva.\n"
        "- **Strategie di gestione**: `class_weight='balanced'` o resampling "
        "(SMOTE) — vedi notebook 03.\n"
    ),
    code(
        "dist = class_distribution(df_train.is_fraud)\n"
        "print('Train class distribution:')\n"
        "for k, v in dist.items():\n"
        "    print(f'  {k}: {v}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "counts = df_train.is_fraud.value_counts()\n"
        "ax.bar(['Legit (0)', 'Fraud (1)'], counts.values, color=['#4C72B0', '#C44E52'])\n"
        "for i, c in enumerate(counts.values):\n"
        "    ax.text(i, c, f'{c:,}', ha='center', va='bottom')\n"
        "ax.set_yscale('log')\n"
        "ax.set_title(f'Distribuzione classi (Train) — frodi: {100*dist[\"positive_rate\"]:.3f}%')\n"
        "plt.show()\n"
    ),
    md(
        "## Distribuzione dell'importo\n\n"
        "Le frodi tipicamente si concentrano in due cluster: **importi piccoli** "
        "(< $1, frodi di test della carta) e **importi medi-grandi** "
        "(massimizzazione del danno prima del blocco)."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "for label, ax in zip([0, 1], axes):\n"
        "    sub = df_train.loc[df_train.is_fraud == label, 'amt']\n"
        "    ax.hist(np.log1p(sub), bins=60, edgecolor='black',\n"
        "            color='#4C72B0' if label == 0 else '#C44E52', alpha=0.85)\n"
        "    ax.set_title(f'log1p(amt) — {\"Legit\" if label == 0 else \"Fraud\"} '\n"
        "                 f'(median ${sub.median():.2f})')\n"
        "    ax.set_xlabel('log1p(amt)')\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(
        "## Pattern temporali\n\n"
        "Decomponiamo il timestamp in ora, giorno della settimana, mese e "
        "guardiamo il fraud rate condizionato. Frodi tipicamente piu' frequenti "
        "di notte (carte rubate usate quando il proprietario dorme)."
    ),
    code(
        "tmp = df_train.assign(\n"
        "    hour=df_train.trans_date_trans_time.dt.hour,\n"
        "    dayofweek=df_train.trans_date_trans_time.dt.dayofweek,\n"
        "    month=df_train.trans_date_trans_time.dt.month,\n"
        ")\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, col in zip(axes, ['hour', 'dayofweek', 'month']):\n"
        "    rate = tmp.groupby(col)['is_fraud'].mean() * 100\n"
        "    ax.bar(rate.index, rate.values, color='#C44E52')\n"
        "    ax.set_title(f'Fraud rate per {col}')\n"
        "    ax.set_xlabel(col)\n"
        "    ax.set_ylabel('Fraud rate (%)')\n"
        "fig.tight_layout(); plt.show()\n"
    ),
    md(
        "## Analisi per categoria\n\n"
        "Le 14 categorie di merchant non hanno tutte lo stesso profilo di rischio."
    ),
    code(
        "cat_stats = (df_train.groupby('category')['is_fraud']\n"
        "             .agg(['mean', 'count']).rename(columns={'mean': 'fraud_rate'}))\n"
        "cat_stats['fraud_rate'] *= 100\n"
        "cat_stats = cat_stats.sort_values('fraud_rate', ascending=False)\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.barh(cat_stats.index, cat_stats.fraud_rate, color='#C44E52')\n"
        "ax.set_xlabel('Fraud rate (%)')\n"
        "ax.set_title('Fraud rate per categoria merchant')\n"
        "for i, v in enumerate(cat_stats.fraud_rate):\n"
        "    ax.text(v, i, f' {v:.2f}%', va='center')\n"
        "plt.tight_layout(); plt.show()\n"
        "cat_stats\n"
    ),
    md(
        "## Distanza geografica cliente-merchant\n\n"
        "Calcoliamo la distanza Haversine fra coordinate cliente e merchant. "
        "Frodi spesso avvengono a distanze maggiori (carte rubate / "
        "card-not-present con localizzazioni anomale)."
    ),
    code(
        "df_train['distance_km'] = haversine_km(\n"
        "    df_train.lat, df_train.long, df_train.merch_lat, df_train.merch_long,\n"
        ")\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "for label, color in [(0, '#4C72B0'), (1, '#C44E52')]:\n"
        "    sns.kdeplot(\n"
        "        df_train.loc[df_train.is_fraud == label, 'distance_km'].clip(upper=200),\n"
        "        ax=ax, label=f'is_fraud={label}', color=color, lw=2,\n"
        "    )\n"
        "ax.set_xlabel('distance_km (clipped a 200)')\n"
        "ax.set_title('Distanza geografica per classe')\n"
        "ax.legend(); plt.show()\n"
    ),
    md(
        "## Conclusioni dell'EDA e implicazioni per il modeling\n\n"
        "| Osservazione | Implicazione |\n"
        "|---|---|\n"
        "| Frodi ~0.5% | Metrica primaria: **AUC-PR**, non accuracy ne' AUC-ROC. |\n"
        "| Importo bimodale | Costruire feature `is_small_amt`, `log_amt`. |\n"
        "| Pattern orario | Costruire `hour`, `is_night`, `is_weekend`. |\n"
        "| Categoria predittiva | Tenere `category` come feature OneHot. |\n"
        "| Distanza geografica | Calcolare `distance_km` con Haversine. |\n"
        "| Sequenzialita' temporale | Split train/test cronologico, walk-forward CV. |\n\n"
        "-> Procedi al notebook **02_feature_engineering**.\n"
    ),
]
write_notebook("01_eda_class_imbalance.ipynb", nb01)


# ============================================================================
# Notebook 02 — Feature engineering
# ============================================================================
nb02 = [
    md(
        "# 02 — Feature engineering\n\n"
        "## Obiettivi didattici\n\n"
        "1. Trasformare il timestamp in **feature temporali** atomiche.\n"
        "2. Calcolare **distanza Haversine** fra cliente e merchant.\n"
        "3. Costruire **aggregati expanding per cliente** SENZA leakage temporale.\n"
        "4. Verificare la composizione del DataFrame post-FE.\n"
        + SETUP_DATASET_NOTE
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd\n"
        "\n"
        "from fraud_pipeline.data import load_train_test, downsample_for_smoke_test\n"
        "from fraud_pipeline.features import FraudFeatureEngineer\n"
        "from fraud_pipeline.preprocessing import build_preprocessor, infer_column_groups\n"
        "from fraud_pipeline.config import DEFAULT_CONFIG\n"
    ),
    md(
        "## Carichiamo un sample (50k righe) per velocita'\n\n"
        "Il feature engineering completo su 1.5M righe richiede ~30 secondi; "
        "su 50k e' istantaneo e didattico."
    ),
    code(
        "df_train, _ = load_train_test()\n"
        "df_train = downsample_for_smoke_test(df_train, n_rows=50_000,\n"
        "                                     random_state=DEFAULT_CONFIG.random_state)\n"
        "print(f'Sample: {df_train.shape}, frodi={df_train.is_fraud.sum()}')\n"
        "df_train.head(2)\n"
    ),
    md(
        "## FraudFeatureEngineer: 3 famiglie di feature\n\n"
        "**Temporali** (da `trans_date_trans_time` e `dob`):\n"
        "- `hour`, `day_of_week`, `month`, `is_weekend`, `is_night`\n"
        "- `customer_age_years`\n\n"
        "**Geografiche** (da `lat/long` cliente e merchant):\n"
        "- `distance_km` (Haversine)\n"
        "- `is_far_tx` (>500 km)\n\n"
        "**Trasformate dell'importo**:\n"
        "- `log_amt`, `is_small_amt`\n\n"
        "**Aggregati cliente** (expanding, NO leakage):\n"
        "- `customer_tx_count_so_far`\n"
        "- `customer_mean_amt_so_far`, `customer_std_amt_so_far`\n"
        "- `customer_amt_zscore` (deviazione vs storico)\n"
    ),
    code(
        "fe = FraudFeatureEngineer()\n"
        "X = df_train.drop(columns=['is_fraud'])\n"
        "X_fe = fe.fit_transform(X)\n"
        "added = sorted(set(X_fe.columns) - set(X.columns))\n"
        "print(f'Feature aggiunte ({len(added)}):')\n"
        "for c in added:\n"
        "    print(f'  - {c}: dtype={X_fe[c].dtype}')\n"
        "print(f'\\nColonne droppate dopo FE: {sorted(set(X.columns) - set(X_fe.columns))}')\n"
    ),
    md(
        "## Verifica no-leakage degli aggregati\n\n"
        "Per ogni cliente, la **prima** transazione cronologica deve avere "
        "`customer_tx_count_so_far == 0`. Se non fosse cosi', l'expanding starebbe "
        "includendo la riga corrente nel suo stesso aggregato."
    ),
    code(
        "verify = X_fe.assign(cc_num=X['cc_num'].values)\n"
        "first_per_customer = verify.sort_index().groupby('cc_num').head(1)\n"
        "violations = (first_per_customer['customer_tx_count_so_far'] != 0).sum()\n"
        "print(f'Violazioni leakage (atteso=0): {violations}')\n"
        "first_per_customer[['customer_tx_count_so_far',\n"
        "                    'customer_mean_amt_so_far',\n"
        "                    'customer_amt_zscore']].head()\n"
    ),
    md(
        "## Z-score vs storico: una feature potente\n\n"
        "`customer_amt_zscore` misura quanto la transazione corrente devia da "
        "quelle storiche dello stesso cliente. Frodi spesso hanno z-score "
        "elevato (importo anomalo). Visualizziamo la distribuzione condizionata."
    ),
    code(
        "df_check = X_fe.copy()\n"
        "df_check['is_fraud'] = df_train['is_fraud'].values\n"
        "fig, ax = plt.subplots(figsize=(9, 4)) if False else (None, None)\n"
        "import matplotlib.pyplot as plt; import seaborn as sns\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "for label, color in [(0, '#4C72B0'), (1, '#C44E52')]:\n"
        "    sub = df_check.loc[df_check.is_fraud == label, 'customer_amt_zscore'].clip(-5, 15)\n"
        "    sns.kdeplot(sub, ax=ax, label=f'is_fraud={label}', color=color, lw=2)\n"
        "ax.set_title('customer_amt_zscore per classe (clipped)')\n"
        "ax.legend(); plt.show()\n"
    ),
    md(
        "## ColumnTransformer (preprocessor)\n\n"
        "Due branch parallele:\n\n"
        "- **numeriche** -> SimpleImputer(median) + StandardScaler\n"
        "- **nominali** -> SimpleImputer('unknown') + OneHotEncoder(min_frequency=10)\n\n"
        "Le colonne ad altissima cardinalita' (`merchant`, 693 unique) sono "
        "droppate per non esplodere la dimensionalita'. In una pipeline avanzata "
        "si userebbe target encoding."
    ),
    code(
        "groups = infer_column_groups(X_fe)\n"
        "print(f'Numeriche: {len(groups[\"numeric\"])}')\n"
        "print(f'Nominali : {len(groups[\"nominal\"])}  ({groups[\"nominal\"]})')\n"
        "preproc = build_preprocessor(\n"
        "    numeric_cols=groups['numeric'],\n"
        "    nominal_cols=groups['nominal'],\n"
        ")\n"
        "preproc\n"
    ),
    code(
        "X_t = preproc.fit_transform(X_fe)\n"
        "n_in = X_fe.shape[1]\n"
        "n_out = X_t.shape[1]\n"
        "print(f'Feature in:  {n_in}')\n"
        "print(f'Feature out: {n_out}  (espansione +{n_out - n_in} colonne dovuta a OneHot)')\n"
    ),
    md(
        "## Conclusione\n\n"
        "Il feature engineer e' un **transformer sklearn**: serializzabile, "
        "componibile in `Pipeline`, garantisce no-leakage in CV. Pronto per il "
        "notebook **03_models_baseline_vs_ensemble**.\n"
    ),
]
write_notebook("02_feature_engineering.ipynb", nb02)


# ============================================================================
# Notebook 03 — Modeling: LogReg vs RF (vs XGBoost)
# ============================================================================
nb03 = [
    md(
        "# 03 — Modeling: baseline vs ensemble\n\n"
        "## Obiettivi didattici\n\n"
        "1. Confrontare **due famiglie** di modelli: lineare regolarizzato "
        "(Logistic Regression) e ensemble di alberi (Random Forest).\n"
        "2. Applicare **time-series cross-validation** (no shuffle).\n"
        "3. Gestire lo sbilanciamento via **`class_weight='balanced'`**.\n"
        "4. (Opzionale) Aggiungere XGBoost se installato.\n"
        "5. Selezionare il miglior modello via **AUC-PR su CV**.\n"
        + SETUP_DATASET_NOTE
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd\n"
        "\n"
        "from fraud_pipeline.data import load_train_test, split_features_target, downsample_for_smoke_test\n"
        "from fraud_pipeline.features import FraudFeatureEngineer\n"
        "from fraud_pipeline.preprocessing import build_preprocessor, infer_column_groups\n"
        "from fraud_pipeline.models import get_all_pipelines\n"
        "from fraud_pipeline.tuning import tune_all_models, summarize_tuning, PRIMARY_SCORING\n"
        "from fraud_pipeline.config import DEFAULT_CONFIG\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n"
    ),
    md("## Setup dati: sample 100k per smoke test"),
    code(
        "df_train, df_test = load_train_test()\n"
        "df_train = downsample_for_smoke_test(df_train, 100_000,\n"
        "                                     random_state=DEFAULT_CONFIG.random_state)\n"
        "X_train, y_train = split_features_target(df_train)\n"
        "X_test, y_test = split_features_target(df_test)\n"
        "\n"
        "fe = FraudFeatureEngineer()\n"
        "X_train_fe = fe.fit_transform(X_train)\n"
        "groups = infer_column_groups(X_train_fe)\n"
        "preproc = build_preprocessor(numeric_cols=groups['numeric'],\n"
        "                             nominal_cols=groups['nominal'])\n"
        "\n"
        "n_pos = max(int(y_train.sum()), 1)\n"
        "scale_pos_weight = (len(y_train) - n_pos) / n_pos\n"
        "print(f'scale_pos_weight = {scale_pos_weight:.1f}')\n"
        "\n"
        "base = get_all_pipelines(preproc, use_class_weight=True,\n"
        "                         include_xgboost=False,  # cambia a True se hai installato xgboost\n"
        "                         scale_pos_weight=scale_pos_weight)\n"
        "candidates = {n: Pipeline([('feature_engineer', FraudFeatureEngineer())] + list(p.steps))\n"
        "              for n, p in base.items()}\n"
        "list(candidates.keys())\n"
    ),
    md(
        "## Baseline cross-validation (no tuning)\n\n"
        "Misura l'AUC-PR dei modelli con iperparametri di default su "
        "**TimeSeriesSplit** (4 fold). Niente shuffle: i fold rispettano "
        "l'ordine temporale."
    ),
    code(
        "cv = TimeSeriesSplit(n_splits=DEFAULT_CONFIG.cv_splits)\n"
        "rows = []\n"
        "for name, pipe in candidates.items():\n"
        "    scores = cross_val_score(pipe, X_train, y_train,\n"
        "                             scoring=PRIMARY_SCORING, cv=cv, n_jobs=-1)\n"
        "    rows.append({'model': name, 'auc_pr_mean': scores.mean(), 'auc_pr_std': scores.std()})\n"
        "pd.DataFrame(rows).sort_values('auc_pr_mean', ascending=False)\n"
    ),
    md(
        "## Tuning iperparametri\n\n"
        "**Strategia per modello**:\n\n"
        "- **LogisticRegression**: GridSearchCV su `C` (forza regolarizzazione).\n"
        "- **RandomForest**: GridSearchCV su `n_estimators`, `max_depth`, "
        "`min_samples_leaf`.\n"
        "- **XGBoost** (se incluso): RandomizedSearchCV su grid combinatoria.\n\n"
        "**Scoring primario**: `average_precision` (= AUC-PR).\n"
    ),
    code(
        "results = tune_all_models(\n"
        "    pipelines=candidates,\n"
        "    X=X_train, y=y_train,\n"
        "    config=DEFAULT_CONFIG,\n"
        "    xgb_n_iter=8,\n"
        ")\n"
        "summary = summarize_tuning(results)\n"
        "summary\n"
    ),
    md(
        "## Discussione\n\n"
        "Tipicamente:\n\n"
        "- **Random Forest** vince su Logistic Regression per AUC-PR di alcuni "
        "punti percentuali: cattura interazioni non lineari (es. importo alto "
        "+ orario notturno + distanza grande).\n"
        "- **Logistic Regression** rimane comunque utile come **modello "
        "interpretabile** per il business: i coefficienti sono leggibili "
        "come log-odds ratio.\n"
        "- **XGBoost** (se incluso) tipicamente sorpassa RF di 1-3 punti su AUC-PR "
        "ma e' piu' costoso da tunare.\n"
    ),
    md("## Salvataggio modelli per il notebook successivo"),
    code(
        "import joblib\n"
        "from pathlib import Path\n"
        "out_dir = Path('../reports/models')\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name, r in results.items():\n"
        "    path = out_dir / f'{name.lower()}_best.joblib'\n"
        "    joblib.dump(r.best_estimator, path)\n"
        "    print(f'salvato: {path.name}  AUC-PR_cv={r.best_score:.4f}')\n"
    ),
]
write_notebook("03_models_baseline_vs_ensemble.ipynb", nb03)


# ============================================================================
# Notebook 04 — Threshold tuning + error analysis
# ============================================================================
nb04 = [
    md(
        "# 04 — Threshold tuning e analisi errori\n\n"
        "## Obiettivi didattici\n\n"
        "1. Tracciare **curva PR** e **curva ROC** del miglior modello.\n"
        "2. Ottimizzare la **soglia decisionale** in base alla matrice dei costi.\n"
        "3. Analizzare le **confusion matrix** alla soglia 0.5 vs ottimale.\n"
        "4. Esaminare **feature importance**.\n"
        "5. Esporre il modello tramite `predict_fraud()`.\n"
        + SETUP_DATASET_NOTE
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "import joblib\n"
        "from pathlib import Path\n"
        "\n"
        "from fraud_pipeline.data import load_train_test, split_features_target\n"
        "from fraud_pipeline.threshold import (\n"
        "    CostMatrix, optimal_threshold_by_cost, threshold_sweep,\n"
        "    confusion_matrix_at_threshold,\n"
        ")\n"
        "from fraud_pipeline.evaluation import (\n"
        "    compute_metrics, plot_pr_curve, plot_roc_curve, plot_confusion_matrix,\n"
        "    get_top_feature_importance, plot_feature_importance,\n"
        "    make_classification_report,\n"
        ")\n"
        "from fraud_pipeline.inference import predict_fraud, example_transaction\n"
    ),
    md("## Caricamento test set + miglior modello"),
    code(
        "_, df_test = load_train_test()\n"
        "X_test, y_test = split_features_target(df_test)\n"
        "\n"
        "models_dir = Path('../reports/models')\n"
        "model_paths = list(models_dir.glob('*_best.joblib'))\n"
        "models = {p.stem.replace('_best', ''): joblib.load(p) for p in model_paths}\n"
        "list(models.keys())\n"
    ),
    md(
        "## Metriche su holdout test (a soglia default 0.5)\n\n"
        "Riepilogo per ogni modello: precision, recall, F1, F2, AUC-PR, AUC-ROC."
    ),
    code(
        "rows = []\n"
        "probas = {}\n"
        "for name, model in models.items():\n"
        "    proba = model.predict_proba(X_test)[:, 1]\n"
        "    probas[name] = proba\n"
        "    m = compute_metrics(y_test.values, proba, threshold=0.5)\n"
        "    rows.append({'model': name, **m.as_dict()})\n"
        "metrics_df = pd.DataFrame(rows).sort_values('auc_pr', ascending=False).set_index('model')\n"
        "metrics_df.style.format({\n"
        "    'precision': '{:.3f}', 'recall': '{:.3f}',\n"
        "    'f1': '{:.3f}', 'f2': '{:.3f}',\n"
        "    'auc_pr': '{:.4f}', 'auc_roc': '{:.4f}',\n"
        "})\n"
    ),
    md(
        "## Curve PR e ROC del miglior modello\n\n"
        "**PR curve** mostra il vero trade-off su problemi sbilanciati. La linea "
        "tratteggiata rappresenta il random classifier (precision = prevalenza)."
    ),
    code(
        "best_name = metrics_df.index[0]\n"
        "print(f'Best model: {best_name}')\n"
        "best_proba = probas[best_name]\n"
        "fig1 = plot_pr_curve(y_test.values, best_proba,\n"
        "                     title=f'{best_name}: Precision-Recall (test)')\n"
        "plt.show()\n"
        "fig2 = plot_roc_curve(y_test.values, best_proba,\n"
        "                      title=f'{best_name}: ROC (test)')\n"
        "plt.show()\n"
    ),
    md(
        "## Ottimizzazione soglia su matrice di costi\n\n"
        "Cost matrix di esempio:\n"
        "- **Costo False Negative (frode mancata)**: $120 (perdita media stimata).\n"
        "- **Costo False Positive (legit bloccata)**: $5 (chargeback friction).\n\n"
        "La funzione `optimal_threshold_by_cost` trova la soglia che minimizza "
        "il costo atteso totale."
    ),
    code(
        "cost = CostMatrix(cost_fn=120.0, cost_fp=5.0)\n"
        "best_threshold, best_cost = optimal_threshold_by_cost(\n"
        "    y_test.values, best_proba, cost=cost,\n"
        ")\n"
        "print(f'Soglia ottimale: t = {best_threshold:.4f}')\n"
        "print(f'Costo atteso  : ${best_cost:,.2f}')\n"
    ),
    md(
        "## Sweep delle soglie\n\n"
        "Visualizziamo precision/recall/F1/F2/cost al variare della soglia per "
        "capire dove si trova il punto di equilibrio."
    ),
    code(
        "sweep = threshold_sweep(y_test.values, best_proba, cost=cost)\n"
        "fig, ax1 = plt.subplots(figsize=(11, 5))\n"
        "ax1.plot(sweep.threshold, sweep.precision, label='precision', color='#4C72B0')\n"
        "ax1.plot(sweep.threshold, sweep.recall, label='recall', color='#C44E52')\n"
        "ax1.plot(sweep.threshold, sweep.f1, label='F1', color='#55A868', linestyle='--')\n"
        "ax1.plot(sweep.threshold, sweep.f2, label='F2', color='#8172B2', linestyle='--')\n"
        "ax1.set_xlabel('threshold')\n"
        "ax1.set_ylabel('metric value')\n"
        "ax1.legend(loc='center left')\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(sweep.threshold, sweep.expected_cost, color='black', alpha=0.6,\n"
        "         label='cost')\n"
        "ax2.axvline(best_threshold, color='red', linestyle=':', label=f't*={best_threshold:.3f}')\n"
        "ax2.set_ylabel('expected_cost ($)')\n"
        "ax2.legend(loc='center right')\n"
        "plt.title('Threshold sweep'); plt.show()\n"
    ),
    md(
        "## Confronto confusion matrix: 0.5 vs ottimale\n\n"
        "Vediamo l'effetto pratico dello spostamento della soglia."
    ),
    code(
        "for t, label in [(0.5, 'default'), (best_threshold, 'optimal')]:\n"
        "    cm = confusion_matrix_at_threshold(y_test.values, best_proba, t)\n"
        "    print(f'Soglia {label} = {t:.4f}: {cm}')\n"
        "    y_pred = (best_proba >= t).astype(int)\n"
        "    plot_confusion_matrix(y_test.values, y_pred,\n"
        "                          title=f'{best_name} @t={t:.3f} ({label})')\n"
        "    plt.show()\n"
    ),
    md(
        "## Classification report a soglia ottimale"
    ),
    code(
        "y_pred_opt = (best_proba >= best_threshold).astype(int)\n"
        "print(make_classification_report(y_test.values, y_pred_opt))\n"
    ),
    md(
        "## Feature importance del miglior modello (se tree-based)"
    ),
    code(
        "model = models[best_name]\n"
        "try:\n"
        "    # Recupera nomi feature post-preprocessor.\n"
        "    fe = model.named_steps['feature_engineer']\n"
        "    preproc = model.named_steps['preprocessor']\n"
        "    feature_names = preproc.get_feature_names_out()\n"
        "    # Per estrarre feature_importances_, dobbiamo passargli la pipeline\n"
        "    # del modello finale, non quella esterna.\n"
        "    inner = type(model)(steps=[('preprocessor', preproc),\n"
        "                                ('model', model.named_steps['model'])])\n"
        "    df_imp = get_top_feature_importance(inner, feature_names, top_n=20)\n"
        "    print(df_imp.to_string(index=False))\n"
        "    plot_feature_importance(df_imp, title=f'{best_name}: top-20 feature')\n"
        "    plt.show()\n"
        "except Exception as e:\n"
        "    print(f'Feature importance non estratta: {e}')\n"
    ),
    md(
        "## API di inferenza: `predict_fraud()`\n\n"
        "Il modello e' ora servito tramite una funzione semplice. Accetta dict "
        "(singola transazione) o DataFrame (batch). La soglia ottimale e' caricata "
        "automaticamente da `reports/models/threshold.json`."
    ),
    code(
        "tx = example_transaction()\n"
        "result = predict_fraud(tx, threshold=best_threshold)\n"
        "print(result)\n"
        "\n"
        "# Batch su 5 transazioni del test set:\n"
        "batch_results = predict_fraud(X_test.head(5), threshold=best_threshold)\n"
        "for i, r in enumerate(batch_results):\n"
        "    actual = int(y_test.iloc[i])\n"
        "    print(f'  tx[{i}]: prob={r[\"fraud_probability\"]:.4f}  '\n"
        "          f'pred={int(r[\"is_fraud\"])}  actual={actual}')\n"
    ),
    md(
        "## Conclusione\n\n"
        "Il pipeline e' ora **production-ready**:\n\n"
        "1. Modello + threshold serializzati in `reports/models/`.\n"
        "2. `predict_fraud()` come unico punto di ingresso per l'inferenza.\n"
        "3. Metriche e analisi errori in `reports/`.\n"
        "4. Soglia decisionale ottimizzata sulla matrice di costi del business.\n\n"
        "Per dettagli teorici (perche' AUC-PR, perche' time-series CV, perche' "
        "log-amt e z-score, ...) vedi `docs/teoria/`.\n"
    ),
]
write_notebook("04_threshold_tuning_and_errors.ipynb", nb04)


print("\nTutti i 4 notebook generati in", NB_DIR)
