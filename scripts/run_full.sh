#!/usr/bin/env bash
# Pipeline completa: setup venv + smoke training + esecuzione notebook + verifica.
# Uso: bash scripts/run_full.sh [--quick]
#
# NOTA: la modalita' "full" (senza --quick) richiede ~470MB di CSV Kaggle in
# data/raw/ e puo' richiedere 30+ minuti su laptop. Per un check rapido usa --quick.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUICK_FLAG="${1:-}"

if [[ ! -d "venv" ]]; then
  echo "[setup] creo venv..."
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install --upgrade pip --quiet
  pip install -e ".[notebooks]" --quiet
else
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# --- Sanity check sul dataset ---
if [[ ! -f "data/raw/fraudTrain.csv" || ! -f "data/raw/fraudTest.csv" ]]; then
  echo "[ERRORE] CSV Kaggle assenti in data/raw/."
  echo "Scarica da https://www.kaggle.com/datasets/kartik2112/fraud-detection"
  echo "e copia fraudTrain.csv e fraudTest.csv in data/raw/."
  exit 1
fi

echo "[1/3] Training pipeline ${QUICK_FLAG:-(full)}..."
fraud-train ${QUICK_FLAG}

echo "[2/3] Rigenerazione + esecuzione notebook..."
python scripts/build_notebooks.py

# Esegue i notebook solo se NON in modalita' quick (potrebbero richiedere
# l'intero dataset). In quick si limita a rigenerare gli stub.
if [[ "${QUICK_FLAG}" != "--quick" ]]; then
  for nb in notebooks/01_eda_class_imbalance.ipynb \
            notebooks/02_feature_engineering.ipynb \
            notebooks/03_models_baseline_vs_ensemble.ipynb \
            notebooks/04_threshold_tuning_and_errors.ipynb; do
    echo "    -> $nb"
    jupyter nbconvert --to notebook --execute --inplace \
        "$nb" --ExecutePreprocessor.timeout=3600 \
        --log-level=ERROR
  done
fi

echo "[3/3] Validazione: smoke-test predict_fraud()..."
python -c "
from fraud_pipeline.inference import predict_fraud, example_transaction
res = predict_fraud(example_transaction())
print(f'  Predizione: prob={res[\"fraud_probability\"]:.4f}, '
      f'is_fraud={res[\"is_fraud\"]}, threshold={res[\"threshold\"]:.3f}')
assert 0.0 <= res['fraud_probability'] <= 1.0
print('  [OK] inference smoke test passed')
"

echo
echo "Pipeline completata. Vedi reports/ per i risultati."
