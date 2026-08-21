#!/usr/bin/env bash
# Runs everything documented in README.md against real WESAD data
# (data/WESAD/). Each block calls train.py then analyze_results.py, and
# copies results/ into results_archive/<name>/, since each next run
# overwrites the same results/history_<mode>.csv, results/plot_*.png files.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Takes ~15-25 min depending on the machine (CNN parts are slowest).
set -e  # stop on first error instead of continuing with broken results

DATA_DIR="data/WESAD"
if [ ! -d "$DATA_DIR" ]; then
  echo "$DATA_DIR not found -- check the dataset is there (S2/, S3/, ... or S2.pkl, ...)."
  exit 1
fi

# Start each run with a clean slate: mlp_main and cnn_main each get
# written to twice (main result, then multi-seed), so this can't live
# inside archive() itself -- it would wipe out the first write.
rm -rf results_archive
mkdir -p results_archive

clean_results() {
  # wipes results/ before each step, so archive() below only ever picks up
  # files that step actually produced, not leftovers from an earlier one
  rm -f results/*.csv results/*.png 2>/dev/null || true
}

archive() {
  # $1 = subfolder name under results_archive/
  mkdir -p "results_archive/$1"
  cp results/*.csv "results_archive/$1/" 2>/dev/null || true
  cp results/*.png "results_archive/$1/" 2>/dev/null || true
  echo "  -> results_archive/$1/"
}

echo "=== 1/8: MLP -- main result (4 methods) ==="
clean_results
python3 train.py --data-dir "$DATA_DIR" --modes fedavg fedper ditto fedper_adaptive --arch mlp --lr 0.01
python3 analyze_results.py
archive "mlp_main"

echo "=== 2/8: MLP -- statistical significance (10 seeds) ==="
clean_results
python3 multi_seed_eval.py --data-dir "$DATA_DIR" --arch mlp --lr 0.01
archive "mlp_main"

echo "=== 3/8: MLP -- hyperparameter ablation ==="
clean_results
python3 ablation_study.py --data-dir "$DATA_DIR" --arch mlp
archive "mlp_ablation"

echo "=== 4/8: MLP -- privacy (DP) and communication efficiency (quantization) ==="
clean_results
python3 privacy_efficiency_study.py --data-dir "$DATA_DIR" --arch mlp --lr 0.01
archive "mlp_privacy"

echo "=== 5/8: CNN -- main result (4 methods, --local-epochs 2!) ==="
clean_results
python3 train.py --data-dir "$DATA_DIR" --modes fedavg fedper ditto fedper_adaptive --arch cnn --local-epochs 2 --lr 0.1 --balanced-loss --ema-alpha 0.5
python3 analyze_results.py
archive "cnn_main"

echo "=== 6/8: CNN -- statistical significance (10 seeds) ==="
clean_results
python3 multi_seed_eval.py --data-dir "$DATA_DIR" --arch cnn --local-epochs 2 --lr 0.1 --balanced-loss --ema-alpha 0.5
archive "cnn_main"

echo "=== 7/8: CNN -- hyperparameter ablation ==="
clean_results
python3 ablation_study.py --data-dir "$DATA_DIR" --arch cnn
archive "cnn_ablation"

echo "=== 8/8: CNN -- privacy (DP) and communication efficiency (quantization) ==="
clean_results
python3 privacy_efficiency_study.py --data-dir "$DATA_DIR" --arch cnn --local-epochs 2 --lr 0.1
archive "cnn_privacy"

echo ""
echo "Done. All results are in results_archive/<mlp_main|mlp_ablation|mlp_privacy|cnn_main|cnn_ablation|cnn_privacy>/"
echo "results/ currently only has the last run (CNN privacy) -- everything else is archived above."
