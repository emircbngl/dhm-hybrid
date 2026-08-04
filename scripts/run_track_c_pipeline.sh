#!/usr/bin/env bash
# End-to-end Track C pipeline: synthetic refs → dataset → train → eval → demo.
# Run from the repo root.
#
# Usage:
#   bash scripts/run_track_c_pipeline.sh [--epochs N] [--skip-train] [--skip-eval] [--skip-demo]
#
# Idempotent: each step skips if its output already exists, unless you
# pass --force.

set -euo pipefail

EPOCHS=${EPOCHS:-80}
FORCE=0
SKIP_TRAIN=0
SKIP_EVAL=0
SKIP_DEMO=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DHM_DATA_ROOT:-${REPO_ROOT}/data/rapor}"
SYNTH_REF_DIR="${DATA_ROOT}/_synthetic_refs"
DATASET_DIR="${DATA_ROOT}/_track_c_dataset"
MODEL_DIR="models/track_c/v0.1"
EVAL_ROOT="${DATA_ROOT}/_track_c_eval"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --skip-demo) SKIP_DEMO=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Track C Pipeline — full end-to-end"
echo "  data root:    $DATA_ROOT"
echo "  synthetic ref:$SYNTH_REF_DIR"
echo "  dataset:      $DATASET_DIR"
echo "  model dir:    $MODEL_DIR"
echo "  epochs:       $EPOCHS"
echo "  force:        $FORCE"
echo "============================================================"

# ─── Step 1: synthetic refs ────────────────────────────────────────
if [[ $FORCE -eq 1 || ! -f "$SYNTH_REF_DIR/manifest.json" ]]; then
    echo ""
    echo "[1/4] Building synthetic median references per session…"
    python3 scripts/build_synthetic_refs.py
else
    echo ""
    echo "[1/4] Synthetic refs already exist, skipping. (--force to rebuild)"
fi

# ─── Step 2: training dataset ─────────────────────────────────────
if [[ $FORCE -eq 1 || ! -f "$DATASET_DIR/index.json" ]]; then
    echo ""
    echo "[2/4] Building Track C training dataset (synthetic-ref GT)…"
    python3 scripts/build_track_c_dataset.py
else
    echo ""
    echo "[2/4] Dataset already exists, skipping. (--force to rebuild)"
fi

# ─── Step 3: train ────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 1 ]]; then
    echo ""
    echo "[3/4] Skipped training (--skip-train)."
elif [[ $FORCE -eq 1 || ! -f "$MODEL_DIR/model.pt" ]]; then
    echo ""
    echo "[3/4] Training Track C residual U-Net ($EPOCHS epochs)…"
    rm -rf "$MODEL_DIR"
    python3 scripts/train_track_c.py \
        --epochs "$EPOCHS" \
        --crop-size 768 \
        --batch-size 1 \
        --base-channels 32 \
        --tv-weight 0.05 \
        --lr 3e-4 \
        --early-stop-patience 25 \
        --out-dir "$MODEL_DIR"
else
    echo ""
    echo "[3/4] Trained model already exists, skipping. (--force to retrain)"
fi

# ─── Step 4: evaluate ─────────────────────────────────────────────
if [[ $SKIP_EVAL -eq 1 ]]; then
    echo ""
    echo "[4/4] Skipped evaluation (--skip-eval)."
elif [[ -f "$MODEL_DIR/model.pt" ]]; then
    echo ""
    echo "[4/4] Evaluating on test split…"
    python3 scripts/eval_track_c.py \
        --model "$MODEL_DIR/model.pt" \
        --split test \
        --out-dir "$EVAL_ROOT/v0.1/test"
    python3 scripts/eval_track_c.py \
        --model "$MODEL_DIR/model.pt" \
        --split val \
        --out-dir "$EVAL_ROOT/v0.1/val"
else
    echo "no trained model at $MODEL_DIR/model.pt — skipping eval"
fi

# ─── Step 5: demo on a single hologram ────────────────────────────
if [[ $SKIP_DEMO -eq 1 ]]; then
    echo ""
    echo "[demo] Skipped (--skip-demo)."
elif [[ -f "$MODEL_DIR/model.pt" ]]; then
    echo ""
    DEMO_HOLO="$DATA_ROOT/session_01/DHM_2026_04_29_14_59_29.png"
    DEMO_OUT="$DATA_ROOT/_track_c_demo"
    if [[ -f "$DEMO_HOLO" ]]; then
        echo "[demo] Reconstructing single demo hologram (no reference)…"
        python3 scripts/reffree_reconstruct.py \
            --model "$MODEL_DIR/model.pt" \
            --hologram "$DEMO_HOLO" \
            --out-dir "$DEMO_OUT"
        echo "[demo] visualization: $DEMO_OUT/visualization.png"
    else
        echo "[demo] no demo hologram at $DEMO_HOLO, skipping"
    fi
fi

echo ""
echo "============================================================"
echo "Track C pipeline complete."
echo "  model:        $MODEL_DIR/model.pt"
echo "  eval reports: $EVAL_ROOT/v0.1/"
echo "  demo:         $DATA_ROOT/_track_c_demo/"
echo "============================================================"
