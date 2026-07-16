#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"

# One-time numerical audit.  A second vanilla CLIP exists only inside this
# comparison process and is released before any Adapter training starts.
EQUIVALENCE_JSON="./output/emotic_ddp_prompt_free_auxiliary_equivalence/equivalence.json"
EQUIVALENCE_OK="$(python - "${EQUIVALENCE_JSON}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
passed = os.path.isfile(path) and json.load(open(path, encoding="utf-8")).get("passed") is True
print("1" if passed else "0")
PY
)"
if [[ "${EQUIVALENCE_OK}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    smoke_emotic_ddp_prompt_free_equivalence.py
else
  echo "Skip completed prompt-free equivalence audit: ${EQUIVALENCE_JSON}"
fi

# Train three seeds from the DDP-owned prompt-free branch.  The first seed
# writes the frozen global-feature cache; later seeds reuse it.
for seed in 0 1 2; do
  GPU="${GPU}" SEED="${seed}" bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_prompt_free_auxiliary_base5.sh
done

# Discard the auxiliary head and evaluate only W1/W2 on prompted CLS.  This
# reuses the existing pure-validation global-scale lock and strict all-task
# DDP evaluation; no external Prototype Adapter checkpoint is read.
GPU="${GPU}" ADAPTER_SOURCE=ddp_aux_full_base5 bash \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_feature_correction.sh
