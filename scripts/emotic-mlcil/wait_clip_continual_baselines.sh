#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${STATE_DIR:?STATE_DIR is required}"
JOB_KEYS="${JOB_KEYS:?JOB_KEYS is required}"
POLL_SECONDS="${POLL_SECONDS:-20}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-}"
RUN_ID="${RUN_ID:-}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-}"

read -r -a jobs <<< "${JOB_KEYS}"
while true; do
  completed=0
  failed=0
  for job in "${jobs[@]}"; do
    if [[ -f "${STATE_DIR}/${job}.done" ]]; then
      completed=$((completed + 1))
    elif [[ -f "${STATE_DIR}/${job}.failed" ]]; then
      failed=$((failed + 1))
    fi
  done
  echo "baseline jobs: completed=${completed}/${#jobs[@]} failed=${failed}"
  if [[ "${failed}" -gt 0 ]]; then
    echo "At least one baseline job failed; inspect ${STATE_DIR}/*.log" >&2
    exit 1
  fi
  if [[ "${completed}" -eq "${#jobs[@]}" ]]; then
    echo "All CLIP continual baseline jobs completed"
    if [[ -n "${RUN_OUTPUT_ROOT}" || -n "${RUN_ID}" || -n "${COLLECT_SCRIPT}" ]]; then
      [[ -n "${RUN_OUTPUT_ROOT}" && -n "${RUN_ID}" && -n "${COLLECT_SCRIPT}" ]] || {
        echo "Result collection requires RUN_OUTPUT_ROOT, RUN_ID, and COLLECT_SCRIPT" >&2
        exit 1
      }
      "${PYTHON}" "${COLLECT_SCRIPT}" \
        --run-output-root "${RUN_OUTPUT_ROOT}" \
        --run-id "${RUN_ID}" \
        --expected-bundles "${#jobs[@]}"
    fi
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
