#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${STATE_DIR:?STATE_DIR is required}"
JOB_KEYS="${JOB_KEYS:?JOB_KEYS is required}"
POLL_SECONDS="${POLL_SECONDS:-20}"

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
    echo "All frozen-CLIP baseline jobs completed"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
