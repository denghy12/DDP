#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${CCIM_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/ccim_official}"
REPOSITORY="https://github.com/ydk122024/CCIM.git"
COMMIT="d6a651f91d1c1c91faca862ddeea915df9314919"
SOURCE_SHA="2e1b2f1178fcdc556aee7efed743c624020d43c34ec2272f7415dee4ce776d26"

mkdir -p "$(dirname "${DESTINATION}")"
if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone "${REPOSITORY}" "${DESTINATION}"
fi

if [[ -n "$(git -C "${DESTINATION}" status --porcelain)" ]]; then
  echo "CCIM source checkout has local changes; refusing to overwrite it" >&2
  exit 2
fi
git -C "${DESTINATION}" fetch origin
git -C "${DESTINATION}" switch --detach "${COMMIT}"

OBSERVED_COMMIT="$(git -C "${DESTINATION}" rev-parse HEAD)"
OBSERVED_SHA="$(sha256sum "${DESTINATION}/CCIM.py" | awk '{print $1}')"
[[ "${OBSERVED_COMMIT}" == "${COMMIT}" ]] || { echo "CCIM commit mismatch" >&2; exit 2; }
[[ "${OBSERVED_SHA}" == "${SOURCE_SHA}" ]] || { echo "CCIM.py SHA-256 mismatch" >&2; exit 2; }

echo "CCIM_SOURCE_ROOT=${DESTINATION}"
echo "CCIM_COMMIT=${OBSERVED_COMMIT}"
echo "CCIM_SOURCE_SHA256=${OBSERVED_SHA}"
