#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE="${DERPP_SOURCE_BASE:-/mnt/haoyuan/workspace/baseline_sources}"
SOURCE_TRANSPORT="${DERPP_SOURCE_TRANSPORT:-auto}"
ARCHIVE_NAME="derpp_neurips2020_cb9a36d.tar.gz"
ARCHIVE_PATH="${SOURCE_BASE}/${ARCHIVE_NAME}"
SOURCE_ROOT="${SOURCE_BASE}/derpp_official"
URL="https://codeload.github.com/aimagelab/mammoth/tar.gz/refs/tags/neurips2020"
EXPECTED_ARCHIVE_SHA="d7cdffefdb7d77939a1055984cb586ad83af0220439cd76e4a106ea201c1695b"
EXPECTED_DERPP_SHA="d38736e8d8c888300a8e5ddcac7cab1ac12e2eb1a0aa4c28f2387bdd79fc973a"
EXPECTED_BUFFER_SHA="3f4c9b416e22241bd6417d1cd2d6ec0625d7309c1d9e308757635debe545d5d3"
EXPECTED_LICENSE_SHA="309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275"

mkdir -p "${SOURCE_BASE}"
[[ "${SOURCE_TRANSPORT}" == "auto" || "${SOURCE_TRANSPORT}" == "ssh" ]] || {
  echo "DERPP_SOURCE_TRANSPORT must be auto or ssh" >&2
  exit 2
}

verify_source() {
  [[ "$(sha256sum "${SOURCE_ROOT}/models/derpp.py" | awk '{print $1}')" == "${EXPECTED_DERPP_SHA}" ]]
  [[ "$(sha256sum "${SOURCE_ROOT}/utils/buffer.py" | awk '{print $1}')" == "${EXPECTED_BUFFER_SHA}" ]]
  [[ "$(sha256sum "${SOURCE_ROOT}/LICENSE" | awk '{print $1}')" == "${EXPECTED_LICENSE_SHA}" ]]
}

if [[ -d "${SOURCE_ROOT}" ]]; then
  if verify_source; then
    echo "Fixed DER++ source is already prepared: ${SOURCE_ROOT}"
    exit 0
  fi
  echo "Existing DER++ source directory failed immutable hash checks: ${SOURCE_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  DOWNLOAD_RC=1
  if [[ "${SOURCE_TRANSPORT}" == "auto" ]]; then
    set +e
    curl --http1.1 -fL \
      --retry 10 \
      --retry-all-errors \
      --retry-delay 3 \
      --connect-timeout 30 \
      --max-time 600 \
      "${URL}" \
      -o "${ARCHIVE_PATH}.download"
    DOWNLOAD_RC=$?
    set -e
  fi
  if [[ "${DOWNLOAD_RC}" -eq 0 ]]; then
    mv "${ARCHIVE_PATH}.download" "${ARCHIVE_PATH}"
  else
    if [[ "${SOURCE_TRANSPORT}" == "auto" ]]; then
      echo "HTTPS archive download failed with exit code ${DOWNLOAD_RC}" >&2
    else
      echo "Skipping HTTPS archive download by request" >&2
    fi
    echo "Falling back to the configured GitHub SSH transport..." >&2
    git clone \
      --depth 1 \
      --branch neurips2020 \
      --single-branch \
      git@github.com:aimagelab/mammoth.git \
      "${SOURCE_ROOT}"
    ACTUAL_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
    if [[ "${ACTUAL_COMMIT}" != "cb9a36d788d6ad051c9eee0da358b25421d909f5" ]]; then
      echo "DER++ SSH clone resolved to an unexpected commit" >&2
      echo "actual=${ACTUAL_COMMIT}" >&2
      exit 2
    fi
    if ! verify_source; then
      echo "SSH-cloned DER++ source failed immutable hash checks" >&2
      exit 2
    fi
    echo "DER++ source prepared successfully through GitHub SSH"
    echo "source=${SOURCE_ROOT}"
    echo "commit=${ACTUAL_COMMIT}"
    exit 0
  fi
fi

ACTUAL_ARCHIVE_SHA="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_ARCHIVE_SHA}" != "${EXPECTED_ARCHIVE_SHA}" ]]; then
  echo "DER++ archive SHA-256 mismatch" >&2
  echo "expected=${EXPECTED_ARCHIVE_SHA}" >&2
  echo "actual=${ACTUAL_ARCHIVE_SHA}" >&2
  exit 2
fi

STAGE="$(mktemp -d "${SOURCE_BASE}/.derpp_stage.XXXXXX")"
tar -xzf "${ARCHIVE_PATH}" --strip-components=1 -C "${STAGE}"
mv "${STAGE}" "${SOURCE_ROOT}"

if ! verify_source; then
  echo "Extracted DER++ source failed immutable hash checks" >&2
  exit 2
fi

echo "DER++ source prepared successfully"
echo "source=${SOURCE_ROOT}"
echo "archive=${ARCHIVE_PATH}"
echo "archive_sha256=${ACTUAL_ARCHIVE_SHA}"
