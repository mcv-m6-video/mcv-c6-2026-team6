#!/usr/bin/env bash
# Unpack Hugging Face SN-BAS-2025 zips into a single merged dataset root.
#
# The Hub repo ships train.zip, valid.zip, test.zip (see SoccerNet/SN-BAS-2025).
# Starter code expects labels_dir/<video>/Labels-ball.json where <video> is like
#   england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich
#
# Usage (after huggingface-cli download):
#   bash "${C6_ROOT}/scripts/prepare_sn_bas_data.sh"
#
# Environment variables:
#   C6_ROOT           — default: parent of scripts/
#   SN_BAS_ZIP_DIR    — override zip location.
#                       If unset, the script looks in data/hf_sn_bas then data/sn_bas_2025.
#   SN_BAS_ZIP_PASSWORD — REQUIRED: SoccerNet NDA password. Split zips use WinZip AES;
#                       GNU unzip -P often still skips all files (empty match dirs). This script
#                       uses scripts/extract_split_zips.py (pyzipper) or 7z if available.

set -euo pipefail

C6_ROOT="${C6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${C6_ROOT}/data/sn_bas_2025"

# Locate zip directory
if [[ -n "${SN_BAS_ZIP_DIR:-}" ]]; then
    ARCH="${SN_BAS_ZIP_DIR}"
elif [[ -f "${C6_ROOT}/data/hf_sn_bas/train.zip" ]]; then
    ARCH="${C6_ROOT}/data/hf_sn_bas"
elif [[ -f "${C6_ROOT}/data/sn_bas_2025/train.zip" ]]; then
    ARCH="${C6_ROOT}/data/sn_bas_2025"
else
    echo "ERROR: Cannot find train.zip. Download with huggingface-cli first." >&2
    echo "  huggingface-cli download SoccerNet/SN-BAS-2025 --repo-type dataset --local-dir ${C6_ROOT}/data/hf_sn_bas" >&2
    exit 1
fi

echo "Using split zips from: ${ARCH}"

if [[ -z "${SN_BAS_ZIP_PASSWORD:-}" ]]; then
    echo "ERROR: SN_BAS_ZIP_PASSWORD is not set." >&2
    echo "The train/valid/test zips from SoccerNet/SN-BAS-2025 are password-protected (NDA)." >&2
    echo "Export the password you received after signing the NDA, then re-run:" >&2
    echo "  export SN_BAS_ZIP_PASSWORD=\$(cat \"\${C6_ROOT}/sn_bas_pass.txt\")" >&2
    echo "  bash \"\${C6_ROOT}/scripts/prepare_sn_bas_data.sh\"" >&2
    exit 1
fi

mkdir -p "${OUT}"

python3 "${C6_ROOT}/scripts/extract_split_zips.py" \
    "${ARCH}" "${OUT}" "${SN_BAS_ZIP_PASSWORD}"

# Verify at least one label file exists
PROBE="${OUT}/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/Labels-ball.json"
if [[ ! -f "${PROBE}" ]]; then
    echo "ERROR: Extraction finished but Labels-ball.json not found at expected path:" >&2
    echo "  ${PROBE}" >&2
    echo "Wrong SN_BAS_ZIP_PASSWORD, or extraction failed. Ensure: pip install pyzipper" >&2
    exit 1
fi

echo "Extraction complete. Verify full layout with:"
echo "  python3 ${C6_ROOT}/scripts/verify_data_layout.py"
