#!/usr/bin/env bash
# Download pinned Packer and Terraform releases into the ignored
# infrastructure/nebius/.tools/ directory. Every archive is verified against
# the published SHA-256 checksum pinned below. This script never runs any
# paid command; it only prepares free local validation tooling.
set -euo pipefail

PACKER_VERSION="1.11.2"
TERRAFORM_VERSION="1.5.7"

packer_sha256_darwin_amd64="107c4334b136ffb5b884bac87f2ef6620f15df7d1d0a646db20b8054f9c607fe"
packer_sha256_darwin_arm64="b89f4944cca27839922a397248b94fc20d92acf15933bb36d58eb6d1283dc254"
packer_sha256_linux_amd64="ced13efc257d0255932d14b8ae8f38863265133739a007c430cae106afcfc45a"
packer_sha256_linux_arm64="dd296d743dd4593304307583cff5290bba9b868fc2b0b605b64566f8141ca728"

terraform_sha256_darwin_amd64="b310ec0e626e9799000cfc8e30247cd827cf7f8030c8e0400257c7f111e93537"
terraform_sha256_darwin_arm64="db7c33eb1a446b73a443e2c55b532845f7b70cd56100bec4c96f15cfab5f50cb"
terraform_sha256_linux_amd64="c0ed7bc32ee52ae255af9982c8c88a7a4c610485cf1d55feeb037eab75fa082c"
terraform_sha256_linux_arm64="f4b4ad7c6b6088960a667e34495cae490fb072947a9ff266bf5929f5333565e4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/../.tools"
mkdir -p "${TOOLS_DIR}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) echo "unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac
PLATFORM_KEY="${OS}_${ARCH}"

if command -v sha256sum >/dev/null 2>&1; then
  SHA256_CHECK() { sha256sum --check --strict "$@"; }
else
  SHA256_CHECK() { shasum -a 256 -c "$@"; }
fi

fetch_verified() {
  local url="$1" destination="$2" expected_sha256="$3"
  if [[ -x "${destination}" ]]; then
    return 0
  fi
  local archive="${destination}.zip"
  curl -fsSL --retry 3 --retry-delay 5 -o "${archive}" "${url}"
  echo "${expected_sha256}  ${archive}" > "${archive}.sha256"
  SHA256_CHECK "${archive}.sha256"
  rm -f "${archive}.sha256"
  unzip -o -q "${archive}" -d "${TOOLS_DIR}"
  rm -f "${archive}"
}

packer_var="packer_sha256_${PLATFORM_KEY}"
terraform_var="terraform_sha256_${PLATFORM_KEY}"

fetch_verified \
  "https://releases.hashicorp.com/packer/${PACKER_VERSION}/packer_${PACKER_VERSION}_${OS}_${ARCH}.zip" \
  "${TOOLS_DIR}/packer" \
  "${!packer_var}"

fetch_verified \
  "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_${OS}_${ARCH}.zip" \
  "${TOOLS_DIR}/terraform" \
  "${!terraform_var}"

chmod +x "${TOOLS_DIR}/packer" "${TOOLS_DIR}/terraform"
"${TOOLS_DIR}/packer" version | sed -n '1p'
"${TOOLS_DIR}/terraform" version | sed -n '1p'
