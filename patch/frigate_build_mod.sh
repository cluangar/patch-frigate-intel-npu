#!/bin/bash

# Intel NPU Driver Installation Script for Frigate Docker Container
# Updated for v1.19.0 (Aug 2025), Level Zero v1.22.4, and OpenVINO 2025.2.0
# Checks for local .deb files first, only downloads if missing
# Allows OpenVINO version override with OPENVINO_VER env var

set -e  # Exit on any error

# Auto-detect Frigate container name from image
CONTAINER_NAME=$(docker ps --filter ancestor=ghcr.io/blakeblackshear/frigate:0.16.0-rc3 --format '{{.Names}}' | head -n1)

# Get absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/openvino.py"

# Download directory
DOWNLOAD_DIR="$HOME/npu-drivers"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Default OpenVINO version (can be overridden via env var)
OPENVINO_VER="${OPENVINO_VER:-2025.2.0}"

echo "🚀 Installing Intel NPU drivers v1.19.0 (Ubuntu22.04) in Frigate container..."
echo "Using OpenVINO version: $OPENVINO_VER"

# Check if container exists and is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Frigate container not found or not running"
    echo "Make sure container '$CONTAINER_NAME' is running with: docker compose up -d"
    exit 1
fi

# Intel NPU driver package list (Ubuntu 22.04 build)
PKGS=(
    "intel-driver-compiler-npu_1.19.0.20250707-16111289554_ubuntu22.04_amd64.deb"
    "intel-fw-npu_1.19.0.20250707-16111289554_ubuntu22.04_amd64.deb"
    "intel-level-zero-npu_1.19.0.20250707-16111289554_ubuntu22.04_amd64.deb"
)

# Level Zero runtime (updated to v1.22.4)
PKGS+=("level-zero_1.22.4+u22.04_amd64.deb")

# Dependencies from Debian repos
PKGS+=(
    "libtbb12_2021.8.0-2_amd64.deb"
    "libtbbbind-2-5_2021.8.0-2_amd64.deb"
    "libtbbmalloc2_2021.8.0-2_amd64.deb"
    "libhwloc15_2.9.0-1_amd64.deb"
)

# URLs for each package
URLS=(
    "https://github.com/intel/linux-npu-driver/releases/download/v1.19.0/${PKGS[0]}"
    "https://github.com/intel/linux-npu-driver/releases/download/v1.19.0/${PKGS[1]}"
    "https://github.com/intel/linux-npu-driver/releases/download/v1.19.0/${PKGS[2]}"
    "https://github.com/oneapi-src/level-zero/releases/download/v1.22.4/${PKGS[3]}"
    "http://ftp.us.debian.org/debian/pool/main/o/onetbb/${PKGS[4]}"
    "http://ftp.us.debian.org/debian/pool/main/o/onetbb/${PKGS[5]}"
    "http://ftp.us.debian.org/debian/pool/main/o/onetbb/${PKGS[6]}"
    "http://ftp.us.debian.org/debian/pool/main/h/hwloc/${PKGS[7]}"
)

# Download missing files
for i in "${!PKGS[@]}"; do
    if [[ -f "${PKGS[$i]}" ]]; then
        echo "📥 Found local ${PKGS[$i]}, skipping download"
    else
        echo "📥 Downloading ${PKGS[$i]}..."
        wget -q "${URLS[$i]}"
    fi
done

echo "📦 Copying packages into Frigate container..."
for deb_file in "${PKGS[@]}"; do
    docker cp "$deb_file" "$CONTAINER_NAME:/tmp/"
done

echo "• Installing packages in container..."
docker exec "$CONTAINER_NAME" bash -c "
cd /tmp
echo 'Removing any conflicting packages...'
dpkg --purge --force-remove-reinstreq intel-fw-npu intel-driver-compiler-npu intel-level-zero-npu level-zero 2>/dev/null || true

echo 'Installing NPU packages...'
dpkg -i *.deb || {
    echo 'Fixing dependencies...'
    apt update -qq
    apt --fix-broken install -y -qq
    dpkg -i *.deb
}

echo 'Updating library cache...'
ldconfig

echo 'Cleaning up temp files...'
rm -f *.deb
"

echo "🔧 Updating OpenVINO in container..."
docker exec "$CONTAINER_NAME" bash -c "
pip install --upgrade --quiet --break-system-packages --resume-retries 5 --timeout 120 openvino==${OPENVINO_VER}
"

# --- Patch Frigate's openvino.py ---
if [ -f "$PATCH_FILE" ]; then
    echo "📦 Copying patched Frigate openvino.py into container..."
    docker exec "$CONTAINER_NAME" bash -c "
        if [ -f /opt/frigate/frigate/detectors/plugins/openvino.py ]; then
            cp /opt/frigate/frigate/detectors/plugins/openvino.py \
               /opt/frigate/frigate/detectors/plugins/openvino.py.bak
        fi
    "
    docker cp "$PATCH_FILE" "$CONTAINER_NAME:/opt/frigate/frigate/detectors/plugins/openvino.py"
    echo "🔧 Patched Frigate openvino.py applied"
else
    echo "❌ No patched openvino.py found at $PATCH_FILE, skipping patch"
fi

echo "🧪 Testing NPU detection..."
docker exec "$CONTAINER_NAME" python3 -c "
try:
    from openvino import Core
    core = Core()
    devices = core.available_devices
    print('Available OpenVINO devices in Frigate:', devices)
    if 'NPU' in devices:
        print('✅ SUCCESS: NPU detected in Frigate container!')
        try:
            npu_name = core.get_property('NPU', 'FULL_DEVICE_NAME')
            print(f'NPU info: {npu_name}')
        except:
            print('NPU detected but detailed info unavailable')
    else:
        print('❌ NPU not detected. Available devices:', devices)
except Exception as e:
    print(f'❌ OpenVINO test failed: {e}')
"

echo "🔍 Checking NPU device access..."
docker exec "$CONTAINER_NAME" bash -c "
if [ -e /dev/accel/accel0 ]; then
    echo '✅ NPU device /dev/accel/accel0 is accessible'
    ls -la /dev/accel/accel0
else
    echo '❌ NPU device not found - check Docker device passthrough'
fi
"

echo "🔄 Restarting Frigate container to complete setup..."
docker restart "$CONTAINER_NAME"
sleep 8

echo "🧪 Final NPU verification..."
docker exec "$CONTAINER_NAME" python3 -c "
from openvino import Core
core = Core()
devices = core.available_devices
print('Final test - Available devices:', devices)
if 'NPU' in devices:
    print('✅ NPU successfully installed and working!')
else:
    print('❌ NPU not detected after restart')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update your Frigate config.yaml detector section:"
echo "   detectors:"
echo "     openvino:"
echo "       type: openvino"
echo "       device: NPU    # Changed from GPU to NPU"
echo ""
echo "2. Restart Frigate to use NPU:"
echo "   docker restart frigate"
echo ""
echo "3. Check Frigate logs for NPU usage:"
echo "   docker logs frigate | grep -i npu"
echo ""
echo "💾 Downloaded drivers saved in: $DOWNLOAD_DIR"
echo "⚠️  NOTE: These container changes are temporary and will be lost on container recreation."
echo "   Re-run this script after any Frigate updates."

# Clean up download directory if desired
read -p "🗑️  Delete downloaded drivers? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$DOWNLOAD_DIR"
    echo "✅ Cleaned up downloaded files"
else
    echo "📁 Drivers saved in $DOWNLOAD_DIR for future use"
fi


