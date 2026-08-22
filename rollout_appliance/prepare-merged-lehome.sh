#!/usr/bin/env bash
set -euo pipefail
# Recreate official+flywheel merged package on the rollout boot disk.
sudo mkdir -p /opt/lehome/merged
sudo docker run --rm -v /opt/lehome/merged:/out --entrypoint bash lehome-rollout:build -lc '
  rm -rf /out/lehome
  mkdir -p /out
  cp -a /opt/lehome-challenge/source/lehome/lehome /out/lehome
'
sudo mkdir -p /opt/lehome/merged/lehome/flywheel
sudo cp -a /opt/lehome/source/lehome/lehome/flywheel/. /opt/lehome/merged/lehome/flywheel/
sudo mkdir -p /opt/lehome/merged/lehome/assets/object
sudo cp -a /opt/lehome/source/lehome/lehome/assets/object/Garment.py /opt/lehome/merged/lehome/assets/object/Garment.py
sudo cp -a /opt/lehome/source/lehome/lehome/assets/collider_audit.py /opt/lehome/merged/lehome/assets/collider_audit.py
sudo mkdir -p /opt/lehome/merged/lehome/assets/scenes
sudo cp -a /opt/lehome/source/lehome/lehome/assets/scenes/bedroom.py /opt/lehome/merged/lehome/assets/scenes/bedroom.py
sudo cp -a /opt/lehome/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py /opt/lehome/merged/lehome/tasks/bedroom/garment_bi_v2.py
sudo cp -a /opt/lehome/source/lehome/lehome/devices/__init__.py /opt/lehome/merged/lehome/devices/__init__.py
if [ -f /opt/lehome/source/lehome/lehome/devices/action_process.py ]; then
  sudo cp -a /opt/lehome/source/lehome/lehome/devices/action_process.py /opt/lehome/merged/lehome/devices/action_process.py
fi
sudo find /opt/lehome/merged -name "__pycache__" -type d -exec rm -rf {} + || true
sudo chmod -R a+rX /opt/lehome/merged
echo MERGED_OK
