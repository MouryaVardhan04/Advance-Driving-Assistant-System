## Running this project on Raspberry Pi 5 (64-bit)

This document explains the recommended, low-risk steps to prepare this repository for running on a Raspberry Pi 5 (aarch64). It includes a setup script, a safe archival/prune helper (does not permanently delete), and notes about PyTorch/OpenCV on ARM.

Assumptions
- You are running a 64-bit Raspberry Pi OS (Bullseye/Bookworm 64-bit) or equivalent aarch64 distribution.
- You have sudo access.

High-level steps
1. Inspect what you need to keep (models, configs, scripts). See the `scan` section below.
2. Run the installer: `scripts/setup_rpi.sh` (this creates a `.venv` and installs Python deps).
3. Optionally run `scripts/prune_repo_for_rpi.sh --apply` to move large/unneeded folders into `archive_unneeded_rpi/` (dry-run by default).
4. Run the project with `scripts/run_on_rpi.sh`.

Important notes and tips
- OpenCV: Pre-built `pip` wheels for OpenCV on aarch64 may not always be available. The setup script installs system OpenCV dependencies via `apt` and then tries to `pip install` packages. If `pip install opencv-python` fails, the system OpenCV libraries will allow importing `cv2` if installed via `apt` + `python3-opencv`.
- PyTorch: Official PyTorch wheels for Raspberry Pi/ARM are not always available on PyPI. The setup script attempts to install a PyTorch aarch64 wheel from the PyTorch download index; if that fails it will skip and note how to install manually or use a CPU-only alternative (e.g., TensorFlow Lite or converting models to TFLite).
- Models: Large model files (e.g., in `Models/`, `best.pt`) are left untouched. Use the prune script to archive them if you don't want them on the device.

Quick commands (on Pi, zsh/bash):
```bash
# update and install system deps + create venv and install pip deps
cd /path/to/repo
sudo ./scripts/setup_rpi.sh

# inspect what would be archived (dry-run)
./scripts/prune_repo_for_rpi.sh

# to actually move those folders to archive_unneeded_rpi/
./scripts/prune_repo_for_rpi.sh --apply

# run the project (activates .venv)
./scripts/run_on_rpi.sh
```

When something fails
- Read the output from the scripts. They try to be tolerant and will instruct manual steps if a critical wheel is not available.
- If PyTorch installation fails and you rely on it, see: https://pytorch.org/get-started/locally/ or search for community built aarch64 wheels for your OS image.

Next steps we can help with
- Convert heavy PyTorch models to TFLite or ONNX for faster CPU inference on Pi 5.
- Add a small test script (`scripts/check_camera_and_cv.py`) to confirm camera and cv2 functionality.

---
Created by automated repo prep tooling. Review scripts in `scripts/` before executing on any machine.
