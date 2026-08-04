# Manual Installation Guide

This guide provides step-by-step instructions for manually installing the LeHome Challenge environment.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- GPU driver and CUDA supporting IsaacSim5.1.0.

### Isaac Sim 5.1 rollout-host gate

Before downloading a policy or starting Isaac on a paid rollout host, run:

```bash
python scripts/check_isaac_runtime.py
```

It prints a deterministic JSON receipt and exits nonzero unless the host is
Linux `x86_64` with a consistent NVIDIA R580 driver at least `580.65.06` and
below `590.0.0`. This is intentionally narrower than “a newer driver”: R590,
including `595.71.05`, is not reviewed for this Isaac Sim 5.1 rollout runtime.
Switch hosts rather than trying to upgrade or downgrade the injected Vast
driver from inside an unprivileged container. Make this command the first
startup action in the normalized image/template, before model downloads.

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### 2. Install Dependencies with uv

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 3. Clone and Configure IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

### 4. Install IsaacLab

Activate the virtual environment and install IsaacLab:

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

### 5. Install LeHome Package

Finally, install the LeHome package in development mode:

```bash
uv pip install -e ./source/lehome
```

---
###
If you are using a server, please download the system dependencies.

```bash
    #step 1
    apt update
    apt install -y \
    libglu1-mesa \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    libx11-6
    #step 2
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
```


## Next Steps

Now that you have installed the environment, you can:

- [Prepare Assets and Data](datasets.md)
- [Start Training](training.md)
- [Evaluate Policies](policy_eval.md)
- [Back to README](../README.md)
