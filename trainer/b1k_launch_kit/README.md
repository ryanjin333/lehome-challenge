# BEHAVIOR-1K GR00T N1.7 training launch kit

This kit prepares a fresh disposable one- or two-GPU Vast.ai training container, begins the full 100-task run without another prompt, uploads a complete verified run bundle, and destroys the instance. It downloads exactly 200 human demonstrations for each of the 100 tasks, with the head and two wrist RGB streams, actions, proprioception, metadata, and annotations. It does not download depth or install the rollout simulator.

## Do these two things before renting

1. Sign into Hugging Face, accept access to [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B), and create a read-only Hugging Face token. Save it as an encrypted `HF_TOKEN` under Vast Account Settings > Environment Variables. The private template persists the injected value to `${HF_HOME}/token` with root-only permissions during startup, and the bootstrap checks this gate before beginning the approximately 1.08 TB dataset transfer.
2. Confirm that the token can write to the private `ryanjin333/lehome-groot-n17-models` repository. The disposable path uploads the final run there and destroys the instance only after an immutable download/readback verifies every artifact. Cloudflare R2 is optional for continuous intermediate-checkpoint backups and the current parent-cycle restore helper.

Never paste tokens into chat or commit them to this directory. Inject them through Vast's environment-variable controls or create a private `.env` directly on the instance.

## Vast selection

Use the requirements in [`config/vast-requirements.json`](config/vast-requirements.json). The important fields are:

- Docker/container instance, not the Ubuntu VM/KVM template
- Image: the immutable GHCR tag in [`config/vast-requirements.json`](config/vast-requirements.json); never use `latest`
- 1× RTX PRO 6000 96 GB for the budget run, or 2× for twice the samples per step
- At least 2 TB disk with at least 1.5 TB free
- At least 128 GB RAM and 24 CPU cores
- `↓1000 Mbps` minimum, preferably `↓2000+ Mbps`
- `↑500 Mbps` minimum
- Direct SSH and a verified datacenter host

The official GR00T N1.7 dGPU stack uses CUDA 12.8 and Python 3.10. The bootstrap creates the pinned Python environment from the challenge fork's lockfile.

## Fastest disposable startup

The fastest path is the prebuilt image in [`docker/training.Dockerfile`](docker/training.Dockerfile). It bakes the transfer tools, `uv`, the GR00T checkout, and the frozen Python environment into the image. Vast does not bill while the image is in the `Loading` state; after launch, the only large unavoidable preparation is the filtered human dataset and gated model download.

Build and publish the image once to your container registry:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/training.Dockerfile \
  -t REGISTRY/b1k-groot-train:2026-08-01 \
  --push .
```

Use that image in the private Vast template. Put this in the template's on-start field:

```bash
mkdir -p /workspace/logs/b1k
nohup /opt/b1k-launch-kit/bin/run_disposable_training.sh \
  >/workspace/disposable-training.log 2>&1 &
```

Inject the variables from [`.env.example`](.env.example) through Vast account/template environment variables. At minimum the initial disposable path requires `HF_TOKEN`, `MODEL_REPOSITORY=ryanjin333/lehome-groot-n17-models`, `CYCLE_ID`, and `AUTO_DESTROY=1`.

For `cycle-001` and later, also set `PARENT_CYCLE_ID` to the preceding successful cycle. The current parent-cycle restore helper additionally requires the optional R2 configuration. The bootstrap downloads the immutable Hugging Face dataset again, restores the prior verified checkpoint and exact normalization/modality metadata, trains a fresh 15,000-step cycle, uploads the new run, verifies it, and destroys itself.

If a registry image is not available, [`bin/vast_onstart.sh`](bin/vast_onstart.sh) can download this small launch archive from a checksummed `B1K_KIT_URL`; that fallback still installs GR00T dependencies on every machine and is slower.

## Manual fallback

Upload this small kit to the instance. From the local directory containing it:

```bash
scp -P VAST_SSH_PORT -r . root@VAST_IP:/workspace/b1k-launch-kit
```

If `HF_TOKEN` was saved as a Vast account environment variable and the private template's on-start hook ran, no token command is needed over SSH. Otherwise, provide it for the current shell at minimum:

```bash
export HF_TOKEN='your-read-token'
```

For the initial cycle, no R2 variables are required. Then start the disposable pipeline inside `tmux`:

```bash
cd /workspace/b1k-launch-kit
tmux new-session -d -s b1k \
  './bin/run_disposable_training.sh 2>&1 | tee /workspace/disposable-training.log'
tmux attach -t b1k
```

Disconnecting SSH does not stop the job. Reattach later with `tmux attach -t b1k`.

The disposable runner performs these operations automatically:

1. Verifies one or two GPUs, 1.5 TB free disk, R2 configuration, and gated-model access.
2. Installs only the small system tools required for training and transfers.
3. Starts the RGB dataset download, GR00T environment creation, and model pre-cache concurrently.
4. Resumes partial Hugging Face downloads after any interruption.
5. Validates all 100 chunks, exactly 200 demonstrations per chunk, and all three RGB streams.
6. Deploys the official R1Pro modality configuration.
7. Restores a verified parent checkpoint and exact metadata when `PARENT_CYCLE_ID` is set.
8. Starts one- or two-GPU training immediately.
9. Archives the final checkpoint, normalization, modality, provenance and logs.
10. Uploads the run bundle, runs `rclone check`, writes the verified marker, and destroys the Vast instance.

At `↓3117 Mbps`, the 1.08 TB filtered download has a 46-minute theoretical floor and will more realistically take roughly 1–2 hours. At `↓1000 Mbps`, expect roughly 3–5 hours.

## Training defaults

The launcher uses the official challenge GR00T N1.7 pipeline:

- `nvidia/GR00T-N1.7-3B` with the gated Cosmos backbone
- One or two GPUs through `torchrun` (detected automatically)
- 15,000 optimizer steps for the initial warm-up (`MAX_STEPS` can override this for later cycles)
- Checkpoint every 1,500 steps; five retained locally
- `--decode-only-used-frames`
- W&B offline by default
- One GPU batch candidates `256 → 128 → 64`; two GPUs `512 → 256 → 128`

One-GPU training starts at global batch 256; two-GPU training starts at 512. If and only if a recognized CUDA out-of-memory error occurs, the launcher retries the smaller candidates. Non-memory errors stop the run instead of silently changing the experiment.

Logs are written under `/workspace/logs/b1k`:

```text
dataset-download.log
groot-setup.log
model-download.log
dataset-validation.json
train-b1k-all100-gbs*.log
checkpoint-watcher.log
```

## Training-to-rollout handoff

The disposable runner uploads only the final checkpoint by default, avoiding repeated transfer of every intermediate checkpoint. Set `ENABLE_CHECKPOINT_WATCHER=1` only when intermediate remote recovery points are worth the extra upload time and storage. The human dataset is never uploaded; every fresh training VM pulls the same immutable RGB-only source from Hugging Face.

The final `runs/<cycle>/` bundle contains the standalone checkpoint, `meta/stats.json`, `meta/modality.json`, dataset and GR00T revisions, logs, a run manifest and SHA-256 checksums. Destruction is fail-closed: missing metadata, upload failure, failed `rclone check`, a missing verified marker, missing `AUTO_DESTROY=1`, or an invalid `CONTAINER_ID` leaves the instance intact for recovery.

To force a particular checkpoint upload:

```bash
./bin/push_artifacts.sh \
  --checkpoint /workspace/outputs/b1k-all100-gbs512/checkpoint-1500 \
  --cycle cycle-000
```

On the rollout machine, install `rclone`, provide the same read credentials, and download only the checkpoint:

```bash
./bin/pull_checkpoint.sh \
  --cycle cycle-000/b1k-all100-gbs512 \
  --destination /workspace/checkpoints/cycle-000
```

The pull script verifies `SHA256SUMS` before reporting the checkpoint ready. Do not destroy the training container until at least one checkpoint has been verified in R2.

## Non-mutating preflight

Before renting, the local kit can be checked without downloading or installing anything:

```bash
HF_TOKEN=placeholder ./bin/bootstrap_training.sh --dry-run
R2_REMOTE=r2 R2_BUCKET=b1k AUTO_DESTROY=1 \
  ./bin/run_disposable_training.sh --dry-run
R2_REMOTE=r2 R2_BUCKET=b1k ./bin/push_artifacts.sh \
  --checkpoint /example/checkpoint-1500 --cycle cycle-000 --dry-run
```

Run the local automated verification with:

```bash
pytest -q
bash -n bin/*.sh
python3 -m compileall -q bin
```
