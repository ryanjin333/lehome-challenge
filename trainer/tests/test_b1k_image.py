from pathlib import Path


TRAINER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINER_ROOT.parent
DOCKERFILE = TRAINER_ROOT / "Dockerfile"
B1K_KIT = TRAINER_ROOT / "b1k_launch_kit"
VERIFY_IMAGE = TRAINER_ROOT / "scripts" / "verify-image.sh"


def test_image_pins_behavior_fork_and_bakes_required_entrypoints() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "https://github.com/wensi-ai/Isaac-GR00T.git" in dockerfile
    assert "ace36d935b376fbf25cd56371e23877b95407c40" in dockerfile
    assert "scripts/b1k/train_b1k.py" in dockerfile
    assert "scripts/b1k/deploy_modality.py" in dockerfile
    assert "examples/b1k/r1pro.py" in dockerfile
    assert "COPY trainer/b1k_launch_kit /opt/b1k-launch-kit" in dockerfile
    dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "!trainer/b1k_launch_kit/**" in dockerignore
    verifier = VERIFY_IMAGE.read_text(encoding="utf-8")
    assert "ace36d935b376fbf25cd56371e23877b95407c40" in verifier
    assert "scripts/b1k/train_b1k.py" in verifier


def test_baked_launch_kit_is_complete_and_uses_the_image_runtime() -> None:
    bootstrap = (B1K_KIT / "bin" / "bootstrap_training.sh").read_text(encoding="utf-8")
    start = (B1K_KIT / "bin" / "start_training.sh").read_text(encoding="utf-8")

    assert (B1K_KIT / "bin" / "run_disposable_training.sh").is_file()
    assert (B1K_KIT / "bin" / "push_run_bundle.sh").is_file()
    assert (B1K_KIT / "bin" / "destroy_instance.sh").is_file()
    assert (B1K_KIT / "config" / "dataset-manifest.json").is_file()
    assert (B1K_KIT / ".env.example").is_file()
    assert "/opt/runtime/bin/python" in bootstrap
    assert "scripts/b1k/train_b1k.py" in bootstrap
    assert '"${GROOT_PYTHON}" -m torch.distributed.run' in start
    push = (B1K_KIT / "bin" / "push_run_bundle.sh").read_text(encoding="utf-8")
    assert "lehome-train" in push
    assert "remotely_verified" in push
    assert "R2_BUCKET is required" not in push
