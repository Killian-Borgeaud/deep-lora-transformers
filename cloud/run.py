"""Run any script in scripts/ on a Modal GPU, exactly as `python scripts/<name>.py` would.

From the repo root:

    uv run --group modal modal run cloud/run.py --script intro
    uv run --group modal modal run cloud/run.py --script stsb_save_checkpoint --gpu A100-80GB

Checkpoints and figures are written to the `dlt-checkpoints` Volume. finetune() skips
experiments that already exist there, so an interrupted script resumes where it stopped.
Download the results with:

    uv run --group modal modal volume get dlt-checkpoints / ./checkpoints
"""

import modal

app = modal.App("deep-lora")

# Paths are relative to the repo root, so `modal run` must be called from there
image = (
    modal.Image.debian_slim()
    .uv_sync()
    .add_local_dir("dlt", remote_path="/root/dlt")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

checkpoints = modal.Volume.from_name("dlt-checkpoints", create_if_missing=True)
# Cache Hugging Face models and datasets across runs instead of downloading each time
hf_cache = modal.Volume.from_name("dlt-hf-cache", create_if_missing=True)

CHECKPOINTS_DIR = "/root/checkpoints"


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 60 * 60,  # Modal's maximum
    volumes={CHECKPOINTS_DIR: checkpoints, "/root/.cache/huggingface": hf_cache},
)
def run(script: str):
    import os
    import runpy

    # Scripts save plots to ./figures; point it into the Volume so they persist
    os.makedirs(f"{CHECKPOINTS_DIR}/figures", exist_ok=True)
    if not os.path.exists("figures"):
        os.symlink(f"{CHECKPOINTS_DIR}/figures", "figures")

    try:
        runpy.run_module(f"scripts.{script}", run_name="__main__")
    finally:
        # Persist whatever finished, even if the script crashed
        checkpoints.commit()
        hf_cache.commit()


@app.local_entrypoint()
def main(script: str = "intro", gpu: str = "A100", matmul_precision: str = ""):
    # Full-width runs (rank=None) on all-dense need more than 40GB, use --gpu A100-80GB.
    # On Ampere+ GPUs JAX uses TF32 for float32 matmuls by default; pass
    # --matmul-precision highest to get full float32, as on the paper's V100.
    env = {"JAX_DEFAULT_MATMUL_PRECISION": matmul_precision} if matmul_precision else {}
    run.with_options(gpu=gpu, env=env).remote(script)
