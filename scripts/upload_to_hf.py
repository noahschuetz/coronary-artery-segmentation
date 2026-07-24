"""
Upload trained checkpoints + model card to the Hugging Face Hub.

Prereqs:
    pip install huggingface_hub
    huggingface-cli login          # or set HF_TOKEN

Usage:
    python scripts/upload_to_hf.py --repo_id <username>/coronary-segmentation
"""

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo

CHECKPOINTS = {
    "baseline_unet.pth": "results/unet_baseline/best_model.pth",
    "att_mamba2_unet.pth": "results/att_mamba2_unet_baseline/best_model.pth",
}
MODEL_CARD = "docs/MODEL_CARD.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. noahschuetz/coronary-segmentation")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # HF renders README.md as the model card
        shutil.copy(MODEL_CARD, os.path.join(tmp, "README.md"))
        for dst, src in CHECKPOINTS.items():
            if os.path.exists(src):
                shutil.copy(src, os.path.join(tmp, dst))
            else:
                print(f"WARNING: missing checkpoint {src}, skipping")

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=tmp,
            commit_message="Add coronary segmentation checkpoints and model card",
        )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
