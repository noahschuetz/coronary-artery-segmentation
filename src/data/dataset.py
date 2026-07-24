"""
Defines the functions for finding, splitting, and loading data.
Creates and returns the PyTorch DataLoaders for train/val/test.
"""

import os
import glob
from sklearn.model_selection import train_test_split
from monai.data import PersistentDataset, DataLoader
from src.data.transforms import get_train_transforms, get_val_transforms


def _get_data_files(data_dir: str):
    """
    Scans the data directory for image/label pairs.
    Supports multiple formats:
    - Direct files: *.img.nii.gz / *.label.nii.gz or *.img.nii / *.label.nii
    - Directory structure: <id>.img.nii/ containing dia_0.nii or diao_0.nii
                          <id>.label.nii/ containing label.nii
    """
    data_files = []

    # Check for directory-based structure first (e.g., 1.img.nii/ folders)
    img_dirs = sorted(glob.glob(os.path.join(data_dir, "*.img.nii")))
    if img_dirs and os.path.isdir(img_dirs[0]):
        print(f"Found {len(img_dirs)} image directories.")
        for img_dir in img_dirs:
            base_name = os.path.basename(img_dir)
            patient_id = base_name.replace(".img.nii", "")

            # Find the actual NIfTI file inside the image directory
            img_file = None
            for candidate in ["dia_0.nii", "diao_0.nii"]:
                candidate_path = os.path.join(img_dir, candidate)
                if os.path.isfile(candidate_path):
                    img_file = candidate_path
                    break

            if img_file is None:
                print(f"Warning: No valid image file in {img_dir}. Skipping.")
                continue

            # Find the label file
            label_dir = os.path.join(data_dir, f"{patient_id}.label.nii")
            label_file = os.path.join(label_dir, "label.nii")

            if os.path.isfile(label_file):
                data_files.append({
                    "image": img_file,
                    "label": label_file
                })
            else:
                print(f"Warning: Missing label for patient {patient_id}. Skipping.")

        print(f"Successfully paired {len(data_files)} image/label sets.")
        return data_files

    # Fallback: direct file structure (*.img.nii.gz or *.img.nii files)
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.img.nii.gz")))
    ext = ".img.nii.gz"
    label_ext = ".label.nii.gz"

    if not image_files:
        image_files = sorted(glob.glob(os.path.join(data_dir, "*.img.nii")))
        ext = ".img.nii"
        label_ext = ".label.nii"

    if not image_files:
        raise FileNotFoundError(f"No image files or directories found in {data_dir}")

    print(f"Found {len(image_files)} potential images.")

    for img_path in image_files:
        base_name = os.path.basename(img_path)
        patient_id = base_name.replace(ext, "")
        label_path = os.path.join(data_dir, f"{patient_id}{label_ext}")

        if os.path.exists(label_path):
            data_files.append({
                "image": img_path,
                "label": label_path
            })
        else:
            print(f"Warning: Missing label for patient {patient_id}. Skipping.")

    print(f"Successfully paired {len(data_files)} image/label sets.")
    return data_files


def get_data_loaders(data_config: dict, training_config: dict):
    """
    Creates and returns the training, validation, and test data loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader or None)
    """
    data_files = _get_data_files(data_dir=data_config["data_dir"])
    if not data_files:
        raise ValueError("No data files found. Check 'data_dir' in your config.")

    seed = int(data_config.get("seed", 42))
    val_split = float(data_config.get("val_split", 0.2))
    test_split = float(data_config.get("test_split", 0.1))

    if val_split < 0 or test_split < 0 or (val_split + test_split) >= 1.0:
        raise ValueError("Invalid split values. Ensure val_split + test_split < 1.0")

    # 1) Train vs (Val+Test)
    train_files, temp_files = train_test_split(
        data_files,
        test_size=(val_split + test_split),
        random_state=seed,
        shuffle=True,
    )

    # 2) Val vs Test (from temp)
    if test_split > 0 and len(temp_files) > 1:
        val_ratio_within_temp = val_split / (val_split + test_split)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1.0 - val_ratio_within_temp),
            random_state=seed,
            shuffle=True,
        )
    else:
        val_files, test_files = temp_files, []

    print(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    # Configure transforms
    preprocess_cfg = data_config.get("preprocess", {})
    augment_cfg = data_config.get("augment", {})

    # MONAI convention: DataLoader batch_size=1 (one full volume at a time),
    # while samples_per_volume controls how many random patches are cropped
    # from each volume by RandCropByPosNegLabeld.
    samples_per_volume = training_config.get(
        "samples_per_volume",
        training_config.get("batch_size", 4),  # backward compat
    )
    train_transforms = get_train_transforms(
        patch_size=data_config["patch_size"],
        num_samples=samples_per_volume,
        preprocess_config=preprocess_cfg,
        augment_config=augment_cfg,
    )
    val_transforms = get_val_transforms(
        preprocess_config=preprocess_cfg,
    )

    # cache dir — include a hash of preprocess config so different pipelines
    # (e.g. vesselness=True vs False) don't collide in PersistentDataset cache
    import hashlib, json
    cfg_hash = hashlib.md5(json.dumps(preprocess_cfg, sort_keys=True).encode()).hexdigest()[:8]
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(project_root, ".cache", cfg_hash)
    os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(train_files, transform=train_transforms, cache_dir=os.path.join(cache_dir, "train"))
    val_ds = PersistentDataset(val_files, transform=val_transforms, cache_dir=os.path.join(cache_dir, "val"))
    test_ds = (
        PersistentDataset(test_files, transform=val_transforms, cache_dir=os.path.join(cache_dir, "test"))
        if len(test_files) > 0
        else None
    )

    num_workers = int(data_config.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2))

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2))

    return train_loader, val_loader, test_loader
