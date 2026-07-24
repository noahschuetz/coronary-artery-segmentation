"""
Segmentation training script.

Reads a YAML config, builds data loaders, model, loss, and optimizer, then
runs training with periodic validation and best-checkpoint saving (by Dice).
After training, reloads the best model and evaluates on the test split if one
was returned by get_data_loaders().

Validation schedule is controlled by config keys:
    training.validation.start_epoch  (default: 1)
    training.validation.interval     (default: 1)
    training.validation.run_last     (default: True)

Outputs written to results_dir/:
    best_model.pt        — checkpoint dict (weights + optimizer + epoch)
    best_model.pth       — raw model weights
    config_snapshot.yaml — copy of the config used for this run
    test_metrics.yaml    — final test set results (if test split exists)
    logs/                — TensorBoard event files

Usage:
    python src/train.py --config configs/att_mamba2_unet.yaml
"""

import os
import yaml
import argparse
import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

# Suppress known deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorch 2.6+ defaults to weights_only=True, which breaks MONAI's
# PersistentDataset cache (it pickles MetaTensor objects). Setting this
# env var restores the old default globally for MONAI compatibility.
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

# MONAI Imports
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU

# Project Imports
from src.data.dataset import get_data_loaders
from src.models.model_factory import get_model


def build_metrics_dict(include_cldice=False):
    """Create a fresh metrics dict (MONAI metrics keep internal state).

    Args:
        include_cldice: If True, include clDice metric (slow, uses CPU skeletonization).
                        Recommended only for final test evaluation.
    """
    d = {
        "dice": DiceMetric(include_background=False, reduction="mean_batch"),
        "iou": MeanIoU(include_background=False, reduction="mean_batch"),
        "precision": ConfusionMatrixMetric(
            metric_name="precision",
            include_background=False,
            reduction="mean_batch",
        ),
        "recall": ConfusionMatrixMetric(
            metric_name="sensitivity",  # Recall = Sensitivity
            include_background=False,
            reduction="mean_batch",
        ),
        "f1": ConfusionMatrixMetric(
            metric_name="f1 score",
            include_background=False,
            reduction="mean_batch",
        ),
        "accuracy": ConfusionMatrixMetric(
            metric_name="accuracy",
            include_background=False,
            reduction="mean_batch",
        ),
    }
    if include_cldice:
        from src.metrics.cl_dice import ClDiceMetric
        d["cldice"] = ClDiceMetric(include_background=False)
    return d


def save_best_checkpoint(save_path, model, optimizer, epoch, best_metric):
    """Save a robust checkpoint dict (recommended)."""
    ckpt = {
        "epoch": epoch,
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, save_path)


def load_model_weights(model, checkpoint_path, device):
    """
    Loads either:
    - a pure state_dict (model weights), OR
    - a checkpoint dict containing 'model_state_dict'
    """
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)
    return model


def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """Runs one training epoch with mixed precision."""
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(loader, desc="Training", unit="batch")
    for batch_data in progress_bar:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()

        # Clip gradients to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (progress_bar.n + 1)})

    return running_loss / max(1, len(loader))


def evaluate_epoch(model, loader, loss_fn, device, patch_size, metrics_dict, phase_name="Validating"):
    """
    Runs one evaluation epoch (validation or test) using sliding window inference.
    Returns (loss, metrics_dict_as_floats)
    """
    model.eval()

    # Reset all metrics
    for metric in metrics_dict.values():
        metric.reset()

    running_loss = 0.0
    progress_bar = tqdm(loader, desc=phase_name, unit="scan")

    with torch.no_grad():
        for data in progress_bar:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            with autocast("cuda", dtype=torch.float16):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=patch_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
                loss = loss_fn(outputs, labels)

            # One-hot for metrics
            pred_classes = outputs.argmax(dim=1, keepdim=True)
            outputs_post = torch.zeros_like(outputs)
            outputs_post.scatter_(1, pred_classes, 1.0)

            labels_post = torch.zeros(
                (labels.shape[0], 2) + labels.shape[2:],
                dtype=labels.dtype,
                device=labels.device,
                )
            labels_post[:, 0] = (labels[:, 0] == 0).float()  # background
            labels_post[:, 1] = (labels[:, 0] == 1).float()  # foreground

            for metric in metrics_dict.values():
                metric(y_pred=outputs_post, y=labels_post)

            running_loss += loss.item()

            # Progress dice (best-effort)
            current_dice = 0.0
            try:
                dice_buf = metrics_dict["dice"].get_buffer()
                if len(dice_buf) > 0:
                    current_dice = dice_buf[-1].mean().item()
            except Exception:
                current_dice = 0.0

            progress_bar.set_postfix(
                {
                    f"{phase_name.lower()}_loss": running_loss / (progress_bar.n + 1),
                    f"{phase_name.lower()}_dice": f"{current_dice:.4f}",
                }
            )

    avg_loss = running_loss / max(1, len(loader))

    # Aggregate all metrics
    results = {}
    for name, metric in metrics_dict.items():
        metric_result = metric.aggregate()

        # MONAI metrics sometimes return list; normalize to tensor
        if isinstance(metric_result, list):
            if len(metric_result) == 0:
                results[name] = 0.0
                continue
            if isinstance(metric_result[0], torch.Tensor):
                metric_result = torch.stack(metric_result)
            else:
                metric_result = torch.tensor(metric_result)

        if hasattr(metric_result, "numel") and metric_result.numel() > 0:
            results[name] = metric_result.mean().item()
        else:
            results[name] = 0.0

    return avg_loss, results


def is_validation_epoch(epoch_num, num_epochs, start_epoch, interval, run_last):
    """
    epoch_num: 1-indexed epoch number
    start_epoch: first epoch to validate (1-indexed)
    interval: validate every N epochs after start_epoch
    run_last: if True, validate on the final epoch regardless of schedule
    """
    if epoch_num < start_epoch:
        return False

    if interval is None or int(interval) <= 0:
        # degenerate case: only validate at start_epoch (and maybe last)
        return (epoch_num == start_epoch) or (run_last and epoch_num == num_epochs)

    if (epoch_num - start_epoch) % int(interval) == 0:
        return True

    if run_last and epoch_num == num_epochs:
        return True

    return False


def main():
    # --- 1. Parse Arguments ---
    parser = argparse.ArgumentParser(description="3D Segmentation Training Engine")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file.",
    )
    args = parser.parse_args()

    # --- 2. Load Configuration ---
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        return

    # --- 3. Setup Environment ---
    print(f"--- Starting Experiment: {config['experiment_name']} ---")

    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Save config snapshot for reproducibility
    with open(os.path.join(results_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs"))

    # --- 4. Load Data ---
    print("Loading data...")
    loaders = get_data_loaders(
        data_config=config["data"],
        training_config=config["training"],
    )

    if isinstance(loaders, (list, tuple)) and len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    elif isinstance(loaders, (list, tuple)) and len(loaders) == 2:
        train_loader, val_loader = loaders
        test_loader = None
    else:
        raise RuntimeError(
            "get_data_loaders(...) must return (train_loader, val_loader) or (train_loader, val_loader, test_loader)."
        )

    if test_loader is None:
        print("NOTE: test_loader was not returned by get_data_loaders(...). Test evaluation will be skipped.")
    else:
        print("Test split detected: test_loader is available. Final test evaluation will run after training.")

    # --- 5. Build Model ---
    print("Building model...")
    model = get_model(config["model"], config["data"]).to(device)

    # --- 6. Setup Training Components ---
    loss_cfg = config["training"].get("loss", {})
    loss_type = loss_cfg.get("type", "dice_ce")

    if loss_type == "dice_ce_cldice":
        from src.losses.combined_loss import DiceCECLDiceLoss
        loss_function = DiceCECLDiceLoss(
            lambda_dice_ce=float(loss_cfg.get("lambda_dice_ce", 0.7)),
            lambda_cldice=float(loss_cfg.get("lambda_cldice", 0.3)),
            cldice_iter=int(loss_cfg.get("cldice_iter", 3)),
        )
        print(f"Loss: DiceCE+clDice (dice_ce={loss_cfg.get('lambda_dice_ce', 0.7)}, cldice={loss_cfg.get('lambda_cldice', 0.3)})")
    else:
        loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.8,
            lambda_ce=0.2,
        )
        print("Loss: DiceCE")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        cooldown=2,
    )

    scaler = GradScaler("cuda")

    val_cfg = config["training"].get("validation", {}) if isinstance(config.get("training", {}), dict) else {}
    start_epoch = int(val_cfg.get("start_epoch", 1))
    interval = int(val_cfg.get("interval", 1))
    run_last = bool(val_cfg.get("run_last", True))

    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    base_lr = float(config["training"]["learning_rate"])

    # --- 7. Training Loop ---
    print("Starting training...")
    num_epochs = int(config["training"]["num_epochs"])
    patch_size = tuple(config["data"]["patch_size"])

    print(
        f"Validation schedule: start_epoch={start_epoch}, interval={interval}, run_last={run_last}\n"
        f"  -> will validate at epochs: {', '.join(str(e) for e in range(1, num_epochs+1) if is_validation_epoch(e, num_epochs, start_epoch, interval, run_last))}"
    )

    best_metric = -1.0
    best_epoch = -1

    best_ckpt_path = os.path.join(results_dir, "best_model.pt")   # checkpoint dict
    best_weights_path = os.path.join(results_dir, "best_model.pth")  # raw weights (optional)

    for epoch in range(num_epochs):
        epoch_num = epoch + 1  # 1-indexed
        print(f"\n--- Epoch {epoch_num}/{num_epochs} ---")

        # Linear LR warmup: scale from base_lr/warmup_epochs to base_lr
        if warmup_epochs > 0 and epoch_num <= warmup_epochs:
            warmup_lr = base_lr * (epoch_num / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss = train_epoch(model, train_loader, optimizer, loss_function, scaler, device)

        # Log train always
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        do_val = is_validation_epoch(epoch_num, num_epochs, start_epoch, interval, run_last)

        if do_val:
            # fresh metrics each validation run
            val_metrics_dict = build_metrics_dict()
            val_loss, val_metrics = evaluate_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_function,
                device=device,
                patch_size=patch_size,
                metrics_dict=val_metrics_dict,
                phase_name="Validating",
            )

            # Only step the plateau scheduler after warmup is complete
            if epoch_num > warmup_epochs:
                scheduler.step(val_loss)

            # Log validation only on validation epochs
            writer.add_scalar("Loss/validation", val_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"Metrics/val_{metric_name}", metric_value, epoch)

            print(f"Epoch {epoch_num} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {val_loss:.4f}")
            print("  Validation Metrics:")
            print(f"    Dice:      {val_metrics['dice']:.4f}")
            print(f"    IoU:       {val_metrics['iou']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall:    {val_metrics['recall']:.4f}")
            print(f"    F1 Score:  {val_metrics['f1']:.4f}")
            print(f"    Accuracy:  {val_metrics['accuracy']:.4f}")

            # Save best model checkpoint based on Dice score
            if val_metrics["dice"] > best_metric:
                best_metric = float(val_metrics["dice"])
                best_epoch = epoch

                # 1) Save robust checkpoint dict
                save_best_checkpoint(best_ckpt_path, model, optimizer, epoch, best_metric)

                # 2) Also save raw weights (optional convenience)
                torch.save(model.state_dict(), best_weights_path)

                print(
                    f"  New best model saved — Dice: {best_metric:.4f} (epoch {best_epoch + 1})\n"
                    f"    checkpoint: {best_ckpt_path}\n"
                    f"    weights:    {best_weights_path}"
                )
        else:
            # No validation this epoch
            print(f"Epoch {epoch_num} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(
                f"  Validation: SKIPPED (scheduled start={start_epoch}, interval={interval}, run_last={run_last})"
            )

    # --- Training done ---
    writer.close()
    print("\n--- Training Complete ---")

    if best_epoch == -1:
        print("WARNING: No validation was run, so no best checkpoint was saved.")
        return

    print(f"Best validation Dice: {best_metric:.4f} at epoch {best_epoch + 1}")

    # --- 8. Load best model + evaluate on TEST ---
    if test_loader is not None:
        if not os.path.exists(best_ckpt_path) and not os.path.exists(best_weights_path):
            print("ERROR: No best model file found. Skipping test evaluation.")
            return

        # Load best model
        print("\nLoading best model for final TEST evaluation...")
        load_path = best_ckpt_path if os.path.exists(best_ckpt_path) else best_weights_path

        model = load_model_weights(model, load_path, device)
        model.to(device)

        # Evaluate on test
        test_metrics_dict = build_metrics_dict(include_cldice=True)
        test_loss, test_metrics = evaluate_epoch(
            model=model,
            loader=test_loader,
            loss_fn=loss_function,
            device=device,
            patch_size=patch_size,
            metrics_dict=test_metrics_dict,
            phase_name="Testing",
        )

        print("\n--- Final TEST Results (Best-Val Model) ---")
        print(f"  Test Loss: {test_loss:.4f}")
        print("  Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"    {metric_name:12s}: {metric_value:.4f}")

        # Save test metrics to disk
        out_path = os.path.join(results_dir, "test_metrics.yaml")
        with open(out_path, "w") as f:
            yaml.dump(
                {
                    "best_val_epoch": int(best_epoch + 1),
                    "best_val_dice": float(best_metric),
                    "test_loss": float(test_loss),
                    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                    "loaded_from": str(load_path),
                },
                f,
                default_flow_style=False,
            )
        print(f"\nSaved test metrics to: {out_path}")


if __name__ == "__main__":
    main()
