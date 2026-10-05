import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


def is_quick_mode():
    return os.getenv("TAILTUNE_QUICK", "0") == "1"


def evaluate_loss(model, loader, device):
    """
    Evaluate average cross-entropy loss.

    Returns NaN if the loader contains no valid examples.
    """
    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for x, target in loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(x)[:, 1:]

            # Targets are 1-indexed item IDs.
            loss = F.cross_entropy(logits, target - 1)

            if not torch.isfinite(loss):
                continue

            batch_size = target.size(0)
            total += loss.item() * batch_size
            count += batch_size

    if count == 0:
        return float("nan")

    return total / count


def train_sasrec(
    model,
    loader,
    epochs,
    lr,
    weight_decay,
    device,
    checkpoint_path=None,
    val_loader=None,
    patience=3,
    max_batches=None,
):
    """
    Train SASRec.

    Quick mode:
        - limits the number of batches
        - skips validation
        - saves the finite smoke-test checkpoint

    Full mode:
        - evaluates validation loss
        - uses validation loss for early stopping
        - saves the best checkpoint
    """

    quick = is_quick_mode()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Mixed precision only on CUDA.
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # Quick mode is intended only to verify that the complete
    # training pipeline works end-to-end.
    if quick:
        epochs = min(int(epochs), 1)
        if max_batches is None:
            max_batches = 20

    best_val = float("inf")
    best_train = float("inf")
    bad_epochs = 0
    checkpoint_saved = False

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()

        running_loss = 0.0
        count = 0

        progress = tqdm(
            loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(progress):

            # Stop early in quick mode / smoke-test mode.
            if max_batches is not None and batch_idx >= max_batches:
                break

            x, target = batch

            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(x)[:, 1:]

                loss = F.cross_entropy(
                    logits,
                    target - 1,
                )

            # Safety check.
            if not torch.isfinite(loss):
                print(
                    f"\nWARNING: non-finite loss at batch "
                    f"{batch_idx + 1}; skipping batch."
                )
                continue

            if use_amp:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = target.size(0)

            running_loss += loss.item() * batch_size
            count += batch_size

            elapsed = time.time() - start_time

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                time=f"{elapsed:.0f}s",
            )

        train_loss = running_loss / max(count, 1)

        # -----------------------------------------------------
        # QUICK MODE
        # -----------------------------------------------------
        if quick:
            elapsed = time.time() - start_time

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"val_loss=SKIPPED "
                f"time={elapsed:.1f}s"
            )

            # For a smoke test, the important thing is that the
            # model trained successfully and produced finite loss.
            if math.isfinite(train_loss):
                best_train = train_loss

                if checkpoint_path:
                    torch.save(
                        model.state_dict(),
                        checkpoint_path,
                    )
                    checkpoint_saved = True

            continue

        # -----------------------------------------------------
        # FULL TRAINING MODE
        # -----------------------------------------------------
        val_loss = float("nan")

        if val_loader is not None:
            val_loss = evaluate_loss(
                model,
                val_loader,
                device,
            )

        elapsed = time.time() - start_time

        if math.isfinite(val_loss):
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"time={elapsed:.1f}s"
            )
        else:
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"val_loss=NaN "
                f"time={elapsed:.1f}s"
            )

        # -----------------------------------------------------
        # Checkpoint using validation loss when valid.
        # -----------------------------------------------------
        if math.isfinite(val_loss):

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0

                if checkpoint_path:
                    torch.save(
                        model.state_dict(),
                        checkpoint_path,
                    )
                    checkpoint_saved = True

            else:
                bad_epochs += 1

                if bad_epochs >= patience:
                    print("Early stopping.")
                    break

        else:
            # Do NOT treat NaN as an improvement.
            #
            # If validation data is unavailable, keep training
            # rather than immediately triggering early stopping.
            print(
                "WARNING: validation loss is not finite. "
                "Skipping validation-based checkpointing."
            )

    # ---------------------------------------------------------
    # Final checkpoint fallback
    # ---------------------------------------------------------
    #
    # This protects against a situation where validation is NaN
    # for the entire run. We still want a usable model rather
    # than reporting a fake successful checkpoint.
    #
    if not checkpoint_saved and math.isfinite(best_train):
        if checkpoint_path:
            torch.save(
                model.state_dict(),
                checkpoint_path,
            )
            checkpoint_saved = True

            print(
                "Saved fallback checkpoint using finite "
                "training loss."
            )

    # ---------------------------------------------------------
    # Final reporting
    # -----------------------------------------------------
    if quick:
        print(f"\nBest training loss: {best_train:.4f}")
    elif math.isfinite(best_val):
        print(f"\nBest validation loss: {best_val:.4f}")
    else:
        print("\nBest validation loss: unavailable (NaN)")

    if checkpoint_saved and checkpoint_path:
        print(f"Saved checkpoint: {checkpoint_path}")
    else:
        print("WARNING: No checkpoint was saved.")

    return (
        best_train
        if quick
        else best_val
    )


def train_btsr(
    model,
    loader,
    epochs,
    lr,
    weight_decay,
    device,
    checkpoint_path=None,
    patience=3,
    max_batches=None,
):
    """
    Train the Barlow-Twins-regularized SASRec model.

    The supervised next-item objective is combined with
    the Barlow Twins representation loss.
    """

    quick = is_quick_mode()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    if quick:
        epochs = min(int(epochs), 1)

        if max_batches is None:
            max_batches = 20

    best = float("inf")
    bad = 0
    checkpoint_saved = False

    from src.models.barlow_twins import barlow_twins_loss

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()

        running = 0.0
        count = 0

        progress = tqdm(
            loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(progress):

            if max_batches is not None and batch_idx >= max_batches:
                break

            x1, x2, target = batch

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                z1 = model.encode(x1)

                logits = model.score_all(z1)[:, 1:]

                ce = F.cross_entropy(
                    logits,
                    target - 1,
                )

                z2 = model.encode(x2)

                bt = barlow_twins_loss(
                    z1,
                    z2,
                    model.lambda_offdiag,
                )

                loss = ce + model.alpha * bt

            if not torch.isfinite(loss):
                print(
                    f"\nWARNING: non-finite BT-SR loss at "
                    f"batch {batch_idx + 1}; skipping batch."
                )
                continue

            if use_amp:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            batch_size = target.size(0)

            running += loss.item() * batch_size
            count += batch_size

            elapsed = time.time() - start_time

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{ce.item():.4f}",
                bt=f"{bt.item():.4f}",
                time=f"{elapsed:.0f}s",
            )

        train_loss = running / max(count, 1)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"time={time.time() - start_time:.1f}s"
        )

        if not math.isfinite(train_loss):
            print(
                "WARNING: training loss is not finite. "
                "Skipping checkpoint."
            )
            continue

        if train_loss < best:
            best = train_loss
            bad = 0

            if checkpoint_path:
                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )
                checkpoint_saved = True

        else:
            bad += 1

            if bad >= patience:
                print("Early stopping.")
                break

    if quick:
        print(f"\nBest BT-SR training loss: {best:.4f}")
    else:
        print(f"\nBest BT-SR training loss: {best:.4f}")

    if checkpoint_saved and checkpoint_path:
        print(f"Saved checkpoint: {checkpoint_path}")
    else:
        print("WARNING: No BT-SR checkpoint was saved.")

    return best