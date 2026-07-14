"""Deterministic two-stage training for native sequence-to-sequence FGL-OFP.

The teacher is a training-only object.  It is first optimized against the
shorter future-window target, restored to its best validation checkpoint, and
then frozen.  The student is optimized with ``alpha * CE + (1-alpha) * KL``.
Student validation deliberately accepts no teacher argument and uses only the
student future-window CE; consequently early stopping has the same causal
information boundary as deployment.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .config import FGOFPConfig
from .data import LengthBucketBatchSampler, collate_native_sequences
from .losses import combined_future_guided_loss, module_normalized_cross_entropy


@dataclass
class TrainingResult:
    """Models and audit metadata produced by :func:`train_models`.

    ``teacher_train_only`` is ``None`` for the CE-only control.  Callers should
    never serialize the teacher as a deployment dependency; it is returned only
    so experiments can audit freezing and reproduce the training checkpoint.
    """

    student: nn.Module
    teacher_train_only: nn.Module | None
    history: pd.DataFrame
    weights: dict[str, float]
    best_epochs: dict[str, int | None]
    best_validation_losses: dict[str, float | None]
    seconds_total: float
    coverage: dict[str, int]
    fgl_coverage_scale: float
    device: str
    mixed_precision_enabled: bool

    @property
    def model(self) -> nn.Module:
        """Compatibility alias: the deployable model is always the student."""

        return self.student

    @property
    def student_positive_weight(self) -> float:
        return float(self.weights["student"])

    @property
    def teacher_positive_weight(self) -> float:
        return float(self.weights["teacher"])

    @property
    def best_student_epoch(self) -> int:
        value = self.best_epochs["student"]
        if value is None:
            raise RuntimeError("student checkpoint metadata is missing")
        return int(value)

    @property
    def best_teacher_epoch(self) -> int | None:
        value = self.best_epochs["teacher"]
        return None if value is None else int(value)

    @property
    def best_student_validation_loss(self) -> float:
        value = self.best_validation_losses["student"]
        if value is None:
            raise RuntimeError("student validation metadata is missing")
        return float(value)

    @property
    def best_teacher_validation_loss(self) -> float | None:
        value = self.best_validation_losses["teacher"]
        return None if value is None else float(value)


@dataclass(frozen=True)
class _DatasetAudit:
    coverage: dict[str, int]
    student_positive_mass: float
    student_negative_mass: float
    teacher_positive_mass: float
    teacher_negative_mass: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch and select deterministic CUDA kernels."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and fail clearly when CUDA was explicitly requested."""

    name = str(requested).lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device='cuda' but CUDA is unavailable")
    return torch.device(name)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataset_lengths(dataset: Dataset) -> np.ndarray:
    lengths = getattr(dataset, "lengths", None)
    if lengths is None:
        lengths = [int(dataset[index]["length"]) for index in range(len(dataset))]
    result = np.asarray(lengths, dtype=np.int64)
    if result.shape != (len(dataset),) or np.any(result <= 0):
        raise ValueError("dataset lengths must be a positive vector aligned with modules")
    return result


def make_sequence_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a padding-efficient loader using deterministic length buckets."""

    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    sampler = LengthBucketBatchSampler(
        _dataset_lengths(dataset),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        seed=int(seed),
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_native_sequences,
        num_workers=workers,
        pin_memory=bool(pin_memory),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _require_tensor(sample: Mapping[str, object], name: str) -> Tensor:
    value = sample.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"dataset sample field {name!r} must be a Tensor")
    return value


def _audit_training_dataset(dataset: Dataset) -> _DatasetAudit:
    if len(dataset) <= 0:
        raise ValueError("training dataset cannot be empty")
    coverage = {
        "modules": 0,
        "ce_modules": 0,
        "student_ce_rows": 0,
        "student_ce_positive_rows": 0,
        "teacher_ce_rows": 0,
        "teacher_ce_positive_rows": 0,
        "fgl_modules": 0,
        "fgl_pairs": 0,
        "positive_fgl_modules": 0,
        "positive_fgl_pairs": 0,
        "first_row_faults": 0,
    }
    student_positive_mass = 0.0
    student_negative_mass = 0.0
    teacher_positive_mass = 0.0
    teacher_negative_mass = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("dataset samples must be mappings")
        student = _require_tensor(sample, "targets_student").to(dtype=torch.long)
        teacher = _require_tensor(sample, "targets_teacher").to(dtype=torch.long)
        loss_mask = _require_tensor(sample, "loss_mask").to(dtype=torch.bool)
        fgl_mask = _require_tensor(sample, "fgl_mask").to(dtype=torch.bool)
        if student.ndim != 1 or teacher.shape != student.shape:
            raise ValueError("student and teacher targets must be aligned vectors")
        if loss_mask.shape != student.shape or fgl_mask.shape != student.shape:
            raise ValueError("loss and FGL masks must align with targets")
        rows = int(loss_mask.sum().item())
        if rows <= 0:
            raise ValueError("every training module must expose at least one CE row")
        valid_student = student[loss_mask]
        valid_teacher = teacher[loss_mask]
        if bool(((valid_student < 0) | (valid_student > 1)).any()) or bool(
            ((valid_teacher < 0) | (valid_teacher > 1)).any()
        ):
            raise ValueError("valid future-window targets must be binary")
        student_positives = int((valid_student == 1).sum().item())
        teacher_positives = int((valid_teacher == 1).sum().item())
        student_positive_mass += student_positives / rows
        student_negative_mass += (rows - student_positives) / rows
        teacher_positive_mass += teacher_positives / rows
        teacher_negative_mass += (rows - teacher_positives) / rows

        pairs = int(fgl_mask.sum().item())
        positive_pairs = int(((student == 1) & fgl_mask).sum().item())
        coverage["modules"] += 1
        coverage["ce_modules"] += 1
        coverage["student_ce_rows"] += rows
        coverage["student_ce_positive_rows"] += student_positives
        coverage["teacher_ce_rows"] += rows
        coverage["teacher_ce_positive_rows"] += teacher_positives
        coverage["fgl_pairs"] += pairs
        coverage["positive_fgl_pairs"] += positive_pairs
        coverage["fgl_modules"] += int(pairs > 0)
        coverage["positive_fgl_modules"] += int(positive_pairs > 0)
        first_fault_index = int(sample.get("first_fault_index", -1))
        coverage["first_row_faults"] += int(first_fault_index == 0)
    return _DatasetAudit(
        coverage=coverage,
        student_positive_mass=student_positive_mass,
        student_negative_mass=student_negative_mass,
        teacher_positive_mass=teacher_positive_mass,
        teacher_negative_mass=teacher_negative_mass,
    )


def _weight_from_mass(positive: float, negative: float, maximum: float) -> float:
    if positive <= 0.0:
        return 1.0
    return float(np.clip(negative / positive, 1.0, float(maximum)))


def module_normalized_positive_weight(
    dataset: Dataset,
    *,
    target: str = "student",
    mode: str = "module_normalized_auto",
    maximum: float = 5.0,
) -> float:
    """Compute class weight in the same equal-module measure as the CE loss.

    Each module first contributes its positive and negative *fractions*.  Those
    masses are then summed across modules and their ratio is capped.  This is
    the class-weight counterpart of averaging rows within a module before
    averaging modules; long modules therefore cannot dominate the estimate.
    """

    if mode == "none":
        return 1.0
    if mode != "module_normalized_auto":
        raise ValueError("mode must be none or module_normalized_auto")
    if float(maximum) < 1.0:
        raise ValueError("maximum must be at least 1")
    audit = _audit_training_dataset(dataset)
    if target == "student":
        return _weight_from_mass(
            audit.student_positive_mass, audit.student_negative_mass, maximum
        )
    if target == "teacher":
        return _weight_from_mass(
            audit.teacher_positive_mass, audit.teacher_negative_mass, maximum
        )
    raise ValueError("target must be student or teacher")


def training_coverage(dataset: Dataset) -> dict[str, int]:
    """Return immutable-by-copy pre-training CE/FGL coverage diagnostics."""

    return dict(_audit_training_dataset(dataset).coverage)


def _move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    required_names = (
        "raw",
        "raw_mask",
        "delta_steps",
        "targets_teacher",
        "targets_student",
        "loss_mask",
        "teacher_index",
        "fgl_mask",
        "lengths",
    )
    moved: dict[str, Tensor] = {}
    for name in required_names:
        value = batch.get(name)
        if not isinstance(value, Tensor):
            raise TypeError(f"collated batch field {name!r} must be a Tensor")
        moved[name] = value.to(device=device, non_blocking=True)
    for name in ("rule_margin", "rule_mask", "rule_hard"):
        value = batch.get(name)
        if value is not None:
            if not isinstance(value, Tensor):
                raise TypeError(f"collated batch field {name!r} must be a Tensor")
            moved[name] = value.to(device=device, non_blocking=True)
    time = int(moved["raw"].shape[1])
    positions = torch.arange(time, device=device).unsqueeze(0)
    moved["padding_mask"] = positions < moved["lengths"].unsqueeze(1)
    return moved


def _forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    arguments = {
        "raw": batch["raw"].to(dtype=torch.float32),
        "raw_mask": batch["raw_mask"].to(dtype=torch.bool),
        "delta_steps": batch["delta_steps"].to(dtype=torch.float32),
        "padding_mask": batch["padding_mask"].to(dtype=torch.bool),
    }
    if bool(getattr(model, "requires_rule_inputs", False)):
        missing = [
            name
            for name in ("rule_margin", "rule_mask", "rule_hard")
            if name not in batch
        ]
        if missing:
            raise ValueError(f"rule-guided model batch missing fields: {missing}")
        arguments.update(
            {
                "rule_margin": batch["rule_margin"].to(dtype=torch.float32),
                "rule_mask": batch["rule_mask"].to(dtype=torch.bool),
                "rule_hard": batch["rule_hard"].to(dtype=torch.bool),
            }
        )
    return model(**arguments)


def _clone_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _freeze_teacher(teacher: nn.Module) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = loader.batch_sampler
    if isinstance(sampler, LengthBucketBatchSampler):
        sampler.set_epoch(int(epoch))


def _new_scaler(enabled: bool) -> Any:
    """Build a CUDA scaler across the supported PyTorch >=2.1 range."""

    amp_namespace = getattr(torch, "amp", None)
    scaler_class = getattr(amp_namespace, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=bool(enabled))
        except TypeError:
            # Some intermediate releases exposed torch.amp.GradScaler without
            # the device positional argument.
            return scaler_class(enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _optimizer_step(
    loss: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    grad_clip: float,
) -> None:
    scaler.scale(loss).backward()
    if float(grad_clip) > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    scaler.step(optimizer)
    scaler.update()


def _run_teacher_epoch(
    teacher: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    use_amp: bool,
    grad_clip: float,
) -> tuple[float, dict[str, int]]:
    training = optimizer is not None
    teacher.train(training)
    weighted_loss = 0.0
    total_modules = 0
    totals = {"ce_modules": 0, "ce_rows": 0, "ce_positive_rows": 0}
    for original in loader:
        batch = _move_batch(original, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = _forward(teacher, batch)
                loss, coverage = module_normalized_cross_entropy(
                    logits,
                    batch["targets_teacher"],
                    batch["loss_mask"],
                    pos_weight=pos_weight,
                    return_coverage=True,
                )
            if training:
                _optimizer_step(loss, teacher, optimizer, scaler, grad_clip)
        modules = int(coverage["ce_modules"])
        weighted_loss += float(loss.detach().cpu()) * modules
        total_modules += modules
        for name in totals:
            totals[name] += int(coverage[name])
    if total_modules <= 0:
        raise ValueError("empty teacher data loader")
    return weighted_loss / total_modules, totals


def _run_student_training_epoch(
    student: nn.Module,
    teacher: nn.Module | None,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: float,
    alpha: float,
    temperature: float,
    fgl_coverage_scale: float,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    grad_clip: float,
) -> tuple[dict[str, float], dict[str, int]]:
    student.train()
    if teacher is not None:
        teacher.eval()
    sums = {"total": 0.0, "cross_entropy": 0.0, "fgl_kl": 0.0}
    denominators = {"total": 0, "cross_entropy": 0, "fgl_kl": 0}
    totals = {
        "ce_modules": 0,
        "ce_rows": 0,
        "ce_positive_rows": 0,
        "fgl_modules": 0,
        "fgl_pairs": 0,
    }
    for original in loader:
        batch = _move_batch(original, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            student_logits = _forward(student, batch)
            if float(alpha) == 1.0:
                ce, coverage = module_normalized_cross_entropy(
                    student_logits,
                    batch["targets_student"],
                    batch["loss_mask"],
                    pos_weight=pos_weight,
                    return_coverage=True,
                )
                total_loss = ce
                kl = ce.detach() * 0.0
                coverage = {
                    **coverage,
                    "fgl_modules": 0,
                    "fgl_pairs": 0,
                }
            else:
                if teacher is None:
                    raise RuntimeError("FGL training requires a frozen teacher")
                with torch.no_grad():
                    teacher_logits = _forward(teacher, batch)
                output = combined_future_guided_loss(
                    student_logits=student_logits,
                    target=batch["targets_student"],
                    target_mask=batch["loss_mask"],
                    teacher_logits=teacher_logits,
                    teacher_index=batch["teacher_index"],
                    fgl_mask=batch["fgl_mask"],
                    alpha=alpha,
                    temperature=temperature,
                    pos_weight=pos_weight,
                    fgl_coverage_scale=fgl_coverage_scale,
                )
                total_loss = output.total
                ce = output.cross_entropy
                kl = output.fgl_kl
                coverage = output.coverage
        _optimizer_step(total_loss, student, optimizer, scaler, grad_clip)
        ce_modules = int(coverage["ce_modules"])
        fgl_modules = int(coverage["fgl_modules"])
        sums["total"] += float(total_loss.detach().cpu()) * ce_modules
        sums["cross_entropy"] += float(ce.detach().cpu()) * ce_modules
        # fgl_kl includes zero-valued modules without a valid future pair, so
        # its epoch aggregation uses the same fixed module denominator as CE.
        sums["fgl_kl"] += float(kl.detach().cpu()) * ce_modules
        denominators["total"] += ce_modules
        denominators["cross_entropy"] += ce_modules
        denominators["fgl_kl"] += ce_modules
        for name in totals:
            totals[name] += int(coverage[name])
    if denominators["total"] <= 0:
        raise ValueError("empty student data loader")
    averages = {
        "total": sums["total"] / denominators["total"],
        "cross_entropy": sums["cross_entropy"] / denominators["cross_entropy"],
        "fgl_kl": (
            sums["fgl_kl"] / denominators["fgl_kl"]
            if denominators["fgl_kl"] > 0
            else 0.0
        ),
    }
    return averages, totals


def _run_student_validation_epoch(
    student: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: float,
    use_amp: bool,
) -> tuple[float, dict[str, int]]:
    """Validate causal student CE only; this function cannot call a teacher."""

    student.eval()
    weighted_loss = 0.0
    total_modules = 0
    totals = {"ce_modules": 0, "ce_rows": 0, "ce_positive_rows": 0}
    with torch.no_grad():
        for original in loader:
            batch = _move_batch(original, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = _forward(student, batch)
                loss, coverage = module_normalized_cross_entropy(
                    logits,
                    batch["targets_student"],
                    batch["loss_mask"],
                    pos_weight=pos_weight,
                    return_coverage=True,
                )
            modules = int(coverage["ce_modules"])
            weighted_loss += float(loss.detach().cpu()) * modules
            total_modules += modules
            for name in totals:
                totals[name] += int(coverage[name])
    if total_modules <= 0:
        raise ValueError("empty student validation data loader")
    return weighted_loss / total_modules, totals


def _is_improvement(value: float, best: float) -> bool:
    return float(value) < float(best) - 1e-7


def train_models(
    student: nn.Module,
    teacher: nn.Module | None,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: FGOFPConfig,
) -> TrainingResult:
    """Train teacher then student without exposing future data at validation.

    Setting ``training.fgl_alpha=1`` is the exact CE-only control: the teacher
    is neither moved to the device nor called, and the result stores ``None``.
    """

    seed_everything(config.seed)
    device = resolve_device(config.training.device)
    use_amp = bool(config.training.mixed_precision and device.type == "cuda")
    audit = _audit_training_dataset(train_dataset)
    coverage = dict(audit.coverage)
    fgl_enabled = float(config.training.fgl_alpha) < 1.0
    if fgl_enabled and teacher is student:
        raise ValueError("student and teacher must be independent model instances")
    if fgl_enabled and coverage["positive_fgl_pairs"] <= 0:
        raise ValueError(
            "FGL is enabled but the training split has zero positive FGL pairs; "
            "check native timestamps, future offset, and alignment tolerance"
        )
    fgl_coverage_scale = (
        float(coverage["modules"]) / float(coverage["fgl_modules"])
        if fgl_enabled and coverage["fgl_modules"] > 0
        else 1.0
    )
    mode = config.training.positive_weight_mode
    maximum = float(config.training.max_positive_weight)
    if mode == "none":
        student_weight = teacher_weight = 1.0
    elif mode == "module_normalized_auto":
        student_weight = _weight_from_mass(
            audit.student_positive_mass, audit.student_negative_mass, maximum
        )
        teacher_weight = _weight_from_mass(
            audit.teacher_positive_mass, audit.teacher_negative_mass, maximum
        )
    else:
        raise ValueError("positive_weight_mode must be none or module_normalized_auto")
    weights = {"student": student_weight, "teacher": teacher_weight}
    print(
        "[training coverage] "
        f"modules={coverage['modules']} "
        f"student_ce_rows={coverage['student_ce_rows']} "
        f"teacher_ce_rows={coverage['teacher_ce_rows']} "
        f"fgl_pairs={coverage['fgl_pairs']} "
        f"positive_fgl_pairs={coverage['positive_fgl_pairs']} "
        f"positive_fgl_modules={coverage['positive_fgl_modules']} "
        f"first_row_faults={coverage['first_row_faults']}",
        flush=True,
    )
    print(
        f"[class weights] student={student_weight:.6f} "
        f"teacher={teacher_weight:.6f} mode={mode}",
        flush=True,
    )
    if fgl_enabled:
        print(
            "[FGL normalization] "
            f"global_covered_module_scale={fgl_coverage_scale:.6f}",
            flush=True,
        )

    train_loader = make_sequence_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_sequence_loader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.seed + 1,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    student = student.to(device)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    best_epochs: dict[str, int | None] = {"teacher": None, "student": None}
    best_losses: dict[str, float | None] = {"teacher": None, "student": None}
    trained_teacher: nn.Module | None = None

    if fgl_enabled:
        if teacher is None:
            raise ValueError("teacher cannot be None when FGL is enabled")
        trained_teacher = teacher.to(device)
        for parameter in trained_teacher.parameters():
            parameter.requires_grad_(True)
        teacher_optimizer = torch.optim.AdamW(
            trained_teacher.parameters(),
            lr=float(config.training.teacher_learning_rate),
            weight_decay=float(config.training.weight_decay),
        )
        teacher_scaler = _new_scaler(use_amp)
        best_teacher_loss = float("inf")
        best_teacher_state: dict[str, Tensor] | None = None
        stale = 0
        for epoch in range(1, int(config.training.teacher_epochs) + 1):
            _set_sampler_epoch(train_loader, epoch - 1)
            train_loss, train_counts = _run_teacher_epoch(
                trained_teacher,
                train_loader,
                device=device,
                pos_weight=teacher_weight,
                optimizer=teacher_optimizer,
                scaler=teacher_scaler,
                use_amp=use_amp,
                grad_clip=config.training.grad_clip,
            )
            validation_loss, validation_counts = _run_teacher_epoch(
                trained_teacher,
                validation_loader,
                device=device,
                pos_weight=teacher_weight,
                optimizer=None,
                scaler=teacher_scaler,
                use_amp=use_amp,
                grad_clip=0.0,
            )
            rows.append(
                {
                    "stage": "teacher",
                    "epoch": epoch,
                    "train_total_loss": train_loss,
                    "train_ce_loss": train_loss,
                    "train_fgl_kl": 0.0,
                    "validation_ce_loss": validation_loss,
                    **{f"train_{key}": value for key, value in train_counts.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_counts.items()
                    },
                }
            )
            print(
                f"[teacher epoch {epoch:02d}] train_ce={train_loss:.6f} "
                f"validation_ce={validation_loss:.6f}",
                flush=True,
            )
            if _is_improvement(validation_loss, best_teacher_loss):
                best_teacher_loss = float(validation_loss)
                best_teacher_state = _clone_state(trained_teacher)
                best_epochs["teacher"] = epoch
                stale = 0
            else:
                stale += 1
            patience = int(config.training.early_stopping_patience)
            if patience > 0 and stale >= patience:
                break
        if best_teacher_state is None:
            raise RuntimeError("teacher training did not produce a checkpoint")
        trained_teacher.load_state_dict(copy.deepcopy(best_teacher_state))
        best_losses["teacher"] = best_teacher_loss
        _freeze_teacher(trained_teacher)

    # Teacher pretraining consumes RNG state (dropout, CUDA kernels). Reset it
    # so the CE-only and CE+FGL students see identical sampler order and
    # dropout streams; their only intended difference is the KL term.
    seed_everything(config.seed + 10_000)
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config.training.student_learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    student_scaler = _new_scaler(use_amp)
    best_student_loss = float("inf")
    best_student_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, int(config.training.student_epochs) + 1):
        _set_sampler_epoch(train_loader, epoch - 1)
        train_losses, train_counts = _run_student_training_epoch(
            student,
            trained_teacher,
            train_loader,
            device=device,
            pos_weight=student_weight,
            alpha=float(config.training.fgl_alpha),
            temperature=float(config.training.temperature),
            fgl_coverage_scale=fgl_coverage_scale,
            optimizer=student_optimizer,
            scaler=student_scaler,
            use_amp=use_amp,
            grad_clip=config.training.grad_clip,
        )
        # This call is intentionally teacher-free.  Early stopping therefore
        # cannot benefit from relative-future inputs or teacher logits.
        validation_loss, validation_counts = _run_student_validation_epoch(
            student,
            validation_loader,
            device=device,
            pos_weight=student_weight,
            use_amp=use_amp,
        )
        rows.append(
            {
                "stage": "student",
                "epoch": epoch,
                "train_total_loss": train_losses["total"],
                "train_ce_loss": train_losses["cross_entropy"],
                "train_fgl_kl": train_losses["fgl_kl"],
                "validation_ce_loss": validation_loss,
                **{f"train_{key}": value for key, value in train_counts.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_counts.items()
                },
            }
        )
        print(
            f"[student epoch {epoch:02d}] train={train_losses['total']:.6f} "
            f"ce={train_losses['cross_entropy']:.6f} "
            f"fgl_kl={train_losses['fgl_kl']:.6f} "
            f"validation_ce={validation_loss:.6f}",
            flush=True,
        )
        if _is_improvement(validation_loss, best_student_loss):
            best_student_loss = float(validation_loss)
            best_student_state = _clone_state(student)
            best_epochs["student"] = epoch
            stale = 0
        else:
            stale += 1
        patience = int(config.training.early_stopping_patience)
        if patience > 0 and stale >= patience:
            break
    if best_student_state is None:
        raise RuntimeError("student training did not produce a checkpoint")
    student.load_state_dict(copy.deepcopy(best_student_state))
    student.eval()
    best_losses["student"] = best_student_loss

    return TrainingResult(
        student=student,
        teacher_train_only=trained_teacher,
        history=pd.DataFrame(rows),
        weights=weights,
        best_epochs=best_epochs,
        best_validation_losses=best_losses,
        seconds_total=float(time.perf_counter() - started),
        coverage=coverage,
        fgl_coverage_scale=fgl_coverage_scale,
        device=str(device),
        mixed_precision_enabled=use_amp,
    )


def train_model(
    student: nn.Module,
    teacher: nn.Module | None,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: FGOFPConfig,
) -> TrainingResult:
    """Singular compatibility alias for :func:`train_models`."""

    return train_models(student, teacher, train_dataset, validation_dataset, config)


__all__ = [
    "TrainingResult",
    "make_sequence_loader",
    "module_normalized_positive_weight",
    "resolve_device",
    "seed_everything",
    "train_model",
    "train_models",
    "training_coverage",
]
