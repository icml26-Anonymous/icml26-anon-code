from Tools import *
from Utils import *
from Models import *
import os
import csv
from tqdm import tqdm
from Utility_functions import *
import random

def init_csv_writer(csv_path: None, fieldnames: List[str]):
    if csv_path is None:
        return None, None
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return f, writer


def init_running_stats(model: nn.Module, num_tasks: int, device: str):
    running_mu, running_nu = {}, {}
    for m in model.modules():
        if isinstance(m, GatedLayer):
            for t in range(num_tasks):
                running_mu[(m, t)] = torch.zeros((), device=device)
                running_nu[(m, t)] = torch.zeros((), device=device)
    return running_mu, running_nu


def freeze_current_task_gates(model: nn.Module, t_idx: int, k_target: float):
    """
    Called at the start of the 'weights' phase.
    Freezes current task gates using *static* k_target (not k-decay).
    """
    print(f"[Gates] Freezing gates for task {t_idx}")
    t_index = t_idx
    for m in model.modules():
        if isinstance(m, GatedLayer):
            # for t_index in range(t_idx + 1):
            hard_mask = apply_topk(m.gate_for(t_index), k_target, hard=True)
            g = m._gate_bank[t_index].mul_(hard_mask)

            m._gate_bank[t_index] = g
            m.gate_for(t_index).requires_grad_(False)


def apply_kappa_decay_and_renorm(model: nn.Module, t_idx: int, k_decay: float):
    """
    Applies hard top-k with k_decay and renormalises gates so that
    sum(gates) matches k_decay * num_channels.
    """
    for m in model.modules():
        if isinstance(m, GatedLayer):
            hard_mask = apply_topk(m.gate_for(t_idx), k_decay, hard=True)
            g = m._gate_bank[t_idx].mul_(hard_mask)
            m._gate_bank[t_idx] = g * ((torch.numel(g) * k_decay) / sum(g))


def compute_kappa_decay(
    epoch_counter: int,
    decay_start: int,
    k_start: float,
    k_target: float,
    epochs_parallel: int,
    step_number: int,
    steps_per_epoch: int,
    power: float = 2.0,
) -> float:
    """
    Non-linear κ-decay:
      - For epoch_counter < decay_start: κ = k_start (no decay yet)
      - After that: κ moves from k_start -> k_target
        using a convex schedule:
            κ(α) = k_target + (k_start - k_target) * (1 - α)^power
        where α ∈ [0, 1] is the progress over the decay window.

    Args:
        epoch_counter: current epoch index (1-based or 0-based consistently with loop)
        decay_start: epoch index at which we *start* decaying
        k_start: initial κ at the beginning of the task
        k_target: final κ to reach by the end of the decay window
        epochs_parallel: total epochs in the parallel phase for this task
        step_number: current batch index within this epoch (0-based)
        steps_per_epoch: number of batches in this epoch
        power: p > 1 gives fast early drop + slow tail (2.0 is usually fine)
    """
    # Before we start decaying: keep a large mask for exploration
    if epoch_counter < decay_start:
        return max(k_target, k_start)

    # Clamp so we don't get negative remaining epochs
    decay_epochs = max(1, epochs_parallel - decay_start)
    total_decay_steps = decay_epochs * max(1, steps_per_epoch)

    # Global step index since decay started
    # (epoch_counter - decay_start) is 0-based number of epochs since start
    steps_since_start = (epoch_counter - decay_start) * max(1, steps_per_epoch) + step_number
    steps_since_start = max(0, min(steps_since_start, total_decay_steps))

    # Progress α in [0, 1]
    alpha = steps_since_start / float(total_decay_steps)

    # Convex decay: fast early, slow near k_target
    # κ(α) = k_target + (k_start - k_target) * (1 - α)^p
    remaining = (1.0 - alpha) ** power
    k = k_target + (k_start - k_target) * remaining

    # Numerical safety / invariant
    return max(k_target, min(k_start, k))


def compute_kappa_decay_oldv2(epoch_counter: int,
                         decay_start: int,
                         k_start: float,
                         k_target: float,
                         epochs_parallel: int,
                         step_number: int,
                         steps_per_epoch: int) -> float:
    """
    Matches existing schedule logic:
    - before decay_start: keep k = k_start
    - afterwards: linear anneal to k_target
    """
    if epoch_counter < decay_start:
        return max(k_target, k_start)

    # avoid negative / div by zero
    kappa_distance = abs(k_start - k_target)

    denom = max(1, (epochs_parallel - 2 - decay_start))

    prev_alpha = (epoch_counter - decay_start - 1) / denom
    prev_alpha = min(max(prev_alpha, 0.0), 1.0) * kappa_distance

    epoch_step = (1 / denom) * kappa_distance
    step_alpha = (step_number / steps_per_epoch) * epoch_step

    k = k_start - prev_alpha - step_alpha
    return max(k_target, k)


def format_per_task(acc_list):
    return "[" + ", ".join(f"{a:.1f}%" for a in acc_list) + "]"


##### EVALUATION FUNCTIONS

# ---------- helper: run model with a chosen task gate ----------
@torch.no_grad()
def forward_masked(model, x, task_id, kappa):
    logits = model(x, task_id=task_id, k_frac=kappa)
    return logits


@torch.no_grad()
def eval_CIL_ptKappa(
        model,
        tasks_data: List[Tuple],
        n_tasks=None,
        device="cpu",
        kappa_targets=None,
        criterion="confidence"   # or: "similarity"
    ):
    """
    Evaluate in Class-IL by testing ALL task gates per batch.

    criterion: "confidence"  -> pick task with max soft-max confidence
               "similarity"  -> pick task with highest gate similarity
    """
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)


    if kappa_targets is None:
        kappa_targets = []
        for k in range(n_tasks):
            kappa_targets.append(1/n_tasks)

    acc = [0.0] * n_tasks
    cnt = [0]   * n_tasks

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        kappa = kappa_targets[task_ref]
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-stand task | {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)

            best_score = -1e9
            best_t     = None
            best_logits = None

            # try every task gate
            for t in range(n_tasks):
                logits = forward_masked(model, xb, t, kappa)
                score = F.softmax(logits, 1).max(1).values.mean()

                if score > best_score:
                    best_score, best_t, best_logits = score, t, logits


            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

    return [100.0 * a/c if c else 0. for a, c in zip(acc, cnt)]


@torch.no_grad()
def batch_task_confidences(model, xb, n_tasks, k_targets):
    """Return [T, B] tensor with per-sample max-softmax for each task t."""
    S = []
    for t in range(n_tasks):
        kappa = k_targets[t]
        logits = forward_masked(model, xb, t, kappa)
        conf = torch.softmax(logits, dim=1).max(1).values  # [B]
        S.append(conf)
    return torch.stack(S, dim=0)  # [T, B]


@torch.no_grad()
def compute_conf_stats(model, tasks_data, n_tasks, kappa_targets, device, max_batches=None):
    """
    Unsupervised stats: per-task mean (mu) and std (sigma) of confidence.
    We can pass our training loaders; ideally use a small held-out split.
    """
    model.eval()
    eps = 1e-8

    sums   = torch.zeros(n_tasks, device=device)
    sums2  = torch.zeros(n_tasks, device=device)
    counts = torch.zeros(n_tasks, device=device)

    for loader, _ in tasks_data[:n_tasks]:
        for i, (xb, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            S = batch_task_confidences(model, xb, n_tasks, kappa_targets)  # [T, B]
            sums   += S.sum(dim=1)
            sums2  += (S**2).sum(dim=1)
            counts += S.size(1)

    mu = sums / (counts + eps)
    var = (sums2 / (counts + eps)) - mu**2
    var.clamp_(min=0.0)
    sigma = (var + eps).sqrt()
    return mu.detach(), sigma.detach()

@torch.no_grad()
def eval_CIL_ptKappa_znorm(
        model,
        tasks_data: List[Tuple],
        device="cpu",
        kappa_targets = None,
        n_tasks=None,
        packed_model=None,
        dataset="CIF10",
        criterion="confidence",     # or: "similarity"
        norm_mode="none",          # "none" | "z" | "mean_ratio" | "platt"
        max_batches=10
    ):
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)

    if kappa_targets is None:
        kappa_targets = []
        for k in range(n_tasks):
            kappa_targets.append(1/n_tasks)

    acc = [0.0] * n_tasks
    cnt = [0] * n_tasks
    eps = 1e-8

    mu, sigma = compute_conf_stats(model, tasks_data, n_tasks=n_tasks, kappa_targets=kappa_targets, device=device, max_batches=max_batches)
    # print("μ:", mu.cpu().numpy().round(3), "σ:",
    #       sigma.cpu().numpy().round(3))

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-znorm task | {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)

            best_score = -1e9
            best_t = None
            best_logits = None

            for t in range(n_tasks):
                kappa = kappa_targets[t]
                logits = forward_masked(model, xb, t, kappa)

                if criterion == "confidence":
                    conf = torch.softmax(logits, dim=1).max(1).values  # [B]
                    if norm_mode == "z":
                        assert mu is not None and sigma is not None
                        score = ((conf - mu[t]) / (sigma[t] + eps)).mean()
                    elif norm_mode == "mean_ratio":
                        assert mu is not None
                        score = (conf / (mu[t] + eps)).mean()
                    else:  # "none"
                        score = conf.mean()
                else:
                    # score = batch_similarity(model, xb, t)
                    print('Batch_Similarity Missing')

                if score > best_score:
                    best_logits = logits
                    best_score, best_t = score, t

            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

        full_model_acc = [100.0 * a / c if c else 0. for a, c in zip(acc, cnt)]

    if packed_model is None:
        return full_model_acc
    else:

        acc = [0.0] * n_tasks
        cnt = [0] * n_tasks
        eps = 1e-8

        for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
            for xb, yb in tqdm(loader, desc=f"Eval | CIL-znorm task | {task_ref}"):
                xb, yb = xb.to(device), yb.to(device)

                best_score = -1e9
                best_t = None
                best_logits = None

                for t in range(n_tasks):
                    packed_m = packed_model[t]
                    logits = packed_m(xb)

                    if criterion == "confidence":
                        conf = torch.softmax(logits, dim=1).max(1).values  # [B]
                        if norm_mode == "z":
                            assert mu is not None and sigma is not None
                            score = ((conf - mu[t]) / (sigma[t] + eps)).mean()
                        elif norm_mode == "mean_ratio":
                            assert mu is not None
                            score = (conf / (mu[t] + eps)).mean()
                        else:  # "none"
                            score = conf.mean()
                    else:
                        # score = batch_similarity(model, xb, t)
                        print('Batch_Similarity Missing')

                    if score > best_score:
                        best_logits = logits
                        best_score, best_t = score, t

                pred = best_logits.argmax(1) + (best_t*10)
                acc[task_ref] += (pred == yb).sum().item()
                cnt[task_ref] += yb.size(0)

        packed_model_acc = [100.0 * a / c if c else 0. for a, c in zip(acc, cnt)]
        return full_model_acc, packed_model_acc





def eval_TIL(model: nn.Module, tasks_data, device="cpu", kappa_targets=None, k_decay=0.2, up2task=5):
    accs = []

    n_tasks = len(tasks_data)
    if kappa_targets is None:
        kappa_targets = []
        for k in range(n_tasks):
            kappa_targets.append(1/n_tasks)

    for t, (_, test_loader) in enumerate(tasks_data):
        k_frac = kappa_targets[t]
        if t < up2task:
            accs.append(eval_task(model, test_loader, t, device, k_frac))
        elif t == up2task:
            accs.append(eval_task(model, test_loader, t, device, k_decay))
    return accs

def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def gates_update_step_allowed(phase: str) -> bool:
    return phase in {"parallel", "gates"}


def valid_loss(model, test_loader, t_idx, k_decay, device):
    v_losses = []
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for step, (xb, yb) in enumerate(test_loader):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, task_id=t_idx, k_frac=k_decay)
            v_loss = F.cross_entropy(logits, yb)
            v_losses.append(v_loss.item())
    mean_loss = sum(v_losses) / len(v_losses)
    print(f"Test loss {sum(v_losses) / len(v_losses):.3f}")
    return mean_loss


@torch.no_grad()
def eval_full_CIL_sequential(
        model,
        tasks_data: List[Tuple],
        device="cpu",
        kappa_targets=None,
        n_tasks=None,
        dataset="CIF10",
        criterion="confidence",  # or: "similarity"
        norm_mode="z",  # "none" | "z" | "mean_ratio" | "platt"
        max_batches=20
):
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)

    if kappa_targets is None:
        kappa_targets = []
        for k in range(n_tasks):
            kappa_targets.append(1 / n_tasks)

    acc = [0.0] * n_tasks
    cnt = [0] * n_tasks

    if (model.mu is None) or (model.sigma is None):
        mu, sigma = compute_conf_stats(model, tasks_data, n_tasks=n_tasks, kappa_targets=kappa_targets, device=device,
                                       max_batches=max_batches)
        model.mu = mu
        model.sigma = sigma

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-znorm task | {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)
            pred = model.class_il_predict_sequential(xb, n_tasks=n_tasks, kappa_targets=kappa_targets, criterion=criterion, norm_mode=norm_mode)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)
        full_model_acc = [100.0 * a / c if c else 0. for a, c in zip(acc, cnt)]

    return full_model_acc


def build_class_slices(n_tasks, classes_per_task, start=0):
    return {t: torch.arange(start + t * classes_per_task, start + (t + 1) * classes_per_task)
            for t in range(n_tasks)}


def save_mu_sigma(path: str, mu: torch.Tensor, sigma: torch.Tensor, extra: dict = None):
    """
    Saves mu/sigma (+ optional metadata) to a single .pt file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "mu": mu.detach().cpu(),
        "sigma": sigma.detach().cpu(),
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


import torch

def load_mu_sigma(path: str, device: str = "cuda"):
    """
    Loads mu/sigma from .pt file. Returns tensors on device.
    """
    payload = torch.load(path, map_location="cpu")
    mu = payload["mu"].to(device, dtype=torch.float32)
    sigma = payload["sigma"].to(device, dtype=torch.float32)
    extra = payload.get("extra", None)
    return mu, sigma, extra
