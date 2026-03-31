# -------------------------------------------------------------
# router_utils.py
# -------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from tqdm import tqdm
from Utility_functions import *
import random


# ---------- 1.  Unmasked forward -------------------------------------------
def forward_unmasked(model, x) -> Dict[torch.nn.Module, torch.Tensor]:
    """
    Run one forward pass disabling *all* task masks.
    Returns a {layer: activation} dict needed for similarity.
    """
    # --- temporarily overwrite apply_topk to return all-ones mask ----------
    original_apply_topk = globals()['apply_topk']   # import from util
    globals()['apply_topk'] = lambda g, k, hard=True: torch.ones_like(g)

    acts = {}
    hooks = []

    def _store_act(module, inp, out):
        acts[module] = out.detach()

    for m in model.modules():
        if isinstance(m, GatedLayer):
            hooks.append(m.register_forward_hook(_store_act))

    _ = model(x, task_id=random.randint(0, 4), k_frac=1.0)   # task_id dummy, masks are 1's

    for h in hooks:
        h.remove()
    globals()['apply_topk'] = original_apply_topk
    return acts


# ---------- 2.  Cosine-similarity router -----------------------------------
def route_task(model, x, kappa: float, tasks_done: int,
               topk_progressive: int = None) -> int:
    """
    Return the most likely task index for a single *batch* x.
    Costs 1 un-masked + 1 masked forward pass (Class-IL single-head).
    """
    acts = forward_unmasked(model, x)           # ungated activations
    device = x.device

    # running_mu = model.running_mu
    # running_nu = model.running_nu
    CURRENT_CLASSES = []
    for i in range(tasks_done):
        start_ = i * 2
        end_ = (i + 1) * 2
        class_list = list(range(start_, end_))
        CURRENT_CLASSES.append(class_list)
    # pre-compute channel energies a_{ℓ,j}
    batch_energy = {}

    for m, A in acts.items():
        if A.dim() == 4:  # and A.shape[1] > 100:                        # [B,C,H,W]
            e = (A).mean((0, 2, 3))
            # e.mul_(apply_topk(e, k_fraction=kappa))
            e = apply_topk(e, k_fraction=kappa)
            batch_energy[m] = e
        # elif A.dim() == 2:                                   # [B,C]
            # e = (A).mean(0)
            # e = apply_topk(e, k_fraction=kappa)
            # print(e)
            # batch_energy[m] = e
            # shape [d_ℓ]

    sim_scores = torch.zeros(tasks_done, device=device)
    # print('SIM SCORES')
    for t in range(tasks_done):
        s_tot = 0.0
        for m, e in batch_energy.items():
            g = m.gate_for(t).to(device)
            g = apply_topk(g, k_fraction=kappa)
            if g.shape[0] == 10:
                fc_g = model.fc._gate_bank[t].clone()
                new_g = torch.zeros_like(fc_g, device=fc_g.device)
                new_g[CURRENT_CLASSES[t]] = 1.0
                g = new_g
            # print(g.shape)
            # print(g)
            # print(e)
            s_tot += torch.dot(e, g) / (e.sum() + 1e-8)
            # raw = torch.dot(e, g) / (e.sum() + 1e-8)
            #
            # # fetch running stats
            # mu = running_mu[(m, t)]
            # nu = running_nu[(m, t)]
            # sigma = (nu - mu * mu).clamp(min=1e-6).sqrt()
            #
            # z = (raw - mu) / sigma
            # s_tot += z

        sim_scores[t] = s_tot
        # print(t)
        # print(sim_scores[t])
    if topk_progressive is not None:
        # keep best-k tasks; here we simply pick the best one
        _, top_t = torch.topk(sim_scores, k=topk_progressive)
        best_t = top_t[0].item()
    else:
        best_t = sim_scores.argmax().item()

    return best_t


# ---------- 3.  Class-IL evaluation ----------------------------------------
@torch.no_grad()
def eval_class_IL(model, tasks_data: List[Tuple],
                  device="cpu", kappa=0.3, topk_progressive=None) -> List[float]:
    """
    Evaluate final Class-IL accuracy (task-ID unknown).
    """
    model.eval()
    acc = [0.0] * len(tasks_data)
    cnt = [0]   * len(tasks_data)

    # for t, m in enumerate(model.modules()):
    #     if isinstance(m, GatedLayer):
    #         print(t, m.gate_for(0).sum().item())  # gate “size”
    #         print(t, m.gate_for(1).sum().item())  # gate “size”

    for task_id, (_, loader) in enumerate(tasks_data):
        for xb, yb in tqdm(loader, desc=f"Task {task_id}"):
            xb, yb = xb.to(device), yb.to(device)

            # 1) route to a task
            t_hat = route_task(model, xb,
                               kappa, tasks_done=len(tasks_data),
                               topk_progressive=topk_progressive)

            print(f"Task {task_id} - Predicted: {t_hat}")
            # 2) run *masked* forward once with predicted task
            logits = model(xb, task_id=t_hat, k_frac=kappa)
            pred   = logits.argmax(1)
            acc[task_id] += (pred == yb).sum().item()
            cnt[task_id] += yb.size(0)

    return [100.0 * a/c if c else 0. for a, c in zip(acc, cnt)]


# ---------- helper: run model with a chosen task gate ----------
@torch.no_grad()
def forward_masked(model, x, task_id, kappa):
    logits = model(x, task_id=task_id, k_frac=kappa)
    return logits


# ---------- helper: layer-wise similarity (used by both eval modes) -----
def batch_similarity(model, x, task_id):
    """
    Return cos-like similarity score for a *batch* x and a given task_id.
    Re-uses the ungated-forward helper 
    """
    acts = forward_unmasked(model, x)  # from router_utils.py
    s = 0.0
    for m, A in acts.items():
        if A.dim() == 4:         # [B,C,H,W]  -> energy per channel
            e = (A**2).mean((0, 2, 3))
        else:                    # [B,C]
            e = (A**2).mean(0)
        g = m.gate_for(task_id).to(x.device)
        num   = torch.dot(e, g)
        denom = e.sum() + 1e-8
        s += num / denom
    return s


# ---------- main exhaustive Class-IL evaluator --------------------------
@torch.no_grad()
def eval_exhaustive(
        model,
        tasks_data: List[Tuple],
        n_tasks=None,
        device="cpu",
        kappa=0.3,
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

    acc = [0.0] * n_tasks
    cnt = [0]   * n_tasks

    for task_ref, (_, loader) in enumerate(tasks_data[:n_tasks]):
        for xb, yb in tqdm(loader, desc=f"Eval | CIL-stand task | {task_ref}"):
            xb, yb = xb.to(device), yb.to(device)

            best_score = -1e9
            best_t     = None
            best_logits = None

            # try every task gate
            for t in range(n_tasks):
                logits = forward_masked(model, xb, t, kappa)
                if criterion == "confidence":
                    score = F.softmax(logits, 1).max(1).values.mean()
                    # print(f"T: {t}, Score: {score}")
                else:                                   # similarity
                    score = batch_similarity(model, xb, t)
                # if score > exit_bound:
                #     best_score, best_t, best_logits = score, t, logits
                #     break
                if score > best_score:
                    best_score, best_t, best_logits = score, t, logits


            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

    return [100.0 * a/c if c else 0. for a, c in zip(acc, cnt)]


@torch.no_grad()
def batch_task_confidences(model, xb, n_tasks, kappa):
    """Return [T, B] tensor with per-sample max-softmax for each task t."""
    S = []
    for t in range(n_tasks):
        logits = forward_masked(model, xb, t, kappa)
        conf = torch.softmax(logits, dim=1).max(1).values  # [B]
        S.append(conf)
    return torch.stack(S, dim=0)  # [T, B]


@torch.no_grad()
def compute_conf_stats(model, tasks_data, n_tasks, kappa, device, max_batches=None):
    """
    Unsupervised stats: per-task mean (mu) and std (sigma) of confidence.
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
            S = batch_task_confidences(model, xb, n_tasks, kappa)  # [T, B]
            sums   += S.sum(dim=1)
            sums2  += (S**2).sum(dim=1)
            counts += S.size(1)

    mu = sums / (counts + eps)
    var = (sums2 / (counts + eps)) - mu**2
    var.clamp_(min=0.0)
    sigma = (var + eps).sqrt()
    return mu.detach(), sigma.detach()



def task_score_from_logits(logits, score_type: str):
    """
    logits: [B, C]
    score_type: 'conf' | 'maxlogit' | 'energy' | 'entropy'
    Returns per-sample scores [B] (higher = more likely this task).
    """
    if score_type == "conf":
        p = torch.softmax(logits, dim=1)
        return p.max(1).values
    elif score_type == "maxlogit":
        return logits.max(1).values
    elif score_type == "energy":
        # More negative means higher confidence; flip sign so higher is better
        e = -torch.logsumexp(logits, dim=1)
        return -e
    elif score_type == "entropy":
        p = torch.softmax(logits, dim=1)
        h = -(p * (p.clamp_min(1e-12).log())).sum(1)
        return -h  # lower entropy = better; flip sign
    else:
        raise ValueError("Unknown score_type")


@torch.no_grad()
def eval_exhaustiveY(
        model,
        tasks_data: List[Tuple],
        device="cpu",
        kappa=0.3,
        n_tasks=None,
        dataset="CIF10",
        criterion="confidence",     # or: "similarity"
        norm_mode="none",           # "none" | "z" | "mean_ratio" | "platt"
        mu=None, sigma=None,        # used by z / mean_ratio
        platt_a=None, platt_b=None  # used by platt
    ):
    model.eval()
    if n_tasks is None:
        n_tasks = len(tasks_data)

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
                logits = forward_masked(model, xb, t, kappa)

                if criterion == "confidence":
                    conf = torch.softmax(logits, dim=1).max(1).values  # [B]
                    if norm_mode == "z":
                        assert mu is not None and sigma is not None
                        score = ((conf - mu[t]) / (sigma[t] + eps)).mean()
                    elif norm_mode == "mean_ratio":
                        assert mu is not None
                        score = (conf / (mu[t] + eps)).mean()
                    elif norm_mode == "platt":
                        assert platt_a is not None and platt_b is not None
                        # Use calibrated *logits* for comparison (no need to sigmoid)
                        score = (platt_a[t] * conf + platt_b[t]).mean()
                    else:  # "none"
                        score = conf.mean()
                else:
                    score = batch_similarity(model, xb, t)

                if score > best_score:
                    best_logits = logits
                    best_score, best_t = score, t

            pred = best_logits.argmax(1)
            acc[task_ref] += (pred == yb).sum().item()
            cnt[task_ref] += yb.size(0)

    return [100.0 * a / c if c else 0. for a, c in zip(acc, cnt)]

