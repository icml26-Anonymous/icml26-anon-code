from typing import Dict, List, Tuple
from Utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility functions (top‑k mask, Hebbian update, usage)
# -----------------------------------------------------------------------------

def apply_topk(g: torch.Tensor, k_fraction: float, hard: bool = True) -> torch.Tensor:
    """Return binary or soft mask built from gating vector g."""
    if not hard:
        return torch.clamp(g, 0.0, 1.0)
    k = max(1, int(k_fraction * g.numel()))
    top_idx = torch.topk(g, k).indices
    mask = torch.zeros_like(g)
    mask[top_idx] = 1.0
    return mask


def scaled_hebbian_update(
    g: torch.Tensor,
    reward: torch.Tensor,
    usage: torch.Tensor,
    *,
    lr: float = 1e-2,
    alpha: float = 1.0,
    beta: float = 1.0,
    scale_mode: str = "linear",   # <-- new arg
    gamma: float = 2.0,  # new arg for power mode
    task_idx: int = 1
) -> torch.Tensor:
    """Hebbian update with selectable usage‑penalty scale."""
    if scale_mode == "linear":
        scale = torch.clamp(1.0 - beta * usage, 0.0, 1.0)

    elif scale_mode == "power":  # (1‑u)^γ    -- HAT‑style squeeze
        scale = (1.0 - usage).clamp(min=0.0) ** gamma

    elif scale_mode == "exp":
        scale = torch.exp(-beta * usage)

    elif scale_mode == "inverse":
        scale = 1.0 / (1.0 + beta * usage)

    elif scale_mode == "step":
        scale = (usage < 1.0 / beta).float()

    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    scaled_r = reward * scale
    delta = lr * (scaled_r - alpha * scaled_r.mean())

    # return torch.clamp(g + delta, 0.0, 3.0)
    return torch.clamp(g + delta, 0.0, 1.0)
    # return g + delta



def gather_usage(model: nn.Module, t_done: int) -> Dict[nn.Module, torch.Tensor]:
    """Sum gate activations over completed tasks (0..t_done‑1)."""
    usage: Dict[nn.Module, torch.Tensor] = {}
    for m in model.modules():
        if isinstance(m, GatedLayer):
            if t_done == 0:
                usage[m] = torch.zeros_like(m.gate_for(0))
            else:
                g_prev = torch.stack([m.gate_for(t) for t in range(t_done)], 0)
                usage[m] = g_prev.mean(0)
    return usage

def init_all_gates(model: nn.Module, num_tasks: int) -> None:
    for m in model.modules():
        if isinstance(m, GatedLayer):
            for _ in range(num_tasks):
                m.new_task_gate()


class GatedLayer:  # mix‑in -----------------------------------------------------
    """Stores task‑specific gating vectors and exposes helper methods."""

    _gate_bank: List[torch.Tensor]
    _dim: int

    def register_gate(self, dim: int, init_low: float = 0.4, init_high: float = 0.6,
                      device: str = "cpu") -> None:
        self._gate_bank = []
        self._dim = dim
        self._init_low, self._init_high = init_low, init_high
        self._device = device

    # ---------------------------------------------------------------------
    # def new_task_gate(self) -> torch.Tensor:
    #     g = (torch.rand(self._dim, device=self._device)
    #           * (self._init_high - self._init_low) + self._init_low)
    #     self._gate_bank.append(g)
    #     return g

    # ---------------------------------------------------------------------
    def gate_for(self, t: int) -> torch.Tensor:
        return self._gate_bank[t]

    def new_task_gate(self, usage=None, kappa=0.15, eps=0.2, theta=0.2):
        device = self.weight.device  # ← current device of the layer
        if usage is None:
            g = torch.rand(self._dim, device=device) * 0.2 + 0.4

        else:
            # print(usage)
            free = (usage < theta)
            # print(usage)
            # print(free)
            eps = torch.clamp(1 - usage, 0, kappa)

            g = torch.empty(self._dim, device=device)
            g[free] = torch.rand(free.sum(), device=device) * 0.3 + 0.7
            # (2) Heavily-used channels: scale by their own eps_j
            rand_vals = torch.rand((~free).sum(), device=device)
            g[~free] = rand_vals * eps[~free]  # ← element-wise product

        self._gate_bank.append(g)
        return g


# Gated linear and convolutional layers ---------------------------------------
class GLinear(nn.Linear, GatedLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GatedLayer.register_gate(self, self.out_features)


class GConv2d(nn.Conv2d, GatedLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        GatedLayer.register_gate(self, self.out_channels)
        self.layer_id = None


# -----------------------------------------------------------------------------
#  Minimal evaluation utilities
# -----------------------------------------------------------------------------

def eval_task(model: nn.Module, loader, task_id: int, device: str = "cpu", k_frac: float = 0.3) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, task_id=task_id, k_frac=k_frac)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total

def eval_task_CIF100(model: nn.Module, loader, task_id: int, device: str = "cpu", k_frac: float = 0.3, oracle_mask: bool = True) -> float:
    model.eval()
    correct, total = 0, 0
    TASK_CLASSES = [list(range(i, i + 10)) for i in range(0, 100, 10)]
    # TASK_CLASSES = [list(range(i, i + 2)) for i in range(0, 10, 2)]
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, task_id=task_id, k_frac=k_frac)

            # ---------- ORACLE MASKING ----------
            if oracle_mask:
                mask = torch.full_like(logits, -1e9)  # –∞
                cls = TASK_CLASSES[task_id]
                mask[:, cls] = 0  # keep current
                logits = logits + mask
            # --------------------------------------

            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return 100.0 * correct / total


def eval_all_runtime(model: nn.Module, tasks_data, device="cpu", k_frac=0.3, k_decay=0.2, up2task=5):
    accs = []
    for t, (_, test_loader) in enumerate(tasks_data):
        if t < up2task:
            accs.append(eval_task(model, test_loader, t, device, k_frac))
        elif t == up2task:
            accs.append(eval_task(model, test_loader, t, device, k_decay))
    return accs

def eval_all(model: nn.Module, tasks_data, device="cpu", k_frac=0.3, up2task=5):
    accs = []
    for t, (_, test_loader) in enumerate(tasks_data):
        if t <= up2task:
            accs.append(eval_task(model, test_loader, t, device, k_frac))
 
    return accs


def eval_all_CIF100(model: nn.Module, tasks_data, device="cpu", k_frac=0.3, up2task=5):
    accs = []
    for t, (_, test_loader) in enumerate(tasks_data):
        if t <= up2task:
            accs.append(eval_task_CIF100(model, test_loader, t, device, k_frac))
    return accs


# --------------------------------------------------------------
# Gate‑overlap analysis
# --------------------------------------------------------------

@torch.no_grad()
def gate_overlap_per_layer(
    model: nn.Module,
    k_frac: float,
    hard: bool = True
) -> Tuple[List[nn.Module], torch.Tensor]:
    """
    Returns
    -------
    layers : list[nn.Module]
        Ordered list of the gated layers (to label rows).
    overlap : torch.Tensor  [L, T, T]
        overlap[ℓ, i, j]  =  |mask_ℓ^i  ∧  mask_ℓ^j| / |mask_ℓ^i|
        where L = #gated layers, T = #tasks.
    """
    # 1)  Gather masks for every (layer, task) pair
    layers: List[nn.Module] = []
    masks: Dict[nn.Module, List[torch.Tensor]] = {}   # layer → list[T] Bool

    n_tasks = None
    for m in model.modules():
        if isinstance(m, GatedLayer):
            if n_tasks is None:
                n_tasks = len(m._gate_bank)
            layers.append(m)
            masks[m] = []
            for t in range(n_tasks):
                mask = apply_topk(m.gate_for(t), k_frac, hard=hard).bool()
                masks[m].append(mask)

    # 2)  Build layer‑wise overlap tensor
    n_layers = len(layers)
    overlap = torch.zeros(n_layers, n_tasks, n_tasks)

    for l, m in enumerate(layers):
        for i in range(n_tasks):
            mi = masks[m][i]
            k = mi.sum().float()                      # size of mask_i
            for j in range(i, n_tasks):
                inter = (mi & masks[m][j]).sum().float()
                score = inter / k                     # asymmetric IoU
                overlap[l, i, j] = overlap[l, j, i] = score

    return layers, overlap        # shape  (L, T, T)


@torch.no_grad()
def gate_overlap_matrix(
    model: nn.Module,
    k_frac: float,
    hard: bool = True
) -> torch.Tensor:
    """
    Returns
    -------
    overlap : torch.Tensor  [T, T]
        overlap[i,j] = mean_layer  |mask_i ∧ mask_j| / |mask_i|
                      where |·| is the 0‑norm (count of ones).
        Diagonal entries are 1.0 by definition.
    """
    # 1.  Collect gate masks per task & per layer
    bank: List[Dict[nn.Module, torch.Tensor]] = []   # length = T
    n_tasks = None
    for m in model.modules():
        if isinstance(m, GatedLayer):
            if n_tasks is None:
                n_tasks = len(m._gate_bank)
                bank = [dict() for _ in range(n_tasks)]
            for t in range(n_tasks):
                mask = apply_topk(m.gate_for(t), k_frac, hard=hard)
                bank[t][m] = mask.bool()           # store as boolean

    # 2.  Compute pair‑wise overlaps
    overlap = torch.zeros(n_tasks, n_tasks)
    for i in range(n_tasks):
        for j in range(i, n_tasks):
            layer_frac = []
            for m in bank[i].keys():               # same keys for all tasks
                mi, mj = bank[i][m], bank[j][m]
                inter = (mi & mj).sum().float()
                k = mi.sum().float()               # = κ·d_ell
                layer_frac.append(inter / k)       # IoU with mask_i as denom
            score = torch.stack(layer_frac).mean() # average over layers
            overlap[i, j] = overlap[j, i] = score
    return overlap


def freeze_norm_layers(model: nn.Module, freeze_affine: bool = True) -> None:
    """
    After task 0:
      • BN: use eval() (freezes running mean/var), and optionally freeze affine params.
      • GN: no running stats; optionally freeze affine params for parity.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()  # freeze running stats
            if freeze_affine:
                if m.weight is not None: m.weight.requires_grad_(False)
                if m.bias   is not None: m.bias.requires_grad_(False)
        elif isinstance(m, nn.GroupNorm):
            if freeze_affine:
                if m.weight is not None: m.weight.requires_grad_(False)
                if m.bias   is not None: m.bias.requires_grad_(False)

def set_bn_eval_only(model: nn.Module) -> None:
    """If you want to keep model.train() but force only BN layers to eval()."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
