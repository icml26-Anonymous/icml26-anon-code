
from Utility_functions import *
from typing import Dict, List, Optional
import torch
import torch.nn as nn

@torch.no_grad()
def build_union_gates(
    model: nn.Module,
    kappa: float,
    hard: bool = False,
    tol_overlap: float = 1e-6,
    eps: float = 1e-12,
    merge: str = "max",          # , "sum", "mean"
) -> Dict[nn.Module, torch.Tensor]:
    """
    For each GatedLayer m, build a union gate vector g_union[m] such that:
      - support: g_union[j] > 0 if any task has g_t[j] > 0
      - values: keeps gate magnitudes via merge reduction across tasks

    If hard=True: g_union is binary {0,1} union support.
    If hard=False: g_union keeps magnitudes (merge over tasks).
    """

    gate_vals: Dict[nn.Module, List[torch.Tensor]] = {}
    n_tasks: Optional[int] = None

    for m in model.modules():
        if isinstance(m, GatedLayer):
            if n_tasks is None:
                n_tasks = len(m._gate_bank)
            else:
                assert n_tasks == len(m._gate_bank), "Inconsistent gate_bank lengths."

            g_list = []
            for t in range(n_tasks):
                g_t = m.gate_for(t)  # expected shape [C], float
                if hard:
                    g_t = (g_t > eps).to(g_t.dtype)  # binary mask but float dtype
                else:
                    # keep magnitudes, but optionally treat tiny values as zero
                    g_t = torch.where(g_t > eps, g_t, torch.zeros_like(g_t))
                g_list.append(g_t)

            gate_vals[m] = g_list

    assert n_tasks is not None, "No GatedLayer found in model."

    union_gates: Dict[nn.Module, torch.Tensor] = {}

    for m, g_list in gate_vals.items():
        # --- overlap check on support (non-zero positions) ---
        support_stack = torch.stack([(g > eps).float() for g in g_list], dim=0)  # [T, C]
        use_counts = support_stack.sum(dim=0)                                    # [C]
        frac_overlap = (use_counts > 1.0 + 1e-6).float().mean().item()

        if frac_overlap > tol_overlap:
            print(f"[WARN] Layer {m} has {frac_overlap*100:.2f}% channels "
                  f"used by >1 task under kappa={kappa}.")

        # --- union value merge ---
        if merge == "max":
            g_union = g_list[0].clone()
            for t in range(1, n_tasks):
                g_union = torch.maximum(g_union, g_list[t])
        elif merge == "sum":
            g_union = torch.zeros_like(g_list[0])
            for g in g_list:
                g_union = g_union + g
        elif merge == "mean":
            g_union = torch.zeros_like(g_list[0])
            for g in g_list:
                g_union = g_union + g
            g_union = g_union / float(n_tasks)
        else:
            raise ValueError(f"Unknown merge='{merge}'. Use 'max', 'sum', or 'mean'.")

        union_gates[m] = g_union

    return union_gates

def infer_task_assignments_from_gates(
    model: nn.Module,
    kappa: float,
    hard: bool = True
) -> Dict[nn.Module, torch.Tensor]:
    """
    Infer a task id for each channel in each GatedLayer, by looking at
    which task's gate selects that channel (hard top-k).

    Returns:
        task_id_per_layer: dict[layer] -> LongTensor [C] with task ids in {0,...,T-1}
                           or -1 for "never selected".
    """
    task_id_per_layer: Dict[nn.Module, torch.Tensor] = {}
    num_tasks: Optional[int] = None

    # First pass: determine num_tasks from any GatedLayer
    for m in model.modules():
        if isinstance(m, GatedLayer):
            if num_tasks is None:
                num_tasks = len(m._gate_bank)
            else:
                assert num_tasks == len(m._gate_bank), "Inconsistent gate_bank length across layers."

    if num_tasks is None:
        return task_id_per_layer  # no gated layers

    T = num_tasks

    # For each gated layer, build a [C] assignment
    l_cnt = 0
    for m in model.modules():
        if not isinstance(m, GatedLayer):
            continue

        C = m.gate_for(0).numel()
        # -1 means "never selected by any task"
        task_ids = torch.full((C,), -1, dtype=torch.long, device=m.gate_for(0).device)

        for t in range(T):

            g_t = m.gate_for(t)  # [C]
            # mask_t = g_t > 0  # boolean [C]
            mask_t = apply_topk(m.gate_for(t), kappa, hard=True).bool()
            # Where this task selects the channel, assign if still unassigned
            # print(f"Task: {t}")
            # print(f"g_t: {g_t}")
            # print(f"mask_t: {mask_t}")
            # print(f"task_ids: {task_ids}")
            newly = mask_t & (task_ids < 0)
            if newly.any():
                task_ids[newly] = t
            #
            # print(f"newly: {newly}")
            # print(f"new task_ids: {task_ids}")


        l_cnt += 1


        # Optional sanity: warn if some channels remain -1
        unused = (task_ids < 0).sum().item()
        if unused > 0:
            print(f"[compile] Layer: {m.layer_id} {m}: {unused} channels never selected by any task.")

        task_id_per_layer[m] = task_ids

    return task_id_per_layer

def compile_hebbgate_for_single_pass(
    model: nn.Module,
    class_indices_per_task: Dict[int, torch.Tensor],
    kappa: float,
    hard: bool = True,
    num_tasks: int = 10
) -> nn.Module:
    """
    Compile a trained HebbGate model into a single-pass inference network by:

    1) Inferring per-channel task assignments from the gate bank.
    2) Zeroing cross-task weights between consecutive gated layers.
    3) Restricting the classifier head rows to their task's channels.
    4) Optionally, you can later fold the union gate into BN/weights if desired.

    Args:
        model: trained model with GatedLayer modules and a single-head classifier.
        class_indices_per_task: mapping task_id -> tensor of class ids (global labels).
        kappa: final sparsity used in inference (e.g. 1 / num_tasks).
        hard: whether gates were applied as hard top-k (True).

    Returns:
        The same model object, modified in-place.
    """
    compile_head = True
    compile_backbone = True
    model.eval()  # just to be safe

    # ---------------------------------------------------------------------
    # 1. Infer task assignment per channel in each gated layer
    # ---------------------------------------------------------------------
    # Helper to get a stable ordered list of gated layers in forward order
    gated_layers: List[nn.Module] = []
    skip_layers: List[nn.Module] = []

    for m in model.modules():
        if isinstance(m, GConv2d):
            gated_layers.append(m)
        elif isinstance(m, nn.Conv2d):
            print(m)
            skip_layers.append(m)

    task_ids_per_layer = infer_task_assignments_from_gates(model, kappa=kappa, hard=hard)

    # ----------------------------------------------------------------------------------------------
    # 2. Zero cross-task weights between consecutive gated layers and appropriate skip connections.
    # ----------------------------------------------------------------------------------------------
    if compile_backbone:
        for pair_nmbr, (prev, nxt) in enumerate(zip(gated_layers[:-1], gated_layers[1:])):

            W = nxt.weight
            C_out, C_in = W.size(0), W.size(1)
            if getattr(prev, "out_channels", None) != C_in:
                continue
            print(f"Pair Number: {pair_nmbr}")
            print(f"Prev Layer: {prev.layer_id}")
            print(f"Next Layer: {nxt.layer_id}")

            # zero_nonoverlap_weights(prev, nxt, task_ids_per_layer, num_tasks)
            before = (nxt.weight != 0).float().mean().item()
            zero_cross_task_weights(prev, nxt, task_ids_per_layer)
            after = (nxt.weight != 0).float().mean().item()
            print("density:", before, "->", after)

        # # zero skip connections
        zero_cross_task_skipweights(gated_layers[4], gated_layers[6], skip_layers[0], task_ids_per_layer)
        zero_cross_task_skipweights(gated_layers[8], gated_layers[10], skip_layers[1], task_ids_per_layer)
        zero_cross_task_skipweights(gated_layers[12], gated_layers[14], skip_layers[2], task_ids_per_layer)


    # ---------------------------------------------------------------------
    # 3. Restrict Classifier head
    # ---------------------------------------------------------------------
    if compile_head:
        keep_unassigned = False
        lastGconv = gated_layers[-1]
        W_head = model.fc.weight.data  # [num_classes, C_last]
        task_in = task_ids_per_layer[lastGconv].to(W_head.device)
        num_tasks = len(class_indices_per_task)
        C_last = W_head.size(1)

        cls_ids = torch.zeros(W_head.shape[0]).to(W_head.device)
        for t in range(num_tasks):
            task_ids = class_indices_per_task[t].to(W_head.device)
            # Ensure cls_ids is a LongTensor of indices, not a boolean mask
            if task_ids.dtype == torch.bool:
                task_ids = task_ids.nonzero(as_tuple=False).view(-1)

            cls_ids[task_ids] = t

        same_task = (cls_ids[:, None] == task_in[None, :])  # [C_out, C_in]

        if keep_unassigned:
            unassigned = (cls_ids[:, None] == -1) | (task_in[None, :] == -1)
            keep = same_task | unassigned
        else:
            keep = same_task

        # broadcast keep to weight shape and zero the rest
        if W_head.dim() == 2:
            W_head.mul_(keep.to(dtype=W_head.dtype))
        elif W_head.dim() == 4:
            W_head.mul_(keep.to(dtype=W_head.dtype).unsqueeze(-1).unsqueeze(-1))
        else:
            raise ValueError(f"Unsupported weight dim: {W_head.dim()}")

    # Pass Gates to Weights
    # fold_union_gates_into_conv_weights(model, num_tasks, task_ids_per_layer)
    # union_gates(model, num_tasks, task_ids_per_layer)

    return model



@torch.no_grad()
def zero_cross_task_weights(prev: nn.Module,
                            nxt: nn.Module,
                            task_ids_per_layer: dict,
                            keep_unassigned: bool = False):
    """
    Zero weights in nxt that connect prev-channels of one task to nxt-channels of another task.

    task_ids_per_layer[m]: Tensor [C] with values in {-1, 0..T-1}
      - For prev: C == C_in of nxt
      - For nxt:  C == C_out of nxt
    """
    W = nxt.weight  # Linear: [C_out, C_in], Conv: [C_out, C_in, kH, kW]

    task_in  = task_ids_per_layer[prev].to(device=W.device)  # [C_in]
    task_out = task_ids_per_layer[nxt].to(device=W.device)   # [C_out]

    # sanity checks
    assert task_in.dim() == 1 and task_out.dim() == 1
    assert task_in.numel() == W.size(1), f"task_in has {task_in.numel()} but W has C_in={W.size(1)}"
    assert task_out.numel() == W.size(0), f"task_out has {task_out.numel()} but W has C_out={W.size(0)}"

    same_task = (task_out[:, None] == task_in[None, :])  # [C_out, C_in]

    if keep_unassigned:
        unassigned = (task_out[:, None] == -1) | (task_in[None, :] == -1)
        keep = same_task | unassigned
    else:
        keep = same_task

    # broadcast keep to weight shape and zero the rest
    if W.dim() == 2:
        W.mul_(keep.to(dtype=W.dtype))
    elif W.dim() == 4:
        W.mul_(keep.to(dtype=W.dtype).unsqueeze(-1).unsqueeze(-1))
    else:
        raise ValueError(f"Unsupported weight dim: {W.dim()}")

    return keep  # optional: return mask for debugging


@torch.no_grad()
def zero_cross_task_skipweights(prev: nn.Module,
                            nxt: nn.Module,
                            skip: nn.Module,
                            task_ids_per_layer: dict,
                            keep_unassigned: bool = True):
    """
    Zero weights in skip that correspond to connected prev-channels of one task to nxt-channels of another task.

    task_ids_per_layer[m]: Tensor [C] with values in {-1, 0..T-1}
      - For prev: C == C_in of nxt
      - For nxt:  C == C_out of nxt
      - For skip: C == C_out of nxt
    """
    print(f'Prev: {prev.layer_id}, Next: {nxt.layer_id}')
    W = skip.weight  # Linear: [C_out, C_in], Conv: [C_out, C_in, kH, kW]

    task_in  = task_ids_per_layer[prev].to(device=W.device)  # [C_in]
    task_out = task_ids_per_layer[nxt].to(device=W.device)   # [C_out]

    # sanity checks
    assert task_in.dim() == 1 and task_out.dim() == 1
    assert task_in.numel() == W.size(1), f"task_in has {task_in.numel()} but W has C_in={W.size(1)}"
    assert task_out.numel() == W.size(0), f"task_out has {task_out.numel()} but W has C_out={W.size(0)}"

    same_task = (task_out[:, None] == task_in[None, :])  # [C_out, C_in]

    if keep_unassigned:
        unassigned = (task_out[:, None] == -1) | (task_in[None, :] == -1)
        keep = same_task | unassigned
    else:
        keep = same_task

    # broadcast keep to weight shape and zero the rest
    if W.dim() == 2:
        W.mul_(keep.to(dtype=W.dtype))
    elif W.dim() == 4:
        W.mul_(keep.to(dtype=W.dtype).unsqueeze(-1).unsqueeze(-1))
    else:
        raise ValueError(f"Unsupported weight dim: {W.dim()}")

    return keep  # optional: return mask for debugging



@torch.no_grad()
def union_gates(model: nn.Module, num_tasks: int, task_ids_per_layer: dict, eps: float = 1e-6):
    FOLD = True
    for m in model.modules():
        if isinstance(m, GConv2d):
            if m.layer_id in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 18, 16]:
                # [T, C_out]
                G = torch.stack([m._gate_bank[t].to(m.weight.device) for t in range(num_tasks)], dim=0)

                g_union = G.max(dim=0).values
                if FOLD:
                    if (g_union - G.sum(0)).sum() > 0:
                        print('Warning: Overlap detected')
                    mask = torch.where(g_union > eps, torch.ones_like(g_union), torch.zeros_like(g_union))
                    g_union = torch.where(g_union > 0.001, g_union, torch.zeros_like(g_union))

                    # fold into weights (output gating => scale output channels)
                    m.weight.mul_(g_union.view(-1, 1, 1, 1))
                    # optional: after folding, set all gates to 1 so forward masking does nothing
                    # ones = torch.ones_like(g_union)
                    for t in range(num_tasks):
                        m._gate_bank[t] = mask.clone()  # ones.clone()  #g_union # ones.clone()
                else:
                    for t in range(num_tasks):
                        m._gate_bank[t] = g_union.clone()  #ones.clone()  #g_union # ones.clone()

            # exit()

@torch.no_grad()
def fold_union_gates_into_conv_weights(model: nn.Module, num_tasks: int, eps: float = 1e-6):

    for m in model.modules():
        if isinstance(m, GConv2d):
            if True:
                # [T, C_out]
                G = torch.stack([m._gate_bank[t].to(m.weight.device) for t in range(num_tasks)], dim=0)


                # union (keep magnitudes)
                g_union = G.max(dim=0).values  # [C_out]
                if (g_union - G.sum(0)).sum() > 0:
                    print('Warning: Overlap detected')
                # mask = torch.where(g_union > eps, torch.ones_like(g_union), torch.zeros_like(g_union))
                g_union = torch.where(g_union > 0.001, g_union, torch.zeros_like(g_union))

                # fold into weights (output gating => scale output channels)
                # m.weight.mul_(g_union.view(-1, 1, 1, 1))

                # optional: after folding, set all gates to 1 so forward masking does nothing
                ones = torch.ones_like(g_union)
                for t in range(num_tasks):
                    m._gate_bank[t] = g_union.clone()  #ones.clone()  #g_union # ones.clone()



def eval_TIL_comp(model: nn.Module, tasks_data, device="cpu", kappa=0.1):
    accs = []

    n_tasks = len(tasks_data)

    for t, (_, test_loader) in enumerate(tasks_data):
        accs.append(eval_task(model, test_loader, t, device, kappa))
    return accs


def eval_TIL_single(model: nn.Module, tasks_data, device="cpu", kappa_targets=None, k_decay=0.2, up2task=5):
    accs = []

    n_tasks = len(tasks_data)
    k_frac = k_decay
    for t, (_, test_loader) in enumerate(tasks_data):
        accs.append(eval_task(model, test_loader, 3, device, k_frac))
    return accs

