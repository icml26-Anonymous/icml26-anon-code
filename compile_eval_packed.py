# -*- coding: utf-8 -*-
"""
    • Works for any nn.Linear / nn.Conv2d that inherits the GatedLayer mix‑in.
    • One gate‑vector per (layer, task) stored inside the layer itself.
    • Forward pass receives a task_id and k_fraction; each gated layer masks
      its own activations on the fly.
    • Scaled Hebbian updates and usage statistics are computed layer‑wise.
"""
from __future__ import annotations

# -----------------------------------------------------------------
# Reward-sign modes
# -----------------------------------------------------------------
REWARD_BATCH = "batch"  # +1 if batch accuracy > 0.5 else −1
REWARD_SAMPLE = "sample"  # sign per sample, then average
REWARD_MARGIN = "margin"  # raw classification margin, no ±1 clippings

from datetime import datetime
from Models import *
from Datasets import *
from Tools import *
from router_utils import *
from Utils import *
from Helper_functions import *


@torch.no_grad()
def gate_overlap_per_layer(model: nn.Module, k_frac: float, hard: bool = True):
    """
    Returns:
      layers: list of GatedLayer modules in order
      overlaps: Tensor [L, T, T] with layer-wise overlaps
    """
    gated_layers = [m for m in model.modules() if isinstance(m, GatedLayer)]
    n_layers = len(gated_layers)
    n_tasks = len(gated_layers[0]._gate_bank)

    overlaps = torch.zeros(n_layers, n_tasks, n_tasks)
    for li, m in enumerate(gated_layers):
        masks = []
        for t in range(n_tasks):
            mask = apply_topk(m.gate_for(t), k_frac, hard=hard).bool()
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # [T, C]

        for i in range(n_tasks):
            for j in range(n_tasks):
                mi, mj = masks[i], masks[j]
                inter = (mi & mj).sum().float()
                k = mi.sum().float().clamp_min(1)
                overlaps[li, i, j] = inter / k
        print(overlaps[li, :, :])
    return gated_layers, overlaps


@torch.no_grad()
def build_union_gates_(
        model: nn.Module,
        k_frac: float,
        hard: bool = True,
        check_disjoint: bool = True,
        tol: float = 1e-6,
) -> Dict[nn.Module, torch.Tensor]:
    """
    Build a 'union' gate per GatedLayer by merging all task-specific gates.

    Assumes that after training:
      - each GatedLayer m has m._gate_bank = [g_t]_{t=0..T-1},
      - the corresponding hard masks are (almost) disjoint across tasks.

    Parameters
    ----------
    model : nn.Module
        Model containing GatedLayer modules.
    k_frac : float
        Fraction κ used for hard top-k when building masks.
    hard : bool
        If True, use hard top-k masks; if False, use clamped soft gates.
    check_disjoint : bool
        If True, asserts that the summed hard masks do not exceed 1+tol
        (i.e. no channel is selected by more than one task).
    tol : float
        Numerical tolerance for the disjointness check.

    Returns
    -------
    union_gates : Dict[nn.Module, torch.Tensor]
        Mapping from each GatedLayer to a 1D tensor g_union
        (same shape as per-task gates), representing the union gate.
        Under perfect disjointness, g_union ≈ sum_t g_t.
    """
    union_gates: Dict[nn.Module, torch.Tensor] = {}

    # Determine number of tasks from the first GatedLayer
    n_tasks = None
    for m in model.modules():
        if isinstance(m, GatedLayer):
            n_tasks = len(m._gate_bank)
            break

    if n_tasks is None:
        raise RuntimeError("No GatedLayer modules found in model.")

    for m in model.modules():
        if not isinstance(m, GatedLayer):
            continue

        # Stack all gates for this layer: [T, C]
        gates = torch.stack([m.gate_for(t) for t in range(n_tasks)], dim=0)

        if hard:
            # Convert each g_t to a binary mask via top-k
            hard_masks = []
            for t in range(n_tasks):
                mask_t = apply_topk(m.gate_for(t), k_frac, hard=True)  # [C]
                hard_masks.append(mask_t.bool())
            hard_masks = torch.stack(hard_masks, dim=0).to(gates.device)  # [T, C]

            # Check disjointness if requested
            if check_disjoint:
                # Each channel count = how many tasks select it
                counts = hard_masks.sum(dim=0).float()  # [C]
                if (counts > 1.0 + tol).any():
                    max_c = counts.max().item()
                    raise RuntimeError(
                        f"Union gates not disjoint for layer {m}: "
                        f"max channel count = {max_c} (>1)."
                    )

            # Union is OR of masks
            union_mask = hard_masks.any(dim=0).float()  # [C]
            union_g = union_mask  # binary union gate

        else:
            # Soft union: sum gates across tasks. Under disjointness,
            # each channel belongs to exactly one task, so this is safe.
            union_g = gates.sum(dim=0)

        union_gates[m] = union_g

    return union_gates


def gate_stats(model: nn.Module, kappa) -> None:
    counter = 0
    any_overlap = 0
    layers, O = gate_overlap_per_layer(model, k_frac=kappa)
    print(O.mean(0))  # should match gate_overlap_matrix
    for layer_o in O:
        for id, t_o in enumerate(layer_o):
            # print(t_o)
            if sum(t_o) > 1:
                print(id, t_o)  # layer-wise view
                any_overlap += 1
    if any_overlap == 0:
        print('No overlap between tasks!')


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
    print(f"Task {t_idx + 1}/{len(tasks_data)} - Test loss {sum(v_losses) / len(v_losses):.3f}")
    return mean_loss


def build_class_slices(n_tasks, classes_per_task, start=0):
    return {t: torch.arange(start + t * classes_per_task, start + (t + 1) * classes_per_task)
            for t in range(n_tasks)}


if __name__ == "__main__":
    device_n = 0
    CUDA_LAUNCH_BLOCKING = 1.
    torch.cuda.set_device(device_n)
    print(torch.cuda.current_device())

    from ResNet_comp_X import *
    from ResNet_Packed import *
    from compile_tools_Packed import *
    from ResNet_Slice import *
    from fused_packed_forward import *
    from compute_metrics import *

    norm = "in"
    per_task_bn = False
    seeds = [11]  # 195] #
    batch_size = 128

    kappa=0.1
    load_ = True
    compile_model = False
    VERIFY = False
    EVAL_FULL = True
    REPORT_COMPUTE = False
    REPORT_ACCURACY = True

    datasets = ['CIF100', 'imagenet']
    dataset = datasets[0]

    if dataset == 'imagenet':

        tasks_data = generate_split_tiny_imagenet(
            root_dir="ADD-PATH",
            n_tasks=10, batch_size=batch_size, num_workers=4,
            shuffle_classes=False, seed=seeds[0]
        )
        num_classes = 200
        epochs_ = [2]  # schedule knobs
        k_fractions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        class_indices_per_task = build_class_slices(n_tasks=10, classes_per_task=20, start=0)

    else:
        tasks_data = generate_split_cifar100(batch_size=batch_size, data_root="ADD-PATH", num_workers=0)
        num_classes = 100
        epochs_ = [2]  # 2, 5, 10, 20]
        k_fractions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        class_indices_per_task = build_class_slices(n_tasks=10, classes_per_task=10, start=0)

    num_tasks = len(tasks_data)
    device_eval = "cuda" if torch.cuda.is_available() else "cpu"
    time_code = datetime.now().strftime(
        "%H%M_%d%m%y")  # e.g. "1642_070325"

    model = GatedResNet18_Flex(
        num_classes=num_classes,
        kappa=kappa,
        stem=dataset,
        norm_type=norm,
        per_task_bn=per_task_bn,
        gate_skip=False,
        gate_head=False
    )
    model.set_class_slices(class_indices_per_task)
    model.configure_head_protection(freeze_old_cols=True,
                                        mask_non_current_logits=False)

    if load_:
        date_code = "0213_260126"
        model_name = f"Resnet_{dataset}_{date_code}"
        model = load_model_with_gates(model, f"checkpoints/{model_name}.pt",
                                      device="cuda")
        try:
            print(f'\nLoading Confidence statistics for model: {model_name}')
            mu, sigma, _ = load_mu_sigma(f"checkpoints/mu_sigma_hebbgate_in_{model_name}.pt", device="cuda")
            model.mu = mu
            model.sigma = sigma
        except:
            print(f"Confidence statistics (mu, sigma) for model: {model_name} not found!")
            print('Computing Confidence stats (mu, sigma) ...')
            mu, sigma = compute_conf_stats(model, tasks_data, n_tasks=num_tasks, kappa_targets=k_fractions, max_batches=5,
                                           device=device_eval)
            save_mu_sigma(f"checkpoints/mu_sigma_hebbgate_in_{model_name}.pt", mu, sigma, extra={"num_tasks": 10, "mode": "confidence"})
            print(f'Done - Saved at: checkpoints/mu_sigma_hebbgate_in_{model_name}.pt')
            model.mu = mu
            model.sigma = sigma

        if compile_model:
            model.eval_mode = False
            model = compile_hebbgate_for_single_pass(
                model,
                class_indices_per_task=class_indices_per_task,
                kappa=kappa,
                hard=True
            )

        # ----------------------------------------------------------------------------------------
        # COMPILING MODEL TO SUBNETWORK PACKED MODELS
        # ----------------------------------------------------------------------------------------
        print('\nBuilding wiring plans for packed models ...')
        packed_model = nn.ModuleList()
        plans_list = build_packed_wiring_plans(model, kappa, num_tasks, verbose=False)
        print('Compiling packed models:')
        for t, (_, test_loader) in enumerate(tasks_data):
            plan_t = plans_list[t]
            packed_t = build_packed_model_for_task(model, plan_t, class_indices_per_task, task_id=t, device="cuda")
            packed_model.append(packed_t)
            print(f'Packed model for Task: {t} Complete')


        print('\nFusing packed models to Parallel Runner ...')
        runner = ParallelPackedRunnerFunctorch(packed_model, mu=mu, sigma=sigma, device="cuda")
        print('Done!')

        # ----------------------------------------------------------------------------------------
        # REPORT COMPUTATIONAL PROTOCOLS (PARAMS / BYTES, MACs, TIME)
        # ----------------------------------------------------------------------------------------
        if REPORT_COMPUTE:
            for t, (_, test_loader) in enumerate(tasks_data):
                xb = next(iter(test_loader))[0]  #
                break
            report_protocols(model, packed_model, runner, xb, T=10)

        # ----------------------------------------------------------------------------------------
        # REPORT TASK AND CLASS INCREMENTAL LEARNING ACCURACIES
        # ----------------------------------------------------------------------------------------
        if REPORT_ACCURACY:
            print('\nParallel Fused PackedModel:')
            acc_packed_TIL = eval_TIL_packed(runner, tasks_data)
            packed_til_avg = sum(acc_packed_TIL) / len(acc_packed_TIL)
            print(
                f"PackedModel TIL avg={packed_til_avg:.2f}  per-task={format_per_task(acc_packed_TIL)}")
            til_avg = sum(acc_packed_TIL) / len(acc_packed_TIL)

            fused_packedcil_acc = fused_eval(runner, tasks_data)
            fused_packedcil_avg = sum(fused_packedcil_acc) / len(fused_packedcil_acc)
            print(
                f"PackedModel CIL (znorm) avg={fused_packedcil_avg:.2f}  per-task={format_per_task(fused_packedcil_acc)}")

            fused_packedcil_acc = fused_eval_standard(runner, tasks_data, norm_mode='mean')
            fused_packedcil_avg = sum(fused_packedcil_acc) / len(fused_packedcil_acc)
            print(
                f"PackedModel CIL (standard) avg={fused_packedcil_avg:.2f}  per-task={format_per_task(fused_packedcil_acc)}")

            if EVAL_FULL:
                print('\nSequential Full Model Performance:')
                acc_TIL = eval_TIL_comp(
                    model, tasks_data,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    kappa=0.1
                )
                til_avg = sum(acc_TIL) / len(acc_TIL)
                print(
                    f"FullModel TIL  avg={til_avg:.2f}  per-task={format_per_task(acc_TIL)}\n")
                til_avg = sum(acc_TIL) / len(acc_TIL)

                acc_CIL_znorm = eval_full_CIL_sequential(
                    model, tasks_data,
                    device=device_eval,
                    kappa_targets=k_fractions,
                    n_tasks=num_tasks,
                    dataset=dataset,
                    criterion="confidence",
                    norm_mode="z"
                )
                cil_znorm = acc_CIL_znorm
                cil_znorm_avg = sum(acc_CIL_znorm) / len(acc_CIL_znorm)
                print(
                    f"    CIL-znorm  avg={cil_znorm_avg:.2f}  per-task={format_per_task(acc_CIL_znorm)}")


        # ---------------------------------------------------------------------------------------
        # IN CASE THAT PERFORMANCE OF PACKED MODEL DOESNT SEEM RIGHT, THE FOLLOWING TESTS VERIFY
        # AT DIFFERENT STAGES IF THE COMPILING AND IMPLEMENTATION OF THE PACKED MODEL WAS
        # COMPLETED SUCCESSFULLY
        # ---------------------------------------------------------------------------------------
        if VERIFY:
            # Progressive Tests to Check equivalence before running
            for t, (_, test_loader) in enumerate(tasks_data):
                xb = next(iter(test_loader))[0]  # or any batch tensor [B,3,H,W]
                print(f"Task: {t}")
                check_plan_conv_slicing(model, plans_list[t], xb, task_id=t, kappa=kappa, device="cuda")

            for t in range(num_tasks):
                for op in plans_list[0][:6]:
                    print(op.name, op.kind,
                          None if op.in_idx is None else op.in_idx.numel(),
                          op.out_idx.numel())

            m = resolve_op_module(model, plans_list[0][0])  # stem
            print(m, m.weight.shape)

            m = resolve_op_module(model, plans_list[0][5])  # layer2.0.conv1 (for example)
            print(plans_list[0][5].name, m.weight.shape)

            preview = preview_sliced_weights_for_task(model, plans_list[0], task_id=0)
            for row in preview:
                print(row)

            for t, (_, test_loader) in enumerate(tasks_data):
                x = next(iter(test_loader))[0]  # or any batch tensor [B,3,H,W]

                plan_t = plans_list[t]

                io = capture_conv_ios_for_task(model, plan_t, x, task_id=t, kappa=kappa, hard=True, device="cuda")
                ok, results = test_layer_equivalence(model, plan_t, io, atol=1e-6, rtol=1e-4, verbose=True)

                print(f"Task: {t}")
                print("ALL OK:", ok)

            for t, (_, test_loader) in enumerate(tasks_data):
                print(f'Task: {t}')
                xb = next(iter(test_loader))[0]  # or any batch tensor [B,3,H,W]
                plan_t = plans_list[t]
                compare_layers(model, packed_model[t], plan_t, xb, t, kappa=kappa, hard=True, device="cuda")
                compare_task_logits(model, packed_model[t], xb, task_id=t, class_indices_per_task=class_indices_per_task,
                                    kappa=kappa, device="cuda")
