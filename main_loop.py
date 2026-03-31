# -*- coding: utf-8 -*-
""" • Works for any nn.Linear / nn.Conv2d that inherits the GatedLayer mix‑in.
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
REWARD_MARGIN = "margin"  # raw classification margin, no ±1 clipping

from datetime import datetime
from Models import *
from ResNet_comp_X import *
from Datasets import *
from Tools import *
from router_utils import *
from Utils import *
from Helper_functions import *


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


def train_tasks(
        model: nn.Module,
        tasks_data: List[Tuple],
        epochs_list: int = 1,
        device: str = "cpu",
        lr: float = 1e-3,
        g_lr: float = 1e-2,
        kappa_targets=None,
        beta: float = 1.0,
        reward_mode: str = REWARD_BATCH,
        scale_mode: str = "linear",
        gating_mode: str = "hard",
        gate_update_rate: int = 10,
        conf: int = 1,
        kappa_peak=0.5,
        kappa_update_rate=1.0,
        csv_path=None
    ):

    model.to(device)

    if epochs_list == 1:
        epochs_weights = 15
        epochs_parallel = 15
        epochs_gates = 0
    elif epochs_list == 2:
        epochs_weights = 25
        epochs_parallel = 55
        epochs_gates = 0
    else:
        epochs_weights = 1
        epochs_parallel = 1
        epochs_gates = 0

    phases = [("parallel", epochs_parallel),
              ("weights",  epochs_weights),
              ("gates",    epochs_gates)]
    total_epochs = epochs_parallel + epochs_weights  # used for deciding when to run CIL eval

    total_epochs = epochs_parallel + epochs_weights

    # LR scheduler state
    model.lr = lr
    patience = model.lr_patience
    orig_lr = model.lr

    # ▼ Move all stored gates to the same device as the model
    for m in model.modules():
        if isinstance(m, GatedLayer):
            m._gate_bank = [g.to(device) for g in m._gate_bank]

    phases = [("parallel", epochs_parallel), ("weights", epochs_weights),
              ("gates", epochs_gates)]

    # CSV logging setup
    csv_fields = [
        "task_idx", "phase", "epoch_in_phase", "global_epoch",
        "kappa", "train_loss", "val_loss", "lr",
        "til_avg", "til_per_task",
        "cil_stand_avg", "cil_stand_per_task",
        "cil_znorm_avg", "cil_znorm_per_task"
    ]
    csv_file, csv_writer = init_csv_writer(csv_path, csv_fields)

    epoch_logs = []
    decay_start = 5  # unchanged

    for t_idx, (train_loader, test_loader) in enumerate(tasks_data):
        epoch_counter = 0  # global across phases and tasks
        k_target = kappa_targets[t_idx]
        best_loss = float("inf")
        theta = 1.0 / (t_idx + 1)  # usage threshold, unchanged

        # Reset optimizer at start of each task
        model.optimizer = model._get_optimizer(orig_lr)

        # ▼ create a fresh gate for the *next* task, using usage so far
        for m in model.modules():  # <‑‑ replace whole block
            if isinstance(m, GatedLayer):
                if t_idx >= len(m._gate_bank):  # layer‑specific length
                    usage_vec = gather_usage(model, t_idx)[m] if t_idx > 0 else None

                    # print(usage_vec)
                    m.new_task_gate(usage_vec, kappa=k_target, theta=theta)

        usage = gather_usage(model, t_idx)  # still returns tensors on correct device

        # kappa schedule start for this task
        capacity_left = (num_tasks - t_idx) / num_tasks
        k_start = min(capacity_left, kappa_peak)
        epoch_counter = 0
        for phase, n_epochs in phases:
            if n_epochs == 0:  # user can pass 0 to skip a phase
                continue

            if phase == "weights":
                set_requires_grad(model, True)  # θ trainable
                print('Freezing Gates for Task {}'.format(t_idx))
                freeze_current_task_gates(model, t_idx, k_target)

            elif phase == "parallel":
                set_requires_grad(model, True)  # θ trainable
            else:  # "gates"
                set_requires_grad(model, False)  # θ frozen

            for epoch in range(n_epochs):
                model.train()
                losses = []
                epoch_counter += 1

                for step, (xb, yb) in enumerate(train_loader):

                    hebb_allowed = (phase == "parallel" and step % gate_update_rate == 0 and step < (
                            len(train_loader) - (gate_update_rate * 2))) or (phase == "gates")
                    # 
                    if hebb_allowed and step % (gate_update_rate * kappa_update_rate) == 0:
                        k_decay = compute_kappa_decay(
                            epoch_counter,
                            decay_start,
                            k_start,
                            k_target,
                            epochs_parallel,
                            step,
                            len(train_loader)
                        )
                        print(k_decay)

                    apply_kappa_decay_and_renorm(model, t_idx, k_decay)
                    model.kappa = k_decay
                    #

                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb, task_id=t_idx, k_frac=k_decay)
                    loss = F.cross_entropy(logits, yb)
                    # ---- weight update ----------------------------------------
                    if phase != "gates":  # skip optimiser when θ frozen
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()

                    # ---- Hebbian gate update ----------------------------------
                    if hebb_allowed:
                        # -------- Hebbian gate updates per layer --------
                        with torch.no_grad():
                            # -------- 2-a  compute SIGN per chosen mode ------------
                            if reward_mode == REWARD_BATCH:
                                sign_scalar = 1.0 if (logits.argmax(1) == yb).float().mean() > 0.5 else -1.0
                            elif reward_mode == REWARD_SAMPLE:
                                sign_scalar = (2.0 * (logits.argmax(1) == yb).float() - 1.0).mean()  # ∈[-1,+1]
                            else:  # margin
                                gt_logits = logits.gather(1, yb.view(-1, 1)).squeeze()
                                max_oth = logits.clone()
                                max_oth[torch.arange(len(yb)), yb] = -1e9
                                margin = (gt_logits - max_oth.max(1).values).mean()
                                sign_scalar = torch.tanh(margin)  # tensor on correct device

                            # -------- 2-b  compute layer-wise rewards --------------
                            layer_acts = model.last_acts

                            for layer, (m, A) in enumerate(layer_acts.items()):  # m is the layer obj
                                if isinstance(m, GConv2d):
                                    # print(layer)
                                    if A.dim() == 4:  # [B, C, H, W]
                                        msq = (A ** 2).mean((0, 2, 3))  # per‑channel energy
                                    else:  # [B, C]
                                        msq = (A ** 2).mean(0)
                                    msq /= msq.mean() + 1e-8
                                    mask = apply_topk(m.gate_for(t_idx), k_decay, hard=True)
                                    reward = mask * (msq * sign_scalar)
                                    new_g = scaled_hebbian_update(
                                        m.gate_for(t_idx), reward, usage[m],
                                        lr=g_lr, beta=beta, scale_mode=scale_mode, task_idx=t_idx)

                                    m._gate_bank[t_idx] = new_g

                    losses.append(loss.item())

                train_loss = float(sum(losses) / max(1, len(losses)))

                # ----------------- Validation loss -----------------
                test_loss = valid_loss(model, test_loader, t_idx, k_decay, device)

                lr = model.lr
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model = get_model(model)
                    patience = model.lr_patience
                    # print(' *',end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        lr /= model.lr_factor
                        print(' lr={:.1e}'.format(lr), end='')
                        if lr < model.lr_min:
                            print()
                        else:
                            patience = model.lr_patience
                            model.optimizer = model._get_optimizer(lr)
                            model.lr = lr


                # ----------------- TIL & CIL metrics -----------------
                acc_TIL = eval_TIL(
                    model, tasks_data,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    kappa_targets=kappa_targets,
                    k_decay=k_decay,
                    up2task=t_idx
                )
                til_avg = sum(acc_TIL) / len(acc_TIL)

                # Pretty console log for this epoch
                header = (f"[Task {t_idx + 1}/{num_tasks} | "
                          f"Phase {phase:^8} | Epoch {epoch + 1:03d}/{n_epochs:03d}]")
                print(
                    f"{header}  "
                    f"train_loss={train_loss:.3f}  val_loss={test_loss:.3f}  "
                    f"kappa={k_decay:.3f}  lr={model.lr:.2e}"
                )
                print(f"    TIL  avg={til_avg:.2f}  per-task={format_per_task(acc_TIL)}")

                # If last epoch for this task (across phases) → run CIL evals
                cil_stand_avg = cil_znorm_avg = None
                cil_stand = cil_znorm = None

                if total_epochs == epoch_counter:
                    print('last_epoch')
                    acc_CIL_stand = eval_CIL_ptKappa(
                        model, tasks_data, n_tasks=t_idx + 1,
                        device="cuda", kappa_targets=kappa_targets
                    )
                    cil_stand = acc_CIL_stand
                    cil_stand_avg = sum(acc_CIL_stand) / len(acc_CIL_stand)

                    if t_idx != 0:
                        acc_CIL_znorm = eval_CIL_ptKappa_znorm(
                            model, tasks_data,
                            device=device_eval,
                            kappa_targets=kappa_targets,
                            n_tasks=t_idx + 1,
                            dataset=dataset,
                            criterion="confidence",
                            norm_mode="z",
                            max_batches=20
                        )
                    else:
                        acc_CIL_znorm = acc_CIL_stand

                    cil_znorm = acc_CIL_znorm
                    cil_znorm_avg = sum(acc_CIL_znorm) / len(acc_CIL_znorm)

                    print(f"    CIL-stand  avg={cil_stand_avg:.2f}  per-task={format_per_task(acc_CIL_stand)}")
                    print(f"    CIL-znorm  avg={cil_znorm_avg:.2f}  per-task={format_per_task(acc_CIL_znorm)}")

                    msg_fin = f"Task: {t_idx + 1}/{num_tasks} - completed"
                    epoch_logs.extend([
                        header, msg_fin,
                        f"TIL | avg={til_avg:.2f} | per-task={format_per_task(acc_TIL)}",
                        f"CIL-stand | avg={cil_stand_avg:.2f} | per-task={format_per_task(acc_CIL_stand)}",
                        f"CIL-znorm | avg={cil_znorm_avg:.2f} | per-task={format_per_task(acc_CIL_znorm)}"
                    ])
                else:
                    epoch_logs.extend([
                        header,
                        f"TIL | avg={til_avg:.2f} | per-task={format_per_task(acc_TIL)}"
                    ])

                # ----------------- CSV logging for this epoch -----------------
                if csv_writer is not None:
                    row = dict(
                        task_idx=t_idx,
                        phase=phase,
                        epoch_in_phase=epoch + 1,
                        global_epoch=epoch_counter,
                        kappa=k_decay,
                        train_loss=train_loss,
                        val_loss=test_loss,
                        lr=model.lr,
                        til_avg=til_avg,
                        til_per_task=format_per_task(acc_TIL),
                        cil_stand_avg=cil_stand_avg if cil_stand_avg is not None else "",
                        cil_stand_per_task=format_per_task(cil_stand) if cil_stand is not None else "",
                        cil_znorm_avg=cil_znorm_avg if cil_znorm_avg is not None else "",
                        cil_znorm_per_task=format_per_task(cil_znorm) if cil_znorm is not None else "",
                    )
                    csv_writer.writerow(row)
                    csv_file.flush()

        # # # end of task 0 training …
        if t_idx == 0:
            freeze_norm_layers(model, freeze_affine=True)

        if t_idx > 0:
            overlap = gate_overlap_matrix(model, k_target)
            print(f"\nMask overlap up to task{t_idx}:")
            M = overlap[:t_idx + 1, :t_idx + 1]
            print(M)

    return acc_TIL, acc_CIL_stand, acc_CIL_znorm, epoch_logs


def build_class_slices(n_tasks, classes_per_task, start=0):
    return {t: torch.arange(start + t * classes_per_task, start + (t + 1) * classes_per_task)
            for t in range(n_tasks)}


if __name__ == "__main__":
    device_n = 0
    CUDA_LAUNCH_BLOCKING = 1.
    torch.cuda.set_device(device_n)
    print(torch.cuda.current_device())

    norm = "in"
    per_task_bn = False
    seeds = [11]  # , , 1952, 30, 7, 1961] # 13, 22
    batch_size = 64

    save = True
    load_ = False
    train_ = True
    save_model = True
    eval_speed = False

    # save = False
    # load_ = True
    # train_ = False
    # save_model = False
    # eval_speed = True

    architectures = ['Resnet', 'WideCNN']
    datasets = ['CIF100', 'imagenet']
    #
    counter = 0
    conf = 1

    gate_update_rates = [10]
    kappa_update_rates = [1]
    reward_modes = [REWARD_MARGIN]
    scale_modes = ['power', 'linear', 'exp']  # 'linear', , 'power'
    gating_modes = ['hard', 'soft']    #  ['hard', 'soft']
    beta = 2

    lrs = [0.001, 0.0005, 0.01] #,

    gating_LRs = [0.0003, 0.0005] 
    kappa_peak = 0.5

    for dataset in datasets:
        if dataset == 'CIF10':
            tasks_data = generate_split_cifar10(batch_size=batch_size, data_root="./data", num_workers=0)
            num_classes = 10
            epochs_ = [3]
            k_fractions = [0.2, 0.2, 0.2, 0.2, 0.2]  

            class_indices_per_task = build_class_slices(n_tasks=5, classes_per_task=2, start=0)
        elif dataset == 'imagenet':
            tasks_data = generate_split_tiny_imagenet(
                root_dir="./../data/tiny-imagenet-200",
                n_tasks=10, batch_size=batch_size, num_workers=4,
                shuffle_classes=False, seed=seeds[0]
            )
            num_classes = 200
            epochs_ = [2]  # your schedule knobs
            k_fractions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            class_indices_per_task = build_class_slices(n_tasks=10, classes_per_task=20, start=0)
        elif dataset == 'IMNET100':
            tasks_data = generate_split_imagenet100(
                root_dir="ADD_PATH",
                n_tasks=10, batch_size=batch_size, num_workers=4,
                image_size=224, shuffle_classes=False, seed=seeds[0]
            )
            num_classes = 100
            epochs_ = [2]
            k_fractions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            class_indices_per_task = build_class_slices(n_tasks=10, classes_per_task=10, start=0)
        else:
            tasks_data = generate_split_cifar100(batch_size=batch_size, data_root="./../data", num_workers=0)
            num_classes = 100
            epochs_ = [2]  # 2, 5, 10, 20]
            k_fractions = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            class_indices_per_task = build_class_slices(n_tasks=10, classes_per_task=10, start=0)


        num_tasks = len(tasks_data)
        device_eval = "cuda" if torch.cuda.is_available() else "cpu"
        time_code = datetime.now().strftime(
            "%H%M_%d%m%y")  # e.g. "1642_070325"

        print(
            f"Running {len(kappa_update_rates) * len(seeds) * len(lrs) * len(gating_LRs) * len(gating_modes) * len(scale_modes) * len(reward_modes) * len(architectures) * len(gate_update_rates) * len(epochs_)} Experiments.")
        for architecture in architectures:
            for seed in seeds:
                counter += 1
                seed_everything(seed)
                for lr in lrs:
                    for g_lr in gating_LRs:
                        for k in k_fractions[:1]:
                            for gate_update_rate in gate_update_rates:
                                for kappa_update_rate in kappa_update_rates: 
                                    reward = reward_modes[0]
                                    for scalemode in scale_modes:
                                        for gating_mode in gating_modes:
                                            for epochs in epochs_:

                                                print(
                                                    f"Experiment: \n Weights_LR: {lr}\n Gating_LR:{g_lr}\n K-fraction: {k}\n"
                                                    f" Gate_Update_Rate: {gate_update_rate}\n Reward_Mode: {reward}\n"
                                                    f" Scale_Mode: {scalemode}\n Gating_Mode: {gating_mode}")
                                                if architecture == 'Resnet':
                                                    model = GatedResNet18_Flex(
                                                        num_classes=num_classes,
                                                        kappa=k,
                                                        stem=dataset,
                                                        norm_type=norm,
                                                        per_task_bn=per_task_bn,
                                                        gate_skip=False,
                                                        gate_head=False
                                                    )
                                                    model.set_class_slices(class_indices_per_task)
                                                    model.configure_head_protection(freeze_old_cols=True,
                                                                                    mask_non_current_logits=False)
                                                elif architecture == 'WideCNN':
                                                    model = WideCNN(num_classes=num_classes)
                                                elif architecture == "AlexNet":
                                                    model = AlexNet(num_classes=num_classes, lr=0.0001, lr_min=0.00005)
                                                else:
                                                    model = AlexNetW(num_classes=num_classes, lr=0.0001, lr_min=0.00005)

                                                if load_:
                                                    model = load_model_with_gates(model,
                                                                                  f"checkpoints/WideCNN_CIF10_1346_010825.pt",
                                                                                  device="cuda")

                                                    acc_TIL = eval_all(model, tasks_data,
                                                                       device="cuda" if torch.cuda.is_available() else "cpu",
                                                                       k_frac=k,
                                                                       up2task=num_tasks)
                                                    print("Task-Incremental Setting")
                                                    print("Per‑task accuracies:", [f"{a:.1f}%" for a in acc_TIL])
                                                    print("Average accuracy:", sum(acc_TIL) / len(acc_TIL))

                                                    acc_CIL = eval_exhaustive(model, tasks_data,
                                                                              device="cuda", kappa=k)
                                                    print("Sequential Evaluation - Class-Incremental Setting")
                                                    print("Per‑task accuracies:", [f"{a:.1f}%" for a in acc_CIL])
                                                    print("Average accuracy:", sum(acc_CIL) / len(acc_CIL))

                                                current_time = datetime.now().strftime(
                                                    "%H%M_%d%m%y")  # e.g. "1642_070325"

                                                if train_:

                                                    acc_TIL, acc_CIL, acc_CIL_musigma, epoch_logs = train_tasks(
                                                        model, tasks_data,
                                                        epochs_list=epochs,
                                                        device="cuda",
                                                        lr=lr,
                                                        g_lr=g_lr,
                                                        kappa_targets=k_fractions,
                                                        beta=beta,
                                                        reward_mode=reward,
                                                        scale_mode=scalemode,
                                                        gating_mode=gating_mode,
                                                        gate_update_rate=gate_update_rate,
                                                        conf=conf,
                                                        kappa_peak=kappa_peak,
                                                        kappa_update_rate=kappa_update_rate,
                                                        csv_path=f"training_logs/X1/{dataset}/{architecture}/{norm}Norm_seed{seed}_kur{kappa_update_rate}_{time_code}.csv"
                                                    )


                                                    file_name = f"{architecture}_{dataset}_{current_time}"
                                                    if save_model and counter == 1:
                                                        save_model_with_gates(model, "checkpoints/" + file_name + ".pt")
                                                        
                                                    if save:
                                                        overlap = gate_overlap_matrix(model, k_frac=k, hard=True)
                                                        # 6) Finally, save to logfile
                                                        save_experiment_log(
                                                            log_file="training_logs/posticlr/" + f"{dataset}/{architecture}/seed{seed}_kur{kappa_update_rate}_" + file_name + ".txt",
                                                            seed=seed,
                                                            dataset=dataset,
                                                            architecture=architecture,
                                                            num_tasks=num_tasks,
                                                            batch_size=batch_size,
                                                            gate_update_rate=gate_update_rate,
                                                            epochs=epochs,
                                                            k_fraction=k,
                                                            gating_lr=g_lr,
                                                            lr=lr,
                                                            reward_mode=reward,  # REWARD_BATCH, MARGIN, SAMPLE
                                                            scale_mode=scalemode,
                                                            gating_mode=gating_mode,
                                                            TIL_acc=acc_TIL,
                                                            CIL_acc=acc_CIL,
                                                            epoch_logs=epoch_logs,  # ← NEW
                                                            overlap_mat=overlap,
                                                            CIL_musigma_acc=acc_CIL_musigma,
                                                            norm=norm,
                                                            per_task_bn=per_task_bn
                                                        )
