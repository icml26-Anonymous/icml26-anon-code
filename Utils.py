from copy import deepcopy

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def save_experiment_log(
    log_file,
    seed,
    dataset,
    architecture,
    num_tasks,
    batch_size,
    gate_update_rate,
    epochs,
    k_fraction,
    gating_lr,
    lr,
    reward_mode,          # REWARD_BATCH, MARGIN, SAMPLE
    scale_mode,
    gating_mode,
    TIL_acc,
    CIL_acc,
    *,
    epoch_logs=None,      
    overlap_mat=None,     
    CIL_plat_acc = None,
    CIL_musigma_acc = None,
    norm=None,
    per_task_bn=None
):
    """
    Appends the experiment configuration, results, epoch-wise traces,
    and (optionally) the gate–overlap matrix to a text logfile.

    Parameters
    ----------
    epoch_logs : list[str] or None
        Lines such as  "Epoch: weights 3  |  per-task acc …".
    overlap_mat : torch.Tensor or None,  shape [T,T]
        The symmetric overlap matrix returned by `gate_overlap_matrix`.
    """
    with open(log_file, "a") as f:
        # -------- 1. static configuration ---------------------------------
        f.write(f"architecture     = {architecture}\n")
        f.write(f"dataset          = {dataset}\n")
        f.write(f"num_tasks        = {num_tasks}\n")
        f.write(f"batch_size       = {batch_size}\n")
        f.write(f"gate_update_rate = {gate_update_rate}\n")
        f.write(f"epochs           = {epochs}\n")
        f.write(f"k_fraction       = {k_fraction}\n")
        f.write(f"gating_LR        = {gating_lr}\n")
        f.write(f"lr               = {lr}\n")
        # f.write(f"beta_v           = {beta_v}\n")
        f.write(f"reward_mode      = {reward_mode}\n")
        f.write(f"scale_mode       = {scale_mode}\n")
        f.write(f"gating_mode      = {gating_mode}\n\n")

        if norm is not None:
            f.write(f"norm = {norm}\n")
        if per_task_bn is not None:
            f.write(f"per_task_bn = {per_task_bn}\n\n")

        f.write(f"seed             = {seed}\n\n")

        # -------- 2. final accuracies --------------------------------------
        TIL_overall_acc = sum(TIL_acc) / len(TIL_acc),
        CIL_overall_acc = sum(CIL_acc) / len(CIL_acc),
        CIL_musigma_overall = sum(CIL_musigma_acc) / len(CIL_musigma_acc),
        
        f.write(f"Task-Incremental Setting\n")
        for i, acc in enumerate(TIL_acc):
            f.write(f"TIL Task {i + 1:2d} | Test Acc = {acc:.2f}%\n")
        if isinstance(TIL_overall_acc, tuple):
            f.write("TIL Overall Test Accuracy = "
                    + ", ".join([f"{a:.2f}%" for a in TIL_overall_acc]) + "\n\n")
        else:
            f.write(f"TIL Overall Test Accuracy = {TIL_overall_acc:.2f}%\n\n")

        f.write(f"Class-Incremental Setting\n")
        for i, acc in enumerate(CIL_acc):
            f.write(f"CIL Task {i + 1:2d} | Test Acc = {acc:.2f}%\n")
        if isinstance(CIL_overall_acc, tuple):
            f.write("CIL Overall Test Accuracy = "
                    + ", ".join([f"{a:.2f}%" for a in CIL_overall_acc]) + "\n\n")
        else:
            f.write(f"CIL Overall Test Accuracy = {CIL_overall_acc:.2f}%\n\n")

        if CIL_musigma_acc is not None:
            f.write(f"Norm (mu-sigma) Class-Incremental Setting\n")
            for i, acc in enumerate(CIL_musigma_acc):
                f.write(f"CIL Task {i + 1:2d} | Test Acc = {acc:.2f}%\n")
            if isinstance(CIL_musigma_overall, tuple):
                f.write("Norm CIL Overall Test Accuracy = "
                        + ", ".join([f"{a:.2f}%" for a in CIL_musigma_overall]) + "\n\n")
            else:
                f.write(f"Norm CIL Overall Test Accuracy = {CIL_musigma_overall:.2f}%\n\n")

        # -------- 3. epoch / phase trace (optional) ------------------------
        if epoch_logs:
            f.write("------ Epoch trace ------\n")
            for line in epoch_logs:
                f.write(line.rstrip() + "\n")
            f.write("\n")

        # -------- 4. overlap matrix  (optional) ----------------------------
        if overlap_mat is not None:
            T = overlap_mat.size(0)
            f.write("------ Gate overlap matrix ------\n")
            # header
            f.write("      " + "  ".join([f"T{j+1:02d}" for j in range(T)]) + "\n")
            for i in range(T):
                row = "  ".join([f"{overlap_mat[i,j]:.3f}" for j in range(T)])
                f.write(f"T{i+1:02d}  {row}\n")
            f.write("\n")
