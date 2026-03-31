import torch
import os
from Utility_functions import *
import numpy as np
import random


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed(),
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model_with_gates(model: nn.Module, save_path: str):
    """
    Saves model weights and all task-specific gates.

    Args:
        model: the trained model (with GatedLayer modules).
        save_path: path to save the checkpoint (e.g. 'checkpoints/model.pt')
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'gate_banks': {},  # store all gates per module name

    }

    for name, module in model.named_modules():
        if isinstance(module, GatedLayer):
            gates = [g.detach().cpu() for g in module._gate_bank]
            checkpoint['gate_banks'][name] = gates

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[✓] Model and gates saved to: {save_path}")


def load_model_with_gates(model: nn.Module, load_path: str, device="cpu"):
    checkpoint = torch.load(load_path, map_location=device)

    # 1. Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)  # ← move model weights to correct device

    # 2. Load gates
    gate_banks = checkpoint.get('gate_banks', {})
    for name, module in model.named_modules():
        if isinstance(module, GatedLayer) and name in gate_banks:
            module._gate_bank = [
                g.to(device) for g in gate_banks[name]
            ]


    print(f"Model and gates loaded from: {load_path}")
    return model


def save_model_with_gates_and_stats(
    model: torch.nn.Module,
    running_mu: Dict[Tuple[object,int], torch.Tensor],
    running_nu: Dict[Tuple[object,int], torch.Tensor],
    save_path: str
):
    """
    Saves model weights, all task-specific gates, and running_mu/running_nu stats.

    Args:
        model: the trained model (with GatedLayer modules).
        running_mu: dict mapping (module, task_idx) -> mu tensor
        running_nu: dict mapping (module, task_idx) -> nu tensor
        save_path: path to save the checkpoint
    """
    # 1) state dict + gates
    checkpoint = {
        'state_dict': model.state_dict(),
        'gate_banks': {},
        'running_mu': {},
        'running_nu': {},
    }

    # build module→name map
    module_to_name = {m: name for name, m in model.named_modules()}

    for name, module in model.named_modules():
        if isinstance(module, GatedLayer):
            # gates
            checkpoint['gate_banks'][name] = [
                g.detach().cpu() for g in module._gate_bank
            ]
            # stats
            n_tasks = len(module._gate_bank)
            mu_list, nu_list = [], []
            for t in range(n_tasks):
                mu_list.append(running_mu[(module, t)].detach().cpu())
                nu_list.append(running_nu[(module, t)].detach().cpu())
            checkpoint['running_mu'][name] = mu_list
            checkpoint['running_nu'][name] = nu_list

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[✓] Saved model, gates, and stats to {save_path}")


def load_model_with_gates_and_stats(
    model: torch.nn.Module,
    load_path: str,
    device: str = "cpu"
) -> Tuple[torch.nn.Module, Dict, Dict]:
    """
    Loads model weights, gates, and running_mu/running_nu stats.

    Returns:
        model (on device), running_mu dict, running_nu dict
    """
    checkpoint = torch.load(load_path, map_location=device)

    # 1. weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # 2. gates
    gate_banks = checkpoint.get('gate_banks', {})
    for name, module in model.named_modules():
        if isinstance(module, GatedLayer) and name in gate_banks:
            module._gate_bank = [
                g.to(device) for g in gate_banks[name]
            ]

    # 3. reconstruct stats
    running_mu: Dict[Tuple[object,int], torch.Tensor] = {}
    running_nu: Dict[Tuple[object,int], torch.Tensor] = {}

    # build name→module map
    name_to_module = {name: m for name, m in model.named_modules()}

    for name, mu_list in checkpoint.get('running_mu', {}).items():
        nu_list = checkpoint['running_nu'][name]
        module = name_to_module[name]
        for t, mu in enumerate(mu_list):
            running_mu[(module, t)] = mu.to(device)
        for t, nu in enumerate(nu_list):
            running_nu[(module, t)] = nu.to(device)

    model.running_mu = running_mu
    model.running_nu = running_nu
    print(f"[✓] Loaded model, gates, and stats from {load_path}")
    return model

