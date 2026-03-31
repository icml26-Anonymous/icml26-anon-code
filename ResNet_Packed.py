import torch.optim as optim
from Utility_functions import *
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18
from typing import Optional
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from compile_tools_Packed import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from typing import Tuple


def build_remap(skip_idx: torch.Tensor, target_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    skip_idx:   [C_skip] global channel ids present in packed skip tensor (current representation)
    target_idx: [C_tgt]  global channel ids we need (conv2_out_idx, canonical order)

    Returns:
      gather_idx: [C_tgt] indices into skip tensor (0..C_skip-1), with dummy 0 for missing
      present:    [C_tgt] bool mask, True where target channel exists in skip
    """
    # work on CPU for python dict speed, then move back
    s = skip_idx.detach().cpu().tolist()
    pos = {int(ch): i for i, ch in enumerate(s)}  # global_id -> position in skip tensor

    t = target_idx.detach().cpu().tolist()
    gather = []
    present = []
    for ch in t:
        if int(ch) in pos:
            gather.append(pos[int(ch)])
            present.append(True)
        else:
            gather.append(0)      # dummy
            present.append(False)

    gather_idx = torch.tensor(gather, dtype=torch.long, device=skip_idx.device)
    present_m  = torch.tensor(present, dtype=torch.bool, device=skip_idx.device)
    return gather_idx, present_m


@torch.no_grad()
def remap_skip(skip: torch.Tensor, gather_idx: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """
    skip: [B, C_skip, H, W]
    returns: [B, C_tgt, H, W] aligned to target_idx order.
    """
    y = skip.index_select(1, gather_idx)  # [B, C_tgt, H, W]
    y = y * present.to(dtype=skip.dtype).view(1, -1, 1, 1)
    return y


def resolve_plan_name_to_model_name(plan_name: str) -> str:
    # model has conv1/norm1 at top-level, not stem.*
    if plan_name == "stem.conv1":
        return "conv1"
    if plan_name == "stem.norm1":
        return "norm1"
    return plan_name

def resolve_op_module(full_model: nn.Module, op):
    named = dict(full_model.named_modules())
    model_name = resolve_plan_name_to_model_name(op.name)
    if model_name not in named:
        raise KeyError(
            f"Plan op name '{op.name}' -> model name '{model_name}' not found. "
            f"Example keys: {list(named.keys())[:30]}"
        )
    return named[model_name]

@torch.no_grad()
def clone_instancenorm_like(full_norm: nn.Module, num_features: int, device: str):
    """
    For InstanceNorm2d(affine=False): just recreate with same eps/momentum/etc.
    If affine=True, this will copy weights/bias too.
    """
    if not isinstance(full_norm, nn.InstanceNorm2d):
        raise TypeError(f"Expected InstanceNorm2d, got {type(full_norm)}")

    new_norm = nn.InstanceNorm2d(
        num_features=num_features,
        eps=full_norm.eps,
        momentum=full_norm.momentum,
        affine=full_norm.affine,
        track_running_stats=full_norm.track_running_stats
    ).to(device)

    if full_norm.affine:
        new_norm.weight.data.copy_(full_norm.weight.data[:num_features])
        new_norm.bias.data.copy_(full_norm.bias.data[:num_features])

    # running stats are typically unused for IN when track_running_stats=False
    if full_norm.track_running_stats:
        new_norm.running_mean.data.copy_(full_norm.running_mean.data[:num_features])
        new_norm.running_var.data.copy_(full_norm.running_var.data[:num_features])

    return new_norm

def get_op(plan_t, codename):
    for op in reversed(plan_t):
        if op.name == codename:
            return op.out_idx
    raise RuntimeError(f"No {codename} op in plan_t")

def get_last_conv2_out_idx(plan_t):
    for op in reversed(plan_t):
        if op.name.endswith(".conv2"):
            return op.out_idx
    raise RuntimeError("No .conv2 op in plan_t")

def assert_block_alignment(op2, opd):
    # op2 is conv2 op, opd is down op
    if op2.out_idx.numel() != opd.out_idx.numel():
        raise RuntimeError(f"down/out mismatch sizes: {op2.name} vs {opd.name}")
    if not torch.equal(op2.out_idx.cpu(), opd.out_idx.cpu()):
        raise RuntimeError(f"Channel order mismatch in residual add: {op2.name}.out_idx != {opd.name}.out_idx")


@torch.no_grad()
def make_post_add_mask_from_conv2(full_conv2: nn.Module, op2, task_id: int, kappa: float, device):
    # full gate vector (length = full channels, e.g. 512)
    g = full_conv2.gate_for(task_id).detach().to(device)

    # binary top-k mask in full channel space
    m_full = apply_topk(g, kappa, hard=True)  # [C_full], values in {0,1}

    # restrict to packed channels (op2.out_idx)
    out_idx = op2.out_idx.to(device)
    m_packed = m_full.index_select(0, out_idx)  # [C_packed]
    return m_packed

@torch.no_grad()
def full_forward_collect_blocks(full_model, xb, task_id, kappa, hard=True, device="cuda"):
    full_model.eval().to(device)
    full_model.eval_mode = False  # IMPORTANT
    xb = xb.to(device)

    outs = {}

    x = F.relu(full_model._apply_norm(full_model.norm1, full_model.conv1(xb), task_id))
    x = x * full_model._mask(full_model.conv1, task_id, hard, kappa)
    x = full_model.pool(x)

    outs["stem.conv1"] = x.detach()
    # blocks in forward order
    for sname, stage in [("layer1", full_model.layer1),
                         ("layer2", full_model.layer2),
                         ("layer3", full_model.layer3),
                         ("layer4", full_model.layer4)]:
        for bi, block in enumerate(stage):
            x = block(x, task_id, kappa, hard, acts_dict=None)
            outs[f"{sname}.{bi}"] = x.detach()

    x = full_model.avg(x).flatten(1)
    return outs, x.detach()  # block outs + pre-fc 512

@torch.no_grad()
def packed_forward_collect_blocks(packed_model, xb, device="cuda"):
    packed_model.eval().to(device)
    xb = xb.to(device)

    outs = {}

    x = F.relu(packed_model.norm1(packed_model.conv1(xb)))
    outs["block-1"] = x.detach()
    for i, b in enumerate(packed_model.blocks):
        x = b(x)
        outs[f"block{i}"] = x.detach()

    x = packed_model.avg(x).flatten(1)
    return outs, x.detach()  # block outs + pre-fc packed

def get_conv2_ops_in_order(plan_t):
    return [op for op in plan_t if op.name.endswith(".conv2")]

@torch.no_grad()
def compare_blocks(full_model, packed_model, plan_t, xb, task_id, kappa, hard=True, device="cuda",
                   atol=1e-5, rtol=1e-4):
    conv2_ops = get_conv2_ops_in_order(plan_t)  # 8 ops for ResNet18 (2 per stage)
    full_block_outs, full_prefc = full_forward_collect_blocks(full_model, xb, task_id, kappa, hard, device)
    pack_block_outs, pack_prefc = packed_forward_collect_blocks(packed_model, xb, device)

    # full block keys are layer1.0, layer1.1, layer2.0, ... layer4.1
    full_keys = ["layer1.0","layer1.1","layer2.0","layer2.1","layer3.0","layer3.1","layer4.0","layer4.1"]
    pack_keys = [f"block{i}" for i in range(len(full_keys))]

    y_full = full_block_outs["stem.conv1"]  # [B, C_full, H, W]
    y_pack = pack_block_outs["block-1"]  # [B, C_pack, H, W]
    out_idx = plan_t[0].out_idx.to(y_full.device)
    y_ref = y_full.index_select(1, out_idx)

    diff = (y_ref - y_pack).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = y_ref.abs().mean().item() + 1e-12
    rel = mean_abs / denom

    ok = torch.allclose(y_ref, y_pack, atol=atol, rtol=rtol)
    print(f"[t={task_id}] stem.conv1 ref={tuple(y_ref.shape)} pack={tuple(y_pack.shape)} "
          f"max={max_abs:.3e} mean={mean_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")

    all_ok = True
    for i, (fk, pk, op2) in enumerate(zip(full_keys, pack_keys, conv2_ops)):
        y_full = full_block_outs[fk]                      # [B, C_full, H, W]
        y_pack = pack_block_outs[pk]                      # [B, C_pack, H, W]
        out_idx = op2.out_idx.to(y_full.device)
        y_ref = y_full.index_select(1, out_idx)

        diff = (y_ref - y_pack).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = y_ref.abs().mean().item() + 1e-12
        rel = mean_abs / denom

        ok = torch.allclose(y_ref, y_pack, atol=atol, rtol=rtol)
        print(f"[t={task_id}] {fk:8s} ref={tuple(y_ref.shape)} pack={tuple(y_pack.shape)} "
              f"max={max_abs:.3e} mean={mean_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # pre-fc compare
    last_out_idx = get_last_conv2_out_idx(plan_t).to(full_prefc.device)
    full_ref_prefc = full_prefc.index_select(1, last_out_idx)  # [B, C_last_pack]
    diff = (full_ref_prefc - pack_prefc).abs()
    print(f"[t={task_id}] pre-fc ref={tuple(full_ref_prefc.shape)} pack={tuple(pack_prefc.shape)} "
          f"max={diff.max().item():.3e} mean={diff.mean().item():.3e}")

    print("ALL BLOCKS OK:", all_ok)
    return all_ok


@torch.no_grad()
def compare_layers(full_model, packed_model, plan_t, xb, task_id, kappa, hard=True, device="cuda",
                   atol=1e-4, rtol=1e-3):
    conv2_ops = get_conv2_ops_in_order(plan_t)  # 8 ops for ResNet18 (2 per stage)
    full_block_outs, full_prefc = full_forward_collect_layers(full_model, xb, task_id, kappa, hard, device)

    pack_block_outs, pack_prefc = packed_forward_collect_layers(packed_model, xb, device)

    # full block keys are layer1.0, layer1.1, layer2.0, ... layer4.1
    counter = 0
    full_keys = ["stem.conv1"]
    pack_keys = ["stem.conv1"]
    for st in range(1, 5):
        for lr in range(2):
            if f"layer{st}.{lr}" in ["layer2.0", "layer3.0", "layer4.0"]:
                convls = ["x", "conv1", "conv2", "down"]
            else:
                convls = ["x", "conv1", "conv2"]
            for convl in convls:
                full_keys.append(f"layer{st}.{lr}.{convl}")
                pack_keys.append(f"block{counter}.{convl}")
            counter += 1

    all_ok = True
    for i, (fk, pk) in enumerate(zip(full_keys, pack_keys)):
        y_full = full_block_outs[fk]                      # [B, C_full, H, W]
        y_pack = pack_block_outs[pk]                      # [B, C_pack, H, W]
        if fk.endswith(".x"):
            fk = full_keys[i-1]
            out_idx = get_op(plan_t, fk.replace(".x", ".conv2")).to(y_full.device)
        else:
            out_idx = get_op(plan_t, fk).to(y_full.device)
        y_ref = y_full.index_select(1, out_idx)

        diff = (y_ref - y_pack).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = y_ref.abs().mean().item() + 1e-12
        rel = mean_abs / denom

        ok = torch.allclose(y_ref, y_pack, atol=atol, rtol=rtol)
        print(f"[t={task_id}] {fk:8s} ref={tuple(y_ref.shape)} pack={tuple(y_pack.shape)} "
              f"max={max_abs:.3e} mean={mean_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # pre-fc compare
    last_out_idx = get_last_conv2_out_idx(plan_t).to(full_prefc.device)
    full_ref_prefc = full_prefc.index_select(1, last_out_idx)  # [B, C_last_pack]
    diff = (full_ref_prefc - pack_prefc).abs()
    print(f"[t={task_id}] pre-fc ref={tuple(full_ref_prefc.shape)} pack={tuple(pack_prefc.shape)} "
          f"max={diff.max().item():.3e} mean={diff.mean().item():.3e}")

    print("ALL BLOCKS OK:", all_ok)
    return all_ok

@torch.no_grad()
def full_forward_collect_layers(full_model, xb, task_id, kappa, hard=True, device="cuda"):
    full_model.eval().to(device)
    full_model.eval_mode = False  # IMPORTANT
    xb = xb.to(device)

    outs = {}

    x = F.relu(full_model._apply_norm(full_model.norm1, full_model.conv1(xb), task_id))
    x = x * full_model._mask(full_model.conv1, task_id, hard, kappa)
    x = full_model.pool(x)

    outs["stem.conv1"] = x.detach()
    # blocks in forward order
    for sname, stage in [("layer1", full_model.layer1),
                         ("layer2", full_model.layer2),
                         ("layer3", full_model.layer3),
                         ("layer4", full_model.layer4)]:
        for bi, block in enumerate(stage):
            outs[f"{sname}.{bi}.x"] = x.detach()
            x = block(x, task_id, kappa, hard, acts_dict={})
            outs[f"{sname}.{bi}.conv1"] = block.last_acts[block.conv1]
            outs[f"{sname}.{bi}.conv2"] = block.last_acts[block.conv2]
            if block.down is not None:
                outs[f"{sname}.{bi}.down"] = block.last_acts[block.down]


    x = full_model.avg(x).flatten(1)
    return outs, x.detach()  # block outs + pre-fc 512

@torch.no_grad()
def packed_forward_collect_layers(packed_model, xb, device="cuda"):
    packed_model.eval().to(device)
    xb = xb.to(device)

    outs = {}

    x = F.relu(packed_model.norm1(packed_model.conv1(xb)))
    outs["stem.conv1"] = x.detach()
    for bi, block in enumerate(packed_model.blocks):
        outs[f"block{bi}.x"] = x.detach()
        x = block(x, acts_dict={})
        outs[f"block{bi}.conv1"] = block.last_acts[block.conv1]
        outs[f"block{bi}.conv2"] = block.last_acts[block.conv2]
        if block.down is not None:
            outs[f"block{bi}.down"] = block.last_acts[block.down]


    x = packed_model.avg(x).flatten(1)
    return outs, x.detach()  # block outs + pre-fc packed


def build_skip_gather(in_full_idx: torch.Tensor, out_full_idx: torch.Tensor):
    """
    in_full_idx:  [Cin_packed]  full channel ids that exist at block input
    out_full_idx: [Cout_packed] full channel ids that exist at block output (after conv2 gate)

    Returns:
      gather: LongTensor [Cout_packed] where gather[j] = position in packed input that matches out_full_idx[j],
              or -1 if that full channel id is not present in the input subset.
    """
    # map full_id -> position in packed input
    mp = {int(v): i for i, v in enumerate(in_full_idx.tolist())}
    gather = torch.full((out_full_idx.numel(),), -1, dtype=torch.long)
    for j, v in enumerate(out_full_idx.tolist()):
        gather[j] = mp.get(int(v), -1)
    return gather


class PackedBasicBlock(nn.Module):
    def __init__(
        self,
        conv1: GConv2d,
        norm1: nn.InstanceNorm2d,
        conv2: GConv2d,
        norm2: nn.InstanceNorm2d,
        down: Optional[nn.Conv2d],
        down_norm: Optional[nn.InstanceNorm2d],
        block_name: str,
        gather_idx=None, present=None
    ):
        super().__init__()
        self.conv1 = conv1
        self.norm1 = norm1
        self.conv2 = conv2
        self.norm2 = norm2
        self.down = down
        self.down_norm = down_norm
        self.block_name = block_name

        if gather_idx is not None:
            self.register_buffer("skip_gather_idx", gather_idx)
            self.register_buffer("skip_present", present)
        else:
            self.skip_gather_idx = None
            self.skip_present = None

    def _apply_norm(self, norm_mod, x, task_id):
        return norm_mod(x)

    def forward(self, x, acts_dict=None):
        # conv1 (gated AFTER norm+relu)
        out1 = F.relu(self._apply_norm(self.norm1, self.conv1(x), 0))
        if acts_dict is not None:
            acts_dict[self.conv1] = out1.detach()
        # conv2 (gated AFTER norm)
        out2 = self._apply_norm(self.norm2, self.conv2(out1), 0)
        if acts_dict is not None:
            acts_dict[self.conv2] = out2.detach()
        # residual
        if self.down is None:
            # identity skip, but must be aligned to conv2_out_idx basis
            if self.skip_gather_idx is not None:
                skip = remap_skip(x, self.skip_gather_idx, self.skip_present)
            else:
                skip = x
        else:
            skip = self.down_norm(self.down(x))
            if acts_dict is not None:
                acts_dict[self.down] = skip.detach()

        out = out2 + skip
        out = F.relu(out)
        self.last_acts = acts_dict
        return out


@torch.no_grad()
def make_packed_conv_from_op(full_mod: nn.Module, op, *, bias_ok=False) -> nn.Conv2d:
    W = full_mod.weight.detach()
    out_idx = op.out_idx.to(W.device)
    in_idx = None if op.in_idx is None else op.in_idx.to(W.device)

    Wp = W.index_select(0, out_idx)
    if in_idx is not None:
        Wp = Wp.index_select(1, in_idx)
    Wp = Wp.contiguous()

    allOK = True
    for id_num, idx in enumerate(out_idx):
        if in_idx is not None:
            Worig = W.index_select(0, idx).index_select(1, in_idx)
        else:
            Worig = W.index_select(0, idx)
        diff = Worig - Wp[id_num, :, :, :]
        if diff.sum() > 0:
            allOK = False
    if not allOK:
        print(f'Layer: {op.name} - ISSUE - Not correct passing!')
        exit()

    conv = nn.Conv2d(
        in_channels=Wp.size(1),
        out_channels=Wp.size(0),
        kernel_size=full_mod.kernel_size,
        stride=full_mod.stride,
        padding=full_mod.padding,
        dilation=full_mod.dilation,
        groups=full_mod.groups,
        bias=(full_mod.bias is not None) if bias_ok else False,
        padding_mode=full_mod.padding_mode
    )
    conv.weight.data.copy_(Wp)

    if conv.bias is not None:
        b = full_mod.bias.detach().index_select(0, out_idx).contiguous()
        conv.bias.data.copy_(b)

    return conv


@torch.no_grad()
def make_packed_Gconv_from_op(full_mod: nn.Module, op, bias_ok=False):
    if isinstance(full_mod, GConv2d):
        W = full_mod.weight.detach()

        out_idx = op.out_idx.to(W.device)
        in_idx = None if op.in_idx is None else op.in_idx.to(W.device)

        Wp = W.index_select(0, out_idx)
        if in_idx is not None:
            Wp = Wp.index_select(1, in_idx)
        Wp = Wp.contiguous()

        if full_mod.bias is not None:
            print(full_mod.bias)

        allOK = True
        for id_num, idx in enumerate(out_idx):
            if in_idx is not None:
                Worig = W.index_select(0, idx).index_select(1, in_idx)
            else:
                Worig = W.index_select(0, idx)
            diff = Worig - Wp[id_num, :, :, :]
            if diff.sum() > 0:
                allOK = False
        if not allOK:
            print(f'Layer: {op.name} - ISSUE - Not correct passing!')
            exit()

        conv = GConv2d(
            in_channels=Wp.size(1),
            out_channels=Wp.size(0),
            kernel_size=full_mod.kernel_size,
            stride=full_mod.stride,
            padding=full_mod.padding,
            bias=False #(full_mod.bias is not None) if bias_ok else False
        )

        # print(conv)
        conv.weight.data.copy_(Wp)

        # if conv.bias is not None:
        #     b = full_mod.bias.detach().index_select(0, out_idx).contiguous()
        #     conv.bias.data.copy_(b)

        return conv, Wp
    else:
        print('Gated Conv is not GConv')
        exit()


class PackedResNet18(nn.Module):
    def __init__(self, stem_conv, stem_norm, blocks, fc):
        super().__init__()
        self.conv1 = stem_conv
        self.norm1 = stem_norm
        self.pool = nn.Identity()
        self.blocks = nn.ModuleList(blocks)  # in forward order
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = fc

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        # stem gate would happen here in full model; packing implies identity
        x = self.pool(x)
        for b in self.blocks:
            x = b(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)


@torch.no_grad()
def make_in_like(full_norm: nn.Module, C: int, device):
    # full_norm is InstanceNorm2d 
    if not isinstance(full_norm, nn.InstanceNorm2d):
        raise TypeError(f"Expected InstanceNorm2d, got {type(full_norm)}")
    return nn.InstanceNorm2d(
        C,
        eps=full_norm.eps,
        momentum=full_norm.momentum,
        affine=full_norm.affine,
        track_running_stats=full_norm.track_running_stats
    ).to(device)

@torch.no_grad()
def build_packed_model_for_task(
    full_model: nn.Module,
    plan_t: List,
    class_indices_per_task: Dict[int, torch.Tensor],
    task_id: int,
    device="cuda",
) -> nn.Module:
    full_model.eval().to(device)

    # ---- resolve modules dict once
    named = dict(full_model.named_modules())

    # ---- STEM
    stem_op = plan_t[0]
    if stem_op.name != "stem.conv1":
        raise ValueError(f"Expected first op 'stem.conv1', got '{stem_op.name}'")

    stem_full_conv = resolve_op_module(full_model, stem_op).to(device)

    stem_conv, _ = make_packed_Gconv_from_op(stem_full_conv, stem_op)
    stem_conv = stem_conv.to(device)
    # stem norm is full_model.norm1 (plan may call it stem.norm1)
    stem_full_norm = named["norm1"]
    stem_norm = clone_instancenorm_like(stem_full_norm, stem_conv.out_channels, device)

    full_keys = ["layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1"]
    # ---- BLOCKS
    blocks = []
    i = 1
    block_num = 0

    while i < len(plan_t):
        op_prev = plan_t[i-1]
        op1 = plan_t[i]
        op2 = plan_t[i + 1]
        if not op1.name.endswith(".conv1"):
            raise ValueError(f"Expected conv1 op at plan[{i}], got '{op1.name}'")
        if not op2.name.endswith(".conv2"):
            raise ValueError(f"Expected conv2 op at plan[{i+1}], got '{op2.name}'")


        # conv modules
        full_conv1 = resolve_op_module(full_model, op1).to(device)
        full_conv2 = resolve_op_module(full_model, op2).to(device)

        conv1, _ = make_packed_Gconv_from_op(full_conv1, op1)
        conv2, Wp = make_packed_Gconv_from_op(full_conv2, op2)

        conv1 = conv1.to(device)
        conv2 = conv2.to(device)

        # corresponding norms live in the BasicBlock as norm1/norm2
        # names: layerX.Y.norm1 / layerX.Y.norm2
        n1_name = op1.name.replace("conv1", "norm1")
        n2_name = op2.name.replace("conv2", "norm2")

        full_norm1 = named[resolve_plan_name_to_model_name(n1_name)]
        full_norm2 = named[resolve_plan_name_to_model_name(n2_name)]

        norm1 = clone_instancenorm_like(full_norm1, conv1.out_channels, device)
        norm2 = clone_instancenorm_like(full_norm2, conv2.out_channels, device)

        # optional downsample
        down = None
        down_norm = None
        if i + 2 < len(plan_t) and plan_t[i + 2].name.endswith(".down"):
            opd = plan_t[i + 2]

            full_down = resolve_op_module(full_model, opd).to(device)
            down = make_packed_conv_from_op(full_down, opd).to(device)

            dn_name = opd.name.replace("down", "down_norm")
            full_down_norm = named[resolve_plan_name_to_model_name(dn_name)]
            down_norm = clone_instancenorm_like(full_down_norm, down.out_channels, device)

            i += 3
            gather_idx = present = None
        else:
            i += 2
            gather_idx, present = build_remap(op_prev.out_idx, op2.out_idx)

        blocks.append(PackedBasicBlock(conv1, norm1, conv2, norm2, down, down_norm, full_keys[block_num], gather_idx=gather_idx, present=present))
        block_num += 1

    # ---- CLASSIFIER (IMPORTANT)
    # last_out_idx should be the channels that exist at the OUTPUT of the last block (layer4.1.conv2 op)
    last_conv2_op = None
    for op in reversed(plan_t):
        if op.name.endswith("layer4.1.conv2") or op.name.endswith(".conv2"):
            last_conv2_op = op
            break
    if last_conv2_op is None:
        raise RuntimeError("Could not find last conv2 op in plan_t to define last_out_idx.")

    last_out_idx = last_conv2_op.out_idx.to(device)  # indices into full 512-d feature channels

    cls = class_indices_per_task[task_id].to(device)
    if cls.dtype == torch.bool:
        cls = cls.nonzero(as_tuple=False).view(-1)

    # full head is SingleHeadWithProtection (a Linear)
    W_full = full_model.fc.weight.detach().to(device)  # [num_classes_total, 512]
    b_full = full_model.fc.bias.detach().to(device) if getattr(full_model.fc, "bias", None) is not None else None

    # packed head: only task classes, only packed feature channels
    Wp = W_full.index_select(0, cls).index_select(1, last_out_idx).contiguous()
    bp = b_full.index_select(0, cls).contiguous() if b_full is not None else None

    fc = nn.Linear(Wp.size(1), Wp.size(0), bias=(bp is not None)).to(device)
    fc.weight.data.copy_(Wp)
    if bp is not None:
        fc.bias.data.copy_(bp)

    packed = PackedResNet18(stem_conv, stem_norm, blocks, fc).to(device)
    packed.eval()
    return packed


@torch.no_grad()
def forward_to_prefc_packed(packed_model, xb, device="cuda"):
    packed_model.eval().to(device)
    xb = xb.to(device)

    x = F.relu(packed_model.norm1(packed_model.conv1(xb)))
    for b in packed_model.blocks:
        x = b(x)
    x = packed_model.avg(x).flatten(1)  # [B,Cpacked]
    return x

@torch.no_grad()
def forward_to_prefc_full(full_model, xb, task_id, kappa, hard=True, device="cuda"):
    full_model.eval().to(device)
    xb = xb.to(device)

    # manual forward to extract pre-fc
    x = xb
    x = F.relu(full_model._apply_norm(full_model.norm1, full_model.conv1(x), task_id))
    if not full_model.eval_mode:
        x = x * full_model._mask(full_model.conv1, task_id, hard, kappa)
    x = full_model.pool(x)

    for stage in [full_model.layer1, full_model.layer2, full_model.layer3, full_model.layer4]:
        for block in stage:
            x = block(x, task_id, kappa, hard, acts_dict=None)

    x = full_model.avg(x).flatten(1)  # [B,512]
    return x


@torch.no_grad()
def compare_full_vs_packed_prefc(full_model, packed_model, plan_t, xb, task_id, kappa, hard=True, device="cuda"):
    full_model.eval_mode = False
    full_model.eval()
    packed_model.eval()
    last_out_idx = get_last_conv2_out_idx(plan_t).to(device)

    x_full_512 = forward_to_prefc_full(full_model, xb, task_id, kappa, hard=hard, device=device)
    x_ref = x_full_512.index_select(1, last_out_idx)

    x_pack = forward_to_prefc_packed(packed_model, xb, device=device)

    diff = (x_ref - x_pack).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = x_ref.abs().mean().item() + 1e-12
    rel = mean_abs / denom

    print(f"[t={task_id}] pre-fc x_ref={tuple(x_ref.shape)} x_pack={tuple(x_pack.shape)} "
          f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel={rel:.3e}")
    return max_abs, mean_abs, rel


@torch.no_grad()
def compare_full_vs_packed_logits(full_model, packed_model, xb, task_id, kappa, device="cuda"):
    full_model.eval().to(device)
    packed_model.eval().to(device)
    xb = xb.to(device)

    # full model produces logits over all classes; 
    logits_full = full_model(xb, task_id=task_id, k_frac=kappa, hard=True)  # shape [B, num_classes_total]
    logits_p = packed_model(xb)  # shape [B, num_task_classes]

    return logits_full, logits_p



@torch.no_grad()
def build_task_channel_indices(model: nn.Module, kappa: float, num_tasks: int):
    # map layer_id -> module (only GConv2d)
    gconvs = []
    for m in model.modules():
        if isinstance(m, GConv2d):
            gconvs.append(m)
    gconvs = sorted(gconvs, key=lambda m: m.layer_id)

    idx_out = {t: {} for t in range(num_tasks)}  # t -> {layer_id: LongTensor}
    for m in gconvs:
        for t in range(num_tasks):
            mask = apply_topk(m.gate_for(t), kappa, hard=True).bool()
            idx = mask.nonzero(as_tuple=False).view(-1)
            idx_out[t][m.layer_id] = idx

    return idx_out

@torch.no_grad()
def build_idx_out_per_task(
    model: nn.Module,
    kappa: float,
    num_tasks: int,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Returns:
      idx_out[t][layer_id] = LongTensor of selected output channel indices for that GConv2d layer_id.
    """
    idx_out: Dict[int, Dict[int, torch.Tensor]] = {t: {} for t in range(num_tasks)}

    for m in model.modules():
        if isinstance(m, GConv2d):
            lid = int(m.layer_id)
            C = m.gate_for(0).numel()
            for t in range(num_tasks):
                mask = apply_topk(m.gate_for(t), kappa, hard=True).to(torch.bool)
                idx = mask.nonzero(as_tuple=False).view(-1)
                # Sort for determinism (optional but nice)
                idx = idx.sort().values
                idx_out[t][lid] = idx

                # quick sanity: mask cardinality
                expected_k = max(1, int(kappa * C))
                if idx.numel() != expected_k:
                    print(f"[WARN] layer {lid} task {t}: idx={idx.numel()} expected={expected_k}")
    return idx_out

@dataclass
class PackedOp:
    kind: str  # "stem", "conv", "down"
    name: str  # (layer2.0.conv1, etc.)
    layer_id: int  # conv1/conv2: GConv2d.layer_id ; down: block.down layer_id if exists else -1
    in_idx: Optional[torch.Tensor]  # LongTensor
    out_idx: torch.Tensor  # LongTensor


@torch.no_grad()
def build_packed_wiring_plan_for_task(
    model: nn.Module,
    idx_out_task: Dict[int, torch.Tensor],
    task_id: int,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[PackedOp]:
    """
    idx_out_task: mapping layer_id -> out_idx for this task (from build_idx_out_per_task()[task_id])

    Returns a list of PackedOp in forward order.
    """
    plan: List[PackedOp] = []

    # ---- stem
    stem = model.conv1  # GConv2d with layer_id 0 
    stem_lid = int(stem.layer_id)

    out_idx = idx_out_task[stem_lid]
    if device is not None:
        out_idx = out_idx.to(device)

    plan.append(PackedOp(
        kind="stem",
        name="stem.conv1",
        layer_id=stem_lid,
        in_idx=None,           # RGB input
        out_idx=out_idx
    ))
    # After stem mask, only these channels exist
    S_in = out_idx  # block input channels for the first block

    # ---- stages
    stages = List[nn.Module] = []
    stages.append(model.layer1)
    stages.append(model.layer2)
    stages.append(model.layer3)
    stages.append(model.layer4)

    for st_num, stage in enumerate(stages):
        stage_name = f"layer{st_num}"
        for bi, block in enumerate(stage):
            c1 = block.conv1
            c2 = block.conv2
            c1_lid = int(c1.layer_id)
            c2_lid = int(c2.layer_id)

            out1 = idx_out_task[c1_lid]
            out2 = idx_out_task[c2_lid]
            if device is not None:
                out1 = out1.to(device)
                out2 = out2.to(device)

            # conv1: in=S_in, out=out1
            plan.append(PackedOp(
                kind="conv",
                name=f"{stage_name}.{bi}.conv1",
                layer_id=c1_lid,
                in_idx=S_in,
                out_idx=out1
            ))

            # conv2: in=out1, out=out2
            plan.append(PackedOp(
                kind="conv",
                name=f"{stage_name}.{bi}.conv2",
                layer_id=c2_lid,
                in_idx=out1,
                out_idx=out2
            ))

            # downsample: if exists, must map S_in -> out2 (so it can be added to conv2 output)
            if getattr(block, "down", None) is not None:
                down = block.down
                down_lid = int(
                    getattr(down, "layer_id", -1))  # plain Conv2d down has no layer_id unless set
                plan.append(PackedOp(
                    kind="down",
                    name=f"{stage_name}.{bi}.down",
                    layer_id=down_lid,
                    in_idx=S_in,
                    out_idx=out2
                ))

            # block output becomes conv2-selected channels
            S_in = out2

            if verbose:
                print(f"[plan t={task_id}] {stage_name}.{bi}: "
                      f"S_in={plan[-1].out_idx.numel()} (after conv2 mask), "
                      f"conv1_out={out1.numel()}, conv2_out={out2.numel()}, "
                      f"down={'yes' if getattr(block, 'down', None) is not None else 'no'}")

    return plan


@torch.no_grad()
def build_packed_wiring_plan_for_task_(
    model: nn.Module,
    idx_out_task: Dict[int, torch.Tensor],  # layer_id -> LongTensor
    task_id: int,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[PackedOp]:

    # --- guard rails ---
    assert isinstance(idx_out_task, dict), "idx_out_task must be a dict: layer_id -> Tensor"
    # try one element
    any_val = next(iter(idx_out_task.values()))
    assert torch.is_tensor(any_val), (
        "idx_out_task values must be torch.Tensors. "
        "You probably passed idx_out (dict-of-dicts) instead of idx_out[task_id]."
    )

    plan: List[PackedOp] = []

    # ---- stem
    stem = model.conv1
    stem_lid = int(stem.layer_id)
    out_idx = idx_out_task[stem_lid]
    if device is not None:
        out_idx = out_idx.to(device)

    plan.append(PackedOp(
        kind="stem",
        name="stem.conv1",
        layer_id=stem_lid,
        in_idx=None,
        out_idx=out_idx
    ))
    S_in = out_idx  # Tensor

    # ---- stages
    for si, stage_name in enumerate(["layer1", "layer2", "layer3", "layer4"]):
        stage = getattr(model, stage_name)
        for bi, block in enumerate(stage):
            c1, c2 = block.conv1, block.conv2
            c1_lid, c2_lid = int(c1.layer_id), int(c2.layer_id)

            out1 = idx_out_task[c1_lid]
            out2 = idx_out_task[c2_lid]

            if device is not None:
                out1 = out1.to(device)
                out2 = out2.to(device)

            plan.append(PackedOp(
                kind="conv",
                name=f"{stage_name}.{bi}.conv1",
                layer_id=c1_lid,
                in_idx=S_in,
                out_idx=out1
            ))

            plan.append(PackedOp(
                kind="conv",
                name=f"{stage_name}.{bi}.conv2",
                layer_id=c2_lid,
                in_idx=out1,
                out_idx=out2
            ))

            if getattr(block, "down", None) is not None:
                down = block.down
                down_lid = int(getattr(down, "layer_id", c2_lid+1))
                plan.append(PackedOp(
                    kind="down",
                    name=f"{stage_name}.{bi}.down",
                    layer_id=down_lid,
                    in_idx=S_in,
                    out_idx=out2
                ))

            # block output channels
            S_in = out2

            if verbose:
                print(
                    f"[plan t={task_id}] {stage_name}.{bi}: "
                    f"S_in={S_in.numel()} | conv1_out={out1.numel()} | conv2_out={out2.numel()} | "
                    f"down={'yes' if getattr(block,'down',None) is not None else 'no'}"
                )

    return plan


@torch.no_grad()
def build_packed_wiring_plans(
    model: nn.Module,
    kappa: float,
    num_tasks: int,
    *,
    verbose: bool = False,
) -> Dict[int, List[PackedOp]]:
    """
    Returns:
      plans[t] = list of PackedOp for task t, in forward order.
    """
    idx_out = build_idx_out_per_task(model, kappa=kappa, num_tasks=num_tasks)

    plans: Dict[int, List[PackedOp]] = {}

    for t in range(num_tasks):
        if verbose:
            print(f"\n=== Building packed wiring plan for task {t} ===")

        plans[t] = build_packed_wiring_plan_for_task_(
            model,
            idx_out_task=idx_out[t],
            task_id=t,
            verbose=verbose
        )

    if verbose:
        sanity_check_idx_out(idx_out, model, kappa, num_tasks)
        check_idx_matches_mask(model, idx_out, kappa, num_tasks)
        report_overlap_from_idx(idx_out, model, num_tasks)
        for t in range(num_tasks):
            verify_plan_out_idx_matches_idx_out(plans[t], idx_out[t])
            verify_plan_wiring_consistency(plan_t=plans[t])
            verify_resnet_block_plan(plans[t])
    return plans


def get_block_by_name(model: nn.Module, stage_name: str, bi: int):
    stage = getattr(model, stage_name)          # nn.Sequential
    return stage[bi]

def resolve_op_module(model: nn.Module, op) -> nn.Module:
    """
    op.name examples:
      "stem.conv1"
      "layer2.0.conv1"
      "layer3.1.conv2"
      "layer4.0.down"
    """
    if op.name == "stem.conv1":
        return model.conv1

    parts = op.name.split(".")
    # parts: ["layer2", "0", "conv1"] or ["layer2","0","down"]
    stage_name = parts[0]
    bi = int(parts[1])
    field = parts[2]

    block = get_block_by_name(model, stage_name, bi)
    return getattr(block, field)

@torch.no_grad()
def preview_sliced_weights_for_task(model: nn.Module, plan, task_id: int, device=None):
    out = []
    for op in plan:
        mod = resolve_op_module(model, op)
        W = mod.weight.data
        out_idx = op.out_idx.to(W.device)
        if op.in_idx is None:
            # stem: in is RGB
            Wp = W.index_select(0, out_idx).contiguous()
        else:
            in_idx = op.in_idx.to(W.device)
            Wp = W.index_select(0, out_idx).index_select(1, in_idx).contiguous()

        out.append((op.name, tuple(W.shape), tuple(Wp.shape), Wp.numel()))
        if device is not None:
            Wp = Wp.to(device)
    return out


# --------------------------------------------------------------
# Step 4: numerical equivalence tests
# --------------------------------------------------------------

# 4.1 Utilities: resolve module + slice weight
def slice_conv_weight(W: torch.Tensor, out_idx: torch.Tensor, in_idx: Optional[torch.Tensor]):
    Wp = W.index_select(0, out_idx)
    if in_idx is not None:
        Wp = Wp.index_select(1, in_idx)
    return Wp.contiguous()


# 4.2 Used to Capture conv inputs/outputs for the original model (one task, one batch)
@torch.no_grad()
def capture_conv_ios_for_task(model, plan, x, task_id: int, kappa: float, hard: bool = True, device="cuda"):
    model.eval()
    model.eval_mode = False  # make sure it uses normal gating path during forward

    x = x.to(device)
    model = model.to(device)

    io = {}  # name -> {"x": tensor, "y": tensor, "mod": module}

    hooks = []
    for op in plan:
        mod = resolve_op_module(model, op)

        def make_hook(name):
            def hook(m, inp, out):
                # inp is a tuple; inp[0] is the tensor input
                io[name] = {"x": inp[0].detach(), "y": out.detach(), "mod": m}
            return hook

        hooks.append(mod.register_forward_hook(make_hook(op.name)))

    _ = model(x, task_id=task_id, k_frac=kappa, hard=hard)

    for h in hooks:
        h.remove()

    return io

# Layer equivalence test: full conv vs packed conv
@torch.no_grad()
def test_layer_equivalence(model, plan, io, atol=1e-6, rtol=1e-4, verbose=True):
    results = []
    ok_all = True

    for op in plan:
        entry = io[op.name]
        mod = entry["mod"]
        x_full = entry["x"]
        y_full = entry["y"]

        W = mod.weight.detach()
        out_idx = op.out_idx.to(W.device)
        in_idx = None if op.in_idx is None else op.in_idx.to(W.device)

        # Build packed input
        if in_idx is None:
            x_sel = x_full
        else:
            x_sel = x_full.index_select(1, in_idx)

        # Slice weights
        Wp = slice_conv_weight(W, out_idx, in_idx)

        # Run packed conv
        # bias should be None in model; keep generic
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        if bias is not None:
            bias = bias.index_select(0, out_idx)

        y_pack = F.conv2d(
            x_sel,
            Wp,
            bias=bias,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups
        )

        # Select corresponding output channels from full conv output
        y_full_sel = y_full.index_select(1, out_idx)

        diff = (y_pack - y_full_sel).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        # relative error (avoid divide by 0)
        denom = y_full_sel.abs().mean().item()
        rel = mean_abs / (denom + 1e-12)

        passed = (max_abs <= atol) or (rel <= rtol)
        ok_all = ok_all and passed

        results.append((op.name, tuple(y_full_sel.shape), max_abs, mean_abs, rel, passed))

    if verbose:
        for name, shape, max_abs, mean_abs, rel, passed in results:
            print(f"{name:16s} y={shape}  max={max_abs:.3e}  mean={mean_abs:.3e}  rel={rel:.3e}  {'OK' if passed else 'FAIL'}")

    return ok_all, results

# -------------------------------------------------------------------
# Full Model verification tests
# -------------------------------------------------------------------

# 5.1 Verify packed logits == original logits for that task’s classes
@torch.no_grad()
def test_full_model_equivalence(
    full_model: nn.Module,
    packed_model: nn.Module,
    x: torch.Tensor,
    task_id: int,
    class_indices_per_task: Dict[int, torch.Tensor],
    kappa: float,
    device="cuda"
):
    full_model.eval()
    packed_model.eval()
    x = x.to(device)

    # full model logits for task t
    logits_full = full_model(x, task_id=task_id, k_frac=kappa, hard=True).detach()

    cls = class_indices_per_task[task_id].to(device)
    if cls.dtype == torch.bool:
        cls = cls.nonzero(as_tuple=False).view(-1)
    logits_full_sel = logits_full.index_select(1, cls)

    logits_pack = packed_model(x).detach()

    diff = (logits_pack - logits_full_sel).abs()
    print("logits:", logits_pack.shape,
          "max_abs", diff.max().item(),
          "mean_abs", diff.mean().item())


@torch.no_grad()
def compare_task_logits(
    full_model: nn.Module,
    packed_model: nn.Module,
    x: torch.Tensor,
    task_id: int,
    class_indices_per_task: Dict[int, torch.Tensor],
    kappa: float,
    device: str = "cuda"
):
    dev = torch.device(device)
    x = x.to(dev)
    full_model = full_model.to(dev).eval()
    packed_model = packed_model.to(dev).eval()

    # Full model logits for task t
    logits_full = full_model(x, task_id=task_id, k_frac=kappa, hard=True).detach()

    cls = class_indices_per_task[task_id].to(dev)
    if cls.dtype == torch.bool:
        cls = cls.nonzero(as_tuple=False).view(-1)

    logits_full_sel = logits_full.index_select(1, cls)  # [B, #classes_in_task]

    # Packed model logits (already only task classes)
    logits_pack = packed_model(x).detach()

    diff = (logits_pack - logits_full_sel).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"[t={task_id}] logits shape full_sel={tuple(logits_full_sel.shape)} packed={tuple(logits_pack.shape)}")
    print(f"[t={task_id}] max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}")

    return max_abs, mean_abs


@torch.no_grad()
def debug_gate_values(model, task_id, kappa):
    model.eval()
    # stem
    g = model.conv1.gate_for(task_id).detach().cpu()
    C = g.numel()
    k = max(1, int(kappa * C))
    print("stem gate raw min/max:", g.min().item(), g.max().item(), "C", C, "k", k)
    
    m = apply_topk(g.to(model.conv1.weight.device), kappa, hard=True).detach().cpu()
    print("stem mask unique:", torch.unique(m), "sum", m.sum().item())

@torch.no_grad()
def full_pre_fc_features(model: nn.Module, x: torch.Tensor, task_id: int, kappa: float, hard: bool = True):
    model.eval()
    # run forward manually to capture the tensor right before fc
    # (matches GatedResNet18_Flex forward)
    x = x.to(next(model.parameters()).device)

    # stem
    z = F.relu(model._apply_norm(model.norm1, model.conv1(x), task_id))
    z = z * model._mask(model.conv1, task_id, hard, kappa)
    z = model.pool(z)

    # stages
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            z = block(z, task_id, kappa, hard, acts_dict=None)

    feat = model.avg(z).flatten(1)  # [B, 512]
    return feat

@torch.no_grad()
def packed_pre_fc_features(packed_model: nn.Module, x: torch.Tensor):
    packed_model.eval()
    x = x.to(next(packed_model.parameters()).device)

    z = F.relu(packed_model.norm1(packed_model.conv1(x)))
    z = packed_model.stem_scale(z) if hasattr(packed_model, "stem_scale") else z
    z = packed_model.pool(z)

    for b in packed_model.blocks:
        z = b(z)

    feat = packed_model.avg(z).flatten(1)  # [B, S_last]
    return feat

@torch.no_grad()
def compare_pre_fc_features(full_model, packed_model, x, plan_t, task_id, kappa, device="cuda"):
    dev = torch.device(device)
    x = x.to(dev)
    full_model = full_model.to(dev).eval()
    packed_model = packed_model.to(dev).eval()

    f_full = full_pre_fc_features(full_model, x, task_id=task_id, kappa=kappa, hard=True)  # [B,512]
    f_pack = packed_pre_fc_features(packed_model, x)                                        # [B,S]

    last_out_idx = plan_t[-1].out_idx.to(dev)  # indices into 512
    f_full_sel = f_full.index_select(1, last_out_idx)

    diff = (f_pack - f_full_sel).abs()
    print(f"[t={task_id}] pre-fc feat full_sel={tuple(f_full_sel.shape)} packed={tuple(f_pack.shape)}")
    print(f"[t={task_id}] pre-fc max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")


def sanity_check_idx_out(idx_out, model, kappa, num_tasks):
    for m in model.modules():
        if not isinstance(m, GConv2d):
            continue
        lid = int(m.layer_id)
        C = m.gate_for(0).numel()
        k_expected = max(1, int(kappa * C))

        for t in range(num_tasks):
            idx = idx_out[t][lid]
            # range
            if idx.min().item() < 0 or idx.max().item() >= C:
                print(f"[BAD] layer {lid} task {t}: idx out of range")
            # size
            if idx.numel() != k_expected:
                print(f"[BAD] layer {lid} task {t}: k={idx.numel()} expected={k_expected}")
            # unique + sorted
            if idx.unique().numel() != idx.numel():
                print(f"[BAD] layer {lid} task {t}: duplicate indices")
            if not torch.all(idx[:-1] <= idx[1:]):
                print(f"[BAD] layer {lid} task {t}: not sorted")


@torch.no_grad()
def check_idx_matches_mask(model, idx_out, kappa, num_tasks):
    for m in model.modules():
        if not isinstance(m, GConv2d):
            continue
        lid = int(m.layer_id)
        C = m.gate_for(0).numel()

        for t in range(num_tasks):
            g = m.gate_for(t)
            mask = apply_topk(g, kappa, hard=True).to(torch.bool)
            idx_from_mask = mask.nonzero(as_tuple=False).view(-1).sort().values

            idx = idx_out[t][lid].to(idx_from_mask.device)
            ok = torch.equal(idx, idx_from_mask)

            if not ok:
                # print a minimal diff
                a = set(idx.cpu().tolist())
                b = set(idx_from_mask.cpu().tolist())
                print(f"[BAD] layer {lid} task {t}: idx mismatch "
                      f"| idx_only={len(a-b)} mask_only={len(b-a)} C={C}")
                return False
    print("idx_out matches apply_topk masks for all layers/tasks.")
    return True


@torch.no_grad()
def report_overlap_from_idx(idx_out, model, num_tasks):
    # collect layer ids
    lids = []
    for m in model.modules():
        if isinstance(m, GConv2d):
            lids.append(int(m.layer_id))
    lids = sorted(lids)

    # average overlap fraction across layers: |Si ∩ Sj| / |Si|
    overlap = torch.zeros(num_tasks, num_tasks)
    for i in range(num_tasks):
        for j in range(num_tasks):
            fracs = []
            for lid in lids:
                Si = set(idx_out[i][lid].cpu().tolist())
                Sj = set(idx_out[j][lid].cpu().tolist())
                fracs.append(len(Si & Sj) / max(1, len(Si)))
            overlap[i, j] = sum(fracs) / len(fracs)

    print("avg overlap |Si∩Sj|/|Si| across layers:")
    print(overlap)
    return overlap


def verify_plan_out_idx_matches_idx_out(
    plan_t: List,                      # list of ops for one task
    idx_out_t: Dict[int, torch.Tensor] # idx_out[task_id]
):
    ok = True
    for op in plan_t:
        # Only check ops that correspond to real conv outputs
        # We assume op has fields: name, layer_id (int), out_idx (LongTensor)
        lid = int(op.layer_id)
        if lid not in idx_out_t:
            # e.g., plain Conv2d downsample might have no gate/layer_id in idx_out
            continue

        a = op.out_idx.detach().cpu()
        b = idx_out_t[lid].detach().cpu()

        if a.numel() != b.numel() or not torch.equal(a, b):
            print(f"[BAD] op={op.name} layer_id={lid}: out_idx differs from idx_out")
            # show small diff
            sa, sb = set(a.tolist()), set(b.tolist())
            print("  only_in_plan:", len(sa - sb), "only_in_idx_out:", len(sb - sa))
            ok = False

    print("plan out_idx matches idx_out for all checked ops:", ok)
    return ok


@torch.no_grad()
def verify_plan_wiring_consistency(plan_t: List):
    """
    Checks the basic invariant:
      For each conv op with in_idx:
        in_idx must equal prev_tensor_out_idx (same values, same order)
    This assumes plan_t is in forward execution order.
    """
    ok = True

    prev_out = None
    prev_name = None

    for op in plan_t:
        # stem sets the initial prev_out
        if op.name == "stem.conv1":
            prev_out = op.out_idx.detach().cpu()
            prev_name = op.name
            continue

        # For normal conv ops, enforce in_idx == prev_out
        # (This is true in a strictly sequential chain.)
        # For residual blocks, plan builder must set in_idx according
        # to the tensor that actually feeds this op.
        if op.kind == "conv":  
            if op.in_idx is not None:
                a = op.in_idx.detach().cpu()
                b = prev_out
                if b is None:
                    print(f"[BAD] op={op.name}: has in_idx but prev_out is None")
                    ok = False
                elif a.numel() != b.numel() or not torch.equal(a, b):
                    print(f"[BAD] op={op.name}: in_idx != prev_out from {prev_name}")
                    print(f"  in_idx len={a.numel()} prev_out len={b.numel()}")
                    # helpful: how different?
                    sa, sb = set(a.tolist()), set(b.tolist())
                    print("  only_in_in_idx:", len(sa - sb), "only_in_prev_out:", len(sb - sa))
                    ok = False

            # update prev_out after this op
            prev_out = op.out_idx.detach().cpu()
            prev_name = op.name

        else:
            # For ops that don't change channels, keep prev_out as is
            pass

    print("plan wiring consistency (sequential check):", ok)
    return ok



@torch.no_grad()
def verify_resnet_block_plan(plan_t: List):
    """
    Assumes per block stored plan entries in order:
      conv1, conv2, optional down
    and each op has: name, in_idx, out_idx
    """
    ok = True
    i = 1  # skip stem at 0
    while i < len(plan_t):
        op1 = plan_t[i]     # *.conv1
        op2 = plan_t[i + 1] # *.conv2

        # conv2 input must match conv1 output
        if op2.in_idx is None:
            print(f"[BAD] {op2.name}: missing in_idx")
            ok = False
        else:
            if op2.in_idx.numel() != op1.out_idx.numel() or not torch.equal(op2.in_idx.cpu(), op1.out_idx.cpu()):
                print(f"[BAD] block {op1.name.split('.conv1')[0]}: conv2.in_idx != conv1.out_idx")
                ok = False

        # optional down
        has_down = (i + 2 < len(plan_t)) and plan_t[i + 2].name.endswith(".down")
        if has_down:
            opd = plan_t[i + 2]

            # down must map block input -> block output
            if opd.in_idx is None or op1.in_idx is None:
                print(f"[BAD] {opd.name}: missing in_idx (or conv1 missing in_idx)")
                ok = False
            else:
                if opd.in_idx.numel() != op1.in_idx.numel() or not torch.equal(opd.in_idx.cpu(), op1.in_idx.cpu()):
                    print(f"[BAD] {opd.name}: down.in_idx != conv1.in_idx (block input selection)")
                    ok = False

            if opd.out_idx.numel() != op2.out_idx.numel() or not torch.equal(opd.out_idx.cpu(), op2.out_idx.cpu()):
                print(f"[BAD] {opd.name}: down.out_idx != conv2.out_idx (block output selection)")
                ok = False

            i += 3
        else:
            i += 2

    print("block-level plan consistency:", ok)
    return ok


import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 0) Name mapping fix
# -----------------------------
def _plan_name_to_model_name(name: str) -> str:
    # Plan uses "stem.conv1" but the model is "conv1"
    if name == "stem.conv1":
        return "conv1"
    return name

# -----------------------------
# 1) Capture conv inputs for one forward
# -----------------------------
@torch.no_grad()
def _capture_conv_inputs(model: nn.Module, xb: torch.Tensor, *, task_id: int, kappa: float, hard: bool, device: str):
    model.eval().to(device)
    xb = xb.to(device)

    inputs = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            inputs[name] = inp[0].detach()
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    # run one forward to populate inputs
    _ = model(xb, task_id=task_id, k_frac=kappa, hard=hard)

    for h in hooks:
        h.remove()

    return inputs

# -----------------------------
# 2) Core check for one op
# -----------------------------
@torch.no_grad()
def _check_one_conv_op(model: nn.Module, op, x_full: torch.Tensor, *,
                       device: str,
                       zero_eps: float,
                       atol: float,
                       rtol: float,
                       cpu_double: bool):
    """
    Checks conv slicing equivalence for one op.
    Also checks that dropped input channels are ~0 if in_idx is used.
    """

    # resolve module
    named = dict(model.named_modules())
    op_name_model = _plan_name_to_model_name(op.name)

    if op_name_model not in named:
        print(f"[MISSING MODULE] plan={op.name} -> model={op_name_model}")
        return False

    full_conv = named[op_name_model]
    if not isinstance(full_conv, nn.Conv2d):
        print(f"[NOT CONV] {op.name} -> {type(full_conv)}")
        return False

    # indices
    out_idx = op.out_idx
    in_idx = op.in_idx if getattr(op, "in_idx", None) is not None else None

    # choose backend for comparison
    if cpu_double:
        # strict + deterministic reference
        x = x_full.detach().cpu().double()
        W = full_conv.weight.detach().cpu().double()
        b = full_conv.bias.detach().cpu().double() if full_conv.bias is not None else None
        out_idx_d = out_idx.cpu()
        in_idx_d = None if in_idx is None else in_idx.cpu()

        # check dropped channels near zero (CPU)
        if in_idx_d is not None:
            Cin = x.size(1)
            keep = torch.zeros(Cin, dtype=torch.bool)
            keep[in_idx_d] = True
            dropped = x[:, ~keep, :, :]
            max_drop = dropped.abs().max().item() if dropped.numel() > 0 else 0.0
            zero_ok = (max_drop <= zero_eps)
        else:
            max_drop = 0.0
            zero_ok = True

        # full ref
        y_full = F.conv2d(x, W, b, stride=full_conv.stride, padding=full_conv.padding,
                          dilation=full_conv.dilation, groups=full_conv.groups)
        y_ref = y_full.index_select(1, out_idx_d)

        # packed
        Wp = W.index_select(0, out_idx_d)
        bp = None if b is None else b.index_select(0, out_idx_d)
        if in_idx_d is not None:
            x_p = x.index_select(1, in_idx_d)
            Wp = Wp.index_select(1, in_idx_d)
        else:
            x_p = x

        y_p = F.conv2d(x_p, Wp.contiguous(), bp, stride=full_conv.stride, padding=full_conv.padding,
                       dilation=full_conv.dilation, groups=full_conv.groups)

        diff = (y_ref - y_p).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = y_ref.abs().mean().item() + 1e-12
        rel = mean_abs / denom

        ok = (max_abs <= atol) or torch.allclose(y_ref, y_p, atol=atol, rtol=rtol)

        print(f"{op.name:16s} CPU64  x={tuple(x.shape)} y={tuple(y_ref.shape)} "
              f"drop_max={max_drop:.3e} {'ZERO' if zero_ok else 'NZ'} "
              f"max={max_abs:.3e} mean={mean_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
        return bool(ok and zero_ok)

    else:
        # CUDA / normal float
        device = device
        x = x_full.to(device)
        full_conv = full_conv.to(device)
        W = full_conv.weight.detach()
        b = full_conv.bias.detach() if full_conv.bias is not None else None
        out_idx_d = out_idx.to(device)
        in_idx_d = None if in_idx is None else in_idx.to(device)

        # check dropped channels ~0
        if in_idx_d is not None:
            Cin = x.size(1)
            keep = torch.zeros(Cin, dtype=torch.bool, device=device)
            keep[in_idx_d] = True
            dropped = x[:, ~keep, :, :]
            max_drop = dropped.abs().max().item() if dropped.numel() > 0 else 0.0
            zero_ok = (max_drop <= zero_eps)
        else:
            max_drop = 0.0
            zero_ok = True

        # ref
        y_full = full_conv(x)
        y_ref = y_full.index_select(1, out_idx_d)

        # packed
        Wp = W.index_select(0, out_idx_d)
        bp = None if b is None else b.index_select(0, out_idx_d)
        if in_idx_d is not None:
            x_p = x.index_select(1, in_idx_d)
            Wp = Wp.index_select(1, in_idx_d)
        else:
            x_p = x

        y_p = F.conv2d(x_p, Wp.contiguous(), bp,
                       stride=full_conv.stride, padding=full_conv.padding,
                       dilation=full_conv.dilation, groups=full_conv.groups)

        diff = (y_ref - y_p).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = y_ref.abs().mean().item() + 1e-12
        rel = mean_abs / denom

        ok = torch.allclose(y_ref, y_p, atol=atol, rtol=rtol)

        print(f"{op.name:16s} CUDA   x={tuple(x.shape)} y={tuple(y_ref.shape)} "
              f"drop_max={max_drop:.3e} {'ZERO' if zero_ok else 'NZ'} "
              f"max={max_abs:.3e} mean={mean_abs:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
        return bool(ok and zero_ok)

# -----------------------------
# 3) One-call runner
# -----------------------------
@torch.no_grad()
def check_plan_conv_slicing(full_model: nn.Module, plan_t, xb: torch.Tensor, *,
                            task_id: int,
                            kappa: float,
                            hard: bool = True,
                            device: str = "cuda",
                            atol: float = 5e-5,
                            rtol: float = 1e-4,
                            # dropped channels should be exactly zero if masking logic is correct.
                            zero_eps: float = 0.0,
                            # if True, checks on CPU float64 (slow but definitive)
                            cpu_double: bool = False):
    """
    Single command: captures full conv inputs for one forward and checks plan conv slicing.
    Returns True only if ALL ops pass (including dropped-channels-zero check when in_idx exists).
    """

    inputs = _capture_conv_inputs(full_model, xb, task_id=task_id, kappa=kappa, hard=hard, device=device)

    all_ok = True
    for op in plan_t:
        # only conv ops in the plan
        name_model = _plan_name_to_model_name(op.name)
        if name_model not in inputs:
            print(f"[WARN] no captured input for plan={op.name} -> model={name_model}")
            all_ok = False
            continue

        x_full = inputs[name_model]
        ok = _check_one_conv_op(full_model, op, x_full,
                                device=device, zero_eps=zero_eps,
                                atol=atol, rtol=rtol, cpu_double=cpu_double)
        all_ok = all_ok and ok

    print(f"\nTask {task_id} ALL CONV OPS OK: {all_ok}")
    return all_ok


@torch.no_grad()
def compare_logits_fullsel_vs_packed(full_model, packed_model, xb, task_id, kappa, class_indices_per_task, last_out_idx, device="cuda"):
    full_model.eval().to(device)
    packed_model.eval().to(device)
    xb = xb.to(device)

    # --- capture pre-fc feature from full model
    feat_holder = {}
    def hook_fn(mod, inp, out):
        feat_holder["feat"] = inp[0].detach()  # fc input is [B, 512]
    h = full_model.fc.register_forward_hook(hook_fn)

    logits_full = full_model(xb, task_id=task_id, k_frac=kappa, hard=True)
    h.remove()

    feat_full = feat_holder["feat"]  # [B, 512]

    # --- build full_sel reference logits using the SAME slicing as packed
    cls = class_indices_per_task[task_id].to(device)
    if cls.dtype == torch.bool:
        cls = cls.nonzero(as_tuple=False).view(-1)

    W = full_model.fc.weight.detach().to(device)   # [num_classes, 512]
    b = full_model.fc.bias.detach().to(device) if full_model.fc.bias is not None else None

    feat_sel = feat_full.index_select(1, last_out_idx.to(device))  # [B, S_last]
    W_sel = W.index_select(0, cls).index_select(1, last_out_idx.to(device))  # [|cls|, S_last]
    b_sel = None if b is None else b.index_select(0, cls)

    logits_ref = feat_sel @ W_sel.t()
    if b_sel is not None:
        logits_ref = logits_ref + b_sel

    # --- packed logits
    logits_packed = packed_model(xb)  # [B, |cls|] if packed model is per-task

    # --- compare
    diff = (logits_ref - logits_packed).abs()
    print(f"[t={task_id}] logits_ref={tuple(logits_ref.shape)} packed={tuple(logits_packed.shape)} "
          f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")
    return logits_ref, logits_packed
