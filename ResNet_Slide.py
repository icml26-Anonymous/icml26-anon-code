import torch.optim as optim
from Utility_functions import *
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- Norm factory -------------------------
def make_norm(norm_type: str, num_features: int):
    norm_type = norm_type.lower()
    if norm_type == "in":
        return nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

def resolve_plan_name_to_model_name(plan_name: str) -> str:
    #  model has conv1/norm1 at top-level, not stem.*
    if plan_name == "stem.conv1":
        return "conv1"
    if plan_name == "stem.norm1":
        return "norm1"
    return plan_name

@torch.no_grad()
def make_packed_Gconv_from_op(full_mod: nn.Module, op, task_id, bias_ok=False) -> nn.Conv2d:
    if isinstance(full_mod, GConv2d):
        W = full_mod.weight.detach()
        print(op.name)
        out_idx = op.out_idx.to(W.device)
        in_idx = None if op.in_idx is None else op.in_idx.to(W.device)

        Wp = W.index_select(0, out_idx)
        if in_idx is not None:
            Wp = Wp.index_select(1, in_idx)
        Wp = Wp.contiguous()

        if full_mod.bias is not None:
            print(full_mod.bias)

        conv = GConv2d(
            in_channels=Wp.size(1),
            out_channels=Wp.size(0),
            kernel_size=full_mod.kernel_size,
            stride=full_mod.stride,
            padding=full_mod.padding,
            bias=(full_mod.bias is not None) if bias_ok else False
        )
        conv.weight.data.copy_(Wp)

        if conv.bias is not None:
            b = full_mod.bias.detach().index_select(0, out_idx).contiguous()
            conv.bias.data.copy_(b)

        conv.new_task_gate()
        gate_value = full_mod._gate_bank[task_id].max()
        gate_ = torch.ones_like(conv._gate_bank[0]) * gate_value.item()
        conv._gate_bank[0] = gate_
        return conv
    else:
        print('Gated Conv is not GConv')
        exit()

def resolve_op_module(full_model: nn.Module, op):
    named = dict(full_model.named_modules())
    model_name = resolve_plan_name_to_model_name(op.name)
    if model_name not in named:
        raise KeyError(
            f"Plan op name '{op.name}' -> model name '{model_name}' not found. "
            f"Example keys: {list(named.keys())[:30]}"
        )
    return named[model_name]

# ------------------------- Configurable Gated ResNet-18 -------------------------
class GatedResNet18_Slice(nn.Module):
    """
    Flags:
      - stem: 'cifar' (3x3 stride1 no pool) | 'imagenet' (7x7/2 + maxpool/2)
      - norm_type: 'bn' | 'gn'
      - per_task_bn: bool (only if norm_type='bn'): use TaskBN2d with separate running stats per task
      - gate_skip: bool (gate downsample branch too)
      - gate_head: bool (apply gating to head as well; typically False for single-head)
      - protect_head: dict(freeze_old_cols: bool, mask_non_current_logits: bool)
    """
    def __init__(
        self,
        full_model: nn.Module,
        plan_t: List,
        class_indices_per_task: Dict[int, torch.Tensor],
        num_classes: int = 10,
        kappa: float = 0.20,
        lr: float = 5e-2,
        lr_min: float = 1e-4,
        lr_patience: int = 5,
        lr_factor: int = 3,
        *,
        stem: str = "CIF100",
        norm_type: str = "in",
        per_task_bn: bool = False,
        gate_skip: bool = False,
        gate_head: bool = False,
        device="cuda",
    ):
        super().__init__()
        self.kappa = kappa
        self.stem_type = stem
        self.norm_type = norm_type
        self.per_task_bn = per_task_bn
        self.gate_skip = gate_skip
        self.gate_head = gate_head

        self.current_l_id = 0

        self.plan = plan_t

        named = dict(full_model.named_modules())

        # ---- STEM
        stem_op = plan_t[0]
        if stem_op.name != "stem.conv1":
            raise ValueError(f"Expected first op 'stem.conv1', got '{stem_op.name}'")

        _, stem_C = self.layer_plan_info(0)
        stem_full_conv = resolve_op_module(full_model, stem_op).to(device)
        print(stem_full_conv)
        # ---- stem
        if self.stem_type == "imagenet":
            self.conv1 = GConv2d(3, stem_C, kernel_size=stem_full_conv.kernel_size, stride=stem_full_conv.stride, padding=stem_full_conv.padding, bias=False)
            self.norm1 = make_norm(norm_type, stem_C)
            self.pool  = nn.Identity()
        elif "CIF" in self.stem_type:
            self.conv1 = GConv2d(3, stem_C, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = make_norm(norm_type, stem_C)
            self.pool  = nn.Identity()
        else:
            raise ValueError("stem must be 'cifar' or 'imagenet'")

        self.conv1 = self.pass_weights(self.conv1, stem_full_conv, stem_op)
        print(self.conv1)
        exit()
        self.conv1.layer_id = self.current_l_id
        self.current_l_id += 1
        # ---- stages
        self.layer1 = self._make_layer(64,  64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128,256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256,512, blocks=2, stride=2)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc = GLinear(512, num_classes)

        # create task-0 gates for all GatedLayers
        for m in self.modules():
            if isinstance(m, GatedLayer):
                m.new_task_gate()

        # Optimizer defaults ( can override externally)
        self.lr_patience = lr_patience
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.optimizer = self._get_optimizer(self.lr)

        # will be filled by trainer if  call set_class_slices
        self.class_indices_per_task = None
        self.eval_mode = False

    def layer_plan_info(self, layer_id):
        out_C = self.plan[layer_id].out_idx.shape[0]
        if self.plan[layer_id].in_idx is not None:
            in_C = self.plan[layer_id].in_idx.shape[0]
        else:
            in_C = None
        return in_C, out_C

    def pass_weights(self, conv: nn.Module, full_mod: nn.Module, op, *, bias_ok=False) -> GConv2d:
        W = full_mod.weight.detach()
        print(op.name)
        out_idx = op.out_idx.to(W.device)
        in_idx = None if op.in_idx is None else op.in_idx.to(W.device)

        Wp = W.index_select(0, out_idx)
        if in_idx is not None:
            Wp = Wp.index_select(1, in_idx)
        Wp = Wp.contiguous()

        allOK = True
        for id_num, idx in enumerate(out_idx):
            diff = W.index_select(0, idx) - Wp[id_num, :, :, :]
            if diff.sum() > 0:
                print(f'Layer: {op.name} - Not correct passing!')
                print(W.index_select(0, out_idx[0]))
                print(Wp[0, :, :, :])
                allOK = False
        if allOK:
            print(f'Layer: {op.name} - All ok!')

        conv.weight.data.copy_(Wp)

        if conv.bias is not None:
            b = full_mod.bias.detach().index_select(0, out_idx).contiguous()
            conv.bias.data.copy_(b)

        return conv

    def set_class_slices(self, class_indices_per_task):
        """class_indices_per_task: Dict[int, LongTensor] (global label ids per task)."""
        self.class_indices_per_task = class_indices_per_task
        if isinstance(self.fc, GLinear):
            self.fc.set_class_slices(class_indices_per_task)

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        # Stronger recipe typically used for Tiny-ImageNet
        # return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [Slided_Block(in_ch, out_ch, stride, kappa=self.kappa,
                                 norm_type=self.norm_type, per_task_bn=self.per_task_bn,
                                 gate_skip=self.gate_skip, layer_id=self.current_l_id)]

        self.current_l_id = layers[-1].cur_layer_id

        for _ in range(1, blocks):
            layers.append(Slided_Block(out_ch, out_ch, 1, kappa=self.kappa,
                                         norm_type=self.norm_type, per_task_bn=self.per_task_bn,
                                         gate_skip=self.gate_skip, layer_id=self.current_l_id))

        self.current_l_id = layers[-1].cur_layer_id
        return nn.Sequential(*layers)

    def _mask(self, layer: GatedLayer, task_id: int, hard: bool, k: float):
        m = apply_topk(layer.gate_for(task_id), k, hard)
        return m[:, None, None]

    def _apply_norm(self, norm_mod, x):
        return norm_mod(x)

    def forward(self, x, *, task_id: int, k_frac: float = None, hard: bool = True, mask_logits_for_loss: bool = False):
        if k_frac is not None:
            self.kappa = k_frac
        acts = {}

        # stem
        x = F.relu(self._apply_norm(self.norm1, self.conv1(x)))
        x = x * self._mask(self.conv1, task_id, hard, self.kappa)
        x = self.pool(x)
        acts[self.conv1] = x.detach()

        # stages
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in stage:
                x = block(x, task_id, self.kappa, hard, acts)

        x = self.avg(x).flatten(1)


        logits = self.fc(x)

        self.last_acts = acts
        return logits


# ------------------------- Residual Block -------------------------
class Slided_Block(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        *,
        kappa: float,
        norm_type: str = "in",
        per_task_bn: bool = False,
        gate_skip: bool = False,  # if True, gate the downsample branch as well
        layer_id: int = 1
    ):
        super().__init__()
        self.conv1 = GConv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = make_norm(norm_type, out_ch)
        self.conv2 = GConv2d(out_ch, out_ch, 3, stride=1,    padding=1, bias=False)
        self.norm2 = make_norm(norm_type, out_ch)

        self.down = None
        self.down_norm = None
        self.down_is_gated = gate_skip

        self.conv1.layer_id = layer_id
        self.conv2.layer_id = layer_id + 1


        if stride != 1 or in_ch != out_ch:
            if gate_skip:
                self.down = GConv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
                self.down.layer_id = layer_id + 2
            else:
                self.down = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            self.down_norm = make_norm(norm_type, out_ch)

            self.cur_layer_id = layer_id + 3
        else:
            self.cur_layer_id = layer_id + 2
        self.kappa = kappa
        self.norm_type = norm_type
        self.per_task_bn = per_task_bn
        self.gate_skip = gate_skip

    def _mask(self, layer: GatedLayer, task_id: int, hard: bool, k: float):
        m = apply_topk(layer.gate_for(task_id), k, hard)
        return m[:, None, None]

    def _apply_norm(self, norm_mod, x):
        return norm_mod(x)

    def forward(self, x, task_id: int, k: float, hard: bool = True, acts_dict=None):
        # conv1 (gated AFTER norm+relu)
        out1 = F.relu(self._apply_norm(self.norm1, self.conv1(x)))
        out1 = out1 * self._mask(self.conv1, task_id, hard, k)
        if acts_dict is not None:
            acts_dict[self.conv1] = out1.detach()

        # conv2 (gated AFTER norm)
        out2 = self._apply_norm(self.norm2, self.conv2(out1))

        if acts_dict is not None:
            acts_dict[self.conv2] = out2.detach()

        # residual
        if self.down is not None:
            skip = self._apply_norm(self.down_norm, self.down(x))
            # print(self.down.layer_id)
            if self.down_is_gated:
                # print(f"Gated skip connection from L: {self.conv1.layer_id-1} to L: {self.conv2.layer_id} connecting via {self.down.layer_id}")
                skip = skip * self._mask(self.down, task_id, hard, k)
        else:
            # print(f"Non-Gated skip connection from L: {self.conv1.layer_id-1} to L: {self.conv2.layer_id}")
            skip = x

        out = out2 + skip

        m = self._mask(self.conv2, task_id, hard, k)
        # if self.conv2.layer_id not in [6, 11, 16]:
        out = out * m
        out = F.relu(out)

        return out


