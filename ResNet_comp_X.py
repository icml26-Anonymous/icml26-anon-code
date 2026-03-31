import torch.optim as optim
from Utility_functions import *
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Optional per-task BatchNorm (keeps separate running stats per task) ---------
class TaskBN2d(nn.Module):
    def __init__(self, num_features: int, max_tasks: int = 50, affine: bool = True, eps: float = 1e-5, momentum: float = 0.1, device='cuda'):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        # Create a BN module per task id on demand; keep in a ModuleDict
        self.bns = nn.ModuleDict().to(device)
        # Shared affine params across tasks (optional; set affine=False to disable)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features)).to(device)
            self.bias   = nn.Parameter(torch.zeros(num_features)).to(device)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _get_bn_for(self, task_id: int) -> nn.BatchNorm2d:
        key = str(int(task_id))
        if key not in self.bns:
            bn = nn.BatchNorm2d(self.num_features, affine=False, eps=self.eps, momentum=self.momentum, track_running_stats=True)
            self.bns[key] = bn
        return self.bns[key]

    def forward(self, x: torch.Tensor, task_id: int, device='cuda'):
        bn = self._get_bn_for(task_id).to(device)
        x = bn(x)
        if self.affine:
            # apply shared affine after task-specific running stats
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            x = x * w + b
        return x

# ------------------------- Norm factory -------------------------
def make_norm(norm_type: str, num_features: int, *, per_task_bn: bool = False):
    norm_type = norm_type.lower()
    if norm_type == "bn":
        if per_task_bn:
            return TaskBN2d(num_features, affine=True)
        else:
            return nn.BatchNorm2d(num_features)
    elif norm_type == "gn":
        # Use groups=min(32, C) to avoid C%G!=0 on small channels
        groups = min(num_features, num_features)
        return nn.GroupNorm(groups, num_features)
    elif norm_type == "in":
        return nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ------------------------- Single-head with optional protection -------------------------
class SingleHeadWithProtection(GLinear):
    """
    Single shared head (num_classes). During training of task t you can:
      * freeze rows (class columns) not in current task (zero their weight/bias grads),
      * optionally mask non-current logits to -inf before CE to avoid acting as negatives.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__(in_dim, num_classes)
        self._freeze_hook_w = None
        self._freeze_hook_b = None
        self.class_indices_per_task = None
        self.protect_mode = dict(
            freeze_old_cols=True,
            mask_non_current_logits=False
        )

    def set_class_slices(self, class_indices_per_task):
        """
        class_indices_per_task: Dict[int, LongTensor] or List[LongTensor]
        mapping task_id -> tensor of class ids that belong to that task (global labels).
        """
        self.class_indices_per_task = class_indices_per_task

    def configure_protection(self, *, freeze_old_cols: bool = True, mask_non_current_logits: bool = False):
        self.protect_mode["freeze_old_cols"] = freeze_old_cols
        self.protect_mode["mask_non_current_logits"] = mask_non_current_logits

    def _install_freeze_hooks(self, t: int):
        if not self.protect_mode["freeze_old_cols"]:
            self._remove_hooks()
            return
        assert self.class_indices_per_task is not None, "Call set_class_slices() first."

        current = self.class_indices_per_task[t].to(self.weight.device)
        allow = torch.zeros(self.out_features, dtype=torch.bool, device=self.weight.device)
        allow[current] = True

        def hook_w(grad):
            g = grad.clone()
            g[~allow] = 0.0
            return g

        def hook_b(grad):
            g = grad.clone()
            g[~allow] = 0.0
            return g

        self._remove_hooks()
        self._freeze_hook_w = self.weight.register_hook(hook_w)
        if self.bias is not None:
            self._freeze_hook_b = self.bias.register_hook(hook_b)

    def _remove_hooks(self):
        if self._freeze_hook_w is not None:
            self._freeze_hook_w.remove()
            self._freeze_hook_w = None
        if self._freeze_hook_b is not None:
            self._freeze_hook_b.remove()
            self._freeze_hook_b = None

    def mask_logits_for_loss(self, logits, t: int):
        if not self.protect_mode["mask_non_current_logits"]:
            return logits
        assert self.class_indices_per_task is not None, "Call set_class_slices() first."
        current = self.class_indices_per_task[t].to(logits.device)
        mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[current] = True
        logits = logits.clone()
        logits[:, ~mask] = -1e9
        return logits

# ------------------------- Configurable Gated ResNet-18 -------------------------
class GatedResNet18_Flex(nn.Module):
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
        num_classes: int = 10,
        kappa: float = 0.20,
        lr: float = 5e-2,
        lr_min: float = 1e-4,
        lr_patience: int = 5,
        lr_factor: int = 3,
        *,
        stem: str = "cifar",
        norm_type: str = "bn",
        per_task_bn: bool = False,
        gate_skip: bool = False,
        gate_head: bool = False,
        mu = None,
        sigma = None
    ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.kappa = kappa
        self.stem_type = stem
        self.norm_type = norm_type
        self.per_task_bn = per_task_bn
        self.gate_skip = gate_skip
        self.gate_head = gate_head

        self.current_l_id = 0

        # ---- stem
        if stem == "imagenet":
            self.conv1 = GConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = make_norm(norm_type, 64, per_task_bn=per_task_bn)
            self.pool  = nn.Identity()
        elif "CIF" in stem:
            self.conv1 = GConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.norm1 = make_norm(norm_type, 64, per_task_bn=per_task_bn)
            self.pool  = nn.Identity()
        else:
            raise ValueError("stem must be 'cifar' or 'imagenet'")

        self.conv1.layer_id = self.current_l_id
        self.current_l_id += 1
        # ---- stages
        self.layer1 = self._make_layer(64,  64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128,256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256,512, blocks=2, stride=2)

        self.avg = nn.AdaptiveAvgPool2d(1)
        if gate_head:
            self.fc = GLinear(512, num_classes)  # gated head (rare for single-head)
        else:
            # single shared head with protection knobs
            self.fc = SingleHeadWithProtection(512, num_classes)

        # create task-0 gates for all GatedLayers
        for m in self.modules():
            if isinstance(m, GatedLayer):
                m.new_task_gate()

        self.lr_patience = lr_patience
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.optimizer = self._get_optimizer(self.lr)

        # will be filled by trainer if call set_class_slices
        self.class_indices_per_task = None
        self.eval_mode = False

    def set_class_slices(self, class_indices_per_task):
        """class_indices_per_task: Dict[int, LongTensor] (global label ids per task)."""
        self.class_indices_per_task = class_indices_per_task
        if isinstance(self.fc, SingleHeadWithProtection):
            self.fc.set_class_slices(class_indices_per_task)

    def configure_head_protection(self, *, freeze_old_cols: bool = True, mask_non_current_logits: bool = False):
        if isinstance(self.fc, SingleHeadWithProtection):
            self.fc.configure_protection(freeze_old_cols=freeze_old_cols, mask_non_current_logits=mask_non_current_logits)

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        # Stronger recipe typically used for Tiny-ImageNet
        # return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlockFlex(in_ch, out_ch, stride, kappa=self.kappa,
                                 norm_type=self.norm_type, per_task_bn=self.per_task_bn,
                                 gate_skip=self.gate_skip, layer_id=self.current_l_id)]

        self.current_l_id = layers[-1].cur_layer_id

        for _ in range(1, blocks):
            layers.append(BasicBlockFlex(out_ch, out_ch, 1, kappa=self.kappa,
                                         norm_type=self.norm_type, per_task_bn=self.per_task_bn,
                                         gate_skip=self.gate_skip, layer_id=self.current_l_id))

        self.current_l_id = layers[-1].cur_layer_id
        return nn.Sequential(*layers)

    def _mask(self, layer: GatedLayer, task_id: int, hard: bool, k: float):
        m = apply_topk(layer.gate_for(task_id), k, hard)
        return m[:, None, None]

    def _apply_norm(self, norm_mod, x, task_id):
        if isinstance(norm_mod, TaskBN2d):
            return norm_mod(x, task_id)
        return norm_mod(x)

    def forward(self, x, *, task_id: int, k_frac: float = None, hard: bool = True, mask_logits_for_loss: bool = False):
        if k_frac is not None:
            self.kappa = k_frac
        acts = {}

        # stem
        x = F.relu(self._apply_norm(self.norm1, self.conv1(x), task_id))
        # if not self.eval_mode:
        x = x * self._mask(self.conv1, task_id, hard, self.kappa)
        x = self.pool(x)
        acts[self.conv1] = x.detach()

        # stages
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in stage:
                x = block(x, task_id, self.kappa, hard, acts)

        x = self.avg(x).flatten(1)

        # head
        if self.eval_mode:
            logits = self.fc(x)
        else:
            if self.gate_head:
                m_fc = apply_topk(self.fc.gate_for(task_id), self.kappa, hard)  # [C_out]
                logits = self.fc(x)
                logits = logits * m_fc  # per-class gating (rare in single-head settings)
            else:
                # install freeze hook for current task (classifier protection)
                if isinstance(self.fc, SingleHeadWithProtection):
                    self.fc._install_freeze_hooks(task_id)
                logits = self.fc(x)
                if mask_logits_for_loss and isinstance(self.fc, SingleHeadWithProtection):
                    logits = self.fc.mask_logits_for_loss(logits, task_id)

        self.last_acts = acts
        return logits

    def class_il_predict_sequential(self, xb, n_tasks, kappa_targets=None, criterion='confidence', norm_mode='z', eps = 1e-8):
        best_score = -1e9
        best_t = None
        best_logits = None
        mu = self.mu
        sigma = self.sigma
        if kappa_targets is None:
            kappa_targets = [1/n_tasks for i in range(n_tasks)]

        for t in range(n_tasks):
            kappa = kappa_targets[t]
            logits = self.forward(xb, task_id=t, k_frac=kappa)

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

        return best_logits.argmax(1)

    @torch.no_grad()
    def task_il_predict(self, x, t, kappa, device: str = "cuda"):
        logits = self.forward(x, task_id=t, k_frac=kappa)
        pred = logits.argmax(1)
        return pred

# ------------------------- Residual Block -------------------------
class BasicBlockFlex(nn.Module):
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
        self.norm1 = make_norm(norm_type, out_ch, per_task_bn=per_task_bn)
        self.conv2 = GConv2d(out_ch, out_ch, 3, stride=1,    padding=1, bias=False)
        self.norm2 = make_norm(norm_type, out_ch, per_task_bn=per_task_bn)

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
            self.down_norm = make_norm(norm_type, out_ch, per_task_bn=per_task_bn)

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

    def _apply_norm(self, norm_mod, x, task_id):
        if isinstance(norm_mod, TaskBN2d):
            return norm_mod(x, task_id)
        return norm_mod(x)

    def forward(self, x, task_id: int, k: float, hard: bool = True, acts_dict=None):
        # conv1 (gated AFTER norm+relu)
        out1 = F.relu(self._apply_norm(self.norm1, self.conv1(x), task_id))
        out1 = out1 * self._mask(self.conv1, task_id, hard, k)
        if acts_dict is not None:
            acts_dict[self.conv1] = out1.detach()

        # conv2 (gated AFTER norm)
        out2 = self._apply_norm(self.norm2, self.conv2(out1), task_id)

        if acts_dict is not None:
            acts_dict[self.conv2] = out2.detach()

        # residual
        if self.down is not None:
            skip = self._apply_norm(self.down_norm, self.down(x), task_id)
            # print(self.down.layer_id)
            if self.down_is_gated:
                # print(f"Gated skip connection from L: {self.conv1.layer_id-1} to L: {self.conv2.layer_id} connecting via {self.down.layer_id}")
                skip = skip * self._mask(self.down, task_id, hard, k)
            if acts_dict is not None:
                acts_dict[self.down] = skip.detach()
        else:
            # print(f"Non-Gated skip connection from L: {self.conv1.layer_id-1} to L: {self.conv2.layer_id}")
            skip = x

        out = out2 + skip

        m = self._mask(self.conv2, task_id, hard, k)
        out = out * m
        out = F.relu(out)

        self.last_acts = acts_dict
        return out


    def forward_(self, x, task_id: int, k: float, hard: bool = True, acts_dict=None):
        # conv1 (gated AFTER norm+relu)
        out1 = F.relu(self._apply_norm(self.norm1, self.conv1(x), task_id))
        out1 = out1 * self._mask(self.conv1, task_id, hard, k)
        if acts_dict is not None:
            acts_dict[self.conv1] = out1.detach()

        # conv2 (gated AFTER norm)
        out2 = self._apply_norm(self.norm2, self.conv2(out1), task_id)

        if acts_dict is not None:
            acts_dict[self.conv2] = out2.detach()

        # residual
        if self.down is not None:
            skip = self._apply_norm(self.down_norm, self.down(x), task_id)
            # print(self.down.layer_id)
            if self.down_is_gated:
                # print(f"Gated skip connection from L: {self.conv1.layer_id-1} to L: {self.conv2.layer_id} connecting via {self.down.layer_id}")
                skip = skip * self._mask(self.down, task_id, hard, k)
            if acts_dict is not None:
                acts_dict[self.down] = skip.detach()
            out = out2 + skip
            m = self._mask(self.conv2, task_id, hard, k)
            # if self.conv2.layer_id not in [6, 11, 16]:
            out = out * m
        else:
            m = self._mask(self.conv2, task_id, hard, k)
            # if self.conv2.layer_id not in [6, 11, 16]:
            out2 = out2 * m
            out = out2 + x

        out = F.relu(out)
        self.last_acts = acts_dict
        return out
