import torch.optim as optim
from Utility_functions import *
import torch.nn as nn
from torchvision.models import resnet18 as tv_resnet18

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 784, hid: int = 256, out_dim: int = 10):
        super().__init__()
        self.fc1 = GLinear(in_dim, hid)
        self.fc2 = GLinear(hid, hid)
        self.fc3 = GLinear(hid, out_dim)

    def forward(self, x: torch.Tensor, task_id: int, k_frac: float = 0.3):
        mask1 = apply_topk(self.fc1.gate_for(task_id).to(x.device), k_frac)
        h1 = F.relu(self.fc1(x) * mask1)
        self.last_h1 = h1.detach()               # keep for reward
        mask2 = apply_topk(self.fc2.gate_for(task_id).to(x.device), k_frac)
        h2 = F.relu(self.fc2(h1) * mask2)
        self.last_h2 = h2.detach()
        logits = self.fc3(h2)                    # ungated head
        self.last_acts = {
            self.fc1: h1.detach(),
            self.fc2: h2.detach(),

        }
        return logits



class WideCNN(nn.Module):
    def __init__(self, num_classes=10, input_dims=(3, 32, 32), lr=0.05, lr_min=1e-4, lr_patience=5, lr_factor=3):
        super().__init__()
        self.conv1 = GConv2d(input_dims[0], 128, 3, padding=1)
        self.bn1   = nn.GroupNorm(128, 128)
        self.conv2 = GConv2d(128, 256, 3, padding=1)
        self.bn2   = nn.GroupNorm(256, 256)
        self.conv3 = GConv2d(256, 512, 3, padding=1)
        self.bn3   = nn.GroupNorm(512, 512)
        self.pool  = nn.MaxPool2d(2)

        # --- derive flatten dim with a dummy forward ---
        with torch.no_grad():
            dummy = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
            dummy = self.dummy_forward(dummy)
            flat_dim = dummy.numel()

        self.fc = GLinear(flat_dim, num_classes)
        self.kappa = 0.25
        self.lr = lr
        self.lr_min = lr_min
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.optimizer = self._get_optimizer(self.lr)

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        # return torch.optim.SGD(self.parameters(),lr=lr)
        return torch.optim.Adam(self.parameters(), lr=lr)

    # ---------- main forward ----------
    def forward(self, x, *, task_id: int, k_frac: float = None, hard: bool = True):
        if k_frac is not None:        # allow per‑call override
            self.kappa = k_frac

        k = self.kappa

        # ---- conv1 ----
        m1 = apply_topk(self.conv1.gate_for(task_id), k, hard)[:, None, None]
        x1 = F.relu(self.bn1(self.conv1(x)) * m1)
        x1p = self.pool(x1)

        # ---- conv2 ----
        m2 = apply_topk(self.conv2.gate_for(task_id), k, hard)[:, None, None]
        x2 = F.relu(self.bn2(self.conv2(x1p)) * m2)
        x2p = self.pool(x2)

        # ---- conv3 ----
        m3 = apply_topk(self.conv3.gate_for(task_id), k, hard)[:, None, None]
        x3 = F.relu(self.bn3(self.conv3(x2p)) * m3)

        # ---- store activations for Hebbian update (before masking is fine) ----
        self.last_acts = {
            self.conv1: x1.detach(),   # [B,64,32,32]
            self.conv2: x2.detach(),   # [B,128,16,16]
            self.conv3: x3.detach(),   # [B,256, 8, 8]
        }

        # ---- classifier head ----
        logits = self.fc(x3.flatten(1))
        return logits

    def dummy_forward(self, x):
        x  = self.pool(F.relu(self.bn1(self.conv1(x))))
        x  = self.pool(F.relu(self.bn2(self.conv2(x))))
        x  = F.relu(self.bn3(self.conv3(x)))
        return x


class AlexNetW(nn.Module):
    def __init__(self, num_classes=10, input_dims=(3, 32, 32), lr=0.05, lr_min=1e-4, lr_patience=5, lr_factor=3):
        super().__init__()
        self.conv1 = GConv2d(input_dims[0], 64, 3, padding=1)
        self.bn1   = nn.GroupNorm(64, 64)
        self.conv2 = GConv2d(64, 128, 3, padding=1)
        self.bn2   = nn.GroupNorm(128, 128)
        self.conv3 = GConv2d(128, 256, 3, padding=1)
        self.bn3   = nn.GroupNorm(256, 256)
        self.pool  = nn.MaxPool2d(2)

        # --- derive flatten dim with a dummy forward ---
        with torch.no_grad():
            dummy = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
            dummy = self.dummy_forward(dummy)
            flat_dim = dummy.numel()

        self.fc = GLinear(flat_dim, num_classes)
        self.kappa = 0.25
        self.lr = lr
        self.lr_min = lr_min
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.optimizer = self._get_optimizer(self.lr)

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        # return torch.optim.SGD(self.parameters(),lr=lr)
        return torch.optim.Adam(self.parameters(), lr=lr)

    # ---------- main forward ----------
    def forward(self, x, *, task_id: int, k_frac: float = None, hard: bool = True):
        if k_frac is not None:        # allow per‑call override
            self.kappa = k_frac

        k = self.kappa

        # ---- conv1 ----
        m1 = apply_topk(self.conv1.gate_for(task_id), k, hard)[:, None, None]
        x1 = F.relu(self.bn1(self.conv1(x)) * m1)
        x1p = self.pool(x1)

        # ---- conv2 ----
        m2 = apply_topk(self.conv2.gate_for(task_id), k, hard)[:, None, None]
        x2 = F.relu(self.bn2(self.conv2(x1p)) * m2)
        x2p = self.pool(x2)

        # ---- conv3 ----
        m3 = apply_topk(self.conv3.gate_for(task_id), k, hard)[:, None, None]
        x3 = F.relu(self.bn3(self.conv3(x2p)) * m3)

        # ---- store activations for Hebbian update (before masking is fine) ----
        self.last_acts = {
            self.conv1: x1.detach(),   # [B,64,32,32]
            self.conv2: x2.detach(),   # [B,128,16,16]
            self.conv3: x3.detach(),   # [B,256, 8, 8]
        }

        # ---- classifier head ----
        logits = self.fc(x3.flatten(1))
        return logits

    def dummy_forward(self, x):
        x  = self.pool(F.relu(self.bn1(self.conv1(x))))
        x  = self.pool(F.relu(self.bn2(self.conv2(x))))
        x  = F.relu(self.bn3(self.conv3(x)))
        return x



class AlexNet(nn.Module):
    def __init__(self, num_classes=10, input_dims=(3, 32, 32), lr=0.05, lr_min=1e-4, lr_patience=5, lr_factor=3):
        super().__init__()
        self.conv1 = GConv2d(input_dims[0], 64, 4)
        self.bn1   = nn.GroupNorm(64, 64)
        self.conv2 = GConv2d(64, 128, 3)
        self.bn2   = nn.GroupNorm(128, 128)
        self.conv3 = GConv2d(128, 256, 2)
        self.bn3   = nn.GroupNorm(256, 256)
        self.pool  = nn.MaxPool2d(2)

        # --- derive flatten dim with a dummy forward ---
        with torch.no_grad():
            dummy = torch.zeros(1, input_dims[0], input_dims[1], input_dims[2])
            dummy = self.dummy_forward(dummy)
            flat_dim = dummy.numel()
            print(f"Flat_Dims: {flat_dim}")

        self.Gfc1 = GLinear(flat_dim, 2048)
        self.Gfc2 = GLinear(2048, 2048)
        self.last = torch.nn.Linear(2048, num_classes)
        self.kappa = 0.25

        self.lr = lr
        self.lr_min = lr_min
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.optimizer = self._get_optimizer(self.lr)
    # ---------- main forward ----------

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        # return torch.optim.SGD(self.parameters(),lr=lr)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, *, task_id: int, k_frac: float = None, hard: bool = True):
        if k_frac is not None:        # allow per‑call override
            self.kappa = k_frac

        k = self.kappa

        # ---- conv1 ----
        m1 = apply_topk(self.conv1.gate_for(task_id), k, hard)[:, None, None]
        x1 = F.relu(self.bn1(self.conv1(x)) * m1)
        x1p = self.pool(x1)

        # ---- conv2 ----
        m2 = apply_topk(self.conv2.gate_for(task_id), k, hard)[:, None, None]
        x2 = F.relu(self.bn2(self.conv2(x1p)) * m2)
        x2p = self.pool(x2)

        # ---- conv3 ----
        m3 = apply_topk(self.conv3.gate_for(task_id), k, hard)[:, None, None]
        x3 = F.relu(self.bn3(self.conv3(x2p)) * m3)
        x3p = self.pool(x3)

        mfc1 = apply_topk(self.Gfc1.gate_for(task_id), k, hard)
        xfc1 = F.relu(self.Gfc1(x3p.flatten(1)) * mfc1)

        mfc2 = apply_topk(self.Gfc2.gate_for(task_id), k, hard)
        xfc2 = F.relu(self.Gfc2(xfc1) * mfc2)

        # ---- store activations for Hebbian update (before masking is fine) ----
        self.last_acts = {
            self.conv1: x1.detach(),   # [B,64,32,32]
            self.conv2: x2.detach(),   # [B,128,16,16]
            self.conv3: x3.detach(),   # [B,256, 8, 8]
            self.Gfc1: xfc1.detach(),
            self.Gfc2: xfc2.detach()
        }

        # ---- classifier head ----
        logits = self.last(xfc2)
        return logits

    def dummy_forward(self, x):
        x  = self.pool(F.relu(self.bn1(self.conv1(x))))
        print(f"Conv1 Output Shape: {x.shape}")
        x  = self.pool(F.relu(self.bn2(self.conv2(x))))
        print(f"Conv2 Output Shape: {x.shape}")
        x  = self.pool(F.relu(self.bn3(self.conv3(x))))
        print(f"Conv3 Output Shape: {x.shape}")

        return x

