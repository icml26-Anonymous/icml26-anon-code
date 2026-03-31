import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
import os, pathlib, pickle
from torchvision import datasets, transforms
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------------
#  Permuted-MNIST for CNN backbones
# ------------------------------------------------------------------
def generate_permuted_mnist_CNN(
        num_tasks: int = 5,
        batch_size: int = 128,
        data_root: str = "./data"):
    """
    Returns a list[(train_loader, test_loader)]   length = num_tasks

    Each task applies a different, fixed pixel permutation to all images and
    keeps the tensor shape (B, 1, 28, 28) so that convolutional models work
    out-of-the-box.
    """
    # ── 1.  Download once, get plain tensors in [0,1] ──────────────────
    to_tensor = transforms.ToTensor()             # (H,W,C)->(C,H,W) /255
    mnist_train = datasets.MNIST(
        root=data_root, train=True, download=True, transform=to_tensor)
    mnist_test  = datasets.MNIST(
        root=data_root, train=False, download=True, transform=to_tensor)

    x_train = mnist_train.data.float() / 255.      # [N,28,28]
    y_train = mnist_train.targets
    x_test  = mnist_test.data.float() / 255.
    y_test  = mnist_test.targets

    # Reshape to (N, 1, 28, 28) once
    x_train = x_train.unsqueeze(1)                 # add channel dim
    x_test  = x_test.unsqueeze(1)

    tasks = []
    H = W = 28
    P = H * W                                     # 784

    # ── 2.  Build tasks ------------------------------------------------
    for _ in range(num_tasks):
        # --- use torch.randperm so dtype is Long, always index-safe ------
        perm = torch.randperm(P)  # <──  replaces np.random.permutation
        # -----------------------------------------------------------------

        # Apply permutation
        x_tr_perm = x_train.view(-1, P)[:, perm].view(-1, 1, H, W)
        x_te_perm = x_test.view(-1, P)[:, perm].view(-1, 1, H, W)

        # TensorDatasets / DataLoaders
        train_ds = TensorDataset(x_tr_perm, y_train)
        test_ds  = TensorDataset(x_te_perm, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, pin_memory=True)
        tasks.append((train_loader, test_loader))

    return tasks


def generate_permuted_mnist(num_tasks=5, batch_size=64):
    """
    Returns a list of (DataLoader, DataLoader) for training and validation
    for each permuted MNIST task.
    """
    # Download MNIST once
    mnist_train = datasets.MNIST(root="./data", train=True, download=True)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True)

    # Convert to numpy arrays
    x_train = mnist_train.data.float().numpy().reshape(-1, 28*28) / 255.0
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.float().numpy().reshape(-1, 28*28) / 255.0
    y_test = mnist_test.targets.numpy()

    # Generate tasks
    tasks_data = []
    for _ in range(num_tasks):
        # Create a random permutation of 0..783 (28*28 - 1)
        perm = np.random.permutation(28*28)
        x_train_perm = x_train[:, perm]
        x_test_perm = x_test[:, perm]

        # Create TensorDatasets
        train_ds = TensorDataset(torch.from_numpy(x_train_perm), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(x_test_perm), torch.from_numpy(y_test))

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        tasks_data.append((train_loader, test_loader))

    return tasks_data


# --------------------------------------------------------------------------
#  Split-CIFAR100 with per-task local-label remapping  (cached)
# --------------------------------------------------------------------------
class LocalLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, g2l_map):
        self.subset, self.map = subset, g2l_map
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, g_label = self.subset[idx]
        return img, self.map[int(g_label)]

def make_split_cifar100(batch_size: int = 128,
                        num_tasks: int = 10,
                        *,
                        seed: int = 0,
                        root: str = "./data",
                        cache_dir: str = "./cache"):

    assert 100 % num_tasks == 0
    classes_per_task = 100 // num_tasks
    cache_dir = pathlib.Path(cache_dir); cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"split_cifar100_T{num_tasks}_seed{seed}.pt"

    # --------------------------------------------------  A) load / build split indices
    if cache_file.exists():
        split_idx = torch.load(cache_file)
        if 'map' not in split_idx:          # <-- upgrade logic
            print("[Split-CIFAR] cache lacks local-label maps; rebuilding ...")
            cache_file.unlink()             # delete and fall through to rebuild
            return make_split_cifar100(batch_size, num_tasks,
                                       seed=seed, root=root, cache_dir=cache_dir)
        print(f"[Split-CIFAR] loaded cache {cache_file}")
    else:
        print(f"[Split-CIFAR] cache miss – building class splits ...")
        trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]),
        ])
        train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=trf)
        test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=trf)

        train_targets = np.array(train_ds.targets)
        test_targets  = np.array(test_ds.targets)

        rng = np.random.RandomState(seed)
        class_order = rng.permutation(100)

        split_idx = {'train': [], 'test': [], 'map': []}
        for t in range(num_tasks):
            cls_subset = class_order[t*classes_per_task:(t+1)*classes_per_task]
            local_map = {g: i for i, g in enumerate(cls_subset)}  # global→local
            split_idx['map'].append(local_map)

            tr_idx = np.where(np.isin(train_targets, cls_subset))[0]
            te_idx = np.where(np.isin(test_targets,  cls_subset))[0]
            split_idx['train'].append(tr_idx)
            split_idx['test'].append(te_idx)

        torch.save(split_idx, cache_file)
        print(f"[Split-CIFAR] splits cached to {cache_file}")

    # --------------------------------------------------  B) build DataLoaders
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=trf)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=trf)

    tasks = []
    for tr_idx, te_idx, g2l in zip(split_idx['train'],
                                   split_idx['test'],
                                   split_idx['map']):
        tr_loader = DataLoader(
            LocalLabelWrapper(Subset(train_ds, tr_idx), g2l),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        te_loader = DataLoader(
            LocalLabelWrapper(Subset(test_ds,  te_idx), g2l),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        tasks.append((tr_loader, te_loader))

    return tasks


# --------------------------------------------------------------
# Split‑CIFAR‑10  (5 tasks × 2 classes each)
# --------------------------------------------------------------

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def _default_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])

def _class_subset(dataset, class_ids):
    """Return a Subset containing only samples with targets in class_ids."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in class_ids]
    return Subset(dataset, idx)

def _default_transforms(train=True):
    tf = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ]
    if train:
        tf = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip()] + tf
    return transforms.Compose(tf)

def generate_split_cifar10(batch_size=128, data_root="./data", num_workers=4
                           ) -> List[Tuple[DataLoader, DataLoader]]:
    train_set = datasets.CIFAR10(data_root, True, download=True,
                                 transform=_default_transforms(True))
    test_set  = datasets.CIFAR10(data_root, False, download=True,
                                 transform=_default_transforms(False))

    # use targets list = no transform call
    cls_idx_tr = {c: [i for i, y in enumerate(train_set.targets) if y == c]
                  for c in range(10)}
    cls_idx_te = {c: [i for i, y in enumerate(test_set.targets)  if y == c]
                  for c in range(10)}

    tasks = []
    for t in range(5):
        pair = [2*t, 2*t+1]
        tr_subset = Subset(train_set, cls_idx_tr[pair[0]] + cls_idx_tr[pair[1]])
        te_subset = Subset(test_set,  cls_idx_te[pair[0]] + cls_idx_te[pair[1]])

        tasks.append((
            DataLoader(tr_subset, batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(te_subset, batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
        ))
        print(f"Task {t}: classes {pair}, "
              f"{len(tr_subset)} train / {len(te_subset)} test")

    return tasks


# 1.  Standard CIFAR transforms ---------------------------------------
def _c100_tf(train=True):
    tf = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ]
    if train:
        tf.insert(0, transforms.RandomHorizontalFlip())
        tf.insert(0, transforms.RandomCrop(32, padding=4))
    return transforms.Compose(tf)

# ---------------------------------------------------------------------
# 2.  Split-CIFAR-100 loader  -----------------------------------------
def generate_split_cifar100(batch_size: int = 128,
                            data_root: str = "./data",
                            num_workers: int = 4,
                            n_tasks: int = 10
                           ) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Returns `tasks`, a list with `n_tasks` tuples (train_loader, test_loader).
    Each task contains 100/n_tasks = 10 CIFAR-100 classes in label order:
        Task 0:  0-9,  Task 1: 10-19, …, Task 9: 90-99
    The label \emph{values are kept as-is} so a single shared head can be used.
    """
    train_set = datasets.CIFAR100(data_root, True, download=True,
                                  transform=_c100_tf(train=True))
    test_set  = datasets.CIFAR100(data_root, False, download=True,
                                  transform=_c100_tf(train=False))

    # Pre-index once — no transforms called
    idx_tr = {c: [] for c in range(100)}
    for i, y in enumerate(train_set.targets):
        idx_tr[y].append(i)
    idx_te = {c: [] for c in range(100)}
    for i, y in enumerate(test_set.targets):
        idx_te[y].append(i)

    classes_per_task = 100 // n_tasks
    tasks = []
    for t in range(n_tasks):
        cls = list(range(t*classes_per_task, (t+1)*classes_per_task))

        tr_subset = Subset(train_set, sum([idx_tr[c] for c in cls], []))
        te_subset = Subset(test_set,  sum([idx_te[c] for c in cls], []))

        train_loader = DataLoader(tr_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        test_loader  = DataLoader(te_subset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True)

        tasks.append((train_loader, test_loader))
        print(f"[Split-CIFAR-100] Task {t}: classes {cls[0]}–{cls[-1]}, "
              f"{len(tr_subset)} train / {len(te_subset)} test")

    return tasks



# ---------------------------------------------------------------------
# 2.  Split ImageNet-200 loader  -----------------------------------------
# ImageNet-style normalization, 64×64
_TINY_MEAN = (0.485, 0.456, 0.406)
_TINY_STD  = (0.229, 0.224, 0.225)


def _tiny_tf(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_TINY_MEAN, _TINY_STD),
        ])
    
def generate_split_tiny_imagenet(
    root_dir: str = "./data/tiny-imagenet-200",
    n_tasks: int = 10,                 # 200/10 = 20 classes per task
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle_classes: bool = False,
    seed: int = 0
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Expects the official layout after reorg:
      tiny-imagenet-200/train/<wnid>/*.JPEG
      tiny-imagenet-200/val/<wnid>/*.JPEG  (val/ reorganized into class folders)
    Returns [(train_loader, val_loader)] * n_tasks with single-head labels.
    """
    train_root = os.path.join(root_dir, "train")
    val_root   = os.path.join(root_dir, "val")

    train_ds = datasets.ImageFolder(train_root, transform=_tiny_tf(train=True))
    val_ds   = datasets.ImageFolder(val_root,   transform=_tiny_tf(train=False))

    # class_to_idx is a dict like {'n01443537':0, ...} in sorted order
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    n_classes = len(idx_to_class)  # should be 200

    class_ids = np.arange(n_classes)
    if shuffle_classes:
        rng = np.random.RandomState(seed)
        rng.shuffle(class_ids)

    assert n_classes % n_tasks == 0, "n_tasks must divide 200"
    per_task = n_classes // n_tasks

    # Pre-index samples by class for both splits (fast, no transform calls)
    cls_tr = {c: [] for c in range(n_classes)}
    for i, (_, y) in enumerate(train_ds.samples):
        cls_tr[y].append(i)
    cls_va = {c: [] for c in range(n_classes)}
    for i, (_, y) in enumerate(val_ds.samples):
        cls_va[y].append(i)

    tasks = []
    for t in range(n_tasks):
        block = class_ids[t*per_task:(t+1)*per_task].tolist()

        tr_idx = sum([cls_tr[c] for c in block], [])
        va_idx = sum([cls_va[c] for c in block], [])

        tr_sub = Subset(train_ds, tr_idx)
        va_sub = Subset(val_ds,   va_idx)

        tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
        va_loader = DataLoader(va_sub, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

        print(f"[Tiny-ImageNet] Task {t}: {len(block)} classes "
              f"({min(block)}–{max(block)}), {len(tr_idx)} train / {len(va_idx)} val")
        tasks.append((tr_loader, va_loader))

    return tasks
