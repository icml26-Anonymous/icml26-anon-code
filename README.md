# icml26-anon-code
Anonymous code snapshot for ICML 2026 rebuttal.

This repository provides implementation details and scripts supporting the experiments reported in the submitted manuscript.
It is provided only to clarify reproducibility details and does not constitute a revised submission.

## To run the main training loop run:
python main_loop.py

## Following training and saving of the .pt checkpoint weights
Configure compile_eval_packed.py accordingly and run it by:
python compile_eval_packed.py

This folder contains only the files needed to run:
- `main_loop.py` (training)
- `compile_eval_packed.py` (packed-model compile + evaluation)

## 1) Environment setup

Create and activate a Python environment, then install dependencies (PyTorch + torchvision are required; optional extras may be needed depending on your local setup):

---

## 2) Data + output folders

Before running scripts, create the directories expected by the code:

```bash
mkdir -p data checkpoints training_logs/
```

Default dataset roots in the scripts are currently:
- Tiny-ImageNet: `./data/tiny-imagenet-200`
- CIFAR: `./data/`
- ImageNet100 (optional branch): `./data/imagenet100`

Adjust those paths in the script variables if your data is elsewhere.

---

## 3) Run training (`main_loop.py`)

```bash
python main_loop.py
```

### Main configuration variables (edit in `if __name__ == "__main__":`)

- **Device/runtime**
  - `device_n`: GPU index selected by `torch.cuda.set_device(device_n)`.
  - `CUDA_LAUNCH_BLOCKING`: debug sync flag.

- **General experiment controls**
  - `norm`: normalization type passed to model (e.g., `"in"`).
  - `per_task_bn`: whether to use per-task BN behavior.
  - `seeds`: random seeds list.
  - `batch_size`: loader batch size.

- **Execution mode toggles**
  - `save`: write experiment log text file.
  - `load_`: load checkpoint before evaluation/training.
  - `train_`: run training loop.
  - `save_model`: save trained checkpoint.
  - `eval_speed`: optional eval-speed toggle.

- **Sweep lists**
  - `architectures`: model choice list (e.g., `['Resnet']`).
  - `datasets`: dataset choice list (e.g., `['imagenet']`, `['CIF10']`, `['CIF100']`, `['IMNET100']`).
  - `lrs`: backbone learning rates.
  - `gating_LRs`: gate learning rates.
  - `gate_update_rates`: gate update cadence.
  - `kappa_update_rates`: kappa schedule update cadence.
  - `reward_modes`: one of `REWARD_BATCH`, `REWARD_SAMPLE`, `REWARD_MARGIN`.
  - `scale_modes`: gate scaling rule (`'linear'`, `'exp'`, `'power'`, depending on implementation).
  - `gating_modes`: usually `'hard'` or `'soft'`.
  - `beta`, `kappa_peak`: additional shaping/schedule hyperparameters.

- **Dataset-specific blocks**
  - `tasks_data`: generated task loaders.
  - `num_classes`: classifier output size.
  - `epochs_`: epochs per task.
  - `k_fractions`: per-task kappa targets.
  - `class_indices_per_task`: class slice map.

- **Training call parameters**
  Inside `train_tasks(...)`, key knobs are:
  - `epochs_list`
  - `lr`
  - `g_lr`
  - `kappa_targets`
  - `reward_mode`
  - `scale_mode`
  - `gating_mode`
  - `gate_update_rate`
  - `kappa_peak`
  - `kappa_update_rate`
  - `csv_path`

### Training outputs

- Checkpoint: `checkpoints/<architecture>_<dataset>_<time>.pt`
- CSV training log: `training_logs/.../*.csv`
- Text summary log: `training_logs/.../*.txt`

---

## 4) Run packed compile + eval (`compile_eval_packed.py`)

```bash
python compile_eval_packed.py
```

(`compile_eval_Packed.py` is also included; both are equivalent copies in this bundle.)

### Main configuration variables (edit in `if __name__ == "__main__":`)

- **Core controls**
  - `norm`, `per_task_bn`, `seeds`, `batch_size`
  - `kappa`: gate sparsity used for packed plan building.

- **Mode toggles**
  - `load_`: load an existing checkpoint.
  - `compile_model`: optional compile path before packed conversion.
  - `VERIFY`: reserved verification switch.
  - `EVAL_FULL`: also run full sequential-model eval.
  - `REPORT_COMPUTE`: print params/MAC/time report.
  - `REPORT_ACCURACY`: print TIL/CIL metrics.

- **Dataset selection**
  - `datasets` and `dataset = datasets[0]` decide branch.
  - Tiny-ImageNet branch uses `./data/tiny-imagenet-200`.
  - Else branch uses CIFAR root `./data`.

- **Checkpoint loading**
  - `date_code` and `model_name` compose `checkpoints/{model_name}.pt`.
  - `load_mu_sigma(...)` loads confidence stats if available.
  - If missing, `compute_conf_stats(...)` is called and then `save_mu_sigma(...)` stores stats.

### Compile/eval flow

1. Load trained checkpoint.
2. Build packed wiring plans (`build_packed_wiring_plans`).
3. Build one packed subnetwork per task (`build_packed_model_for_task`).
4. Fuse into parallel runner (`ParallelPackedRunnerFunctorch`).
5. Report accuracy (`eval_TIL_packed`, `fused_eval`, `fused_eval_standard`) and optionally full-model metrics.

---

## 5) Minimal reproducible usage pattern

1. Train and save a checkpoint with `main_loop.py`.
2. In `compile_eval_packed.py`, set:
   - `dataset` to the same training dataset,
   - `date_code` so `model_name` points to your saved checkpoint,
   - matching `kappa` and class/task setup.
3. Run `python compile_eval_packed.py`.

If the checkpoint filename does not match, update:

```python
model_name = f"Resnet_{dataset}_{date_code}"
```

or replace with a fixed filename directly.
