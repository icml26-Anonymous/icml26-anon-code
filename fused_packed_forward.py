import torch
from functorch import vmap, make_functional_with_buffers

class ParallelPackedRunnerFunctorch:
    """
    Parallel inference over T packed models in a single vmap call.
    Requires: all packed models have identical param/buffer shapes.
    """
    def __init__(self, packed_models, criterion="confidence", norm_mode="z", mu=None, sigma=None, device="cuda"):
        self.device = torch.device(device)
        self.T = len(packed_models)
        assert self.T > 0

        self.packed_models = packed_models
        self.mu = mu
        self.sigma = sigma
        self.criterion = criterion
        self.norm_mode = norm_mode
        # Convert each model to functional form
        fmodels = []
        params_list = []
        buffers_list = []

        for m in packed_models:
            m = m.to(self.device).eval()
            fmodel, params, buffers = make_functional_with_buffers(m)
            fmodels.append(fmodel)
            params_list.append(params)
            buffers_list.append(buffers)

        # Use the first functional model (all must share structure)
        self.fmodel = fmodels[0]

        # Stack params across tasks: list of tensors shaped [T, ...]
        n_params = len(params_list[0])
        n_bufs = len(buffers_list[0])

        # Sanity: same lengths
        for i in range(1, self.T):
            assert len(params_list[i]) == n_params
            assert len(buffers_list[i]) == n_bufs

        self.params_T = []
        for p_i in range(n_params):
            stacked = torch.stack([params_list[t][p_i] for t in range(self.T)], dim=0)
            self.params_T.append(stacked)

        self.buffers_T = []
        for b_i in range(n_bufs):
            stacked = torch.stack([buffers_list[t][b_i] for t in range(self.T)], dim=0)
            self.buffers_T.append(stacked)

    @torch.no_grad()
    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits_all: [T, B, K]
        """
        x = x.to(self.device)

        def f(params, buffers, x_in):
            return self.fmodel(params, buffers, x_in)

        logits_all = vmap(f, in_dims=(0, 0, None))(self.params_T, self.buffers_T, x)
        return logits_all

    def _to_task_tensor(self, x, name: str):
        """
        Ensure mu/sigma are torch tensors on device with shape [T].
        Accepts: list, tuple, numpy, torch.Tensor.
        """
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device, dtype=torch.float32)
        if x.dim() == 0:
            # scalar -> expand to [T]
            x = x.repeat(self.T)
        assert x.shape[0] == self.T, f"{name} must have shape [T], got {tuple(x.shape)}"
        return x

    @torch.no_grad()
    def task_il_predict(self, packed_model, x, t, device: str = "cuda"):
        logits = packed_model(x)
        n_classes_ptask = logits.shape[1]
        pred = logits.argmax(1) + (t * n_classes_ptask)
        return pred

    @torch.no_grad()
    def class_il_predict(self, x: torch.Tensor, n_tasks, norm_mode=None, eps: float = 1e-8):
        """
        Vectorized version selection logic (no for loop).

        Assumptions:
          logits_all: [T,B,K] where K = classes-per-task (same for all tasks)
          mu, sigma are per-task stats (shape [T]) over the SAME score computed here.
        Returns:
          pred_global: [B] in global class space (task offset applied)
          best_t: [ ] scalar tensor (task chosen for this batch)
          scores: [T] task scores (useful for debugging)
        """
        logits_all = self.forward_all(x)  # [T,B,K]
        T, B, K = logits_all.shape

        # if criterion is None:
        criterion = self.criterion
        if norm_mode is None:
            norm_mode = self.norm_mode

        # if criterion != "confidence":
        #     raise NotImplementedError("Only criterion='confidence' is implemented in the vectorized path.")

        # conf[t,b] = max softmax prob for task t on sample b
        # softmax over K, then max over classes -> [T,B]
        conf = torch.softmax(logits_all, dim=2).amax(dim=2)  # [T,B]

        # per-task normalization
        if norm_mode == "z":
            mu = self._to_task_tensor(self.mu, "mu")  # [T]
            sigma = self._to_task_tensor(self.sigma, "sigma")  # [T]
            conf_norm = (conf - mu[:, None]) / (sigma[:, None] + eps)  # [T,B]
            # scores = conf_norm.max(1).values
            scores = conf_norm.mean(dim=1)
            # scores = (conf_norm * conf_norm.abs()).mean(dim=1)  # [T]
        elif norm_mode == "mean_ratio":
            mu = self._to_task_tensor(self.mu, "mu")  # [T]
            conf_norm = conf / (mu[:, None] + eps)  # [T,B]
            scores = conf_norm.mean(dim=1)  # [T]
        elif norm_mode == "mean":
            scores = conf.mean(dim=1)  # [T]
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        best_t = scores.argmax(dim=0)  # scalar tensor

        # pick logits for best task: [B,K]
        best_logits = logits_all[best_t]  # indexing with scalar tensor is fine

        # per-sample class prediction within task -> global id
        pred_local = best_logits.argmax(dim=1)  # [B]
        pred_global = pred_local + best_t * K  # [B]

        return pred_global, best_t, scores


    @torch.no_grad()
    def class_il_predict_loop(self, x: torch.Tensor, n_tasks=10):
        """
        Class-IL by max over tasks.
        Returns:
          logits: [B, K]
          pred: [B]
          winning_task: [B]
        """
        criterion = self.criterion
        norm_mode = self.norm_mode
        mu = self.mu
        sigma = self.sigma

        logits_all = self.forward_all(x)

        best_score = -1e9
        best_t = None
        best_logits = None

        acc = [0.0] * n_tasks
        cnt = [0] * n_tasks
        n_classes_ptask = logits_all.shape[2]
        eps = 1e-8

        for t in range(n_tasks):
            logits = logits_all[t]

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

        pred = best_logits.argmax(1) + (best_t * n_classes_ptask)
        return pred

    @torch.no_grad()
    def class_il_predict_sequential(self, x: torch.Tensor, n_tasks=10):
        """
        Class-IL by max over tasks.
        Returns:
          logits: [B, K]
          pred: [B]
          winning_task: [B]
        """
        criterion = self.criterion
        norm_mode = self.norm_mode
        mu = self.mu
        sigma = self.sigma

        best_score = -1e9
        best_t = None
        best_logits = None

        acc = [0.0] * n_tasks
        cnt = [0] * n_tasks

        eps = 1e-8

        for t in range(n_tasks):
            packed_model = self.packed_models[t]
            logits = packed_model(x)
            n_classes_ptask = logits.shape[1]
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

        pred = best_logits.argmax(1) + (best_t * n_classes_ptask)
        return pred


def fused_eval(runner, tasks_data, device="cuda"):
    accs = []
    num_tasks = len(tasks_data)
    for t, (_, test_loader) in enumerate(tasks_data):
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _ = runner.class_il_predict(xb, num_tasks)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(100.0 * correct / total)
    return accs

def fused_eval_standard(runner, tasks_data, norm_mode, device="cuda"):
    accs = []
    num_tasks = len(tasks_data)
    for t, (_, test_loader) in enumerate(tasks_data):
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _, _ = runner.class_il_predict(xb, num_tasks, norm_mode=norm_mode)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(100.0 * correct / total)
    return accs

def eval_TIL_packed(runner, tasks_data, device="cuda"):
    accs = []

    for t, (_, test_loader) in enumerate(tasks_data):
        packed_model = runner.packed_models[t]
        packed_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = runner.task_il_predict(packed_model, xb, t, device)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(100.0 * correct / total)
    return accs





