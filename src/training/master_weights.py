"""AdamW over fp32 master copies of low-precision parameters.

The model keeps its parameters in the compute dtype (bf16), so the forward and
backward are the same bf16 modules the sampler runs. The optimizer keeps an
fp32 master copy of each parameter next to its fp32 AdamW moments, applies the
update to the master, and writes the master back, rounded, into the parameter.
Small updates accumulate in the master until they move the bf16 value, instead
of being rounded away on every step.

Everything the optimizer owns is a tensor in optimizer.state[param], keyed by
the model's parameter like torch.optim.AdamW's state. The FFT trainer's CPU
offload, which moves every tensor in the optimizer state, and state_dict /
load_state_dict carry the masters with no special case.
"""

import math
from collections.abc import Iterable

import torch


class MasterWeightAdamW(torch.optim.Optimizer):
  """torch.optim.AdamW (decoupled weight decay, no amsgrad) with the update
  computed on an fp32 master copy of each parameter."""

  def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    master_dtype: torch.dtype = torch.float32,
  ):
    if lr < 0.0 or eps < 0.0 or weight_decay < 0.0 or not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
      raise ValueError(f"Invalid AdamW hyperparameters: lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}")
    super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
    self.master_dtype = master_dtype

  def load_state_dict(self, state_dict: dict) -> None:
    """torch.optim.Optimizer.load_state_dict casts floating state to its
    parameter's dtype, which would round the masters and moments to bf16.
    Put the saved tensors back in master_dtype after it."""
    super().load_state_dict(state_dict)
    # The saved state is keyed by each parameter's position across the groups.
    params = [param for group in self.param_groups for param in group["params"]]
    for index, saved in state_dict["state"].items():
      param = params[index]
      for key in ("master", "exp_avg", "exp_avg_sq"):
        if key in saved:
          self.state[param][key] = saved[key].to(device=param.device, dtype=self.master_dtype, copy=True)

  def init_state(self, param: torch.nn.Parameter) -> dict:
    state = self.state[param]
    if not state:
      state["step"] = torch.tensor(0.0)
      state["master"] = param.detach().to(self.master_dtype, copy=True)
      state["exp_avg"] = torch.zeros_like(state["master"])
      state["exp_avg_sq"] = torch.zeros_like(state["master"])
    return state

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      lr, (beta1, beta2), eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
      for param in group["params"]:
        if param.grad is None:
          continue
        if param.grad.is_sparse:
          raise RuntimeError("MasterWeightAdamW does not support sparse gradients")
        state = self.init_state(param)
        master, exp_avg, exp_avg_sq = state["master"], state["exp_avg"], state["exp_avg_sq"]
        grad = param.grad.to(master.dtype)
        state["step"] += 1
        step = state["step"].item()

        # The same arithmetic as torch.optim.AdamW's single-tensor path.
        master.mul_(1 - lr * weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        master.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        param.copy_(master)
    return loss
