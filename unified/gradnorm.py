"""GradNorm (Chen et al., ICML 2018) with the four platforms as the tasks.

One shared network, one shared velocity head; the "tasks" are the platform
subsets of each batch, whose losses would otherwise be summed with equal weight.
Measured on f5_s42 the drone's parameter gradient is 3-4x the others', so the
sum is a drone-weighted sum.  GradNorm learns positive weights w_i so that the
gradient norms w_i * ||d L_i / dW|| at a shared layer W are balanced, tilted by
each task's relative training rate (L_i / L_i(0)) to the power alpha.

W is the last shared layer before the heads (mixer_norm's gain), as in the
paper; the norms then cost four small autograd passes through the head, not
through the trunk.  Inference is untouched: the weights only scale losses.
"""
from __future__ import annotations

import torch


class GradNorm:
    def __init__(self, n_tasks: int, alpha: float, shared_param: torch.Tensor, lr: float = 0.025):
        self.n, self.alpha, self.W = n_tasks, alpha, shared_param
        self.w = torch.ones(n_tasks, device=shared_param.device, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=lr)
        self.L0 = None
        self.last = {}

    def weighted_total(self, losses: torch.Tensor) -> torch.Tensor:
        """losses (n,), NaN for a task absent from this batch -> sum_i w_i L_i.

        Call before backward(); the weights inside the total are detached so the
        main backward never touches them.  The weights are updated here for the
        next step."""
        present = torch.isfinite(losses)
        L = torch.where(present, losses, torch.zeros_like(losses))
        if self.L0 is None:
            self.L0 = L.detach().clamp(min=1e-8)
        total = (self.w.detach().clone() * L).sum()   # clone: w is updated in place below

        # G_i = w_i * ||dL_i/dW||: the norm is a plain number, the dependence
        # on w_i is explicit, so no second-order graph is needed
        norms = []
        for i in range(self.n):
            if not present[i]:
                norms.append(torch.zeros((), device=L.device)); continue
            g, = torch.autograd.grad(L[i], self.W, retain_graph=True, allow_unused=True)
            norms.append(torch.zeros((), device=L.device) if g is None else g.detach().norm())
        norms = torch.stack(norms)
        G = self.w * norms
        with torch.no_grad():
            rate = L.detach() / self.L0
            rate = rate / rate[present].mean().clamp(min=1e-8)
            target = G.detach()[present].mean() * rate.pow(self.alpha)
        l_grad = (G - target).abs()[present].sum()
        self.opt.zero_grad(set_to_none=True)
        gw, = torch.autograd.grad(l_grad, self.w)
        self.w.grad = gw
        self.opt.step()
        with torch.no_grad():
            self.w.clamp_(min=1e-3)
            self.w.mul_(self.n / self.w.sum())          # renormalise to sum n
        self.last = {"G": norms, "w": self.w.detach().clone()}
        return total
