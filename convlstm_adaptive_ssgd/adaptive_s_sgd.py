import torch
from torch.optim.optimizer import Optimizer

class AdaptiveSSGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0,
                 alpha_max=0.5, alpha_min=0.0, total_steps=1000):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.total_steps = max(1, total_steps)
        self._step_idx = 0
        self._leader = None
        self._have_leader = False

    @torch.no_grad()
    def set_leader(self, params_state_dict):
        self._leader = [p.detach().clone() for p in params_state_dict]
        self._have_leader = True

    @torch.no_grad()
    def _alpha(self):
        t = min(self._step_idx / self.total_steps, 1.0)
        import math
        return self.alpha_min + 0.5*(self.alpha_max - self.alpha_min)*(1 + math.cos(math.pi * t))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad
                if wd != 0: d_p = d_p.add(p, alpha=wd)
                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)

        self._step_idx += 1

        if self._have_leader:
            alpha = self._alpha()
            li = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        leader_p = self._leader[li].to(p.device)
                        p.mul_(1 - alpha).add_(leader_p, alpha=alpha)
                        li += 1
        return loss
