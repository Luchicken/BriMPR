import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math


class READ(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "READ requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model.module, self.optimizer)

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler)
            return outputs, loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model.module, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    with autocast():
        outputs, _ = model(a=x[0], v=x[1], mode=args.testmode, test=True)

    # adapt
    p_sum = outputs.softmax(dim=-1).sum(dim=-2)
    loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()

    pred = outputs.softmax(dim=-1)
    pred_max = pred.max(dim=-1)[0]
    gamma = math.exp(-1)
    t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
    loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()

    loss = loss_ra - 1 * loss_bal
    
    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        with autocast():
            outputs2, _ = model(a=x[0], v=x[1], mode=args.testmode, test=True)

    return outputs2, (loss_ra.item(), loss_bal.item())


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    """
    params_fusion_qkv = []
    names_fusion_qkv = []
    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv


def configure_model(model):
    """Configure model for use with read."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    # enable grad for qkv
    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)

    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    # =====================================================
    # remove the q/k/v parameters
    del model_state['blocks_u.0.attn.q.weight']
    del model_state['blocks_u.0.attn.q.bias']
    del model_state['blocks_u.0.attn.k.weight']
    del model_state['blocks_u.0.attn.k.bias']
    del model_state['blocks_u.0.attn.v.weight']
    del model_state['blocks_u.0.attn.v.bias']
    # =====================================================
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=False)
    model.blocks_u[0].attn.first_batch = True
    optimizer.load_state_dict(optimizer_state)
