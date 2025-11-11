"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange

from torch.cuda.amp import autocast, GradScaler


class DeYO(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, device, args, steps=1, episodic=False,
                 deyo_margin=0.5 * math.log(1000), margin_e0=0.4 * math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "DeYO requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        self.deyo_margin = deyo_margin  # Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)
        self.margin_e0 = margin_e0  # Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model.module, self.optimizer)

    def forward(self, x, adapt_flag, targets=None, flag=True, group=None):
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, loss, backward, final_backward = forward_and_adapt_deyo(x, self.model, self.args,
                                                                               self.optimizer, self.scaler, self.deyo_margin,
                                                                               self.margin_e0, targets, flag, group)
                else:
                    outputs = forward_and_adapt_deyo(x, self.model, self.args,
                                                     self.optimizer, self.scaler, self.deyo_margin,
                                                     self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_deyo(x, self.model,
                                                                                                     self.args, self.optimizer,
                                                                                                     self.scaler,
                                                                                                     self.deyo_margin,
                                                                                                     self.margin_e0,
                                                                                                     targets, flag, group)
                else:
                    outputs = forward_and_adapt_deyo(x, self.model,
                                                     self.args, self.optimizer,
                                                     self.scaler,
                                                     self.deyo_margin,
                                                     self.margin_e0,
                                                     targets, flag, group, self)
        if targets is None:
            if flag:
                # return outputs, backward, final_backward
                return outputs, loss
            else:
                return outputs, (0, 0)
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model.module, self.optimizer, self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # temprature = 1.1 #0.9 #1.2
    # x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo(x, model, args, optimizer, scaler, deyo_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    with autocast():
        outputs, _ = model(a=x[0], v=x[1], mode=args.testmode, test=True)
    if not flag:
        return outputs

    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where((entropys < deyo_margin))  # Eqn. (8)
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward == 0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, (0, 0), 0, 0

    a_prime, v_prime = x[0][filter_ids_1], x[1][filter_ids_1]
    a_prime = a_prime.detach()
    v_prime = v_prime.detach()

    # aug_type: patch
    a_prime = a_prime.unsqueeze(1)
    resize_t = torchvision.transforms.Resize(
        ((x[0].shape[-2] // args.patch_len) * args.patch_len, (x[0].shape[-1] // args.patch_len) * args.patch_len))
    resize_o = torchvision.transforms.Resize((x[0].shape[-2], x[0].shape[-1]))
    a_prime = resize_t(a_prime)
    a_prime = rearrange(a_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
    perm_idx = torch.argsort(torch.rand(a_prime.shape[0], a_prime.shape[1]), dim=-1)
    a_prime = a_prime[torch.arange(a_prime.shape[0]).unsqueeze(-1), perm_idx]
    a_prime = rearrange(a_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
    a_prime = resize_o(a_prime)
    a_prime = a_prime.squeeze(1)

    resize_t = torchvision.transforms.Resize(
        ((x[1].shape[-1] // args.patch_len) * args.patch_len, (x[1].shape[-1] // args.patch_len) * args.patch_len))
    resize_o = torchvision.transforms.Resize((x[1].shape[-1], x[1].shape[-1]))
    v_prime = resize_t(v_prime)
    v_prime = rearrange(v_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
    perm_idx = torch.argsort(torch.rand(v_prime.shape[0], v_prime.shape[1]), dim=-1)
    v_prime = v_prime[torch.arange(v_prime.shape[0]).unsqueeze(-1), perm_idx]
    v_prime = rearrange(v_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
    v_prime = resize_o(v_prime)

    # if args.aug_type == 'occ':
    #     first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
    #     final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
    #     occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
    #     x_prime[:, :, args.row_start:args.row_start + args.occlusion_size,
    #     args.column_start:args.column_start + args.occlusion_size] = occlusion_window
    # elif args.aug_type == 'patch':
    #     resize_t = torchvision.transforms.Resize(
    #         ((x.shape[-1] // args.patch_len) * args.patch_len, (x.shape[-1] // args.patch_len) * args.patch_len))
    #     resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
    #     x_prime = resize_t(x_prime)
    #     x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
    #     perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
    #     x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
    #     x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
    #     x_prime = resize_o(x_prime)
    # elif args.aug_type == 'pixel':
    #     x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
    #     x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
    #     x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        with autocast():
            outputs_prime, _ = model(a=a_prime, v=v_prime, mode=args.testmode, test=True)

    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
    plpd = plpd.reshape(-1)

    filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)

    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()

    if final_backward == 0:
        del a_prime, v_prime
        del plpd

        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, (0, 0), backward, 0

    plpd = plpd[filter_ids_2]

    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    coeff = 1 / (torch.exp(((entropys.clone().detach()) - margin))) + 1 / (torch.exp(-1. * plpd.clone().detach()))
    entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad()

    del a_prime, v_prime
    del plpd

    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, (loss.item(), 0), backward, final_backward


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: blocks9-11 for Vit-Base
        if 'blocks_a.9' in nm or 'blocks_a.10' in nm or 'blocks_v.9' in nm or 'blocks_v.10' in nm or 'blocks_u.0' in nm:
            continue
        if 'norm_a' in nm or 'norm_v' in nm:
            break
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
