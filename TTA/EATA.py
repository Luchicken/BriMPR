from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import os
import random
import dataloader as dataloader
import pickle
import logging
logger = logging.getLogger(__name__)


def get_source_loader(args, fisher_size=2000):
    data_source = os.path.join(args.json_root, 'clean', 'train.json')  # ks50 (29204), vggsound (183730)
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': 224}
    source_dataset = dataloader.AudiosetDataset(data_source, label_csv=args.label_csv, audio_conf=val_audio_conf)
    indices = list(range(len(source_dataset)))
    random.shuffle(indices)
    source_dataset = torch.utils.data.Subset(source_dataset, indices[:fisher_size])
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    logger.info(f"Number of samples and batches in source loader: #samples = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_loader


class EATA(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False,
                 fisher_alpha=2000.0, fisher_size=2000, d_margin=0.05, params=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        self.e_margin = math.log(args.n_class) * 0.40  # hyperparameter E_0 (Eqn. 3)
        self.d_margin = d_margin  # hyperparameter \epsilon for consine simlarity thresholding (Eqn. 5)
        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = fisher_alpha  # trade-off \beta for two losses (Eqn. 8)
        self.fishers = None

        if self.fisher_alpha > 0:
            fisher_dir_path = os.path.join("./ckpt", "fisher")
            fname = os.path.join(fisher_dir_path, f"fisher_{args.dataset}.pkl")
            if os.path.exists(fname):
                logger.info("Loading the fisher matrices...")
                with open(fname, 'rb') as f:
                    self.fishers = pickle.load(f)
                logger.info("Finished loading the fisher matrices...")
            else:
                os.makedirs(fisher_dir_path, exist_ok=True)
                # compute fisher informatrix
                logger.info("Computing the fisher matrices...")
                fisher_loader = get_source_loader(args, fisher_size)
                ewc_optimizer = torch.optim.SGD(params, 0.001)
                self.fishers = {}
                train_loss_fn = nn.CrossEntropyLoss().to(self.device)

                for iter_, (a_input, v_input, labels) in enumerate(fisher_loader, start=1):
                    a_input = a_input.to(device, non_blocking=True)
                    v_input = v_input.to(device, non_blocking=True)
                    outputs, _ = self.model(a=a_input, v=v_input, mode=self.args.testmode, test=True)
                    _, targets = outputs.max(1)
                    loss = train_loss_fn(outputs, targets)
                    loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if iter_ > 1:
                                fisher = param.grad.data.clone().detach() ** 2 + self.fishers[name][0]
                            else:
                                fisher = param.grad.data.clone().detach() ** 2
                            if iter_ == len(fisher_loader):
                                fisher = fisher / iter_
                            self.fishers.update({name: [fisher, param.data.clone().detach()]})
                    ewc_optimizer.zero_grad()

                logger.info("Finished computing the fisher matrices...")
                del ewc_optimizer
                with open(fname, 'wb') as f:
                    pickle.dump(self.fishers, f)
        else:
            logger.info("Not using EWC regularization. EATA decays to ETA!")

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model.module, self.optimizer)

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            outputs, loss, updated_probs = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler,
                                           self.fishers, self.e_margin, self.current_model_probs, self.fisher_alpha, self.d_margin)
            self.current_model_probs = updated_probs
            return outputs, loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model.module, self.optimizer, self.model_state, self.optimizer_state)
        self.current_model_probs = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler, fishers, e_margin, current_model_probs, fisher_alpha, d_margin):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    with autocast():
        outputs, _ = model(a=x[0], v=x[1], mode=args.testmode, test=True)
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
    loss = entropys.mean(0)

    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[0][ids1][ids2].size(0) != 0:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return outputs, (loss.item(), 0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with eata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    # enable grad for layernorm
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
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