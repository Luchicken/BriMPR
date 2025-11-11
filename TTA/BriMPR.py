from copy import deepcopy
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
import math
import os
import numpy as np
from tqdm import tqdm
import random
import dataloader as dataloader
import logging

logger = logging.getLogger(__name__)


def get_source_loader(args, num_samples=32):
    data_source = os.path.join(args.json_root, 'clean', 'train.json')  # ks50 (29204), vggsound (183730)
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': 224}
    source_dataset = dataloader.AudiosetDataset(data_source, label_csv=args.label_csv, audio_conf=val_audio_conf)
    indices = list(range(len(source_dataset)))
    random.shuffle(indices)
    source_dataset = torch.utils.data.Subset(source_dataset, indices[:num_samples])
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    logger.info(f"Number of samples and batches in source loader: #samples = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_loader


class BriMPR(nn.Module):
    def __init__(self, pva_model, optimizer, device, args):
        super().__init__()
        self.pva_model = pva_model  # pva_model.module: PromptVA
        self.optimizer = optimizer
        self.steps = 1
        self.args = args
        self.scaler = GradScaler()
        self.device = device
        self.optimizer_state = deepcopy(optimizer.state_dict())

        # -------------------------------------- cmmtta --------------------------------------
        from collections import deque
        window_size = 10
        self.window_a, self.window_v = deque(maxlen=window_size), deque(maxlen=window_size)
        self.z_scores_a, self.z_scores_v = [], []
        # -------------------------------------- cmmtta --------------------------------------

        # get mean & variance of source embeddings
        srcstat_dir_path = os.path.join("./ckpt", f"src_stat_{args.num_samples_source}")
        fname = os.path.join(srcstat_dir_path, f"src_stat_{args.dataset}.pth")
        if os.path.exists(fname):
            logger.info("Loading mean and variance...")
            (self.src_stat_a, self.src_stat_v, self.src_stat_u) = torch.load(fname)  # (std, mean)
            logger.info("Finished loading mean and variance.")
        else:
            os.makedirs(srcstat_dir_path, exist_ok=True)
            logger.info('Calculating mean and variance...')
            src_loader = get_source_loader(args, num_samples=args.num_samples_source)

            features_a, features_v, features_u = [], [], []
            with torch.no_grad():
                data_bar = tqdm(src_loader)
                for i, (a_input, v_input, labels) in enumerate(data_bar):
                    a_input = a_input.to(device)
                    v_input = v_input.to(device)
                    feature_a, feature_v, feature_u = self.pva_model(a=a_input, v=v_input, mode='features')  # (64, 11, 768) (64, 11, 768) (64, 1, 768)
                    features_a.append(feature_a)
                    features_v.append(feature_v)
                    features_u.append(feature_u)
                    data_bar.set_description(f'Batch#{i}')
                features_a = torch.cat(features_a, dim=0)  # (num_samples, 11, 768)
                features_v = torch.cat(features_v, dim=0)  # (num_samples, 11, 768)
                features_u = torch.cat(features_u, dim=0)  # (num_samples, 1, 768)
                # (std, mean)
                self.src_stat_a = torch.std_mean(features_a, dim=0)  # tuple ((11, 768), (11, 768))
                self.src_stat_v = torch.std_mean(features_v, dim=0)  # tuple ((11, 768), (11, 768))
                self.src_stat_u = torch.std_mean(features_u, dim=0)  # tuple ((1, 768), (1, 768))
            del features_a, features_v, features_u
            torch.save((self.src_stat_a, self.src_stat_v, self.src_stat_u), fname)
            logger.info('Finished calculating mean and variance.')

    def forward(self, x, adapt_flag, labels=None):
        for _ in range(self.steps):
            outputs, loss = forward_and_adapt(x, self.pva_model, self.optimizer, self.args, self.scaler,
                                              self.src_stat_a, self.src_stat_v, self.src_stat_u, labels)
            # -------------------------------------- cmmtta --------------------------------------
            # outputs, loss, dist_a, dist_v = forward_and_adapt(x, self.pva_model, self.optimizer, self.args, self.scaler,
            #                                                   self.src_stat_a, self.src_stat_v, self.src_stat_u, labels)
            # self.detect_window(dist_a, dist_v)
            # -------------------------------------- cmmtta --------------------------------------

            return outputs, loss

    def reset(self):
        if self.optimizer_state is None:
            raise Exception("cannot reset without saved optimizer state")
        self.optimizer.load_state_dict(self.optimizer_state)
        self.pva_model.module.reset()  # reset prompts

    def detect_window(self, dist_a, dist_v, threshold=5):
        window_a = np.array(self.window_a)
        window_v = np.array(self.window_v)
        mu_a = window_a.mean()
        sigma_a = window_a.std(ddof=1) if len(window_a) > 1 else 0
        mu_v = window_v.mean()
        sigma_v = window_v.std(ddof=1) if len(window_v) > 1 else 0

        if sigma_a == 0:
            z_a = 0
        else:
            z_a = (dist_a - mu_a) / sigma_a
        self.z_scores_a.append(z_a)
        if sigma_v == 0:
            z_v = 0
        else:
            z_v = (dist_v - mu_v) / sigma_v
        self.z_scores_v.append(z_v)

        print(dist_a, mu_a, z_a)
        print(dist_v, mu_v, z_v)
        if (z_a > threshold) and (len(self.window_a) == self.window_a.maxlen):
            logger.info(f"Anomaly detected in audio representation: {z_a} ({self.num_batch})")
            self.pva_model.module.restore(modal='a')
            self.window_a.clear()
        else:
            self.window_a.append(dist_a)
        if (z_v > threshold) and (len(self.window_v) == self.window_v.maxlen):
            logger.info(f"Anomaly detected in video representation: {z_v} ({self.num_batch})")
            self.pva_model.module.restore(modal='v')
            self.window_v.clear()
        else:
            self.window_v.append(dist_v)


@torch.jit.script
def cross_entropy(x, x_ema, adatp):
    return -((x_ema / adatp).softmax(1) * x.log_softmax(1)).sum(1)
@torch.jit.script
def contrastive(audio_rep, video_rep, bidirect_contrast: bool = False, tau: float = 0.07):
    # calculate nce loss for mean-visual representation and mean-audio representation
    audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
    video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
    total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / tau

    # by default we use single directional
    if not bidirect_contrast:
        nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
    else:
        nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
        nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
        c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        nce = (nce_1 + nce_2) / 2
        c_acc = (c_acc_1 + c_acc_2) / 2
    return nce, c_acc

def stat_euclid(src_a, src_v, src_u, tst_a, tst_v, tst_u, num_layers=11):
    """Calculate the Euclidean distance between source and target statistics for each layer."""
    # std & mean
    dist_a = F.pairwise_distance(src_a[0][:num_layers], tst_a[0][:num_layers], p=2) + F.pairwise_distance(src_a[1][:num_layers], tst_a[1][:num_layers], p=2)
    dist_v = F.pairwise_distance(src_v[0][:num_layers], tst_v[0][:num_layers], p=2) + F.pairwise_distance(src_v[1][:num_layers], tst_v[1][:num_layers], p=2)
    dist_u = F.pairwise_distance(src_u[0], tst_u[0], p=2) + F.pairwise_distance(src_u[1], tst_u[1], p=2)
    return dist_a, dist_v, dist_u


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler, src_stat_a, src_stat_v, src_stat_u):
    with autocast():
        outputs, _, latent_c_a, latent_c_v, features_a, features_v, features_u, \
            outputs_avm, outputs_amv = model(a=x[0], v=x[1], mode='all', prompt=True, mask=args.mask)

    # Prompt-driven Modality-specific Global Feature Alignment (PMGFA)
    dist_a, dist_v, dist_u = stat_euclid(src_stat_a, src_stat_v, src_stat_u, torch.std_mean(features_a, 0),
                                         torch.std_mean(features_v, 0), torch.std_mean(features_u, 0), num_layers=11)  # std & mean
    loss_align = dist_a.mean() + dist_v.mean()
    # d_a, d_v = dist_a.mean().item(), dist_v.mean().item()  # cmmtta

    # Cross-modal Masked Embedding Recombination (CMER)
    b = torch.tensor(args.b)
    adatp = 1 + args.a / (1 + torch.exp(b - dist_u.mean().item()))
    loss_amv = cross_entropy(outputs_amv, outputs.clone().detach(), adatp=adatp).mean(0)
    loss_avm = cross_entropy(outputs_avm, outputs.clone().detach(), adatp=adatp).mean(0)
    lambda_a = dist_v.mean().item() / (dist_a.mean().item() + dist_v.mean().item())
    lambda_v = dist_a.mean().item() / (dist_a.mean().item() + dist_v.mean().item())
    loss_consistency = lambda_a * loss_amv + lambda_v * loss_avm

    # Inter-modal Instance-wise Contrastive Learning (IICL)
    loss_ctr, _ = contrastive(latent_c_a, latent_c_v, bidirect_contrast=True, tau=args.tau)

    loss = loss_align + loss_consistency + loss_ctr
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return outputs, (loss.item(), 0)
    # return outputs, (loss.item(), 0), d_a, d_v  # cmmtta


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if nm == 'module':
            for np, p in m.named_parameters():
                if 'prompts' in np:
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def configure_model(model):
    model.eval()
    model.requires_grad_(False)

    return model
