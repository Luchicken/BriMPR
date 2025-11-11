import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import logging
logger = logging.getLogger(__name__)


class T3A(nn.Module):
    def __init__(self, model, device, args, filter_K):
        super().__init__()
        self.model = model
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        warmup_supports = self.model.module.mlp_head[1].weight.data  # shape: (num_classes, 768)
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.module.mlp_head[1](self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=args.n_class).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = filter_K
        self.num_classes = args.n_class
        self.softmax = nn.Softmax(-1)

    def forward(self, x, adapt_flag):
        # z = self.featurizer(x)
        with autocast():
            z, _ = self.model(a=x[0], v=x[1], mode="features", test=True)
        # online adaptation
        # p = self.classifier(z)
        p = self.model.module.mlp_head[1](z)
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        # prediction
        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = F.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ F.normalize(weights, dim=0), 0

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).cuda()

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def predict(self, x, adapt_flag=False):
        return self(x, adapt_flag)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
