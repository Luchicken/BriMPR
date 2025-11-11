import torch.nn as nn
from torch.cuda.amp import autocast


class Source(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        self.model = model
        self.args = args
        self.device = device

    def forward(self, x, adapt_flag):
        with autocast():
            outputs, _ = self.model(a=x[0], v=x[1], mode=self.args.testmode, test=True)
        return outputs, None


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
