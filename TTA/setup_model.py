import torch
import torch.nn as nn
import math
from TTA import Source, Tent, T3A, EATA, SAR, DeYO, READ, BriMPR
from models.vapt import PromptVA
import logging
logger = logging.getLogger(__name__)


def param_count(model):
    trainables = [p for p in model.parameters() if p.requires_grad]
    logger.info('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    logger.info('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    logger.info('Trainable parameter ratio is : {:.3f}%'.format(sum(p.numel() for p in trainables) / sum(p.numel() for p in model.parameters()) * 100))


def setup_source(model, device, args):
    model = Source.configure_model(model)
    source_model = Source.Source(model, device, args)
    return source_model


def setup_tent(model, device, args):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0., betas=(0.9, 0.999))
    tent_model = Tent.Tent(model, optimizer, device, args)
    param_count(model)
    return tent_model


def setup_t3a(model, device, args):
    model = T3A.configure_model(model)
    t3a_model = T3A.T3A(model, device, args, filter_K=args.filter_K)
    return t3a_model


def setup_eata(model, device, args):
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0., betas=(0.9, 0.999))
    eata_model = EATA.EATA(model, optimizer, device, args,
                           fisher_alpha=args.fisher_alpha, fisher_size=args.fisher_size, d_margin=args.d_margin, params=params)
    param_count(model)
    return eata_model


def setup_sar(model, device, args):
    model = SAR.configure_model(model)
    params, param_names = SAR.collect_params(model)
    base_optimizer = torch.optim.Adam
    optimizer = SAR.SAM(params, base_optimizer, lr=1e-4, weight_decay=0., betas=(0.9, 0.999))
    sar_model = SAR.SAR(model, optimizer, device, args)
    param_count(model)
    return sar_model


def setup_deyo(model, device, args):
    model = DeYO.configure_model(model)
    params, param_names = DeYO.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0., betas=(0.9, 0.999))
    args.deyo_margin *= math.log(args.n_class) # for thresholding
    args.deyo_margin_e0 *= math.log(args.n_class) # for reweighting tuning
    deyo_model = DeYO.DeYO(model, optimizer, device, args, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
    param_count(model)
    return deyo_model


def setup_read(model, device, args):
    model = READ.configure_model(model)  # configure the parameters with gradient enabled
    params, param_names = READ.collect_params(model)  # collect the parameters for optimization
    optimizer = torch.optim.Adam([{'params': params, 'lr': 1e-4}], weight_decay=0., betas=(0.9, 0.999))
    read_model = READ.READ(model, optimizer, device, args)
    read_model.eval()
    param_count(model)
    return read_model


def setup_brimpr(model, device, args):
    model = BriMPR.configure_model(model.module)  # model.module -> CAVMAEFT
    pva_model = PromptVA(model, args.num_prompts_a, args.num_prompts_v, args.prompt_layers)
    pva_model = nn.DataParallel(pva_model).to(device)
    # set up optimizer
    params, param_names = BriMPR.collect_params(pva_model)
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0., betas=(0.9, 0.999))
    # set up BriMPR
    brimpr_model = BriMPR.BriMPR(pva_model, optimizer, device, args)
    param_count(brimpr_model)
    return brimpr_model
