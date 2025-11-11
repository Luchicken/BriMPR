import argparse
import os
import sys
import torch
import torch.nn as nn
import dataloader as dataloader
import models
import numpy as np
import time
import warnings
from tqdm import tqdm
from utilities import accuracy, seed_everything
from TTA import setup_model
from utilities.util import get_logger
import logging
os.environ['MPLCONFIGDIR'] = './plt/'
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)


def get_args():
    # TTA for the cav-mae-finetuned model
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str, default='0, 1, 2', help="gpu device number")
    parser.add_argument('--tta_method', type=str, default='BriMPR', choices=['None', 'Tent', 'T3A', 'EATA', 'SAR',
                                                                           'DeYO', 'READ', 'BriMPR'], help='which TTA method to be used')
    parser.add_argument('--corruption_modality', type=str, default='audio', choices=['video', 'audio', 'none'], help='which modality to be corrupted')
    parser.add_argument('--severity_start', type=int, default=5, help='the start severity of the corruption')
    parser.add_argument('--severity_end', type=int, default=5, help='the end severity of the corruption')
    parser.add_argument('--iters', type=int, default=3, help='number of iterations')
    parser.add_argument('--dataset', type=str, default='ks50', choices=['vggsound', 'ks50'], help='dataset name')
    # parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50'], help='dataset name')
    parser.add_argument("--json_root", type=str, default='code_path/json_csv_files/ks50', help="validation data json")
    # parser.add_argument("--json_root", type=str, default='code_path/json_csv_files/vgg', help="validation data json")
    parser.add_argument("--label_csv", type=str, default='code_path/json_csv_files/class_labels_indices_ks50.csv', help="csv with class labels")
    # parser.add_argument("--label_csv", type=str, default='code_path/json_csv_files/class_labels_indices_vgg.csv', help="csv with class labels")
    parser.add_argument("--pretrain_path", type=str, default='code_path/pretrained_model/cav_mae_ks50.pth', help="pretrained model path")
    # parser.add_argument("--pretrain_path", type=str, default='code_path/pretrained_model/vgg_65.5.pth', help="pretrained model path")

    parser.add_argument("--n_class", type=int, default=50, help="number of classes")  # don't need
    parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
    parser.add_argument("--dataset_mean", type=float, default=-5.081, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", type=float, default=4.4849, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", type=int, default=1024, help="the input length in frames")
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num_workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--testmode", type=str, default='multimodal', help="how to test the model")
    parser.add_argument("--output", type=str, default='./logs', help="output directory")

    # T3A parameters
    parser.add_argument('--filter_K', default=20, type=int)  # 20 / 50
    # EATA parameters
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=1.0, help='the trade-off between entropy and regularization loss') # 1. / 1. / 2000.
    parser.add_argument('--d_margin', type=float, default=0.1, help='\epsilon for filtering redundant samples')  # 0.4 / 0.1 / 0.05
    # DeYO parameters
    parser.add_argument('--patch_len', default=4, type=int, help='The number of patches per row/column')
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--deyo_margin', default=0.5, type=float, help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float, help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')

    # BriMPR parameters
    parser.add_argument('--num_prompts_a', type=int, default=10, help='number of inserted prompts for audio')
    parser.add_argument('--num_prompts_v', type=int, default=10, help='number of inserted prompts for video')
    parser.add_argument('--prompt_layers', type=int, default=11, help='number of layers for prompt insertion')
    parser.add_argument("--mask", type=float, default=0.5, help="mask ratio")
    parser.add_argument("--a", type=float, default=0.2, help="hyperparameter for AdaTp")
    parser.add_argument("--b", type=float, default=5, help="hyperparameter for AdaTp")
    parser.add_argument("--tau", type=float, default=0.07, help="tau for contrastive loss")
    parser.add_argument('--num_samples_source', type=int, default=32, help='number of source samples')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'vggsound':
        args.n_class = 309
    elif args.dataset == 'ks50':
        args.n_class = 50

    if args.corruption_modality == 'video':
        corruption_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                           'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
                           'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.corruption_modality == 'audio':
        corruption_list = ['gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind']
    elif args.corruption_modality == 'none':
        corruption_list = ['clean']
        args.severity_start = args.severity_end = 0

    logger = logging.getLogger(__name__)
    get_logger(args)

    ###############################################################
    if args.model == 'cav-mae-ft':
        logger.info('test a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
        va_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
    else:
        raise ValueError('model not supported')
    if args.pretrain_path == 'None':
        warnings.warn("Note no pre-trained models are specified.")
    else:
        # TTA based on a CAV-MAE fine-tuned model
        mdl_weight = torch.load(args.pretrain_path)
        if not isinstance(va_model, nn.DataParallel):
            va_model = nn.DataParallel(va_model).to(device)  # for multi-gpu
        miss, unexpected = va_model.load_state_dict(mdl_weight, strict=False)
        logger.info(f'now load cav-mae fine-tuned weights from {args.pretrain_path}')
        logger.info(f'[miss], [unexpected]: {miss}, {unexpected}')  # check if all weights are correctly loaded

    # setup model
    adapt_flag = False if args.tta_method == 'None' else True
    if args.tta_method == 'None':
        model = setup_model.setup_source(va_model, device, args)
    elif args.tta_method == 'Tent':
        model = setup_model.setup_tent(va_model, device, args)
    elif args.tta_method == 'T3A':
        model = setup_model.setup_t3a(va_model, device, args)
    elif args.tta_method == 'EATA':
        model = setup_model.setup_eata(va_model, device, args)
    elif args.tta_method == 'SAR':
        model = setup_model.setup_sar(va_model, device, args)
    elif args.tta_method == 'DeYO':
        model = setup_model.setup_deyo(va_model, device, args)
    elif args.tta_method == 'READ':
        model = setup_model.setup_read(va_model, device, args)
    elif args.tta_method == 'BriMPR':
        model = setup_model.setup_brimpr(va_model, device, args)
    else:
        raise ValueError('TTA method not supported')

    ###############################################################
    domain_accs = []
    for corruption in corruption_list:  # corruption
        for severity in range(args.severity_start, args.severity_end + 1):  # severity
            epoch_accs = []

            if args.corruption_modality == 'none':  # code_path/json_csv_files/[ks50, vgg]/clean/severity_0.json
                data_val = os.path.join(args.json_root, corruption, 'severity_{}.json').format(severity)
            else:
                data_val = os.path.join(args.json_root, args.corruption_modality, '{}', 'severity_{}.json').format(corruption, severity)
            logger.info(f'===> Now handling: {corruption}-{severity}')

            for itr in range(1, args.iters + 1):
                seed = int(str(itr)*3)
                seed_everything(seed=seed)
                logger.info("### Round {}, Seed={} ###".format(itr, seed))
                if adapt_flag:
                    model.reset()
                    torch.cuda.empty_cache()
                    logger.info("resetting model")

                ###############################################################
                # all exp in this work is based on 224 * 224 image
                im_res = 224
                val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                                  'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}
                tta_loader = torch.utils.data.DataLoader(
                    dataloader.AudiosetDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
                ###############################################################
                with torch.no_grad():
                    for epoch in range(1):
                        data_bar = tqdm(tta_loader)
                        batch_accs = []

                        for i, (a_input, v_input, labels) in enumerate(data_bar):
                            a_input = a_input.to(device)
                            v_input = v_input.to(device)
                            outputs, loss = model((a_input, v_input), adapt_flag=adapt_flag)  # infer and adapt
                            batch_acc = accuracy(outputs, labels, topk=(1, ))
                            batch_acc = round(batch_acc[0].item(), 2)
                            batch_accs.append(batch_acc)

                            if adapt_flag:
                                data_bar.set_description(f'Batch#{i}: Loss#{np.sum(loss):.2f}, Acc#{batch_acc:.1f}')
                            else:
                                data_bar.set_description(f'Batch#{i}: Acc#{batch_acc:.1f}')

                        epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)
                        epoch_accs.append(epoch_acc)
                        logger.info(f'Epoch{epoch}: all acc is {epoch_acc}')

            domain_accs.append(np.mean(epoch_accs))
            logger.info(f'===> {corruption}-{severity}, mean: {np.round(np.mean(epoch_accs), 2)}, std: {np.round(np.std(epoch_accs), 2)}')
    logger.info(f'===> Final result: {np.round(domain_accs, 2)}')
    logger.info(f'===> Mean result: {np.round(np.mean(domain_accs), 2)}')
