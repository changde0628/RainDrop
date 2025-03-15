import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from DiffIR.models import lr_scheduler as lr_scheduler
from torch import nn
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

@MODEL_REGISTRY.register()
class DefocusDiffIRS2(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(DefocusDiffIRS2, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        else:
            raise ValueError('Please specify the pretrain_network_S1 path in the config file.')
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            self.model_Es1 = self.net_g_S1.module.E
        else:
            self.model_Es1 = self.net_g_S1.E
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        if self.is_train:
            self.encoder_iter = opt["train"]["encoder_iter"]
            self.lr_encoder = opt["train"]["lr_encoder"]
            self.lr_sr = opt["train"]["lr_sr"]
            self.gamma_encoder = opt["train"]["gamma_encoder"]
            self.gamma_sr = opt["train"]["gamma_sr"]
            self.lr_decay_encoder = opt["train"]["lr_decay_encoder"]
            self.lr_decay_sr = opt["train"]["lr_decay_sr"]

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized in the second stage.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        parms=[]
        for k,v in self.net_g.named_parameters():
            if "denoise" in k or "condition" in k:
                parms.append(v)
        if 'optim_e' in train_opt:
            optim_type_e = train_opt['optim_e'].pop('type')
            optim_params_e = train_opt['optim_e']
        else:
            optim_params_e = train_opt['optim_g'].copy()
        self.optimizer_e = self.get_optimizer(optim_type, parms, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_e)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None

        if train_opt.get('freq_opt'):
            self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(DefocusDiffIRS2, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def pad_test(self, window_size):        
        # scale = self.opt.get('scale', 1)
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        return lq,gt,mod_pad_h,mod_pad_w

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq,gt,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq
            gt=self.gt
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq)
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def optimize_parameters(self, current_iter):
        l_total = 0
        loss_dict = OrderedDict()
        _, S1_IPR = self.model_Es1(self.lq, self.gt)

        # 根據分散式訓練狀態獲取正確的 diffusion 物件
        if self.opt['dist']:
            diffusion = self.net_g.module.diffusion
        else:
            diffusion = self.net_g.diffusion
        
        # 第一階段訓練 - 主要針對CPEN_S2和去噪網絡
        if current_iter < self.encoder_iter:
            self.optimizer_e.zero_grad()
            
            # 如果是新的DefocusCPEN並且有kernel_branch
            if (self.opt['dist'] and hasattr(self.net_g.module.condition, 'kernel_branch')) or \
            (not self.opt['dist'] and hasattr(self.net_g.condition, 'kernel_branch')):
                
                # 使用DefocusCPEN時需要處理可能的額外輸出
                outputs = diffusion(self.lq, S1_IPR[0])
                
                # 檢查返回值是否包含模糊核參數
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    pred_IPR_list, kernel_params = outputs
                    
                    # 如果配置了模糊核損失並有Ground Truth核
                    if hasattr(self, 'cri_kernel') and hasattr(self, 'kernel_gt'):
                        l_kernel = self.cri_kernel(kernel_params, self.kernel_gt)
                        l_total += self.opt['train']['kernel_weight'] * l_kernel
                        loss_dict['l_kernel'] = l_kernel
                else:
                    _, pred_IPR_list = outputs
            else:
                # 原始行為
                _, pred_IPR_list = diffusion(self.lq, S1_IPR[0])
            
            # 原有的KD損失計算
            i = len(pred_IPR_list) - 1
            S2_IPR = [pred_IPR_list[i]]
            l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
            l_total += l_abs
            loss_dict['l_kd_%d' % (i)] = l_kd
            loss_dict['l_abs_%d' % (i)] = l_abs

            l_total.backward()
            self.optimizer_e.step()
        
        # 第二階段訓練 - 整個網絡聯合訓練
        else:
            self.optimizer_g.zero_grad()
            
            # 檢查是否使用DefocusCPEN
            if (self.opt['dist'] and hasattr(self.net_g.module.condition, 'kernel_branch')) or \
            (not self.opt['dist'] and hasattr(self.net_g.condition, 'kernel_branch')):
                
                # 使用更新的forward方法處理可能的額外輸出
                outputs = self.net_g(self.lq, S1_IPR[0])
                
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    self.output, pred_IPR_list, kernel_params = outputs
                    
                    # 如果配置了模糊核損失並有Ground Truth核
                    if hasattr(self, 'cri_kernel') and hasattr(self, 'kernel_gt'):
                        l_kernel = self.cri_kernel(kernel_params, self.kernel_gt)
                        l_total += self.opt['train']['kernel_weight'] * l_kernel
                        loss_dict['l_kernel'] = l_kernel
                else:
                    self.output, pred_IPR_list = outputs
            else:
                # 原始行為
                self.output, pred_IPR_list = self.net_g(self.lq, S1_IPR[0])
            
            # 原有的像素損失
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            # 原有的KD損失
            i = len(pred_IPR_list) - 1
            S2_IPR = [pred_IPR_list[i]]
            l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
            l_total += l_abs
            loss_dict['l_kd_%d' % (i)] = l_kd
            loss_dict['l_abs_%d' % (i)] = l_abs
            
            # 添加頻域損失
            if hasattr(self, 'cri_freq') and self.cri_freq is not None:
                l_freq = self.cri_freq(self.output, self.gt)
                # 從配置中獲取權重或使用默認值
                freq_weight = self.opt['train'].get('freq_weight', 0.2)
                l_total += freq_weight * l_freq
                loss_dict['l_freq'] = l_freq

            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)