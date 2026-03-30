"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
import re
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None 
    return m.to(device) 


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 

    def _setup(self, ):
        """Avoid instantiating unnecessary classes 
        """
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model
        
        # NOTE (lyuwenyu): must load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self._apply_freeze_settings()

        self.model = dist_utils.warp_model(self.model.to(device), sync_bn=cfg.sync_bn, \
            find_unused_parameters=cfg.find_unused_parameters)

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch
        
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

    def _apply_freeze_settings(self):
        patterns = []
        freeze_mode = getattr(self.cfg, 'freeze_mode', '') or ''

        preset_patterns = {
            'freeze_classifier': [
                r'^decoder\.enc_score_head\.',
                r'^decoder\.dec_score_head\.',
                r'^decoder\.denoising_class_embed\.',
            ],
            'freeze_bbox': [
                r'^decoder\.enc_bbox_head\.',
                r'^decoder\.dec_bbox_head\.',
            ],
            'heads_only': [
                r'^backbone\.',
                r'^encoder\.',
                r'^decoder\.input_proj\.',
                r'^decoder\.encoder\.',
                r'^decoder\.decoder\.',
                r'^decoder\.enc_output\.',
                r'^decoder\.query_pos_head\.',
            ],
        }

        if freeze_mode:
            assert freeze_mode in preset_patterns, \
                f'Unsupported freeze_mode: {freeze_mode}. Expected one of {list(preset_patterns.keys())}'
            patterns.extend(preset_patterns[freeze_mode])

        extra_patterns = getattr(self.cfg, 'freeze_modules', None) or []
        patterns.extend(extra_patterns)

        if not patterns:
            return

        frozen_names = []
        pattern_hits = {pattern: 0 for pattern in patterns}

        for name, param in self.model.named_parameters():
            if any(re.findall(pattern, name) for pattern in patterns):
                param.requires_grad = False
                frozen_names.append(name)
                for pattern in patterns:
                    if re.findall(pattern, name):
                        pattern_hits[pattern] += 1

        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params

        print(f'Applied freeze settings: freeze_mode={freeze_mode or "custom"}')
        print(f'Frozen parameter tensors: {len(frozen_names)}')
        print(f'Frozen parameters: {frozen_params} / {total_params}')
        print(f'Trainable parameters remaining: {trainable_params}')
        for pattern, hits in pattern_hits.items():
            print(f'  pattern {pattern!r}: matched {hits} tensors')

    def cleanup(self, ):
        if self.writer:
            atexit.register(self.writer.close)

    def train(self, ):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self, ):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator
        
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                v.to(device)

    def state_dict(self):
        """state dict, train/eval
        """
        state = {}
        state['date'] = datetime.now().isoformat()
        
        # TODO for resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict() 

        return state


    def load_state_dict(self, state):
        """load state dict, train/eval
        """
        # TODO
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')


    def load_resume_state(self, path: str):
        """load resume
        """
        # for cuda:0 memory
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        self.load_state_dict(state)

    
    def load_tuning_state(self, path: str,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist_utils.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')


    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')


    def val(self, ):
        raise NotImplementedError('')
