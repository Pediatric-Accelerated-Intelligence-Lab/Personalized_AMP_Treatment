"""
Usage:
    python pipeline.py -c config.json --stage prepare
    python pipeline.py -c config.json --stage train
    python pipeline.py -c config.json -r path/to/checkpoint.pth --stage test
    python pipeline.py -c config.json --stage all
"""

import argparse
import collections
import copy
import datetime
import json
import logging
import math
import os
import random
import shutil
import socket
import sys
import time
from datetime import datetime as dt
from functools import reduce
from operator import getitem
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vtk
from scipy.spatial import distance
from skimage import transform
from torch.utils.data import DataLoader, Dataset
from vtk.util.numpy_support import vtk_to_numpy

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    import itertools
    for loader in itertools.repeat(data_loader):
        yield from loader


# ===========================================================================
# logger.py
# ===========================================================================

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s - %(asctime)s - %(name)s - %(module)s]: %(message)s"
    log_console_format = "[%(levelname)s - %(name)s]: %(message)s"

    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(log_file_format)
    file_handler = logging.FileHandler(Path(log_dir) / 'log.log')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_console_format))
    console_handler.setLevel(logging.WARNING)

    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)


# ===========================================================================
# parse_config.py
# ===========================================================================

class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()
        self._name = None

        if hasattr(args, 'device') and args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        self.cfg_fname = None
        self.resume = None
        if getattr(args, 'resume', None):
            self.resume = Path(args.resume)
            if getattr(args, 'config', None) is not None:
                self.cfg_fname = Path(args.config)
            else:
                self.cfg_fname = self.resume.parent / 'config.json'

        if self.cfg_fname is None:
            assert args.config is not None, \
                "Configuration file need to be specified. Add '-c config.json', for example."
            self.cfg_fname = Path(args.config)

        if getattr(args, 'name', None):
            self._name = str(args.name)

        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

        save_dir = Path(self.config['trainer']['save_dir'])
        ts = dt.now().strftime(r'%d%m%y_%H%M%S') if timestamp else ''
        exper_name = self.config['name']

        self._save_dir = save_dir / 'models' / exper_name / ts
        self._log_dir = save_dir / 'log' / exper_name / ts
        self._temp_dir = save_dir / 'temp' / exper_name / ts

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        write_json(self.config, self.save_dir / 'config.json')
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def initialize(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all(k not in module_args for k in kwargs), \
            'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def temp_dir(self):
        return self._temp_dir

    @property
    def name(self):
        return self._name


def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    return reduce(getitem, keys, tree)


# ===========================================================================
# base/base_model.py
# ===========================================================================

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


# ===========================================================================
# base/base_data_loader.py
# ===========================================================================

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=None):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


# ===========================================================================
# base/base_trainer.py
# ===========================================================================

class BaseTrainer:
    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        self.writer = None
        if cfg_trainer.get('tensorboard', False):
            self.writer = _NullWriter()

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            shutil.copyfile(filename, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    def _progress(self, batch_idx):
        raise NotImplementedError


class _NullWriter:
    class _Inner:
        def add_scalar(self, *a, **k):
            pass

        def add_image(self, *a, **k):
            pass

        def add_histogram(self, *a, **k):
            pass

        def set_step(self, *a, **k):
            pass

    def __init__(self):
        self.writer = self._Inner()

    def set_step(self, *a, **k):
        pass


# ===========================================================================
# trainer.py
# ===========================================================================

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            if self.writer is not None:
                self.writer.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, sample_batched in enumerate(self.data_loader):
            data, target = sample_batched['image'], sample_batched['heat_map_stack']
            data = data.permute(0, 3, 1, 2)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            output = output.permute(1, 0, 2, 3, 4)
            target = target.permute(0, 1, 4, 2, 3)

            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            if self.writer is not None:
                self.writer.writer.add_scalar('train/loss', loss.item())
            total_loss += loss.item()

            time_per_test = (time.time() - start_time) / (batch_idx + 1)
            time_left = (self.len_epoch - batch_idx) * time_per_test

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time per batch: {:.5} Time left in epoch: {}'.format(
                    epoch, self._progress(batch_idx), loss.item(), time_per_test,
                    str(datetime.timedelta(seconds=time_left))))

            if batch_idx == self.len_epoch:
                break

        log = {'loss': total_loss / self.len_epoch}

        print('Debug saving checkpoint')
        self._save_checkpoint(epoch, save_best=False)

        print('Doing validation')
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        n_validation = len(self.valid_data_loader)
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(self.valid_data_loader):
                data, target = sample_batched['image'], sample_batched['heat_map_stack']
                data = data.permute(0, 3, 1, 2)
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                output = output.permute(1, 0, 2, 3, 4)
                target = target.permute(0, 1, 4, 2, 3)

                loss = self.loss(output, target)
                total_val_loss += loss.item()

                time_per_test = (time.time() - start_time) / (batch_idx + 1)
                time_left = (n_validation - batch_idx) * time_per_test

                if batch_idx % self.log_step == 0:
                    self.logger.debug(
                        'Validation: {}/{} Loss: {:.6f} Time per batch: {:.5} Time left in validation: {}'.format(
                            batch_idx, n_validation, loss, time_per_test,
                            str(datetime.timedelta(seconds=time_left))))

        if self.writer is not None:
            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            self.writer.writer.add_scalar('validation/loss', avg_val_loss, epoch)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @property
    def device(self):
        return next(self.model.parameters()).device


# ===========================================================================
# model/loss.py
# ===========================================================================

def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


# ===========================================================================
# model/metric.py  (placeholders)
# ===========================================================================

def my_metric(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).item() if output.numel() else 0.0


def my_metric2(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2).item() if output.numel() else 0.0


# ===========================================================================
# model/model.py  (GLM)
# ===========================================================================

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.resample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.resample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.resample is not None:
            residual = self.resample(residual)

        out3 += residual
        return out3


class HourGlassModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = num_features
        for i in range(1, 21):
            setattr(self, 'rb{}'.format(i), ResidualBlock(self.features, self.features))

    def forward(self, x):
        up1 = self.rb1(x)
        lowt1 = F.max_pool2d(x, 2)
        low1 = self.rb2(lowt1)

        up11 = self.rb3(low1)
        lowt11 = F.max_pool2d(low1, 2)
        low11 = self.rb4(lowt11)

        up12 = self.rb5(low11)
        lowt12 = F.max_pool2d(low11, 2)
        low12 = self.rb6(lowt12)

        up13 = self.rb7(low12)
        lowt13 = F.max_pool2d(low12, 2)
        low13 = self.rb8(lowt13)

        up14 = self.rb9(low13)
        lowt14 = F.max_pool2d(low13, 2)
        low14 = self.rb10(lowt14)

        low2 = self.rb11(low14)
        low3 = self.rb12(low2)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        add1 = up2 + up14

        low21 = self.rb13(add1)
        low31 = self.rb14(low21)
        up21 = F.interpolate(low31, scale_factor=2, mode='nearest')
        add2 = up21 + up13

        low22 = self.rb15(add2)
        low32 = self.rb16(low22)
        up22 = F.interpolate(low32, scale_factor=2, mode='nearest')
        add3 = up22 + up12

        low23 = self.rb17(add3)
        low33 = self.rb18(low23)
        up23 = F.interpolate(low33, scale_factor=2, mode='nearest')
        add4 = up23 + up11

        low24 = self.rb19(add4)
        low34 = self.rb20(low24)
        up24 = F.interpolate(low34, scale_factor=2, mode='nearest')
        add5 = up24 + up1
        return add5


class GLMModel(BaseModel):
    def __init__(self, n_landmarks=73, n_features=256, dropout_rate=0.2, image_channels="geometry"):
        super().__init__()
        self.out_features = n_landmarks
        self.features = n_features
        self.dropout_rate = dropout_rate

        channel_map = {
            "geometry": 1, "RGB": 3, "depth": 1,
            "RGB+depth": 4, "geometry+depth": 2,
        }
        self.in_channels = channel_map.get(image_channels, 1)
        if image_channels not in channel_map:
            print("Image channels should be: geometry, RGB, depth, RGB+depth or geometry+depth")

        self.conv1 = nn.Conv2d(self.in_channels, int(self.features / 4), kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(self.features / 4))
        self.conv2 = ResidualBlock(int(self.features / 4), int(self.features / 2))
        self.conv3 = ResidualBlock(int(self.features / 2), int(self.features / 2))
        self.conv4 = ResidualBlock(int(self.features / 2), self.features)
        self.hg1 = HourGlassModule(self.features)
        self.hg2 = HourGlassModule(self.features)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.conv5 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.features)
        self.conv6 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(self.out_features, self.features, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.conv9 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.features)
        self.conv10 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        r3 = self.conv4(x)
        x = self.hg1(r3)
        x = self.dropout1(x)
        ll1 = F.relu(self.bn2(self.conv5(x)), True)
        x = self.conv6(ll1)
        up_temp = F.interpolate(x, scale_factor=2, mode='nearest')
        up_out = self.conv8(up_temp)
        x = self.conv7(x)

        sum_temp = r3 + ll1 + x

        x = self.hg2(sum_temp)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv9(x)), True)
        x = self.conv10(x)
        up_temp2 = F.interpolate(x, scale_factor=2, mode='nearest')
        up_out2 = self.conv11(up_temp2)

        outputs = torch.stack([up_out, up_out2])
        return outputs


# ===========================================================================
# data_loader: GenericHeatmapDataset + HeatmapDataLoader
# ===========================================================================

class GenericHeatmapDataset(Dataset):
    """
    Generic 2D-landmark heatmap dataset.

    Directory layout under root_dir:
        images/<id>[_suffix].png                 (any of: RGB, geometry, depth, ...)
        keypoints/<id>.txt                       (whitespace-separated x y per landmark,
                                                  one landmark per line)

    `<id>` entries come from csv_file, one per line (paths may be nested).
    For each entry, `n_views` augmented variants are expected, named
    `<id>_0`, `<id>_1`, ... on disk (e.g. `foo_3_geometry.png`).
    """

    def __init__(self, csv_file, root_dir, heatmap_size=256, image_size=256,
                 image_channels="RGB", n_views=96, keypoints_subdir="keypoints",
                 images_subdir="images", suffix_map=None, tfrm=None):
        self.file_ids = []
        with open(csv_file) as f:
            for line in f:
                line = line.strip().strip("\n")
                if len(line) > 0:
                    self.file_ids.append(line)
        print('Read ', len(self.file_ids), ' file ids')

        self.root_dir = Path(root_dir)
        self.transform = tfrm
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_views = n_views
        self.keypoints_subdir = keypoints_subdir
        self.images_subdir = images_subdir

        # Per-channel filename suffix (before .png). If a channel maps to "",
        # the bare `<id>.png` is used. Add/remove channels as needed.
        default_suffix_map = {
            "RGB": [""],
            "geometry": ["_geometry"],
            "depth": ["_zbuffer"],
            "RGB+depth": ["", "_zbuffer"],
            "geometry+depth": ["_geometry", "_zbuffer"],
        }
        self.suffix_map = suffix_map or default_suffix_map

        if image_channels not in self.suffix_map:
            raise ValueError(
                "image_channels '{}' has no suffix mapping. "
                "Provide one via suffix_map.".format(image_channels)
            )

        # Build the id table with per-view augmentations
        self.id_table = []
        for f_name in self.file_ids:
            clean_name, _ = os.path.splitext(f_name)
            for n in range(self.n_views):
                self.id_table.append(clean_name + '_' + str(n))
        print('Generated ', len(self.id_table), ' file ids including augmentations')
        self._check_files()

    def _check_if_valid_file(self, file_name):
        if not os.path.isfile(file_name):
            print(file_name, " is not a file!")
            return False
        elif os.stat(file_name).st_size < 10:
            print(file_name, " is not valid (length less than 10 bytes)")
            return False
        return True

    def _image_path(self, file_name, suffix):
        return os.path.join(self.root_dir, self.images_subdir, file_name + suffix + '.png')

    def _keypoints_path(self, file_name):
        return os.path.join(self.root_dir, self.keypoints_subdir, file_name + '.txt')

    def _check_files(self):
        suffixes = self.suffix_map[self.image_channels]
        print('Checking if all files are there')
        new_id_table = []
        for file_name in self.id_table:
            ok = True
            for suffix in suffixes:
                if not self._check_if_valid_file(self._image_path(file_name, suffix)):
                    ok = False
                    break
            if ok:
                new_id_table.append(file_name)
        print('Checking done')
        self.id_table = new_id_table
        print('Final ', len(self.id_table), ' file ids including augmentations')

    def _make_gaussian(self, height, width, sigma=3, center=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_heat_maps(self, height, width, lms, max_length):
        num_lms = lms.shape[0]
        hm = np.zeros((height, width, num_lms), dtype=np.float32)
        for i in range(num_lms):
            if not (np.array_equal(lms[i], [-1, -1])):
                s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
                hm[:, :, i] = self._make_gaussian(height, width, sigma=s, center=(lms[i, 0], lms[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _safe_read_and_scale_image(self, image_file, img_size):
        img_in = None
        org_size = 1024
        if self._check_if_valid_file(image_file):
            try:
                img_t = imageio.imread(image_file)
                org_size = img_t.shape[0]
                if org_size == img_size:
                    img_in = img_t / 255
                else:
                    img_in = transform.resize(img_t, (img_size, img_size), mode='constant')
            except IOError as e:
                print("File ", image_file, " raises exception")
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except ValueError:
                print("File ", image_file, " raises exception")
                print("ValueError")
        return img_in, org_size

    def __len__(self):
        return len(self.id_table)

    def __getitem__(self, idx):
        file_name = self.id_table[idx]
        rendering_type = self.image_channels
        img_size = self.image_size
        org_img_size = 1024  # default

        suffixes = self.suffix_map[rendering_type]
        n_channels = len(suffixes)
        image = np.zeros((img_size, img_size, n_channels), dtype=np.float32)

        for ch, suffix in enumerate(suffixes):
            img_in, org_img_size = self._safe_read_and_scale_image(
                self._image_path(file_name, suffix), img_size)
            if img_in is None:
                continue
            # Single-channel outputs: take channel 0 to be safe
            if img_in.ndim == 3 and n_channels == 1:
                image[:, :, ch] = img_in[:, :, 0]
            elif img_in.ndim == 3:
                # Assume colour data in the first channel group
                image[:, :, ch] = img_in[:, :, 0] if ch > 0 else img_in.mean(axis=2)
            else:
                image[:, :, ch] = img_in

        lm_name = self._keypoints_path(file_name)
        try:
            input_file = open(lm_name, 'r')
        except IOError:
            print('Cannot open ', lm_name)
            return None, None
        landmarks = np.array([line.rstrip().split(' ') for line in input_file])
        landmarks = landmarks.astype(float)

        hm_size = self.heatmap_size
        scaled_lm = landmarks / org_img_size * hm_size
        heat_map = self._generate_heat_maps(hm_size, hm_size, scaled_lm, hm_size)

        n_stacks = 2
        heat_map = np.expand_dims(heat_map, axis=0)
        heat_map = np.repeat(heat_map, n_stacks, axis=0)

        sample = {'image': image, 'heat_map_stack': heat_map}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HeatmapDataLoader(BaseDataLoader):
    def __init__(self, data_dir, heatmap_size=256, image_size=256, image_channels="RGB",
                 n_views=96, batch_size=8, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        self.data_dir = data_dir
        self.csv_file_name = os.path.join(self.data_dir, 'dataset_train.txt')
        self.dataset = GenericHeatmapDataset(
            csv_file=self.csv_file_name, root_dir=data_dir,
            heatmap_size=heatmap_size, image_size=image_size,
            image_channels=image_channels, n_views=n_views)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# ===========================================================================
# prediction.py
# ===========================================================================

class Predict2D:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

    def find_heat_map_maxima(self, heatmaps, sigma=None, method="simple"):
        out_dim = heatmaps.shape[0]
        hm_size = heatmaps.shape[1]
        coordinates = np.zeros((out_dim, 3), dtype=np.float32)

        if method == "simple":
            for k in range(out_dim):
                hm = copy.copy(heatmaps[k, :, :])
                highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
                px, py = highest_idx
                value = hm[px, py]
                coordinates[k, :] = (px - 1, py - 0.5, value)

        if method == "moment":
            for k in range(out_dim):
                hm = heatmaps[k, :, :]
                px, py = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
                value = np.max(hm)

                sz = 15
                a_len = 2 * sz + 1
                if px > sz and hm_size - px > sz and py > sz and hm_size - py > sz:
                    slc = hm[px - sz:px + sz + 1, py - sz:py + sz + 1]
                    ar = np.arange(a_len)
                    sum_x = np.sum(slc, axis=1)
                    s = np.sum(np.multiply(ar, sum_x))
                    ss = np.sum(sum_x)
                    pos = s / ss - sz
                    px = px + pos

                    sum_y = np.sum(slc, axis=0)
                    s = np.sum(np.multiply(ar, sum_y))
                    ss = np.sum(sum_y)
                    pos = s / ss - sz
                    py = py + pos

                coordinates[k, :] = (px - 1, py - 0.5, value)

        return coordinates

    def find_maxima_in_batch_of_heatmaps(self, heatmaps, cur_id, heatmap_maxima):
        write_heatmaps = False
        heatmaps = heatmaps.numpy()
        batch_size = heatmaps.shape[0]

        f = None
        for idx in range(batch_size):
            if write_heatmaps:
                name_hm_maxima = self.config.temp_dir / ('hm_maxima' + str(cur_id + idx) + '.txt')
                f = open(name_hm_maxima, 'w')

            coordinates = self.find_heat_map_maxima(heatmaps[idx, :, :, :], method='moment')
            for lm_no in range(coordinates.shape[0]):
                px = coordinates[lm_no][0]
                py = coordinates[lm_no][1]
                value = coordinates[lm_no][2]
                if value > 1.2:
                    print("Found heatmap with value > 1.2 LM {} value {} pos {} {}  ".format(lm_no, value, px, py))
                    value = 0
                heatmap_maxima[lm_no, cur_id + idx, :] = (px, py, value)
                if write_heatmaps:
                    out_str = str(px) + ' ' + str(py) + ' ' + str(value) + '\n'
                    f.write(out_str)

            if write_heatmaps:
                f.close()

    def generate_image_with_heatmap_maxima(self, image, heat_map):
        im_size = image.shape[0]
        hm_size = heat_map.shape[2]
        i = image.copy()

        coordinates = self.find_heat_map_maxima(heat_map, method='moment')

        factor = im_size / hm_size
        for c in range(coordinates.shape[0]):
            px = coordinates[c][0]
            py = coordinates[c][1]
            if not np.isnan(px) and not np.isnan(py):
                cx = int(px * factor)
                cy = int(py * factor)
                for x in range(cx - 2, cx + 2):
                    for y in range(cy - 2, cy + 2):
                        i[x, y, 0] = 0
                        i[x, y, 1] = 0
                        i[x, y, 2] = 1
        return i

    def show_image_and_heatmap(self, image, heat_map):
        heat_map = heat_map.numpy()
        im_size = image.size(2)
        hm_size = heat_map.shape[2]

        i = np.zeros((im_size, im_size, 3))
        i[:, :, 0] = image[0, :, :]
        i[:, :, 1] = image[0, :, :]
        i[:, :, 2] = image[0, :, :]

        hm = np.zeros((hm_size, hm_size, 3))
        n_lm = heat_map.shape[0]
        for lm in range(n_lm):
            r, g, b = random.random(), random.random(), random.random()
            length = math.sqrt(r * r + g * g + b * b)
            r, g, b = r / length, g / length, b / length
            hm[:, :, 0] += heat_map[lm, :, :] * r
            hm[:, :, 1] += heat_map[lm, :, :] * g
            hm[:, :, 2] += heat_map[lm, :, :] * b

        im_marked = self.generate_image_with_heatmap_maxima(i, heat_map)

        plt.figure()
        plt.imshow(i)
        plt.figure()
        plt.imshow(hm)
        plt.figure()
        plt.imshow(im_marked)
        plt.axis('off')
        plt.ioff()
        plt.show()

    def write_batch_of_heatmaps(self, heatmaps, images, cur_id):
        batch_size = heatmaps.shape[0]

        for idx in range(batch_size):
            name_hm_maxima = str(self.config.temp_dir / ('heatmap' + str(cur_id + idx) + '.png'))
            name_hm_maxima_2 = str(self.config.temp_dir / ('heatmap_max' + str(cur_id + idx) + '.png'))
            heatmap = heatmaps[idx, :, :, :]
            heatmap = heatmap.numpy()
            hm_size = heatmap.shape[2]

            hm = np.zeros((hm_size, hm_size, 3))
            n_lm = heatmap.shape[0]
            for lm in range(n_lm):
                r, g, b = random.random(), random.random(), random.random()
                length = math.sqrt(r * r + g * g + b * b)
                r, g, b = r / length, g / length, b / length
                hm[:, :, 0] += heatmap[lm, :, :] * r
                hm[:, :, 1] += heatmap[lm, :, :] * g
                hm[:, :, 2] += heatmap[lm, :, :] * b

            imageio.imwrite(name_hm_maxima, hm)

            im = images[idx]
            im_marked = self.generate_image_with_heatmap_maxima(im, heatmap)

            imageio.imwrite(name_hm_maxima_2, im_marked)

    def predict_heatmaps_from_images(self, image_stack):
        n_views = self.config['data_loader']['args']['n_views']
        batch_size = self.config['data_loader']['args']['batch_size']
        n_landmarks = self.config['arch']['args']['n_landmarks']

        write_heatmaps = False
        show_result_image = False
        heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

        print('Predicting heatmaps for all views')
        start = time.time()
        cur_id = 0
        while cur_id + batch_size <= n_views:
            cur_images = image_stack[cur_id:cur_id + batch_size, :, :, :]

            data = torch.from_numpy(cur_images)
            data = data.permute(0, 3, 1, 2)

            with torch.no_grad():
                data = data.to(self.device)
                output = self.model(data)

                if cur_id == 0 and show_result_image:
                    image = data[0, :, :, :].cpu()
                    heat_map = output[1, 0, :, :, :].cpu()
                    self.show_image_and_heatmap(image, heat_map)

                heatmaps = output[1, :, :, :, :].cpu()
                self.find_maxima_in_batch_of_heatmaps(heatmaps, cur_id, heatmap_maxima)
                if write_heatmaps:
                    self.write_batch_of_heatmaps(heatmaps, cur_images, cur_id)

            cur_id = cur_id + batch_size

        end = time.time()
        print("Model prediction time: " + str(end - start))
        return heatmap_maxima


# ===========================================================================
# utils3d.py
# ===========================================================================

class Utils3D:
    def __init__(self, config):
        self.config = config
        self.heatmap_maxima = None
        self.transformations_3d = None
        self.lm_start = None
        self.lm_end = None
        self.landmarks = None
        self.logger = config.get_logger('Utils3D')

    def read_heatmap_maxima(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_landmarks = self.config['arch']['args']['n_landmarks']
        n_views = self.config['data_loader']['args']['n_views']

        self.heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/hm_maxima' + str(idx) + '.txt'
            with open(name_hm_maxima) as f:
                id_lm = 0
                for line in f:
                    x, y, val = np.double(line.strip().split(" "))
                    self.heatmap_maxima[id_lm, idx, :] = (x, y, val)
                    id_lm = id_lm + 1
                    if id_lm > n_landmarks:
                        print('Too many landmarks in file ', name_hm_maxima)
                        break

            if id_lm != n_landmarks:
                print('Too few landmarks in file ', name_hm_maxima)

    def read_3d_transformations(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_views = self.config['data_loader']['args']['n_views']
        self.transformations_3d = np.zeros((n_views, 6))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/transform' + str(idx) + '.txt'
            rx, ry, rz, s, tx, ty = np.loadtxt(name_hm_maxima)
            self.transformations_3d[idx, :] = (rx, ry, rz, s, tx, ty)

    def compute_lines_from_heatmap_maxima(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        n_views = self.heatmap_maxima.shape[1]

        self.lm_start = np.zeros((n_landmarks, n_views, 3))
        self.lm_end = np.zeros((n_landmarks, n_views, 3))

        img_size = self.config['data_loader']['args']['image_size']
        hm_size = self.config['data_loader']['args']['heatmap_size']
        winsize = img_size

        x_min = -150
        x_max = 150
        y_min = -150
        y_max = 150
        x_len = x_max - x_min
        y_len = y_max - y_min

        pd = vtk.vtkPolyData()
        for idx in range(n_views):
            rx, ry, rz, s, tx, ty = self.transformations_3d[idx, :]

            t = vtk.vtkTransform()
            t.Identity()
            t.Update()

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()

            for lm_no in range(n_landmarks):
                y = self.heatmap_maxima[lm_no, idx, 0]
                x = self.heatmap_maxima[lm_no, idx, 1]

                y = y / hm_size * img_size
                x = x / hm_size * img_size

                p_wc_s = np.zeros((3, 1))
                p_wc_e = np.zeros((3, 1))

                p_wc_s[0] = (x / winsize) * x_len + x_min
                p_wc_s[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_s[2] = 500

                p_wc_e[0] = (x / winsize) * x_len + x_min
                p_wc_e[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_e[2] = -500

                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

                lines.InsertNextCell(2)
                pid = points.InsertNextPoint(p_wc_s)
                lines.InsertCellPoint(pid)
                pid = points.InsertNextPoint(p_wc_e)
                lines.InsertCellPoint(pid)

                pd.SetPoints(points)
                del points
                pd.SetLines(lines)
                del lines

                tfilt = vtk.vtkTransformPolyDataFilter()
                tfilt.SetTransform(t.GetInverse())
                tfilt.SetInputData(pd)
                tfilt.Update()

                lm_out = vtk.vtkPolyData()
                lm_out.DeepCopy(tfilt.GetOutput())

                self.lm_start[lm_no, idx, :] = lm_out.GetPoint(0)
                self.lm_end[lm_no, idx, :] = lm_out.GetPoint(1)

                del tfilt
            del t
        del pd

    def visualise_one_landmark_lines(self, lm_no, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Writing to', dir_name)

        lm_name = dir_name + '/lm_lines_' + str(lm_no) + '.vtk'

        n_views = self.heatmap_maxima.shape[1]
        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        verts = vtk.vtkCellArray()
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfValues(2 * n_views)

        for idx in range(n_views):
            lines.InsertNextCell(2)
            pid = points.InsertNextPoint(self.lm_start[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
            pid = points.InsertNextPoint(self.lm_end[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            scalars.SetValue(idx * 2, self.heatmap_maxima[lm_no, idx, 2])
            scalars.SetValue(idx * 2 + 1, self.heatmap_maxima[lm_no, idx, 2])

        pd.SetPoints(points)
        del points
        pd.SetLines(lines)
        del lines
        pd.SetVerts(verts)
        del verts
        pd.GetPointData().SetScalars(scalars)
        del scalars

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(lm_name)
        writer.Write()

        del writer
        del pd

    def compute_intersection_between_lines(self, pa, pb):
        n_lines = pa.shape[0]
        si = pb - pa
        ni = np.divide(si, np.transpose(np.sqrt(np.sum(si ** 2, 1)) * np.ones((3, n_lines))))
        nx = ni[:, 0]
        ny = ni[:, 1]
        nz = ni[:, 2]
        sxx = np.sum(nx ** 2 - 1)
        syy = np.sum(ny ** 2 - 1)
        szz = np.sum(nz ** 2 - 1)
        sxy = np.sum(np.multiply(nx, ny))
        sxz = np.sum(np.multiply(nx, nz))
        syz = np.sum(np.multiply(ny, nz))
        s = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
        cx = np.sum(np.multiply(pa[:, 0], (nx ** 2 - 1)) + np.multiply(pa[:, 1], np.multiply(nx, ny)) +
                    np.multiply(pa[:, 2], np.multiply(nx, nz)))
        cy = np.sum(np.multiply(pa[:, 0], np.multiply(nx, ny)) + np.multiply(pa[:, 1], (ny ** 2 - 1)) +
                    np.multiply(pa[:, 2], np.multiply(ny, nz)))
        cz = np.sum(np.multiply(pa[:, 0], np.multiply(nx, nz)) + np.multiply(pa[:, 1], np.multiply(ny, nz)) +
                    np.multiply(pa[:, 2], (nz ** 2 - 1)))

        c = np.array([[cx], [cy], [cz]])
        p_intersect = np.matmul(np.linalg.pinv(s), c)
        return p_intersect[:, 0]

    def compute_intersection_between_lines_ransac(self, pa, pb):
        iterations = 100
        best_error = 100000000
        best_p = (0, 0, 0)
        dist_thres = 10 * 10
        n_lines = len(pa)
        d = n_lines / 3
        used_lines = -1

        for i in range(iterations):
            ran_lines = np.random.choice(range(n_lines), 3, replace=False)
            p_est = self.compute_intersection_between_lines(pa[ran_lines, :], pb[ran_lines, :])
            top = np.cross((np.transpose(p_est) - pa), (np.transpose(p_est) - pb))
            bottom = pb - pa
            distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1)) ** 2
            n_inliners = np.sum(distances < dist_thres)
            if n_inliners > d:
                idx = distances < dist_thres
                p_est = self.compute_intersection_between_lines(pa[idx, :], pb[idx, :])

                top = np.cross((np.transpose(p_est) - pa[idx, :]), (np.transpose(p_est) - pb[idx, :]))
                bottom = pb[idx, :] - pa[idx, :]
                distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1)) ** 2

                sum_squared = np.sum(distances) / n_inliners
                if sum_squared < best_error:
                    best_error = sum_squared
                    best_p = p_est
                    used_lines = n_inliners

        if used_lines == -1:
            self.logger.warning('Ransac failed - estimating from all lines')
            best_p = self.compute_intersection_between_lines(pa, pb)

        return best_p, best_error

    def filter_lines_based_on_heatmap_value_using_quantiles(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        q = self.config['process_3d']['heatmap_max_quantile']
        threshold = np.quantile(max_values, q)
        idx = max_values > threshold
        return pa[idx], pb[idx]

    def filter_lines_based_on_heatmap_value_using_absolute_value(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        threshold = self.config['process_3d']['heatmap_abs_threshold']
        idx = max_values > threshold
        return pa[idx], pb[idx]

    def compute_all_landmarks_from_view_lines(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        self.landmarks = np.zeros((n_landmarks, 3))

        sum_error = 0
        for lm_no in range(n_landmarks):
            pa = self.lm_start[lm_no, :, :]
            pb = self.lm_end[lm_no, :, :]
            if self.config['process_3d']['filter_view_lines'] == "abs_value":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_absolute_value(lm_no, pa, pb)
            elif self.config['process_3d']['filter_view_lines'] == "quantile":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_quantiles(lm_no, pa, pb)
            p_intersect = (0, 0, 0)
            if len(pa) < 3:
                print('Not enough valid view lines for landmark ', lm_no)
            else:
                p_intersect, best_error = self.compute_intersection_between_lines_ransac(pa, pb)
                sum_error = sum_error + best_error
            self.landmarks[lm_no, :] = p_intersect
        print("Ransac average error ", sum_error / n_landmarks)

    @staticmethod
    def multi_read_surface(file_name):
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()
        if file_extension == ".obj":
            obj_in = vtk.vtkOBJReader()
            obj_in.SetFileName(file_name)
            obj_in.Update()
            return obj_in.GetOutput()
        elif file_extension == ".wrl":
            vrmlin = vtk.vtkVRMLImporter()
            vrmlin.SetFileName(file_name)
            vrmlin.Update()
            return vrmlin.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
        elif file_extension == ".vtk":
            pd_in = vtk.vtkPolyDataReader()
            pd_in.SetFileName(file_name)
            pd_in.Update()
            return pd_in.GetOutput()
        elif file_extension == ".vtp":
            pd_in = vtk.vtkXMLPolyDataReader()
            pd_in.SetFileName(file_name)
            pd_in.Update()
            return pd_in.GetOutput()
        elif file_extension == ".stl":
            pd_in = vtk.vtkSTLReader()
            pd_in.SetFileName(file_name)
            pd_in.Update()
            return pd_in.GetOutput()
        elif file_extension == ".ply":
            pd_in = vtk.vtkPLYReader()
            pd_in.SetFileName(file_name)
            pd_in.Update()
            return pd_in.GetOutput()
        else:
            print("Can not read files with extension", file_extension)
            return None

    @staticmethod
    def multi_read_texture(file_name, texture_file_name=None):
        if texture_file_name is None:
            base = os.path.splitext(file_name)[0]
            for ext in (".bmp", ".png", ".jpg", ".jpeg"):
                candidate = base + ext
                if os.path.isfile(candidate):
                    texture_file_name = candidate
                    break

        if texture_file_name is not None:
            _, file_extension = os.path.splitext(texture_file_name)
            file_extension = file_extension.lower()
            if file_extension == ".bmp":
                texture_image = vtk.vtkBMPReader()
            elif file_extension == ".png":
                texture_image = vtk.vtkPNGReader()
            elif file_extension in (".jpg", ".jpeg"):
                texture_image = vtk.vtkJPEGReader()
            else:
                return None
            texture_image.SetFileName(texture_file_name)
            texture_image.Update()
            return texture_image.GetOutput()

        return None

    def apply_pre_transformation(self, pd):
        translation = [0, 0, 0]
        if self.config['pre-align']['align_center_of_mass']:
            vtk_cm = vtk.vtkCenterOfMass()
            vtk_cm.SetInputData(pd)
            vtk_cm.SetUseScalarsAsWeights(False)
            vtk_cm.Update()
            cm = vtk_cm.GetCenter()
            translation = [-cm[0], -cm[1], -cm[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = self.config['pre-align']['rot_x']
        ry = self.config['pre-align']['rot_y']
        rz = self.config['pre-align']['rot_z']
        s = self.config['pre-align']['scale']

        t.Scale(s, s, s)
        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        if self.config['pre-align']['write_pre_aligned']:
            name_out = str(self.config.temp_dir / 'pre_transform_mesh.vtk')
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(trans.GetOutput())
            writer.SetFileName(name_out)
            writer.Write()

        return trans.GetOutput(), t

    def transform_landmarks_to_original_space(self, landmarks, t):
        points = vtk.vtkPoints()
        pd = vtk.vtkPolyData()

        for lm in landmarks:
            points.InsertNextPoint(lm)
        pd.SetPoints(points)

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t.GetInverse())
        trans.Update()
        pd_trans = trans.GetOutput()

        n_landmarks = pd_trans.GetNumberOfPoints()
        new_landmarks = np.zeros((n_landmarks, 3))
        for lm_no in range(pd_trans.GetNumberOfPoints()):
            p = pd_trans.GetPoint(lm_no)
            new_landmarks[lm_no, :] = (p[0], p[1], p[2])
        return new_landmarks

    def project_landmarks_to_surface(self, mesh_name):
        pd = self.multi_read_surface(mesh_name)
        if pd is None:
            return

        pd, t = self.apply_pre_transformation(pd)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        clean.Update()

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(clean.GetOutput())
        locator.SetNumberOfCellsPerBucket(1)
        locator.BuildLocator()

        projected_landmarks = np.copy(self.landmarks)
        n_landmarks = self.landmarks.shape[0]

        for i in range(n_landmarks):
            p = self.landmarks[i, :]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.reference(0)
            tcp = np.zeros(3)
            locator.FindClosestPoint(p, tcp, cell_id, sub_id, dist2)
            projected_landmarks[i, :] = tcp

        self.landmarks = self.transform_landmarks_to_original_space(projected_landmarks, t)

        del pd, clean, locator

    def write_landmarks_as_vtk_points(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Writing to', dir_name)

        lm_name = dir_name + '/lms_as_points.vtk'
        n_landmarks = self.heatmap_maxima.shape[0]

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        for lm_no in range(n_landmarks):
            pid = points.InsertNextPoint(self.landmarks[lm_no, :])
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)

        pd.SetPoints(points)
        del points
        pd.SetVerts(verts)
        del verts

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(lm_name)
        writer.Write()

        del writer, pd

    @staticmethod
    def write_landmarks_as_vtk_points_external(landmarks, file_name):
        n_landmarks = landmarks.shape[0]

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        for lm_no in range(n_landmarks):
            pid = points.InsertNextPoint(landmarks[lm_no, :])
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)

        pd.SetPoints(points)
        del points
        pd.SetVerts(verts)
        del verts

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(file_name)
        writer.Write()

        del writer, pd

    @staticmethod
    def write_landmarks_as_text_external(landmarks, file_name):
        with open(file_name, 'w') as f:
            for lm in landmarks:
                f.write(str(lm[0]) + ' ' + str(lm[1]) + ' ' + str(lm[2]) + '\n')

    @staticmethod
    def get_mesh_files_in_dir(directory):
        names = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.obj', '.wrl', '.vtk', '.vtp', '.ply', '.stl')):
                    full_name = os.path.join(root, filename)
                    if os.path.isfile(full_name) and os.stat(full_name).st_size > 5:
                        names.append(full_name)
        return names


class Render3D:
    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('Render3D')

    def random_transform(self):
        min_x = self.config['process_3d']['min_x_angle']
        max_x = self.config['process_3d']['max_x_angle']
        min_y = self.config['process_3d']['min_y_angle']
        max_y = self.config['process_3d']['max_y_angle']
        min_z = self.config['process_3d']['min_z_angle']
        max_z = self.config['process_3d']['max_z_angle']

        rx = np.double(np.random.randint(min_x, max_x, 1))
        ry = np.double(np.random.randint(min_y, max_y, 1))
        rz = np.double(np.random.randint(min_z, max_z, 1))
        scale = np.double(np.random.uniform(1.4, 1.9, 1))
        tx = np.double(np.random.randint(-20, 20, 1))
        ty = np.double(np.random.randint(-20, 20, 1))
        return rx, ry, rz, scale, tx, ty

    def generate_3d_transformations(self):
        n_views = self.config['data_loader']['args']['n_views']
        transform_stack = np.zeros((n_views, 6), dtype=np.float32)
        for idx in range(n_views):
            rx, ry, rz, s, tx, ty = self.random_transform()
            transform_stack[idx, :] = (rx, ry, rz, s, tx, ty)
        return transform_stack

    def compute_pre_transformation(self, file_name):
        translation = [0, 0, 0]
        if self.config['pre-align']['align_center_of_mass']:
            pd = Utils3D.multi_read_surface(file_name)
            if pd.GetNumberOfPoints() < 1:
                print('Could not read', file_name)
                return None

            vtk_cm = vtk.vtkCenterOfMass()
            vtk_cm.SetInputData(pd)
            vtk_cm.SetUseScalarsAsWeights(False)
            vtk_cm.Update()
            cm = vtk_cm.GetCenter()
            translation = [-cm[0], -cm[1], -cm[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = self.config['pre-align']['rot_x']
        ry = self.config['pre-align']['rot_y']
        rz = self.config['pre-align']['rot_z']

        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()
        return t

    def render_3d_obj_rgb(self, transform_stack, file_name):
        write_image_files = self.config['process_3d']['write_renderings']
        off_screen_rendering = self.config['process_3d']['off_screen_rendering']
        n_views = self.config['data_loader']['args']['n_views']
        img_size = self.config['data_loader']['args']['image_size']
        win_size = img_size

        n_channels = 3
        image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)

        mtl_name = os.path.splitext(file_name)[0] + '.mtl'
        obj_dir = os.path.dirname(file_name)
        obj_in = vtk.vtkOBJImporter()
        obj_in.SetFileName(file_name)
        obj_in.SetFileNameMTL(mtl_name)
        obj_in.SetTexturePath(obj_dir)
        obj_in.Update()

        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.GetActiveCamera().SetPosition(0, 0, 1)
        ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        ren.GetActiveCamera().SetViewUp(0, 1, 0)
        ren.GetActiveCamera().SetParallelProjection(1)

        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(ren)
        ren_win.SetSize(win_size, win_size)
        ren_win.SetOffScreenRendering(off_screen_rendering)

        obj_in.SetRenderWindow(ren_win)
        obj_in.Update()

        props = vtk.vtkProperty()
        props.SetDiffuse(0)
        props.SetSpecular(0)
        props.SetAmbient(1)

        actors = ren.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            actor.SetProperty(props)
            actor = actors.GetNextItem()
        del props

        t_pre_trans = self.compute_pre_transformation(file_name)

        t = vtk.vtkTransform()
        t.Identity()
        t.Update()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(ren_win)
        writer_png = vtk.vtkPNGWriter()
        writer_png.SetInputConnection(w2if.GetOutputPort())

        start = time.time()
        for idx in range(n_views):
            name_rendering = self.config.temp_dir / ('rendering' + str(idx) + '_RGB.png')

            rx, ry, rz, s, tx, ty = transform_stack[idx]

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Concatenate(t_pre_trans)
            t.Update()

            xmin, xmax = -150, 150
            ymin, ymax = -150, 150
            xlen = xmax - xmin
            ylen = ymax - ymin

            cx, cy = 0, 0
            s = self.config['pre-align']['scale']
            extend_factor = 1.0 / s
            side_length = max([xlen, ylen]) * extend_factor

            ren.GetActiveCamera().SetParallelScale(side_length / 2)
            ren.GetActiveCamera().SetPosition(cx, cy, 500)
            ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
            ren.GetActiveCamera().SetViewUp(0, 1, 0)
            ren.GetActiveCamera().ApplyTransform(t.GetInverse())
            ren.ResetCameraClippingRange()

            ren_win.Render()

            if write_image_files:
                w2if.Modified()
                writer_png.SetFileName(str(name_rendering))
                writer_png.Write()
            else:
                w2if.Modified()
                w2if.Update()

            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)

            image_stack[idx, :, :, :] = a[:, :, :]

        end = time.time()
        print("Pure RGB rendering time: " + str(end - start))

        del obj_in, writer_png, w2if, ren, ren_win, t
        return image_stack

    def apply_pre_transformation(self, pd):
        translation = [0, 0, 0]
        if self.config['pre-align']['align_center_of_mass']:
            vtk_cm = vtk.vtkCenterOfMass()
            vtk_cm.SetInputData(pd)
            vtk_cm.SetUseScalarsAsWeights(False)
            vtk_cm.Update()
            cm = vtk_cm.GetCenter()
            translation = [-cm[0], -cm[1], -cm[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = self.config['pre-align']['rot_x']
        ry = self.config['pre-align']['rot_y']
        rz = self.config['pre-align']['rot_z']
        s = self.config['pre-align']['scale']

        t.Scale(s, s, s)
        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        if self.config['pre-align']['write_pre_aligned']:
            name_out = str(self.config.temp_dir / ('pre_transform_mesh.vtk'))
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(trans.GetOutput())
            writer.SetFileName(name_out)
            writer.Write()

        return trans.GetOutput()

    def render_3d_multi_rgb_geometry_depth(self, transform_stack, file_name):
        write_image_files = self.config['process_3d']['write_renderings']
        off_screen_rendering = self.config['process_3d']['off_screen_rendering']
        n_views = self.config['data_loader']['args']['n_views']
        img_size = self.config['data_loader']['args']['image_size']
        win_size = img_size
        slack = 5

        start = time.time()
        self.logger.debug('Rendering')

        n_channels = 5
        image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)

        pd = Utils3D.multi_read_surface(file_name)
        if pd.GetNumberOfPoints() < 1:
            print('Could not read', file_name)
            return None

        pd = self.apply_pre_transformation(pd)

        texture_img = Utils3D.multi_read_texture(file_name)
        if texture_img is not None:
            pd.GetPointData().SetScalars(None)
            texture = vtk.vtkTexture()
            texture.SetInterpolate(1)
            texture.SetQualityTo32Bit()
            texture.SetInputData(texture_img)

        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.GetActiveCamera().SetPosition(0, 0, 1)
        ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        ren.GetActiveCamera().SetViewUp(0, 1, 0)
        ren.GetActiveCamera().SetParallelProjection(1)

        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(ren)
        ren_win.SetSize(win_size, win_size)
        ren_win.SetOffScreenRendering(off_screen_rendering)

        t = vtk.vtkTransform()
        t.Identity()
        t.Update()

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(trans.GetOutput())

        actor_text = vtk.vtkActor()
        actor_text.SetMapper(mapper)
        if texture_img is not None:
            actor_text.SetTexture(texture)
            actor_text.GetProperty().SetColor(1, 1, 1)
            actor_text.GetProperty().SetAmbient(1.0)
            actor_text.GetProperty().SetSpecular(0)
            actor_text.GetProperty().SetDiffuse(0)
        ren.AddActor(actor_text)

        actor_geometry = vtk.vtkActor()
        actor_geometry.SetMapper(mapper)
        ren.AddActor(actor_geometry)

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(ren_win)
        writer_png = vtk.vtkPNGWriter()
        writer_png.SetInputConnection(w2if.GetOutputPort())

        scale = vtk.vtkImageShiftScale()
        scale.SetOutputScalarTypeToUnsignedChar()
        scale.SetInputConnection(w2if.GetOutputPort())
        scale.SetShift(0)
        scale.SetScale(-255)

        writer_png_2 = vtk.vtkPNGWriter()
        writer_png_2.SetInputConnection(scale.GetOutputPort())

        for view in range(n_views):
            name_rgb = str(self.config.temp_dir / ('rendering' + str(view) + '_RGB.png'))
            name_depth = str(self.config.temp_dir / ('rendering' + str(view) + '_zbuffer.png'))
            name_geometry = str(self.config.temp_dir / ('rendering' + str(view) + '_geometry.png'))

            rx, ry, rz, s, tx, ty = transform_stack[view]

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()
            trans.Update()

            xmin, xmax = -150, 150
            ymin, ymax = -150, 150
            zmin = trans.GetOutput().GetBounds()[4]
            zmax = trans.GetOutput().GetBounds()[5]
            xlen = xmax - xmin
            ylen = ymax - ymin

            cx, cy = 0, 0
            extend_factor = 1.0
            side_length = max([xlen, ylen]) * extend_factor

            ren.GetActiveCamera().SetParallelScale(side_length / 2)
            ren.GetActiveCamera().SetPosition(cx, cy, 500)
            ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
            ren.GetActiveCamera().SetClippingRange(500 - zmax - slack, 500 - zmin + slack)

            w2if.SetInputBufferTypeToRGB()

            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            mapper.Modified()
            ren.Modified()
            ren_win.Render()

            if write_image_files:
                w2if.Modified()
                writer_png.SetFileName(name_rgb)
                writer_png.Write()
            else:
                w2if.Modified()
                w2if.Update()

            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)
            image_stack[view, :, :, 0:3] = a[:, :, :]

            actor_text.SetVisibility(False)
            actor_geometry.SetVisibility(True)
            mapper.Modified()
            ren.Modified()
            ren_win.Render()

            if write_image_files:
                w2if.Modified()
                writer_png.SetFileName(name_geometry)
                writer_png.Write()
            else:
                w2if.Modified()
                w2if.Update()

            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)
            image_stack[view, :, :, 3:4] = a[:, :, 0:1]

            ren.Modified()
            ren_win.Render()
            w2if.SetInputBufferTypeToZBuffer()
            w2if.Modified()

            if write_image_files:
                w2if.Modified()
                writer_png_2.SetFileName(name_depth)
                writer_png_2.Write()
            else:
                w2if.Modified()
                w2if.Update()

            scale.Update()
            im = scale.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)
            image_stack[view, :, :, 4:5] = a[:, :, 0:1]

            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            ren.Modified()

        del writer_png_2, writer_png, ren_win, actor_geometry, actor_text, mapper, w2if, t, trans
        if texture_img is not None:
            del texture_img
            del texture
        end = time.time()
        self.logger.debug("File load and rendering time: " + str(end - start))

        return image_stack

    def render_3d_file(self, file_name):
        image_channels = self.config['data_loader']['args']['image_channels']
        file_type = (os.path.splitext(file_name)[1]).lower()

        image_stack = None
        transformation_stack = None
        n_views = self.config['data_loader']['args']['n_views']
        win_size = self.config['data_loader']['args']['image_size']

        if file_type == ".obj" and image_channels == "RGB":
            transformation_stack = self.generate_3d_transformations()
            image_stack = self.render_3d_obj_rgb(transformation_stack, file_name)
            image_stack = image_stack / 255
        elif file_type == ".obj" and image_channels == "RGB+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_rgb = self.render_3d_obj_rgb(transformation_stack, file_name)
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 4
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_rgb / 255
            image_stack[:, :, :, 3:4] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl"]) and image_channels == "RGB":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 3
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_full[:, :, :, 0:3] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "geometry":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 1
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 3:4] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 1
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl"]) and image_channels == "RGB+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 4
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_full[:, :, :, 0:3] / 255
            image_stack[:, :, :, 3:4] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "geometry+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
            n_channels = 2
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 3:4] / 255
            image_stack[:, :, :, 1:2] = image_stack_full[:, :, :, 4:5] / 255
        else:
            print("Can not render filetype ", file_type, " using image_channels ", image_channels)

        return image_stack, transformation_stack


# ===========================================================================
# Data preparation
# ===========================================================================

def create_lock_file(name):
    with open(name, "w") as f:
        f.write(socket.gethostname())


def delete_lock_file(name):
    if os.path.exists(name):
        os.remove(name)


def random_transform(config):
    """Angle-only transform used by the data-prep renderer."""
    min_x = config['process_3d']['min_x_angle']
    max_x = config['process_3d']['max_x_angle']
    min_y = config['process_3d']['min_y_angle']
    max_y = config['process_3d']['max_y_angle']
    min_z = config['process_3d']['min_z_angle']
    max_z = config['process_3d']['max_z_angle']

    rx = np.double(np.random.randint(min_x, max_x, 1))
    ry = np.double(np.random.randint(min_y, max_y, 1))
    rz = np.double(np.random.randint(min_z, max_z, 1))
    scale = np.double(np.random.uniform(1.4, 1.9, 1))
    tx = np.double(np.random.randint(-20, 20, 1))
    ty = np.double(np.random.randint(-20, 20, 1))
    return rx, ry, rz, scale, tx, ty


def process_file_for_dataset(config, file_name, output_dir):
    """
    Generic single-mesh -> (RGB / geometry / depth) + 2D-projected-landmark renderer.

    Required config keys under 'preparedata':
        raw_data_dir      : where meshes and landmark text files live
        processed_data_dir: where rendered images / 2D landmarks / split lists go
        mesh_subdir       : subdir under raw_data_dir containing meshes (optional)
        landmark_subdir   : subdir under raw_data_dir containing per-mesh
                            landmark text files (optional)
        texture_subdir    : subdir under raw_data_dir containing textures (optional)
        mesh_ext          : mesh extension, e.g. ".stl" (default)
        landmark_ext      : landmark file extension (default ".txt")

    Each landmark file must contain one "x y z" triple per line (3D coordinates).
    Each mesh must have a matching texture found via Utils3D.multi_read_texture.
    """
    raw_dir = Path(config['preparedata']['raw_data_dir'])
    mesh_subdir = config['preparedata'].get('mesh_subdir', '')
    lm_subdir = config['preparedata'].get('landmark_subdir', '')
    mesh_ext = config['preparedata'].get('mesh_ext', '.stl')
    lm_ext = config['preparedata'].get('landmark_ext', '.txt')

    base_name = os.path.basename(file_name)
    mesh_path = raw_dir / mesh_subdir / (file_name + mesh_ext)
    lm_path = raw_dir / lm_subdir / (file_name + lm_ext)

    name_path = os.path.dirname(file_name)
    o_dir_image = Path(output_dir) / 'images' / name_path
    o_dir_lm = Path(output_dir) / '2D LM' / name_path
    o_dir_image.mkdir(parents=True, exist_ok=True)
    o_dir_lm.mkdir(parents=True, exist_ok=True)

    lock_file = o_dir_image / (base_name + '.lock')

    for f in (mesh_path, lm_path):
        if not os.path.isfile(str(f)):
            print(f, ' could not read')
            return False
    if os.path.isfile(str(lock_file)):
        print(file_name, ' is locked - skipping')
        return True

    create_lock_file(str(lock_file))
    print('Rendering ', file_name)

    win_size = config['data_loader']['args']['image_size']
    off_screen_rendering = config['preparedata']['off_screen_rendering']
    n_views = config['data_loader']['args']['n_views']
    slack = 5

    # Load landmarks
    points = vtk.vtkPoints()
    lms = vtk.vtkPolyData()
    with open(str(lm_path)) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            x, y, z = np.double(parts[:3])
            points.InsertNextPoint(x, y, z)
    lms.SetPoints(points)
    del points

    stl_reader = Utils3D.multi_read_surface(str(mesh_path))
    if stl_reader is None or stl_reader.GetNumberOfPoints() < 1:
        print('Could not read mesh', mesh_path)
        delete_lock_file(str(lock_file))
        return False

    pd = stl_reader
    pd.GetPointData().SetScalars(None)

    texture_img = Utils3D.multi_read_texture(str(mesh_path))
    texture = None
    if texture_img is not None:
        texture = vtk.vtkTexture()
        texture.SetInterpolate(1)
        texture.SetQualityTo32Bit()
        texture.SetInputData(texture_img)

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.GetActiveCamera().SetPosition(0, 0, 1)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 1, 0)
    ren.GetActiveCamera().SetParallelProjection(1)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(win_size, win_size)
    ren_win.SetOffScreenRendering(off_screen_rendering)

    t = vtk.vtkTransform()
    t.Identity()
    t.Update()

    trans = vtk.vtkTransformPolyDataFilter()
    trans.SetInputData(pd)
    trans.SetTransform(t)
    trans.Update()

    trans_lm = vtk.vtkTransformPolyDataFilter()
    trans_lm.SetInputData(lms)
    trans_lm.SetTransform(t)
    trans_lm.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(trans.GetOutput())

    actor_text = vtk.vtkActor()
    actor_text.SetMapper(mapper)
    if texture is not None:
        actor_text.SetTexture(texture)
    actor_text.GetProperty().SetColor(1, 1, 1)
    actor_text.GetProperty().SetAmbient(1.0)
    actor_text.GetProperty().SetSpecular(0)
    actor_text.GetProperty().SetDiffuse(0)
    ren.AddActor(actor_text)

    actor_geometry = vtk.vtkActor()
    actor_geometry.SetMapper(mapper)
    ren.AddActor(actor_geometry)

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    writer_png = vtk.vtkPNGWriter()
    writer_png.SetInputConnection(w2if.GetOutputPort())

    scale = vtk.vtkImageShiftScale()
    scale.SetOutputScalarTypeToUnsignedChar()
    scale.SetInputConnection(w2if.GetOutputPort())
    scale.SetShift(0)
    scale.SetScale(-255)

    writer_png_2 = vtk.vtkPNGWriter()
    writer_png_2.SetInputConnection(scale.GetOutputPort())

    for view in range(n_views):
        name_rgb = str(o_dir_image / (base_name + '_' + str(view) + '.png'))
        name_geometry = str(o_dir_image / (base_name + '_' + str(view) + '_geometry.png'))
        name_depth = str(o_dir_image / (base_name + '_' + str(view) + '_zbuffer.png'))
        name_2dlm = str(o_dir_lm / (base_name + '_' + str(view) + '.txt'))

        if not os.path.isfile(name_rgb):
            rx, ry, rz, s, tx, ty = random_transform(config)

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()
            trans.Update()
            trans_lm.Update()

            xmin, xmax = -150, 150
            ymin, ymax = -150, 150
            zmin = trans.GetOutput().GetBounds()[4]
            zmax = trans.GetOutput().GetBounds()[5]
            xlen = xmax - xmin
            ylen = ymax - ymin

            cx, cy = 0, 0
            side_length = max([xlen, ylen])
            zoom_fac = win_size / side_length

            ren.GetActiveCamera().SetParallelScale(side_length / 2)
            ren.GetActiveCamera().SetPosition(cx, cy, 500)
            ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
            ren.GetActiveCamera().SetClippingRange(500 - zmax - slack, 500 - zmin + slack)

            w2if.SetInputBufferTypeToRGB()

            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            mapper.Modified()
            ren.Modified()
            ren_win.Render()

            w2if.Modified()
            writer_png.SetFileName(name_rgb)
            writer_png.Write()

            actor_text.SetVisibility(False)
            actor_geometry.SetVisibility(True)
            mapper.Modified()
            ren.Modified()
            ren_win.Render()

            w2if.Modified()
            writer_png.SetFileName(name_geometry)
            writer_png.Write()

            ren.Modified()
            ren_win.Render()
            w2if.SetInputBufferTypeToZBuffer()
            w2if.Modified()

            writer_png_2.SetFileName(name_depth)
            writer_png_2.Write()
            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            ren.Modified()

            with open(name_2dlm, 'w') as f:
                t_lm = trans_lm.GetOutput()
                for i in range(t_lm.GetNumberOfPoints()):
                    x_pos = t_lm.GetPoint(i)[0]
                    y_pos = t_lm.GetPoint(i)[1]
                    x_pos_screen = (x_pos - cx) * zoom_fac + win_size / 2
                    y_pos_screen = -(y_pos - cy) * zoom_fac + win_size / 2
                    f.write(f'{x_pos_screen} {y_pos_screen}\n')

    del writer_png_2, writer_png, ren_win, actor_geometry, actor_text, mapper, w2if, t, trans, stl_reader
    if texture_img is not None:
        del texture, texture_img
    del lms, trans_lm

    delete_lock_file(str(lock_file))
    return True


def split_data_into_train_and_test(base_file_names, output_dir, train_ratio=0.8, seed=0):
    """Generic train/test split by prefix (purely a deterministic shuffle + cut)."""
    train_file = os.path.join(output_dir, "dataset_train.txt")
    test_file = os.path.join(output_dir, "dataset_test.txt")

    rng = random.Random(seed)
    names = list(base_file_names)
    rng.shuffle(names)
    n_train = int(len(names) * train_ratio)

    train_names = names[:n_train]
    test_names = names[n_train:]

    with open(train_file, 'w') as f1:
        for n in train_names:
            f1.write(n + '\n')
    with open(test_file, 'w') as f2:
        for n in test_names:
            f2.write(n + '\n')

    return train_names


def prepare_dataset_data(config):
    print('Preparing dataset')
    file_id_list = config['preparedata']['file_id_list']
    output_dir = config['preparedata']['processed_data_dir']
    image_out_dir = os.path.join(output_dir, 'images')
    lm_out_dir = os.path.join(output_dir, '2D LM')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(lm_out_dir, exist_ok=True)

    base_file_names = []
    with open(file_id_list) as f:
        for line in f:
            line = line.strip().strip("\n")
            if len(line) > 0:
                base_file_names.append(line)
    print('Read ', len(base_file_names), ' file ids')

    train_ratio = config['preparedata'].get('train_ratio', 0.8)
    base_file_names = split_data_into_train_and_test(
        base_file_names, output_dir, train_ratio=train_ratio)
    print('Processing ', len(base_file_names), ' file ids for training')

    for base_name in base_file_names:
        print('Processing ', base_name)
        process_file_for_dataset(config, base_name, output_dir)


def run_prepare(config):
    prepare_dataset_data(config)


# ===========================================================================
# Device / model helpers
# ===========================================================================

def get_working_device(config):
    if config['n_gpu'] >= 1 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 3:
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_and_load_model(config):
    logger = config.get_logger('test')
    logger.debug('Initialising model')
    model = config.initialize('arch', globals())

    if config.resume is None:
        logger.error('Expecting model to be specified using the --r flag')
        return None, None

    device = get_working_device(config)
    checkpoint = torch.load(str(config.resume), map_location=device)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1 and device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.debug('Model was trained for %s epochs', checkpoint['epoch'])

    model = model.to(device)
    model.eval()
    return device, model


# ===========================================================================
# Training
# ===========================================================================

def run_train(config):
    print('Initialising data loader')
    data_loader = config.initialize('data_loader', globals())
    print('Initialising validation data')
    valid_data_loader = data_loader.split_validation()

    print('Initialising model')
    model = config.initialize('arch', globals())

    print('Initialising loss')
    loss = globals()[config['loss']]
    metrics = [globals()[met] for met in config['metrics']]

    print('Initialising optimizer')
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    print('Initialising scheduler')
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    print('Initialising trainer')
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    print('starting to train')
    trainer.train()


# ===========================================================================
# Testing
# ===========================================================================

def read_3d_landmarks(file_name):
    lms = []
    with open(file_name) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            x, y, z = np.double(parts[:3])
            lms.append((x, y, z))
    return lms


def write_landmark_accuracy(gt_lm, pred_lm, file):
    if len(gt_lm) != len(pred_lm):
        print('Number of gt landmarks ', len(gt_lm), ' does not match number of predicted lm ', len(pred_lm))
        return None

    sum_dist = 0
    for idx in range(len(gt_lm)):
        dst = distance.euclidean(gt_lm[idx], pred_lm[idx])
        sum_dist += dst
        file.write(str(dst))
        if idx != len(gt_lm) - 1:
            file.write(', ')
    file.write('\n')
    print('Average landmark error ', sum_dist / len(gt_lm))


def save_predicted_landmarks(pred_lms, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for lm in pred_lms:
            f.write(f"{lm[0]} {lm[1]} {lm[2]}\n")


def get_landmark_bounds(lms):
    xs = [lm[0] for lm in lms]
    ys = [lm[1] for lm in lms]
    zs = [lm[2] for lm in lms]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def get_landmarks_bounding_box_diagonal_length(lms):
    x_min, x_max, y_min, y_max, z_min, z_max = get_landmark_bounds(lms)
    return math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2)


def visualise_landmarks_as_spheres_with_accuracy(gt_lm, pred_lm, file_out):
    diag_len = get_landmarks_bounding_box_diagonal_length(gt_lm)
    sphere_size = diag_len * 0.010

    append = vtk.vtkAppendPolyData()
    for idx in range(len(gt_lm)):
        gt_p = gt_lm[idx]
        pr_p = pred_lm[idx]
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pr_p)
        sphere.SetRadius(sphere_size)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.Update()
        scalars.SetNumberOfValues(sphere.GetOutput().GetNumberOfPoints())

        dst = distance.euclidean(gt_p, pr_p)
        for s in range(sphere.GetOutput().GetNumberOfPoints()):
            scalars.SetValue(s, dst)

        sphere.GetOutput().GetPointData().SetScalars(scalars)
        append.AddInputData(sphere.GetOutput())
        del sphere, scalars

    append.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(append.GetOutput())
    writer.SetFileName(file_out)
    writer.Write()


def predict_one_subject(config, file_name):
    device, model = get_device_and_load_model(config)
    render_3d = Render3D(config)
    image_stack, transform_stack = render_3d.render_3d_file(file_name)

    predict_2d = Predict2D(config, model, device)
    heatmap_maxima = predict_2d.predict_heatmaps_from_images(image_stack)

    u3d = Utils3D(config)
    u3d.heatmap_maxima = heatmap_maxima
    u3d.transformations_3d = transform_stack
    u3d.compute_lines_from_heatmap_maxima()
    u3d.compute_all_landmarks_from_view_lines()
    u3d.project_landmarks_to_surface(file_name)
    return u3d.landmarks


def run_test(config):
    test_set_file = os.path.join(config['data_loader']['args']['data_dir'], 'dataset_test.txt')
    result_file = config.temp_dir / 'results.csv'

    device, model = get_device_and_load_model(config)

    files = []
    with open(test_set_file) as f:
        for line in f:
            clean_name = os.path.splitext(line.strip().strip("\n"))[0]
            if len(clean_name) > 0:
                files.append(clean_name)
    print('Read', len(files), 'files to run test on')

    raw_dir = Path(config['preparedata']['raw_data_dir'])
    mesh_subdir = config['preparedata'].get('mesh_subdir', '')
    lm_subdir = config['preparedata'].get('landmark_subdir', '')
    mesh_ext = config['preparedata'].get('mesh_ext', '.stl')
    lm_ext = config['preparedata'].get('landmark_ext', '.txt')

    idx = 0
    res_f = open(result_file, "w")
    start_time = time.time()
    for f_name in files:
        mesh_name = str(raw_dir / mesh_subdir / (f_name + mesh_ext))
        lm_name = str(raw_dir / lm_subdir / (f_name + lm_ext))

        gt_lms = read_3d_landmarks(lm_name)
        if not os.path.isfile(mesh_name):
            print('File', mesh_name, ' does not exist')
            continue

        print('Computing file ', idx, ' of ', len(files))
        render_3d = Render3D(config)
        image_stack, transform_stack = render_3d.render_3d_file(mesh_name)

        predict_2d = Predict2D(config, model, device)
        heatmap_maxima = predict_2d.predict_heatmaps_from_images(image_stack)

        print('Computing 3D landmarks')
        u3d = Utils3D(config)
        u3d.heatmap_maxima = heatmap_maxima
        u3d.transformations_3d = transform_stack
        u3d.compute_lines_from_heatmap_maxima()
        u3d.compute_all_landmarks_from_view_lines()
        u3d.project_landmarks_to_surface(mesh_name)
        pred_lms = u3d.landmarks

        predicted_landmarks_file = config.temp_dir / (f_name + '_predicted_landmarks.txt')
        save_predicted_landmarks(pred_lms, str(predicted_landmarks_file))

        res_f.write(f_name + ', ')
        write_landmark_accuracy(gt_lms, pred_lms, res_f)
        res_f.flush()

        base_name = os.path.basename(f_name)
        sphere_file = config.temp_dir / (base_name + '_landmarkAccuracy.vtk')
        visualise_landmarks_as_spheres_with_accuracy(gt_lms, pred_lms, str(sphere_file))

        idx += 1
        time_per_test = (time.time() - start_time) / idx
        time_left = (len(files) - idx) * time_per_test
        print('Time left in test: ', str(datetime.timedelta(seconds=time_left)))

    res_f.close()


# ===========================================================================
# Debug / utility stages
# ===========================================================================

def get_cuda_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if torch.cuda.is_available():
        print('Selected cuda device: ', torch.cuda.current_device())
        print('Number of GPUs available: ', torch.cuda.device_count())
        print('Cuda device name: ', torch.cuda.get_device_name(0))
        print('Cuda capabilities: ', torch.cuda.get_device_capability(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        print('Max allocated:   ', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')


def show_batch(sample_batched, config):
    images_batch, heat_map_batch = sample_batched['image'], sample_batched['heat_map_stack']
    heat_map_batch = heat_map_batch.numpy()
    im_size = images_batch.size(2)
    hm_size = heat_map_batch.shape[2]
    channels = config['data_loader']['args']['image_channels']

    i = np.zeros((im_size, im_size, 3))
    if channels in ("geometry", "depth"):
        i[:, :, 0] = images_batch[0][:, :, 0]
        i[:, :, 1] = images_batch[0][:, :, 0]
        i[:, :, 2] = images_batch[0][:, :, 0]
    elif channels == "RGB":
        i[:, :, :] = images_batch[0][:, :, :]

    hm = np.zeros((hm_size, hm_size, 3))
    n_lm = heat_map_batch.shape[4]
    for lm in range(n_lm):
        r, g, b = random.random(), random.random(), random.random()
        v_len = math.sqrt(r * r + g * g + b * b)
        r, g, b = r / v_len, g / v_len, b / v_len
        hm[:, :, 0] += heat_map_batch[0, 0, :, :, lm] * r
        hm[:, :, 1] += heat_map_batch[0, 0, :, :, lm] * g
        hm[:, :, 2] += heat_map_batch[0, 0, :, :, lm] * b

    plt.figure()
    plt.imshow(i)
    plt.figure()
    plt.imshow(hm)
    plt.axis('off')
    plt.ioff()
    plt.show()


def test_dataloader(config):
    data_loader = config.initialize('data_loader', globals())
    for batch_idx, sample_batched in enumerate(data_loader):
        print('Batch id: ', batch_idx)
        show_batch(sample_batched, config)
        break


def test_model_glm(config):
    logger = config.get_logger('train')
    model = config.initialize('arch', globals())
    logger.info(model)


# ===========================================================================
# Entry point
# ===========================================================================

def build_arg_parser():
    args = argparse.ArgumentParser(description='Unified pipeline')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--stage', default='all', type=str,
                      choices=['prepare', 'train', 'test', 'all',
                               'show_batch', 'show_model', 'cuda_info'],
                      help='which pipeline stage to run')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    args.custom_options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    return args


def _read_stage():
    for i, a in enumerate(sys.argv):
        if a == '--stage' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 'all'


def main():
    args = build_arg_parser()
    options = args.custom_options
    config = ConfigParser(args, options)

    stage = _read_stage()
    if stage == 'prepare':
        run_prepare(config)
    elif stage == 'train':
        run_train(config)
    elif stage == 'test':
        run_test(config)
    elif stage == 'all':
        run_prepare(config)
        run_train(config)
        run_test(config)
    elif stage == 'show_batch':
        test_dataloader(config)
    elif stage == 'show_model':
        test_model_glm(config)
    elif stage == 'cuda_info':
        get_cuda_info()


if __name__ == '__main__':
    main()