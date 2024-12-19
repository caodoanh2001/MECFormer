import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import json

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--model_path', default = None, type=str)
    parser.add_argument('--tau', default = 5.0, type=float)
    parser.add_argument('--beta', default = 1.0, type=float)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)
    dm.setup(); vocab_size = dm.vocab_size
    print("Vocab size:", vocab_size)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'vocab_size': vocab_size,
                            'vocab_path': cfg.Data.vocab_path,
                            'max_seq_len': cfg.Data.max_seq_len,
                            'vocab': json.load(open(cfg.Data.vocab_path, 'r')),
                            'bos_tag': cfg.Data.bos_tag,
                            'eos_tag': cfg.Data.eos_tag,
                            'padding_idx': cfg.Data.padding_idx,
                            }
    
    model = ModelInterface(**ModelInterface_dict)
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    tau = args.tau
    beta = args.beta

    #---->train or test
    if cfg.General.server == 'train':
        model.model.update_fc_layers(tau=tau, beta=beta)
        if args.model_path is not None:
            ckpt = torch.load(args.model_path)['state_dict']
            ckpt = {'.'.join(k.split('.')[1:]):ckpt[k] for k in ckpt.keys()}
            model.model.load_state_dict(ckpt)

            for param in model.model.parameters():
                param.requires_grad_(False)

            for param in model.model._fc2.parameters():
                param.requires_grad_(True)

        trainer.fit(model = model, datamodule = dm)
    else:
        print("Number of parameters:", count_parameters(model.model))
        model.model.update_fc_layers(tau=tau, beta=beta)
        ckpt = torch.load(args.model_path)['state_dict']
        ckpt = {'.'.join(k.split('.')[1:]):ckpt[k] for k in ckpt.keys() if 'cum_samples' not in k}
        model.model.load_state_dict(ckpt)
        trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    #---->main
    main(cfg)