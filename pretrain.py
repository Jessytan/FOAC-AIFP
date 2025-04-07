import os
from functools import partial
from trainers import Pretrainer, C2_Net_train
from models.Network import My_Net
from datasets import dataloaders
from utils.util import *
import torch
import yaml
import argparse
from datasets.nsynth import NDS
from datasets.FMC import FSDCLIPS
from datasets.librispeech import LBRS
import torch.nn.functional as F


def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

args = Pretrainer.train_parser()
with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
cfg = cfg['train']
cfg.update(vars(args))
args = dict2namespace(cfg)
if args.dataset == 'librispeech':
    train_set = LBRS(root=args.dataroot, phase="train",index=args.train_classes, base_sess=True, args=args)
    save_model_path = os.path.join(args.save_folder, f'pretrain_model_librispeech.pth')
elif args.dataset == 'Nsynth':
    train_set = NDS(root=args.dataroot,index=args.train_classes,args=args,phase='train')
    test_set = NDS(root=args.dataroot,index=args.train_classes,args=args,phase='test')
    save_model_path = os.path.join(args.save_folder, f'pretrain_model_nysnth.pth')
elif args.dataset == 'FMC':
    train_set = FSDCLIPS(root=args.dataroot,index=args.train_classes,phase='train')
    save_model_path = os.path.join(args.save_folder, f'pretrain_model_fmc.pth')    

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                        num_workers=8, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=100,num_workers=8, pin_memory=True)
eval_loader = dataloaders.meta_test_dataloader(args)
dataset_trainer = train_loader
tm = Pretrainer.BaseTrainer(args,dataset_trainer)
tm.train(eval_loader)



