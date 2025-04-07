import os
from functools import partial
from trainers import trainer, C2_Net_train
from models.Network import My_Net
from datasets import dataloaders
from utils.util import *
from models.Network import Backbone
import torch
import yaml
import argparse



def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

args = trainer.train_parser()
with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
cfg = cfg['train']
cfg.update(vars(args))
args = dict2namespace(cfg)

train_loader = dataloaders.meta_train_dataloader(args)
eval_loader = dataloaders.meta_test_dataloader(args)
train_func = partial(C2_Net_train.default_train, train_loader=train_loader)
tm = trainer.Train_Manager(args, train_func=train_func)

if not args.pretrain:
    model = My_Net(args=args,mode='train')
    model = model.to('cuda')
    
    if args.test:
        model.eval()
        state_dict = torch.load(os.path.join(args.save_folder, 'model_Nsynth_max_auroc.pth'))
        model.weight_base = state_dict['weight_base'].to('cuda')
        model.weight_base_open=state_dict['weight_base_open'].to('cuda')
        model.load_state_dict(state_dict,strict=False)
        result = tm.run_test_fsl(model,eval_loader)
        print(result)
        exit()
    state_dict = torch.load(args.pretrained_model_path)['feature_params']
    full_params = torch.load(args.pretrained_model_path)

    model.load_state_dict(state_dict,strict=False)
    model.init_representation(full_params)
    tm.train(model,eval_loader)
else:
    model = Backbone(args)#My_Net(args=args,mode='pretrain')
    model.to('cuda')
    tm.Pretrain(model)

