import torch
from datasets.librispeech import Openlbrs
import numpy as np
from .nsynth import Opennds
from .FMC import Openfmc

def meta_train_dataloader(args):
    class_index = np.arange(args.train_classes)
    if args.dataset == 'librispeech':
        trainset = Openlbrs(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    elif  'Nsynth' in args.dataset:
        trainset = Opennds(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    elif args.dataset == 'FMC':
        trainset =  Openfmc(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8)

    return loader



def meta_test_dataloader(args):
    if args.dataset == 'librispeech':
        class_new = np.arange(args.train_classes,100)
        testset = Openlbrs(root=args.dataroot,index=class_new,args=args,partition='test', fix_seed=True)
    elif args.dataset == 'Nsynth':
        testset = Opennds(root=args.dataroot,index=np.arange(args.train_classes,100),args=args,partition='test', fix_seed=True)
    elif args.dataset == 'FMC':
        testset = Openfmc(root=args.dataroot,index=np.arange(args.train_classes,89),args=args,partition='test', fix_seed=True)


    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    return loader