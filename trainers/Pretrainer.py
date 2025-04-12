import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import importlib
import torch.backends.cudnn as cudnn
import time
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from models.Network import Backbone
from utils.util import adjust_learning_rate,  AverageMeter
import scipy
from scipy.stats import t


def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam','sgd'], default='sgd')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=0.001)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=20)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-4)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=1)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", default = True)
    parser.add_argument('--restype', type=str, default='ResNet12', help='Network Structure')
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int,default=128)
    parser.add_argument('--decay_epoch', help='epochs that cut lr', default='10')
    parser.add_argument('--lr_decay_epochs', type=str, default='15,55', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')

    parser.add_argument("--train_way", help="training way", type=int,default=5)
    parser.add_argument("--test_way", help="test way", type=int, default=5)
    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int)
    parser.add_argument("--test_shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int, default=15)
    parser.add_argument("--val_trial", help="number of meta-testing episodes during validation", type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='Nsynth',
                        choices=['FMC', 'Nsynth',  'librispeech',
                        'f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n'])
    parser.add_argument('-config', type=str, default="/data/jessy/MyNew (1)/default.yml") 
    
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument('--dataroot', type=str, default="/data/datasets/The_NSynth_Dataset/")#librispeech_fscil The_NSynth_Dataset
    parser.add_argument('--data_root', type=str, default='/data/datasets', help='path to data root')
    parser.add_argument("--finetune", help="whether use resnet12 as backbone or not", default = False)

    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
    parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
    parser.add_argument('--temp', type=float, default=0.1, help="temp")
    parser.add_argument('--num-centers', type=int, default=1)
    parser.add_argument('--open_weight_sum_cali', type=float, default=0.5)


    args = parser.parse_args()

    return args

class BaseTrainer(object):
    def __init__(self, args, dataset_trainer):
        args.logroot = os.path.join(args.save_folder, 'pre_%s.log' % (args.dataset))
        
        
        # set the path according to the environment
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))
        
        
        self.save_path = os.path.join(args.save_folder, 'pret_model_%s' % (args.dataset))

        self.args = args
        self.train_loader = dataset_trainer

        # model & optimizer
        self.model = Backbone(args)
        state_dict = torch.load(args.pretrained_model_path1)['params']
        # model_dict =  self.model.state_dict()
        # feat_params = {k: v for k, v in state_dict.items() if 'encoder' in k}
        # model_dict.update(feat_params)
        self.model.load_state_dict(state_dict,strict=False)

              
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        self.criterion = {'logit':nn.BCEWithLogitsLoss() } #

        ################################################
        options = vars(args)
        Loss = importlib.import_module('loss.RPLoss')
        self.RPLcriterion = getattr(Loss, 'RPLoss')(**options).cuda()
    
        Loss2 = importlib.import_module('loss.GCPLoss')
        self.GCPLcriterion = getattr(Loss2, 'GCPLoss')(**options).cuda()


        params_list = [
                {'params': self.RPLcriterion.parameters()},
                {'params': self.GCPLcriterion.parameters()}]


        self.RPL_GCPL_optimizer = torch.optim.Adam(params_list, lr=args.learning_rate)
        ################################################

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = {name:loss.cuda() for name,loss in self.criterion.items()}
            cudnn.benchmark = True
    #####################################################################
    def RPL_train_epoch(self, epoch, train_loader, model, criterion_cls, criterion_open, optimizer, args,optimizer2=None):
        self.model.eval()
        
        losses = AverageMeter()
        rpl_accs = AverageMeter()

        torch.cuda.empty_cache()
        loss_all = 0
       

        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, (image,target) in enumerate(pbar):
                batch_idx = idx
                data, labels = image.cuda(), target.cuda()

                with torch.set_grad_enabled(True):
                    
                    x, y = model(data)
                    
                    logits, loss = criterion_cls(x, y, labels)
                    ##################
                    logits2, loss2 = criterion_open(x, y, labels,epoch=epoch)
                    loss += loss2
                    ##################
                    #loss = criterion_cls['logit'](y, labels)
                
                    
                    optimizer.zero_grad()
                    #optimizer2.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #optimizer2.step()

                
                losses.update(loss.item(), labels.size(0))

                rpl_pred = logits2.argmax(dim=1)
                rpl_acc = torch.eq(rpl_pred,target.cuda()).sum().float().item() / len(rpl_pred)
                rpl_accs.update(rpl_acc, labels.size(0))

                pbar.set_postfix({"Epoch {} Loss".format(epoch) :'{0:.2f}'.format(losses.avg),"rpl acc":'{0:.2f}'.format(rpl_accs.avg)})
                
                #loss_all += losses.avg
        message = 'Epoch {} Train_Loss {:.3f}'.format(epoch, losses.avg)
        return message 
    
    def train(self, eval_loader=None):

        trlog = {'args':vars(self.args), 'max_1shot_meta':0.0, 'max_5shot_meta':0.0, 'max_1shot_epoch':0, 'max_5shot_epoch':0}        
        

        # routine: supervised pre-training
        for epoch in range(1, self.args.epoch + 1):

            adjust_learning_rate(epoch, self.args, self.optimizer)
            #train_loss, train_msg = self.train_epoch(epoch, self.train_loader, self.model, self.criterion, self.optimizer, self.args)
            train_msg = self.RPL_train_epoch(epoch, self.train_loader, self.model,self.GCPLcriterion,self.RPLcriterion, self.RPL_GCPL_optimizer, self.args, self.optimizer)

            #evaluate
            if eval_loader is not None and (epoch % 5 == 0 or epoch >= 55):
                start = time.time()
                
                meta_5shot_acc, meta_5shot_std = self.meta_test(self.model, eval_loader)
                test_time = time.time() - start
                
                meta_msg = 'Meta Test Acc:  5-shot {:.4f}, Meta Test std:  {:.4f}, Time: {:.1f}'.format( meta_5shot_acc,meta_5shot_std, test_time)
                train_msg = train_msg + ' | ' + meta_msg
                if trlog['max_5shot_meta'] < meta_5shot_acc:
                    trlog['max_5shot_meta'] = meta_5shot_acc
                    trlog['max_5shot_epoch'] = epoch
                    self.save_3_model(epoch,'pre2'+self.args.dataset) # will not use
            
            print(train_msg)
            if epoch % 5 == 0 or epoch==self.args.epoch:
                self.save_3_model(epoch,'mini_last')
                print('The Best Meta 1(5)-shot Acc {:.4f}({:.4f}) in Epoch {}({})'.format(trlog['max_1shot_meta'],trlog['max_5shot_meta'],trlog['max_1shot_epoch'],trlog['max_5shot_epoch']))
            if epoch % 15 == 0 :
                self.save_3_model(epoch,'mini_15epoch')
            if epoch % 40 == 0 :
                self.save_3_model(epoch,'mini_40epoch')
            if epoch % 50 == 0 :
                self.save_3_model(epoch,'mini_50epoch')

                
            
    
    def save_model(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'params': self.model.state_dict()
        }     
        file_name = '{}.pth'.format('epoch_'+str(epoch) if name is None else name)
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.save_path, file_name))

    def meta_test(self,net, metaloader):
        net = net.eval()
        acc = []
        with torch.no_grad():
            with tqdm(metaloader, total=len(metaloader), leave=False) as pbar:
                for idx, data in enumerate(pbar):
                    # Data Preparation
                    support_data, support_label, query_data, query_label,_,_,_,_,_,_ = data
                    support_data = support_data.cuda().squeeze()
                    query_data = query_data.cuda().squeeze()
                    # Data Reorganization
                    
                    support_label = support_label.view(-1).numpy()
                    query_label = query_label.view(-1).numpy()
                    
                    # Feature Extracdtion
                    support_features,_ = net(support_data)
                    query_features,_ = net(query_data)
                    support_features = F.normalize(support_features,p=2,dim=-1).detach().cpu().numpy()
                    query_features = F.normalize(query_features,p=2,dim=-1).detach().cpu().numpy()                
                    query_pred = Proto(support_features, support_label, query_features, query_label)
                    acc.append(metrics.accuracy_score(query_label, query_pred))
                    pbar.set_postfix({"Few-Shot MetaEval Acc":'{0:.2f}'.format(acc[-1])})
        return mean_confidence_interval(acc) 

    def save_3_model(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'feature_params': self.model.state_dict(),
            'RPL_params': self.RPLcriterion.Dist.state_dict(),
            'GCPL_params': self.GCPLcriterion.Dist.state_dict()
        }     
        file_name = '{}.pth'.format('epoch_'+str(epoch) if name is None else name)
        print('==> Saving', file_name)
        torch.save(state, os.path.join('/data/jessy/MyNew (1)/fixed', file_name))  

def Proto(support, support_ys, query, query_label):
    proto_ys = sorted(np.unique(support_ys).tolist())
    proto = []
    for cls_id in proto_ys:
        the_feat = support[support_ys==cls_id].mean(axis=0)
        proto.append(the_feat)
    proto = np.stack(proto)

    proto_norm = np.linalg.norm(proto, axis=1, keepdims=True)
    proto = proto / proto_norm
    cosine_distance = query @ proto.transpose()

    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [proto_ys[idx] for idx in max_idx]
    return pred

def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h