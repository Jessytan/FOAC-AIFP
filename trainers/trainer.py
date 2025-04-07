import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
sys.path.append('..')
from utils.util import *
import torch.nn.functional as F
from datasets.librispeech import LBRS
from scipy.stats import t
from sklearn import metrics
import scipy
from sklearn.metrics import f1_score,roc_curve, roc_auc_score, average_precision_score
from datasets.nsynth import NDS
from datasets.FMC import FSDCLIPS
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import simps
from sklearn.manifold import TSNE


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam','sgd'], default='sgd')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=0.001)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=20)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-4)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=1)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", default = True)
    parser.add_argument('--restype', type=str, default='ResNet12', help='Network Structure')
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int,default=64)
    parser.add_argument('--decay_epoch', help='epochs that cut lr', default='10')
    parser.add_argument('--lr_decay_epochs', type=str, default='15,55', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep", action="store_true")
    parser.add_argument("--train_way", help="training way", type=int,default=5)
    parser.add_argument("--test_way", help="test way", type=int, default=5)
    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int)
    parser.add_argument("--test_shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int, default=15)

    parser.add_argument('--dataset', type=str, default='Nsynth',
                        choices=['FMC', 'Nsynth',  'librispeech'])
    parser.add_argument('--config', type=str, default="./default.yml") 
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--test', default=True, type=bool)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument('--dataroot', type=str, default="/data/datasets/The_NSynth_Dataset/")#librispeech_fscil The_NSynth_Dataset #FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data  
    parser.add_argument('--data_root', type=str, default='/data/datasets', help='path to data root')
    parser.add_argument("--finetune", help="whether use resnet12 as backbone or not", default = False)
    parser.add_argument('--open_weight_sum_cali', type=float, default=0.5)


    args = parser.parse_args()

    return args


def get_opt(model, args):

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    elif args.opt == 'sgd':
        other_params = [param for name, param in model.named_parameters() if not name.startswith('encoder')]
        optim_param = [{'params': other_params},
                     {'params': model.encoder.parameters(), 'lr': 0.0002}]

        optimizer = optim.SGD(optim_param,lr=args.lr,momentum=0.9,weight_decay=args.weight_decay,nesterov=args.nesterov)
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
            args.lr_decay_epochs.append(int(it))
    if args.lr_decay_epochs is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_decay_epochs, gamma=args.gamma)

    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 30], gamma=args.gamma)

    

    return optimizer, scheduler


class Path_Manager:
    def __init__(self, fewshot_path, args):

        self.train = os.path.join(fewshot_path,'train')

        if args.pre:
            self.test = os.path.join(fewshot_path,'test_pre')
            self.val = os.path.join(fewshot_path,'val_pre') if not args.no_val else self.test

        else:
            self.test = os.path.join(fewshot_path,'test')
            self.val = os.path.join(fewshot_path,'val') if not args.no_val else self.test


class Train_Manager:
    def __init__(self, args, train_func):

        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        name = args.dataset

        check_dir(args.save_folder)
        self.logger = get_logger(os.path.join(args.save_folder, '%s.log' % (name)))
        self.save_path = os.path.join(args.save_folder, 'model_%s' % (name))
        self.writer = SummaryWriter(os.path.join(args.save_folder, 'log_%s' % (name)))


        self.args = args
        self.train_func = train_func
        

    def train(self, model,eval_loader):
        args = self.args
        train_func = self.train_func
        writer = self.writer
        save_path = self.save_path
        logger = self.logger

        optimizer, scheduler = get_opt(model, args)

        val_shot = args.n_shots
        test_way = args.test_way

        best_val_acc = 0
        best_epoch = 0
        best_auroc = 0
        best_fscore = 0

        model.train()
        total_epoch = args.epoch
        logger.info("start training!")
        val_loss = 0.0
        iter_counter = 0
        for e in tqdm(range(total_epoch)):
            iter_counter, train_acc, train_auroc, train_loss = train_func(model=model,
                                                     optimizer=optimizer,
                                                     writer=writer,
                                                     iter_counter=iter_counter,
                                                     args = args)
            # adjust_learning_rate(e, self.args, optimizer)
            if (e+1) % args.val_epoch == 0:

                    logger.info("")
                    logger.info("epoch %d/%d, iter %d:" % (e+1,total_epoch,iter_counter))
                    logger.info("train_loss: %.5f" % (train_loss))
                    logger.info("train_acc: %.3f" % (train_acc))
                    logger.info("train_auroc: %.3f" % (train_auroc))

                    model.eval()
                    with torch.no_grad():
                        result,loss = self.run_test_fsl(model,eval_loader) 

                        writer.add_scalar('val_%d-way-%d-shot_acc' % (test_way, val_shot), result[0][0], iter_counter)
                        writer.add_scalar('val_%d-way-%d-shot_auroc' % (test_way, val_shot), result[1][0], iter_counter)
                        writer.add_scalar('val_%d-way-%d-shot_fscore' % (test_way, val_shot), result[2][0], iter_counter)

                    logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f' % (test_way, val_shot, result[0][0], result[0][1]))
                    logger.info('val_%d-way-%d-shot_auroc: %.3f\t%.3f' % (test_way, val_shot, result[1][0], result[1][1]))
                    logger.info('val_%d-way-%d-shot_fscore: %.3f\t%.3f' % (test_way, val_shot, result[2][0], result[2][1]))

                    if result[0][0] > best_val_acc:
                        best_val_acc = result[0][0]
                        best_epoch = e+1
                        if not args.no_val:
                            torch.save(model.state_dict(), save_path+'_max_acc.pth')
                        logger.info('BEST ACC!')
                    if result[1][0]>best_auroc:
                        best_auroc = result[1][0]
                        best_epoch = e+1
                        torch.save(model.state_dict(), save_path+'_max_auroc.pth')
                        logger.info('BEST Auroc!')
                    if result[2][0]>best_fscore:
                        best_fscore = result[2][0]
                        best_epoch = e+1
                        torch.save(model.state_dict(), save_path+'_max_fscore.pth')
                        logger.info('BEST Fscore!')
                    model.train()

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {e+1}/{total_epoch}, Current Learning Rate: {current_lr:.6f}")

        logger.info('training finished!')
        if args.no_val:
            torch.save(model.state_dict(), save_path)

        logger.info('------------------------')
        logger.info(('the best epoch is %d/%d') % (best_epoch,total_epoch))
        logger.info(('the best %d-way %d-shot val acc is %.3f') % (test_way,val_shot,best_val_acc))

    def meta_test(self,net, metaloader):
        net = net.eval()
        acc = []
        with torch.no_grad():
            with tqdm(metaloader, total=len(metaloader), leave=False) as pbar:
                for idx, data in enumerate(pbar):
                    # Data Preparation
                    support_data, support_label, query_data, query_label,_,_,_,_ = data
                    support_data = support_data.cuda()
                    query_data = query_data.cuda()
                    # Data Reorganization
                    _, _, height, width, channel = support_data.size()
                    support_data = support_data.view(-1, height, width, channel)
                    query_data = query_data.view(-1, height, width, channel)
                    support_label = support_label.view(-1).numpy()
                    query_label = query_label.view(-1).numpy()
                    
                    # Feature Extracdtion
                    support_features = net(support_data).view(support_data.size(0), -1)
                    query_features = net(query_data).view(query_data.size(0), -1)
                    support_features = F.normalize(support_features,p=2,dim=-1).detach().cpu().numpy()
                    query_features = F.normalize(query_features,p=2,dim=-1).detach().cpu().numpy()                
                    query_pred = Proto(support_features, support_label, query_features, query_label)
                    acc.append(metrics.accuracy_score(query_label, query_pred))
                    pbar.set_postfix({"Few-Shot MetaEval Acc":'{0:.2f}'.format(acc[-1])})
        return mean_confidence_interval(acc)    

    def run_test_fsl(self,net, openloader):
        net = net.eval()
        
        with tqdm(openloader, total=len(openloader), leave=False) as pbar:
            acc_trace = []
            auroc_trace = []
            fscore_trace = []
            loss_trace = []
            for idx, data in enumerate(pbar):

                labels, probs,loss = self.compute_feats(self.args,net, data)
                acc, auroc,fscore = self.eval_fsl_nplus1(labels, probs)
 
                loss = loss.cpu().numpy()
                acc_trace.append(acc)
                auroc_trace.append(auroc)
                fscore_trace.append(fscore)
                loss_trace.append(loss)

                pbar.set_postfix({
                        "OpenSet MetaEval Acc":'{0:.2f}'.format(acc),
                        "AUROC-auroc MetaEval:":'{0:.2f}'.format(auroc),
                        "AUROC-fscore MetaEval:":'{0:.2f}'.format(fscore)
                    })

            acc = mean_confidence_interval(acc_trace)
            auroc = mean_confidence_interval(auroc_trace)
            fscore=mean_confidence_interval(fscore_trace)

            loss = mean_confidence_interval(loss_trace)
            result=[acc,auroc,fscore]
        return result,loss[0]/100.0

    def compute_feats(self,args,net, data):
        with torch.no_grad():
            # Data Preparation

            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label,supp_idx, open_idx= data
            support_data,support_label              = support_data.float().cuda(),support_label.cuda().long()
            query_data,query_label                  = query_data.float().cuda(),query_label.cuda().long()
            suppopen_data,suppopen_label            = suppopen_data.float().cuda(),suppopen_label.cuda().long()
            openset_data,openset_label              = openset_data.float().cuda(),openset_label.cuda().long()
            supp_idx, open_idx = supp_idx.cuda().long(), open_idx.cuda().long()
            
            openset_label = args.n_ways * torch.ones_like(openset_label)
            support_data, query_data, suppopen_data, openset_data = support_data.squeeze(), query_data.squeeze(), suppopen_data.squeeze(), openset_data.squeeze()
            the_label = tuple(x.squeeze() for x in (support_label, query_label, suppopen_label, openset_label))
            the_img     = (support_data, query_data, suppopen_data, openset_data)
            # Tensor Input Preparation
            cosine_probs,loss_cls,loss_fake= net(the_img,the_label,supp_idx, open_idx,True)
            loss = loss_cls+loss_fake


            # Numpy Input Preparation
            
            supplabel_numpy = support_label.squeeze().cpu().numpy()
            querylabel_numpy = query_label.squeeze().cpu().numpy()
            
            open_label = np.concatenate((np.ones(query_label.size(1)),np.zeros(openset_label.size(1))))

            # Numpy Probs Preparation
            query_cls_probs, openset_cls_probs = cosine_probs
            query_cls_probs = query_cls_probs.cpu().numpy()
            openset_cls_probs = openset_cls_probs.cpu().numpy()
            cosine_probs = (query_cls_probs, openset_cls_probs)
                    
        return (supplabel_numpy, querylabel_numpy, open_label), cosine_probs,loss

    def eval_fsl_nplus1(self,labels, probs):
        supp_label, query_label, open_label = labels
        num_query = query_label.shape[0]
        supp_label = supp_label.view()
        all_probs = np.concatenate(probs, axis=0)

        known_scores = np.max(all_probs[:num_query,:-1], axis=-1)
        unknown_scores = np.max(all_probs[num_query:,:-1], axis=-1)
        auroc_result,_,_,fscore= calc_auroc(known_scores,unknown_scores)

        # assert all_probs.shape[-1] == 6
        num_query = query_label.shape[0]
        query_pred = np.argmax(all_probs[:num_query,:-1], axis=-1)
        acc = metrics.accuracy_score(query_label, query_pred)
        
        return acc, auroc_result,fscore

    def Pretrain(self,model):
        if self.args.dataset == 'librispeech':
            train_set = LBRS(root=self.args.dataroot, phase="train",index=self.args.train_classes, base_sess=True, args=self.args)
            save_model_path = os.path.join(self.args.save_folder, f'pretrain_model_librispeech.pth')
        elif self.args.dataset == 'Nsynth':
            train_set = NDS(root=self.args.dataroot,index=self.args.train_classes,args=self.args,phase='train')
            save_model_path = os.path.join(self.args.save_folder, f'pretrain_model_nysnth.pth')
        elif self.args.dataset == 'FMC':
            train_set = FSDCLIPS(root=self.args.dataroot,index=self.args.train_classes,phase='train')
            save_model_path = os.path.join(self.args.save_folder, f'pretrain_model_fmc.pth')    

        trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True)
        optimizer, scheduler = get_opt(model, self.args)
        for epoch in range(self.args.epochs_pre):
            model.train()
            self.Pre_train_epoch(epoch,model,trainloader,scheduler,optimizer) 
        
        torch.save(dict(params=model.state_dict()), save_model_path)
    def save_model(self, epoch,model, name=None):
        state = {
            'epoch': epoch,
            'params': model.state_dict()
        }     
        file_name = '{}.pth'.format('epoch_'+str(epoch) if name is None else name)
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.args.save_folder, file_name))

    def Pre_train_epoch(self,epoch,model,trainloader,scheduler,optimizer):
        tl = Averager()
        ta = Averager()
        total_loss = AverageMeter()
        acc_avg = AverageMeter()
        model = model.train()
        tqdm_gen = tqdm(trainloader)
        
        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.to('cuda') for _ in batch]

            logits= model(data)
            
            # loss = nn.BCEWithLogitsLoss(logits, train_label.repeat(4))+0.7*nn.BCEWithLogitsLoss(rot_logits, rot_labe.long().cuda())
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            total_loss.update(loss.item())
            acc_avg.update(acc)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                    'Pre train, epo {}, lrc={:.4f},total loss={:.2f} acc={:.2f}'.format(epoch, lrc, total_loss.avg, acc_avg.avg))
            # tl.add(loss)
            # ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # tl = tl.item()
        # ta = ta.item()
        # return tl, ta




def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h



def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    y_pred = np.where(y_score >= np.sort(y_score)[75], 1, 0) #不是1way要改成75
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    auc_pr = average_precision_score(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    f_score = f1_score(y_true, y_pred, average="macro")

    return auc_score, fpr95, auc_pr, f_score


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


def calc_fscore(known, unknown):
    known=np.array(known)
    unknown=np.array(unknown)
    y_true= np.append(np.where(known<5, 0, 1), np.where(unknown>=5, 1, 0))
    y_pred = np.append(np.zeros(len(known)),np.ones(len(unknown)) )
    f_score = f1_score(y_true, y_pred, average="binary")
    return f_score