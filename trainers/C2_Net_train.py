import torch
from torch.nn import NLLLoss
from tqdm import tqdm
from sklearn import metrics
import numpy as np

def default_train(train_loader, model, optimizer, writer, iter_counter, args):

    avg_loss = 0
    avg_acc = 0
    avg_auroc=0
    with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
        for i, data in enumerate(pbar):
            iter_counter += 1

            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label,supp_idx, open_idx= data
            support_data,support_label              = support_data.float(),support_label.long()
            query_data,query_label                  = query_data.float(),query_label.long()
            suppopen_data,suppopen_label            = suppopen_data.float(),suppopen_label.long()
            openset_data,openset_label              = openset_data.float(),openset_label.long()
            supp_idx, open_idx = supp_idx.long(), open_idx.long()
            #tuple(x.squeeze() for x in (support_data, query_data, suppopen_data, openset_data))
            support_data, query_data, suppopen_data, openset_data = support_data.squeeze(), query_data.squeeze(), suppopen_data.squeeze(), openset_data.squeeze()
            the_label = tuple(x.squeeze() for x in (support_label, query_label, suppopen_label, openset_label))
            the_img = [support_data, query_data,  suppopen_data,openset_data]
  
            probs,loss= model(the_img,the_label,supp_idx, open_idx)
            query_cls_probs, openset_cls_probs = probs
            (loss_cls, loss_funit) = loss
            loss_total = loss_cls+loss_funit
           
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            loss_value = loss_total.item()

            ### Closed Set Accuracy
            close_pred = np.argmax(probs[0][:,:args.n_ways].view(-1,args.n_ways).detach().cpu().numpy(),-1)
            close_label = query_label.view(-1).cpu().numpy()
            acc = metrics.accuracy_score(close_label, close_pred)

            ### Open Set AUROC
            open_label_binary = np.concatenate((np.ones(close_pred.shape),np.zeros(openset_cls_probs.shape[0])))
            query_cls_probs = query_cls_probs.view(-1, args.n_ways+1)
            openset_cls_probs = openset_cls_probs.view(-1,args.n_ways+1)
            open_scores = torch.cat([query_cls_probs,openset_cls_probs], dim=0).detach().cpu().numpy()[:,:5]
            open_scores = np.max(open_scores,axis=-1)
            auroc = metrics.roc_auc_score(open_label_binary,open_scores)
 
            avg_acc += acc
            avg_loss += loss_value
            avg_auroc += auroc

            pbar.set_postfix({"Acc":'{0:.2f}'.format(avg_acc / (i + 1)), 
                                "Auroc":'{0:.2f}'.format(avg_auroc / (i + 1)), 
                                "loss" :'{0:.2f}'.format(avg_loss / (i + 1))
                                })
            torch.cuda.empty_cache()

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)
    avg_auroc = avg_auroc / (i + 1)

    writer.add_scalar('C2_Net_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)
    writer.add_scalar('train_auroc', avg_auroc, iter_counter)

    return iter_counter, avg_acc, avg_auroc, avg_loss
