from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torchaudio
import torch

class LBRS(Dataset):

    def __init__(self, root='./', phase='train', 
                 index_path=None, index=None, k=5, base_sess=None, data_type='audio', args=None):
        self.root = os.path.expanduser(root)
        self.root = root
        self.data_type = data_type
        # self.make_extractor()
        self.phase = phase
        # self.train = train  # training set or test set
        self.all_train_df = pd.read_csv(os.path.join(root, "librispeech_fscil_train.csv"))
        self.all_val_df = pd.read_csv(os.path.join(root, "librispeech_fscil_val.csv"))
        self.all_test_df = pd.read_csv(os.path.join(root, "librispeech_fscil_test.csv"))
        index = np.arange(index)
        if phase == 'train':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        elif phase == 'val':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_val_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=k)
        elif phase =='test':
            self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        


    def SelectfromClasses(self, df, index, per_num=None):
        data_tmp = []
        targets_tmp = []

        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            # random.shuffle(ind_cl)

            k = 0
            # ind_cl is the index list whose label equals i, 
            # start_idx make sure 
            # there is no intersection between train and test set  
            for j in ind_cl:
                filename = os.path.join(df['filename'][j])
                path = os.path.join(self.root+"100spks_segments/", filename)
                data_tmp.append(path)
                targets_tmp.append(df['label'][j])
                k += 1
                if per_num is not None:
                    if k >= per_num:
                        break
              
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        audio,sr = torchaudio.load(path)
        return audio.squeeze(0), targets

class Openlbrs(Dataset):
    def __init__(self, args, index,root,partition='test',fix_seed=True):
        super(Openlbrs, self).__init__()
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_episodes = args.test_times if partition == 'test' else args.n_train_runs
        self.index = index
        self.root=root
        self.partition = partition
        self.args = args
        self.all_train_df = pd.read_csv(os.path.join(root, "librispeech_fscil_train.csv"))
        self.all_test_df = pd.read_csv(os.path.join(root, "librispeech_fscil_test.csv"))

        if self.partition=='train':
           self.datapath, self.labels = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        else:
            self.datapath, self.labels = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        self.data={}
        for idx in range(len(self.datapath)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.datapath[idx])

    def SelectfromClasses(self, df, index, per_num=None):
        data_tmp = []
        targets_tmp = []
        
        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            k=0
            for j in ind_cl:
                filename = os.path.join(df['filename'][j])
                path = os.path.join(self.root+"100spks_segments/", filename)
                data_tmp.append(path)
                targets_tmp.append(df['label'][j])
                k += 1
                if per_num is not None:
                    if k >= per_num:
                        break
        return data_tmp, targets_tmp
    
    def __getitem__(self, item):
            return self.get_episode(item)

    
    def get_episode(self, item):
        
        if self.fix_seed:
            np.random.seed(item+1)
        cls_sampled = np.random.choice(self.index, self.n_ways, False)
        support_xs = []
        support_ys = []
        suppopen_xs = []
        suppopen_ys = []
        query_xs = []
        query_ys = []
        openset_xs = []
        openset_ys = []
        
        # Close set preparation
        for idx, the_cls in enumerate(cls_sampled):
            path= self.data[the_cls]
            audio=[]
            for i in range(len(path)):
                audio_single,_ = torchaudio.load(path[i])
                audio.append(audio_single.squeeze())
            support_xs_ids_sampled = np.random.choice(range(len(audio)), self.n_shots, False)
            support_xs.extend([audio[the_id].view(1,-1) for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(audio)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend(audio[the_id].view(1,-1) for the_id in query_xs_ids)
            query_ys.extend([idx] * self.n_queries)
        support_xs = torch.cat(support_xs, dim=0)
        query_xs = torch.cat(query_xs, dim=0)
        # Open set preparation

        cls_open_ids = np.setxor1d(self.index, cls_sampled)
        cls_open_ids = np.random.choice(cls_open_ids, self.n_open_ways, False)
        for idx, the_cls in enumerate(cls_open_ids):
            path= self.data[the_cls]
            audio=[]
            for i in range(len(path)):
                audio_single,_ = torchaudio.load(path[i])
                audio.append(audio_single.squeeze())
            suppopen_xs_ids_sampled = np.random.choice(range(len(audio)), self.n_shots, False)
            suppopen_xs.extend(audio[the_id].view(1,-1) for the_id in suppopen_xs_ids_sampled)
            suppopen_ys.extend([idx] * self.n_shots)
            openset_xs_ids_sampled = np.random.choice(range(len(audio)), self.n_queries, False)
            openset_xs.extend(audio[the_id].view(1,-1) for the_id in openset_xs_ids_sampled)
            openset_ys.extend([the_cls] * self.n_queries)
        suppopen_xs = torch.cat(suppopen_xs, dim=0)
        openset_xs = torch.cat(openset_xs, dim=0)
        
        support_ys,query_ys,openset_ys = np.array(support_ys),np.array(query_ys),np.array(openset_ys)
        suppopen_ys = np.array(suppopen_ys)
        cls_sampled, cls_open_ids = np.array(cls_sampled), np.array(cls_open_ids)
        base_ids = np.setxor1d(self.index, np.concatenate([cls_sampled,cls_open_ids]))
        base_ids = np.array(sorted(base_ids))

        return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys,base_ids,cls_open_ids
      
    def __len__(self):
        return self.n_episodes

