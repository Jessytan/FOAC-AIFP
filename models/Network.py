import torch
import torch.nn as nn
from .module.prototype_dynamic_aggregation import PrototypeDynamicAggregation
from .module.conditional_information_coupling import ConditionalInformationCouplingModule
import math
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
import torch.nn.functional as F
from torch.nn import NLLLoss
from .module.transformer import OpenSetGenerater,TransformerEncoder
from .module.resnet18_encoder import resnet18


class Backbone(nn.Module):
    def __init__(self,args):
        super(Backbone,self).__init__()
        self.args = args
        self.encoder = resnet18(True,args)
        self.fc = nn.Linear(512,self.args.train_classes, bias=True)
        self.set_module_for_audio()
        
    def forward(self, x):
        x = self.spectrogram_extractor(x)   
        x = self.logmel_extractor(x)    
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        resfeat,_ = self.encoder(x)
        cls_logit = self.fc(resfeat)
        return resfeat, cls_logit
    
    def set_module_for_audio(self):
            center = True
            pad_mode = 'reflect'
            ref = 1.0
            amin = 1e-10
            top_db = None
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
                win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
                n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)
            self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

            # speechbrain tools 
            self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                                win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                                hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                                n_fft=self.args.extractor.window_size)
            self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)

class My_Net(nn.Module):
    def __init__(self, args=None,mode=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.metric  = Metric_Cosine()
        self.num_channel = 512
        self.dim = 512 * 52 
        self.encoder = resnet18(True,args)
        self.PAM = PrototypeDynamicAggregation(self.num_channel)
        self.CIAM = ConditionalInformationCouplingModule(512,512,1)
        self.NPM = OpenSetGenerater(self.num_channel, n_head=1,agg='mlp') 

        self.set_module_for_audio()
        self.fc = nn.Linear(self.num_channel,self.args.train_classes, bias=True)
        # self.criterion = NLLLoss()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self,input,labels=None,supp_ids=None,open_ids=None,test=False):
         if self.mode == 'pretrain':
              input,_ = self.encode(input)
              input = self.fc(input)
              return input
         elif self.mode == 'train':
        
            support_data, query_data, suppopen_data, openset_data = input
            (support_label,query_label,supopen_label,openset_label) = labels

            support_feat,s1= self.encode(support_data.cuda())
            q1,query_feat = self.encode(query_data.cuda())
            supopen_feat,so1= self.encode(suppopen_data.cuda())
            openset_feat,q2 = self.encode(openset_data.cuda())
            #Task 1
            open_label = self.args.n_ways * torch.ones_like(openset_label)
            cls_label = torch.cat([query_label, open_label])

            if test:
                 loss_cls,loss_fake,prediction = self.task(s1,support_feat,query_feat,q1,openset_feat,support_label.cuda(),cls_label.cuda(),query_label.cuda())
                 return prediction,loss_cls,loss_fake
            loss_cls,loss_fake,prediction = self.task(s1,support_feat,query_feat,q1,openset_feat,support_label.cuda(),cls_label.cuda(),query_label.cuda(),supp_ids.cuda())
            #Task 2要改变label
            binary_labels = torch.cat([torch.full((15,), i) for i in range(5)])
            loss_cls_aug,loss_aug_fake,_= self.task(so1,supopen_feat,q2,openset_feat,q1,support_label.cuda(),cls_label.cuda(),binary_labels.cuda(),supp_ids.cuda())


            return prediction,(loss_cls+loss_cls_aug,loss_fake+loss_aug_fake)

    def task(self,s1,support_feat,query_feat,q1,openset_feat,support_label,cls_label,query_label,supp_ids=None):
        aug_supp = self.CIAM(s1,query_feat)
        supp_protos= self.PAM(aug_supp,support_label)
       
        base_weights,base_open_weights = self.get_representation(supp_ids)

        recip_units, fake_center = self.NPM(supp_protos,base_weights,base_open_weights)
        cls_protos = torch.cat([supp_protos.unsqueeze(0), fake_center], dim=1)

        query_score = self.metric(cls_protos,q1.unsqueeze(0)).squeeze()
        open_score = self.metric(cls_protos,openset_feat.unsqueeze(0)).squeeze()
    
        cls_score =torch.cat([query_score.squeeze(),open_score.squeeze()],dim=0)
        loss_cls =F.cross_entropy(cls_score, cls_label)
        
        # funit_distance = self.metric(recip_units.transpose(0,1),q1.unsqueeze(0)).squeeze()
        # qopen_funit_distance = self.metric(recip_units.transpose(0,1), openset_feat.unsqueeze(0)).squeeze()
        # funit_distance = torch.cat([funit_distance,qopen_funit_distance],dim=0)
       
        loss_fake = 0.0#fakeunit_compare(funit_distance,self.args.n_ways,cls_label)

        query_score = F.softmax(query_score.detach(), dim=-1).squeeze()
        open_score = F.softmax(open_score.detach(), dim=-1).squeeze()
        
        return loss_cls,loss_fake,(query_score,open_score)      

    def encode(self, x):
        x = self.spectrogram_extractor(x)   
        x = self.logmel_extractor(x)    
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x,x1 = self.encoder(x)
        # if self.mode == 'pretrain':
        #     x = F.adaptive_avg_pool2d(x, 1)
        #     x = x.squeeze(-1).squeeze(-1)
            # x = self.PAM(x)
        return x,x1
        
    def init_representation(self, params):
        params_RPL = params['RPL_params']
        params_GCPL = params['GCPL_params']
        base_open = params_RPL['centers']#.view(60,4,512).mean(1)
        self.weight_base_open = nn.Parameter( base_open * self.args.open_weight_sum_cali, requires_grad=True)
        base = params_GCPL['centers'].view(60,512)                                                                                                                                                                                                                                            
        self.weight_base = nn.Parameter(base * self.args.open_weight_sum_cali , requires_grad=True)
    
    def get_representation(self, base_ids=None):
        if base_ids is not None:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
            base_open_weights = self.weight_base_open[base_ids,:] 
        else:
            base_weights = self.weight_base.unsqueeze(0)
            base_open_weights = self.weight_base_open[base_ids,:]

        return base_weights,base_open_weights
       

    def set_module_for_audio(self):
            center = True
            pad_mode = 'reflect'
            ref = 1.0
            amin = 1e-10
            top_db = None
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
                win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True)

            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
                n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
                freeze_parameters=True)

            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                freq_drop_width=8, freq_stripes_num=2)
            self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

            # speechbrain tools 
            self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                                win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                                hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                                n_fft=self.args.extractor.window_size)
            self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)


class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10.0):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1,2))
        # logits = torch.cosine_similarity(query_feature, supp_center, dim=-1)
        return logits * self.temp
    


def fakeunit_compare(funit_distance,n_ways,cls_label):
    # cls_label_binary = F.one_hot(cls_label).float()
    cls_label_binary = F.one_hot(cls_label.unsqueeze(0))[:,:,:-1].float().squeeze()
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance, target=cls_label_binary))
    return loss   




    