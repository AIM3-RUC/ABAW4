'''
以video为单位加载数据，适用于时序模型
'''
import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import sys
from data.base_dataset import BaseDataset#
from utils.bins import get_center_and_bounds

class SeqDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--cv_no', type=str, default='6', help='the cross-validation number"')
        parser.add_argument('--supplyment_label', default=False, action='store_true', help='whether to supplement label using the label of the cloest frame')
        return parser

    def __init__(self, opt, set_name):
        ''' Sequential Dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst, trn_eval]
        '''
        super().__init__(opt)
        self.root = 'data_root' # fix me
        self.label_name_list = ["valence", "arousal", "expression",
                 "AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"
                 ]
        self.cv_no = opt.cv_no

        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.norm_method = opt.norm_method
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.set_name = set_name
        self.supplyment_label = opt.supplyment_label
        
        bin_centers, bin_bounds = get_center_and_bounds(opt.cls_weighted)
        self.bin_centers = dict([(key, np.array(value)) for key, value in bin_centers.items()])
        self.bin_bounds = dict([(key, np.array(value)) for key, value in bin_bounds.items()])
        
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"Aff-Wild2 Sequential dataset {set_name} created with total length: {len(self)}")

    def normalize_on_trn(self, feature_name, features):
        '''
        features的shape：[seg_len, ft_dim]
        mean_f与std_f的shape：[ft_dim,]，已经经过了去0处理
        '''
        mean_std_file = h5py.File(os.path.join(self.root, 'features', 'mean_std_on_trn', feature_name + '.h5'), 'r')
        mean_trn = np.array(mean_std_file['train']['mean'])
        std_trn = np.array(mean_std_file['train']['std'])
        features = (features - mean_trn) / std_trn
        return features

    def normalize_on_batch(self, features):
        '''
        输入张量的shape：[bs, seq_len, ft_dim]
        mean_f与std_f的shape：[bs, 1, ft_dim]
        '''
        mean_f = torch.mean(features, dim=1).unsqueeze(1).float()
        std_f = torch.std(features, dim=1).unsqueeze(1).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def load_label(self):
        '''
        video_dict['label']为list，包含为各图片对应的label dict
        video_dict['length']代表该视频对应图片的张数
        '''
        set_name = 'trn' if self.set_name == 'trn' else self.set_name
        if self.supplyment_label and set_name == 'trn':
            label_path = os.path.join(self.root, 'targets/{}_target_supplyement.h5'.format(set_name))
        elif set_name == 'trn_val':
            label_path = os.path.join(self.root, 'targets/trn_val_target.h5')
        elif set_name == 'tst':
            label_path = os.path.join(self.root, 'targets/partition.h5')
        else:
            label_path = os.path.join(self.root, 'targets/resplit/{}_split{}_target.h5'.format(set_name, self.cv_no))
            
        if set_name == 'tst':
            label_h5f = h5py.File(label_path, 'r')['tst']
        else:
            label_h5f = h5py.File(label_path, 'r')
        self.video_id_list = list(label_h5f.keys())

        self.target_list = []
        for video_id in tqdm(self.video_id_list, desc='loading {} label'.format(set_name)):
            video_dict = {}
            video_dict['label'] = []
            if self.set_name == 'tst':
                video_dict['length'] = len(set(label_h5f[video_id][()]))
            else:
                video_dict['label'] = torch.from_numpy(label_h5f[video_id]['label'][()])
                video_dict['length'] = video_dict['label'].shape[0]
            self.target_list.append(video_dict)

    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            self.feature_data[feature_name] = []
            feature_path = os.path.join(self.root, 'features/{}.h5'.format(feature_name))
            feature_h5f = h5py.File(feature_path, 'r')
            for idx, video_id in enumerate(tqdm(self.video_id_list, desc='loading {} feature'.format(feature_name))):
                video_dict = {}
                video_dict['feature'] = np.array(feature_h5f[video_id]['feature'][()]) #shape:(seg_len, ft_dim)
                
                assert len(video_dict['feature']) == int(self.target_list[idx]['length']), '\
                    Data Error: In feature {}, video_id: {}, frame does not match label frame'.format(feature_name, video_id)
                # normalize on trn:
                if (self.norm_method=='trn') and (feature_name in self.norm_features):
                    video_dict['feature'] = self.normalize_on_trn(feature_name, video_dict['feature'])
                self.feature_data[feature_name].append(video_dict)

    def __getitem__(self, index):
        target_data = self.target_list[index]
        feature_list = []
        feature_dims = []
        for feature_name in self.feature_set:
            data = torch.from_numpy(self.feature_data[feature_name][index]['feature']).float()
            feature_list.append(data)
            feature_dims.append(self.feature_data[feature_name][index]['feature'].shape[1])
        feature_dims = torch.from_numpy(np.array(feature_dims)).long()
        
        return {**{"feature_list": feature_list, "feature_dims": feature_dims, "video_id": self.video_id_list[index]},
                **target_data, **{"feature_names": self.feature_set}}

    
    def __len__(self):
        return len(self.video_id_list)
    

    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        '''
        1 2 4 7
        1 2 2 4 4 4 7
        
        '''
        feature_num = len(batch[0]['feature_list'])
        feature = []
        for i in range(feature_num):
            feature_name = self.feature_set[i]
            pad_ft = pad_sequence([sample['feature_list'][i] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            pad_ft = pad_ft.float()
            # normalize on batch:
            if (self.norm_method=='batch') and (feature_name in self.norm_features):
                pad_ft = self.normalize_on_batch(pad_ft)
            feature.append(pad_ft)
        feature = torch.cat(feature, dim=2) # pad_ft: (bs, seq_len, ft_dim)，将各特征拼接起来

        length = torch.tensor([sample['length'] for sample in batch])
        video_id = [sample['video_id'] for sample in batch]

        if self.set_name != 'tst':
            valence = pad_sequence([sample['label'][:,0] for sample in batch], padding_value=torch.tensor(-5), batch_first=True)
            arousal = pad_sequence([sample['label'][:,1] for sample in batch], padding_value=torch.tensor(-5), batch_first=True)
            expression = pad_sequence([sample['label'][:,2] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU1 = pad_sequence([sample['label'][:,3] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU2 = pad_sequence([sample['label'][:,4] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU4 = pad_sequence([sample['label'][:,5] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU6 = pad_sequence([sample['label'][:,6] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU7 = pad_sequence([sample['label'][:,7] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU10 = pad_sequence([sample['label'][:,8] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU12 = pad_sequence([sample['label'][:,9] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU15 = pad_sequence([sample['label'][:,10] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU23 = pad_sequence([sample['label'][:,11] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU24 = pad_sequence([sample['label'][:,12] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU25 = pad_sequence([sample['label'][:,13] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            AU26 = pad_sequence([sample['label'][:,14] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        
        feature_dims = batch[0]['feature_dims']
        feature_names = batch[0]['feature_names']
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0
        
        return {
            'feature': feature.float(), 
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'expression': expression.long(),
            'AU1': AU1.long(),
            'AU2': AU2.long(),
            'AU4': AU4.long(),
            'AU6': AU6.long(),
            'AU7': AU7.long(),
            'AU10': AU10.long(),
            'AU12': AU12.long(),
            'AU15': AU15.long(),
            'AU23': AU23.long(),
            'AU24': AU24.long(),
            'AU25': AU25.long(),
            'AU26': AU26.long(),
            'mask': mask.float(),
            'length': length,
            'feature_dims': feature_dims,
            'feature_names': feature_names,
            'video_id': video_id
        } if self.set_name != 'tst' else {
            'feature': feature.float(), 
            'mask': mask.float(),
            'length': length,
            'feature_dims': feature_dims,
            'feature_names': feature_names,
            'video_id': video_id
        }