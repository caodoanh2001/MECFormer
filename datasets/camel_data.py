import random
import torch
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader
import os
import json
import numpy as np

class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.cluster_dir = self.dataset_cfg.cluster_dir
        self.bos_tag = self.dataset_cfg.bos_tag
        self.eos_tag = self.dataset_cfg.eos_tag
        self.padding_idx = self.dataset_cfg.padding_idx
        self.name_tasks = self.dataset_cfg.name_tasks
        self.classification = self.dataset_cfg.classification
        
        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        
        self.state = state

        self.data = list()
        self.label = list()
        self.tasks = list()
        self.data_name = list() # track data name for seperate evaluation

        self.mapping_dict = self.dataset_cfg.mapping_dict
        self.max_seq_len = self.dataset_cfg.max_seq_len
        self.gpt_pretrained = self.dataset_cfg.gpt_pretrained

        dataset_stats = dict()

        if self.classification:
            all_classes = list()
            for key in self.mapping_dict.keys():
                all_classes += self.mapping_dict[key]

            cls2idx = {all_classes[i]:i for i in range(len(all_classes))}

        for i_data, data_name in enumerate(self.dataset_cfg.data_name):
            feature_dir = self.dataset_cfg.data_dir[i_data]
            term_dict = self.mapping_dict[data_name]
            dataset_stats[data_name] = dict()
            # CAMELYON dataset
            if data_name in ['camel', 'esca']:
                csv_dir = self.dataset_cfg.label_dir[i_data] + f'fold{self.fold}.csv'
                slide_data = pd.read_csv(csv_dir, index_col=0)
                
                #---->split dataset
                if state == 'train':
                    data = [feature_dir + '/' + filename for filename in slide_data.loc[:, 'train'].dropna()]
                    label = [term_dict[int(l)] for l in slide_data.loc[:, 'train_label'].dropna()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]
                    
                if state == 'val':
                    data = [feature_dir + '/' + filename for filename in slide_data.loc[:, 'val'].dropna()]
                    label = [term_dict[int(l)] for l in slide_data.loc[:, 'val_label'].dropna()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'test':
                    data = [feature_dir + '/' + filename for filename in slide_data.loc[:, 'test'].dropna()]
                    label = [term_dict[int(l)] for l in slide_data.loc[:, 'test_label'].dropna()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]
                
                labels = [term_dict[int(l)] for l in slide_data.loc[:, 'train_label'].dropna()] + [term_dict[int(l)] for l in slide_data.loc[:, 'val_label'].dropna()] + [term_dict[int(l)] for l in slide_data.loc[:, 'test_label'].dropna()]
                for l in labels:
                    if l not in dataset_stats[data_name]:
                        dataset_stats[data_name][l] = 1
                    else:
                        dataset_stats[data_name][l] += 1

            # TCGA-BRCA
            elif data_name == 'brca':
                csv_dir = './wsi_dataset_annotation/tcga_brca/tcga_brca_subset.csv.zip'
                slide_data = pd.read_csv(csv_dir)
                label_dict = {'IDC':0, 'ILC':1}
                ignore = ['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']
                label_col = 'oncotree_code'
                slide_data = self.df_prep(slide_data, label_dict, ignore, label_col)
                split_dir = self.dataset_cfg.label_dir[i_data] + f'fold{self.fold}.csv'
                split_info = pd.read_csv(split_dir)

                train_list = split_info.train.dropna()
                val_list = split_info.val.dropna()
                test_list = split_info.test.dropna()

                if state == 'train':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if slide_id.split('.svs')[0] in train_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in train_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'val':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if slide_id.split('.svs')[0] in val_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in val_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'test':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if slide_id.split('.svs')[0] in test_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in test_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                labels = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in train_list.values.tolist()] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in val_list.values.tolist()] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in test_list.values.tolist()]
                
                for l in labels:
                    if l not in dataset_stats[data_name]:
                        dataset_stats[data_name][l] = 1
                    else:
                        dataset_stats[data_name][l] += 1

            # TCGA-RCC
            elif data_name == 'rcc':
                csv_dir = './wsi_dataset_annotation/tcga_rcc/tcga_kidney_subset.csv.zip'
                slide_data = pd.read_csv(csv_dir)
                label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2}
                ignore = []
                label_col = 'oncotree_code'
                slide_data = self.df_prep(slide_data, label_dict, ignore, label_col)
                split_dir = self.dataset_cfg.label_dir[i_data] + f'fold{self.fold}.csv'
                split_info = pd.read_csv(split_dir)

                train_list = split_info.train.dropna()
                val_list = split_info.val.dropna()
                test_list = split_info.test.dropna()

                if state == 'train':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in train_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in train_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'val':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in val_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in val_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'test':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in test_list.values.tolist()]
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in test_list.values.tolist()]
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                labels = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in train_list.values.tolist()] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in val_list.values.tolist()] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if slide_data.iloc[i].slide_id.split('.svs')[0] in test_list.values.tolist()]
                for l in labels:
                    if l not in dataset_stats[data_name]:
                        dataset_stats[data_name][l] = 1
                    else:
                        dataset_stats[data_name][l] += 1

            # TCGA-RCC
            elif data_name == 'nsclc':
                csv_dir = './wsi_dataset_annotation/tcga_nsclc/tcga_lung_subset.csv.zip'
                slide_data = pd.read_csv(csv_dir)
                label_dict = {'LUAD':0, 'LUSC':1}
                ignore = []
                label_col = 'oncotree_code'
                slide_data = self.df_prep(slide_data, label_dict, ignore, label_col)
                split_dir = self.dataset_cfg.label_dir[i_data] + f'fold{self.fold}.csv'
                split_info = pd.read_csv(split_dir)

                train_list = split_info.train.dropna()
                val_list = split_info.val.dropna()
                test_list = split_info.test.dropna()
                
                if state == 'train':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in train_list.values.tolist() and os.path.splitext(slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in train_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'val':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in val_list.values.tolist() and os.path.splitext(slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in val_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                if state == 'test':
                    data = [feature_dir + '/' + slide_id.split('.svs')[0] for slide_id in slide_data['slide_id'] if os.path.splitext(slide_id)[0] in test_list.values.tolist() and os.path.splitext(slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    label = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in test_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                    self.tasks += [self.name_tasks[data_name]] * len(label)
                    self.data_name += [data_name] * len(label)
                    if self.classification:
                        label = [cls2idx[l] for l in label]

                labels = [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in train_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663'] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in val_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663'] + [term_dict[slide_data.iloc[i].label] for i in range(len(slide_data)) if os.path.splitext(slide_data.iloc[i].slide_id)[0] in test_list.values.tolist() and os.path.splitext(slide_data.iloc[i].slide_id)[0] != 'TCGA-52-7622-01Z-00-DX1.cb3bb056-27dd-4c15-9004-b06cc8923663']
                for l in labels:
                    if l not in dataset_stats[data_name]:
                        dataset_stats[data_name][l] = 1
                    else:
                        dataset_stats[data_name][l] += 1
                
            self.data += data
            self.label += label

            json.dump(dataset_stats, open('dataset_stats.json', 'w'), indent=4)
        
        # shuffle train data
        if state == 'train':
            idx = np.random.RandomState(seed=2000).permutation(len(self.data))
            self.data, self.label = np.array(self.data)[idx], np.array(self.label)[idx]
            self.tasks = np.array(self.tasks)[idx]

        if not os.path.exists(self.dataset_cfg.vocab_path):
            print("Building vocabolary ...")
            if self.gpt_pretrained:
                self.term_vocab = list(set([word for tokens in [l.split(' ') for l in self.label] for word in tokens]))
                # get token ids
                list_token_ids = []
                for term in self.term_vocab:
                    list_token_ids.extend(self.tokenizer.encode(term)[1:-1])
                
                list_token_ids = list(set(list_token_ids)) # unique items
                self.dict_vocab2idx = dict()

                self.term_vocab = []
                
                for token_id in list_token_ids:
                    token = self.tokenizer.decode([token_id])
                    self.term_vocab.append(token)

                self.term_vocab = ['<bos>', '<unk>', '<eos>', '<pad>'] + self.term_vocab

                for idx in range(len(self.term_vocab)):
                    self.dict_vocab2idx[self.term_vocab[idx]] = idx
                
            else:
                self.term_vocab = ['<bos>', '<unk>', '<eos>', '<pad>'] + list(set([word for tokens in [l.split(' ') for l in self.label] for word in tokens]))
                self.dict_vocab2idx = dict()
                for idx in range(len(self.term_vocab)):
                    self.dict_vocab2idx[self.term_vocab[idx]] = idx
                    
            json.dump(self.dict_vocab2idx, open(self.dataset_cfg.vocab_path, 'w'), indent=4)
        else:
            self.dict_vocab2idx = json.load(open(self.dataset_cfg.vocab_path, 'r'))

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data
    
    def create_cluster_matrix(self, cluster_ids):
        # Check if items are in the same cluster using broadcasting
        A = (cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)).float()
        return A

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = self.label[idx]
        task = self.tasks[idx]

        if not self.classification:
            term = label.split(' ')
            term = torch.Tensor([self.dict_vocab2idx[self.bos_tag]] + [self.dict_vocab2idx[t] for t in term] + [self.dict_vocab2idx[self.eos_tag]] + [self.padding_idx] * (self.max_seq_len - len(term) - 2))
            label = term
        
        full_path = slide_id + '.pt'
        features = torch.load(full_path)

        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        if self.state in ['test']:
            data_name = self.data_name[idx]
            return features, label, data_name, os.path.basename(slide_id), task

        return features, label, task