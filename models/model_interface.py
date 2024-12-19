import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl
import time
# from calflops import calculate_flops
from torch.nn import NLLLoss
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
from transformers import AutoTokenizer
from torch import Tensor
from prettytable import PrettyTable
import os
from calflops import calculate_flops

class ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, vocab_size, vocab_path, max_seq_len, bos_tag='<bos>', eos_tag='<eos>', padding_idx=3, **kwargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = NLLLoss(ignore_index=kwargs["data"]["padding_idx"])
        # self.loss1 = SeesawLoss(ignore_index=kwargs["data"]["padding_idx"], num_classes=vocab_size)
        self.classification = False
        self.n_classes = vocab_size
        try:
            self.classification = kwargs["data"]["classification"]
            if self.classification:
                self.loss = create_loss(loss)
            self.n_classes = 11
        except:
            pass
        self.task_loss = create_loss(loss)
        self.optimizer = optimizer
        self.log_path = kwargs['log']
        csv_result_file_path = './csv_outputs/' + str(self.log_path).split('/')[-2] + '.csv'
        self.result_writer = open(csv_result_file_path, 'w')
        self.word2idx = json.load(open(vocab_path, 'r'))
        self.idx2word = {self.word2idx[k]:k for k in self.word2idx}
        self.max_length = max_seq_len
        self.bos_tag = bos_tag
        self.eos_tag = eos_tag
        self.padding_idx = padding_idx

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2:
            self.ACC = torchmetrics.Accuracy(num_classes = len(self.word2idx), average = 'micro')
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1Score(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.ACC = torchmetrics.Accuracy(num_classes = 2, average = 'micro')
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kwargs['data'].data_shuffle
        self.count = 0

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference

        data, label, task = batch

        if self.classification:
            results_dict = self.model(data=data, label=label, task=task)
            logits = results_dict['logits']
            Y_prob = results_dict['Y_prob']
            Y_hat = results_dict['Y_hat']

            #---->loss
            loss = self.loss(logits, label)

            #---->acc log
            Y_hat = int(Y_hat)
            Y = int(label)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat == Y)

            return {'loss': loss} 

        results_dict = self.model(data=data, label=label, task=task)
        Y_prob = results_dict['Y_prob']

        #---->loss
        out = Y_prob[:, :-1].contiguous()
        label = label[:, 1:].contiguous()

        out = out.view(-1, out.shape[-1])
        label = label.view(-1)

        # loss = self.loss(out, label.long()) + self.loss1(out, label.long())
        loss = self.loss(out, label.long())

        # gate_weights

        if 'gate_weights' in results_dict:
            gate_weights = results_dict['gate_weights'].squeeze(0)
            loss += self.task_loss(gate_weights, task.repeat(gate_weights.shape[0]))

        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):

        if self.classification:
            for c in range(self.n_classes):
                count = self.data[c]["count"]
                correct = self.data[c]["correct"]
                if count == 0: 
                    acc = None
                else:
                    acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        return

    def validation_step(self, batch, batch_idx):
        try:
            data, label, task = batch
        except:
            data, label, _, _, task = batch

        if self.classification:
            results_dict = self.model(data=data, label=label, task=task)
            logits = results_dict['logits']
            Y_prob = results_dict['Y_prob']
            Y_hat = results_dict['Y_hat']

            #---->acc log
            Y = int(label)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat.item() == Y)

            return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}
        
        results_dict = self.model(data=data, label=label, task=task)
        
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def validation_epoch_end(self, val_step_outputs):

        if self.classification:
            logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
            probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
            max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
            target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
            
            #---->
            self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
            # self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
            self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                            on_epoch = True, logger = True)

            #---->acc log
            for c in range(self.n_classes):
                count = self.data[c]["count"]
                correct = self.data[c]["correct"]
                if count == 0: 
                    acc = None
                else:
                    acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        else:
            probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
            target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
            
            out = probs[:, :-1].contiguous()
            label = target[:, 0, 1:].contiguous()

            out = out.view(-1, out.shape[-1])
            label = label.view(-1)
            #---->
            self.log('val_loss', self.loss(out, label.long()), prog_bar=True, on_epoch=True, logger=True)
            self.log('acc', self.ACC(torch.argmax(out, dim=-1), label.long()), prog_bar=True, on_epoch=True, logger=True)
        
            #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def label2term(self, label, data, task, slide_id):
        # Get label terms <---- need better implementation ---->
        bos_idx = self.word2idx[self.bos_tag]
        label_term = ''
        for i, idx_term in enumerate(label[0]):
            if int(idx_term) != bos_idx and int(idx_term) != self.padding_idx:
                word = self.idx2word[int(idx_term)]
                if word != self.eos_tag and i > 0:
                    label_term += (word + ' ')
                else:
                    break
        
        # Get predicted terms
        start_tag = self.bos_tag
        seq_token = torch.Tensor([self.word2idx[start_tag]]).unsqueeze(0).cuda()

        predicted_term = ''
        attns = []

        # ecn_flops = 0.
        # encoder_flops = 0.
        # decoder_flops = 0.
        # logit_flops = 0.

        inputs = dict(data=data, label=label, task=task)
        flops, macs, params = calculate_flops(model=self.model,
                                        kwargs=inputs,
                                        print_results=False)
        
        flops = float(flops.split(" GFLOPS")[0])
        for i in range(self.max_length):
            # logits, gate_weights, _, flops_info, number_of_patches \
            logits, gate_weights, _ \
                = self.model.forward_test(data=data, label=seq_token, task=task)
            seq_token = torch.cat([seq_token, torch.argmax(logits, -1)[:, -1].unsqueeze(0)], dim=-1)
            idx_term = int(torch.argmax(logits, -1)[:, -1])
            word = self.idx2word[idx_term]

            # ecn_flops = flops_info["ecn_flops"]
            # encoder_flops = flops_info["encoder_flops"]
            # decoder_flops += flops_info["decoder_flops"]
            # logit_flops += flops_info["logit_flops"]

            if word != self.eos_tag:
                attns.append(gate_weights.unsqueeze(2))
                predicted_term += (word + ' ')
            else:
                break

        attns = torch.cat(attns, dim=2)
        # total_flops = ecn_flops + encoder_flops + decoder_flops + logit_flops

        return label_term, predicted_term, gate_weights, attns, flops \
            # total_flops, number_of_patches

    def test_step(self, batch, batch_idx):
        data, label, data_name, slide_id, task = batch

        if self.classification:
            inputs = dict()
            inputs["data"] = data

            inputs = dict(data=data, label=label, task=task)
            results_dict = self.model(data=data, label=label, task=task)
            flops, macs, params = calculate_flops(model=self.model,
                                        kwargs=inputs,
                                        print_results=False)
        
            flops = float(flops.split(" GFLOPS")[0])

            logits = results_dict['logits']
            Y_prob = results_dict['Y_prob']
            Y_hat = results_dict['Y_hat']

            #---->acc log
            Y = int(label)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat.item() == Y)

            return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'data_name': data_name[0], 'flops': flops}
        
        inputs = dict()
        inputs["data"] = data

        # , flops, number_of_patches
        label_term, predicted_term, gate_weights, attns, flops = self.label2term(label, data, task, slide_id[0])
        
        return {'predicted_term' : predicted_term.strip(), 
                'label_term': label_term.strip(), 
                'data_name': data_name[0], 
                'slide_id': slide_id, 
                'gate_weights': gate_weights,
                'attns': attns,
                'flops': flops,
                # 'number_of_patches': number_of_patches
                }

    def test_epoch_end(self, output_results):

        self.result_writer.write('dataset,acc,f1,recall,precision\n')

        if self.classification:
            probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
            max_probs = torch.stack([x['Y_hat'] for x in output_results])
            target = torch.stack([x['label'] for x in output_results], dim = 0)
            data_names = [x['data_name'] for x in output_results]
            flops = [x['flops'] for x in output_results]
            
            list_dataset_name = sorted(list(set(data_names)))

            #---->
            # Overall results
            metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
            for keys, values in metrics.items():
                print(f'{keys} = {values}')
                metrics[keys] = values.cpu().numpy()

            dict_seperate = dict()

            for data_name in list_dataset_name:
                dict_seperate[data_name] = dict(max_probs=list(), target=list())

            for max_prob, target_, data_name in zip(max_probs, target, data_names):
                dict_seperate[data_name]['max_probs'].append(max_prob)
                dict_seperate[data_name]['target'].append(target_)
                

            for data_name in dict_seperate:
                dict_seperate[data_name]['max_probs'] = torch.cat(dict_seperate[data_name]['max_probs'])
                dict_seperate[data_name]['target'] = torch.cat(dict_seperate[data_name]['target'])

            result = PrettyTable(["TASK", "ACC", "F1", "RECALL", "PRECISION", "KW SCORE"])

            for data_name in list_dataset_name:
                # print("-" * 30)
                # print(data_name)
                dict_results = dict_seperate[data_name]
                dict_results['target'] = dict_results['target'].cpu().detach().numpy()
                dict_results['max_probs'] = dict_results['max_probs'].cpu().detach().numpy()
                                
                acc = accuracy_score(dict_results['target'], dict_results['max_probs'])
                f1 = f1_score(dict_results['target'], dict_results['max_probs'], average='macro')
                precision = precision_score(dict_results['target'], dict_results['max_probs'], average='macro')
                recall = recall_score(dict_results['target'], dict_results['max_probs'], average='macro')
                kw_score = cohen_kappa_score(dict_results['target'], dict_results['max_probs'], weights='quadratic')

                self.result_writer.write(','.join([data_name, str(acc), str(f1), str(recall), str(precision)]) + '\n')

                result.add_row([data_name,
                            "{:.3f}".format(float(acc) * 100), 
                            "{:.3f}".format(float(f1)),
                            "{:.3f}".format(float(recall)),
                            "{:.3f}".format(float(precision)),
                            "{:.3f}".format(float(kw_score))])
            
            torch.save(torch.Tensor(flops), "flops_TransMIL.pth")
            print(result)
            sys.exit()

        predicted_terms = [x['predicted_term'] for x in output_results]
        label_terms = [x['label_term'] for x in output_results]
        data_names = [x['data_name'] for x in output_results]
        slide_ids = [x['slide_id'] for x in output_results]
        gate_weights = [x['gate_weights'] for x in output_results]
        attns = [x['attns'] for x in output_results]
        flops = [x['flops'] for x in output_results]
        # number_of_patches = [x['number_of_patches'] for x in number_of_patches]

        torch.save(torch.Tensor(flops), "flops_MECFormer.pth")
        # torch.save(torch.Tensor(flops), "flops_MECFormer.pth")

        list_dataset_name = sorted(list(set(data_names)))
        dict_gate_weights = dict()

        result_dict = dict()
        for predict_term, label_term, slide_id, data_name, attn in zip(predicted_terms, label_terms, slide_ids, data_names, attns):
            result_dict[slide_id[0]] = dict(
                predict_term=predict_term,
                label_term=label_term,
                data_name=data_name
            )
        
        json.dump(result_dict, open('results_caption.json', 'w'), indent=4)

        dict_seperate = dict()
        for data_name in list_dataset_name:
            dict_seperate[data_name] = dict(predicted_terms=list(), label_terms=list())
            dict_gate_weights[data_name] = list()

        for predicted_term, label_term, data_name, gate_weight in zip(predicted_terms, label_terms, data_names, gate_weights):
            dict_seperate[data_name]['predicted_terms'].append(predicted_term)
            dict_seperate[data_name]['label_terms'].append(label_term)
            dict_gate_weights[data_name].append(gate_weight)
        
        for data_name in dict_gate_weights:
            dict_gate_weights[data_name] = torch.cat(dict_gate_weights[data_name], dim=0)

        torch.save(dict_gate_weights, './gate_weights/dict_gate_weights.pth')

        result = PrettyTable(["TASK", "ACC", "F1", "RECALL", "PRECISION", "KW SCORE"]) 

        # print("Overall results:")
        oa = accuracy_score(label_terms, predicted_terms)
        of1 = f1_score(label_terms, predicted_terms, average='macro')
        oprecision = precision_score(label_terms, predicted_terms, average='macro')
        orecall = recall_score(label_terms, predicted_terms, average='macro')
        okw = cohen_kappa_score(label_terms, predicted_terms, weights='quadratic')
        
        for data_name in dict_seperate:
            dict_results = dict_seperate[data_name]
            acc = accuracy_score(dict_results['label_terms'], dict_results['predicted_terms'])
            f1 = f1_score(dict_results['label_terms'], dict_results['predicted_terms'], average='macro')
            precision = precision_score(dict_results['label_terms'], dict_results['predicted_terms'], average='macro')
            recall = recall_score(dict_results['label_terms'], dict_results['predicted_terms'], average='macro')
            kw_score = cohen_kappa_score(dict_results['label_terms'], dict_results['predicted_terms'], weights='quadratic')

            result.add_row([data_name,
                            "{:.3f}".format(float(acc) * 100), 
                            "{:.3f}".format(float(f1)),
                            "{:.3f}".format(float(recall)),
                            "{:.3f}".format(float(precision)),
                            "{:.3f}".format(float(kw_score))])

            self.result_writer.write(','.join([data_name, str(acc), str(f1), str(recall), str(precision)]) + '\n')
        
        result.add_row(["Overall",
                            "{:.3f}".format(float(oa) * 100), 
                            "{:.3f}".format(float(of1)),
                            "{:.3f}".format(float(orecall)),
                            "{:.3f}".format(float(oprecision)),
                            "{:.3f}".format(float(okw))])

        print(result)

        self.result_writer.write('overall,{},{},{},{}\n'.format(str(oa), str(of1), str(precision), str(recall)))
        self.result_writer.close()
        sys.exit()

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        self.hparams.model['vocab_size'] = self.hparams['vocab_size']
        self.hparams.model['max_seq_len'] = self.hparams.data["max_seq_len"]
        # self.hparams.data['bos_tag'] = self.hparams.data["bos_tag"]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)