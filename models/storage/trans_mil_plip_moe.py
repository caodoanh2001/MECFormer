import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .nystrom_attention import NystromAttention
from .attention import MultiHeadAttention
from .plip_decoder import CLIPTextTransformer, plip_config, load_plip_weight
from torch import Tensor, nn
from typing import List, Optional
from copy import deepcopy

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, clusters=None):
        x = x + self.attn(self.norm(x), clusters=clusters)
        return x

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out

class TransEncoder(nn.Module):
    def __init__(self, d_model):
        super(TransEncoder, self).__init__()
        self.layer1 = TransLayer(dim=d_model)
        self.layer2 = TransLayer(dim=d_model)

    def forward(self, x):
        #---->Translayer x1
        x = self.layer1(x) #[B, N, 512]
        
        #---->Translayer x2
        x = self.layer2(x) #[B, N, 512]
        return x

class TransDecoder(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransDecoder, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class DictMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        assert num_hidden_layers <= 2
        self.input_dim = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        elif num_hidden_layers == 1:
            self.fc1 = nn.Identity()

        if num_hidden_layers >= 1:
            self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.constant_(self.fc2.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.ones(num_experts) * init_lambda, requires_grad=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight

        if self.num_hidden_layers == 2:
            hidden_states = F.relu(self.fc1(hidden_states))
        gate_weights = self.fc2(hidden_states)
        return gate_weights
    
class DictMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        fix_base_model_and_experts: bool = True,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.input_dim = hidden_size
        self.batch_first = batch_first

        self.gate = DictMoEGate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        self.base_model = deepcopy(base_model)
        experts = [deepcopy(e) for e in expert_models]
        base_sd = self.base_model.state_dict()
        experts_params = []
        experts_sd = [e.state_dict() for e in experts]

        for name in base_sd.keys():
            task_vectors = []
            for e_sd in experts_sd:
                with torch.no_grad():
                    _task_vector = e_sd[name] - base_sd[name]
                    task_vectors.append(_task_vector)
            task_vectors = torch.stack(task_vectors)
            experts_params.append(nn.Parameter(task_vectors, requires_grad=not fix_base_model_and_experts))
        self.expert_parms = nn.ParameterList(experts_params)

        if fix_base_model_and_experts:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            for p in self.expert_parms.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_states: Tensor):
        if not self.batch_first:
            hidden_states = hidden_states.permute(1, 0, 2)
        batch_size, seq_len, hidden_size = hidden_states.shape
        gate_weights: Tensor = self.gate(hidden_states)
        if self.gate.num_hidden_layers == 0:
            base_sd = self.base_model.state_dict(keep_vars=True)
            sd = {}
            for param_idx, (name, param) in enumerate(base_sd.items()):
                expert_params: nn.Parameter = self.expert_parms[param_idx]
                task_vector = expert_params * gate_weights.view([-1] + [1] * (expert_params.dim() - 1))
                task_vector = task_vector.sum(dim=0)
                sd[name] = param + task_vector
            final_hidden_states = torch.func.functional_call(self.base_model, sd, hidden_states)
        else:
            gate_weights = gate_weights.mean(dim=1)
            final_hidden_states = []
            base_sd = self.base_model.state_dict(keep_vars=True)
            for sample_idx in range(batch_size):
                sd = {}
                for param_idx, (name, param) in enumerate(base_sd.items()):
                    if 'layer_norm' not in name:
                        expert_params: nn.Parameter = self.expert_parms[param_idx]
                        task_vector = expert_params * gate_weights[sample_idx].view([-1] + [1] * (expert_params.dim() - 1))
                        task_vector = task_vector.sum(dim=0)
                        sd[name] = param + task_vector
                    else:
                        sd[name] = param
                        for expert_norm_params in self.expert_parms[param_idx]:
                            sd[name] = sd[name] + (expert_norm_params - param) * 0.03
                
                _final_hidden_states = torch.func.functional_call(self.base_model, sd, hidden_states[sample_idx : sample_idx + 1])
                final_hidden_states.append(_final_hidden_states)
            final_hidden_states = torch.cat(final_hidden_states, dim=0)
        
        if not self.batch_first:
            final_hidden_states = final_hidden_states.permute(1, 0, 2)
        
        return final_hidden_states
    
class TransMilPlip(nn.Module):
    def __init__(self, vocab_size, max_seq_len, f_dim, d_model, vocab=None):
        super(TransMilPlip, self).__init__()
        
        self.f_dim = f_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.num_encode_layers = 2
        self.num_decode_layers = 2

        self._fc1 = nn.Sequential(nn.Linear(self.f_dim, self.d_model), nn.ReLU())
        self.encoder = nn.ModuleList()
        
        for _ in range(self.num_encode_layers):
            self.encoder.append(TransLayer(dim=self.d_model))
        
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decode_layers):
            self.decoder.append(TransDecoder(d_model=self.d_model))
        
        # Load PLIP pre-training weights for word embedding
        decoder = CLIPTextTransformer(plip_config)
        decoder = load_plip_weight(decoder, ckpt_path='./models/tokenizers/pytorch_model.bin')
        self.word_emb = decoder.embeddings.token_embedding
        self.pos_emb = decoder.embeddings.position_embedding

        self.padding_idx = plip_config.pad_token_id
        self._fc2 = nn.Linear(self.d_model, vocab_size, bias=True)

    def masking_seq(self, gt_term):
        b_s, seq_len = gt_term.shape[:2]
        mask_queries = (gt_term != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=gt_term.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (gt_term == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(gt_term.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        return mask_queries, seq, mask_self_attention

    def forward(self, **kwargs):
        wsi_input = kwargs['data'].float()
        gt_term = kwargs['label']
        
        # Encoder
        x = self._fc1(wsi_input)
        for i in range(self.num_encode_layers):
            x = self.encoder[i](x)

        # Decoder
        mask_queries, seq, mask_self_attn = self.masking_seq(gt_term)
        mask_encoder = torch.ones((x.shape[1], x.shape[1]), dtype=torch.uint8, device=x.device)
        out = self.word_emb(gt_term.long()) + self.pos_emb(seq) # bs, max_seq, dim
        
        for i in range(self.num_decode_layers):
            out = self.decoder[i](out, x, mask_queries, mask_self_attn, mask_encoder)
        
        results_dict = {'logits': out, 'Y_prob': out, 'Y_hat': gt_term}
        return results_dict
    
    def forward_test(self, **kwargs):
        wsi_input = kwargs['data'].float()
        gt_term = kwargs['label']

        # Encoder
        x = self._fc1(wsi_input)
        for i in range(self.num_encode_layers):
            x = self.encoder[i](x)

        # Decoder
        mask_queries = (gt_term != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_queries, seq, mask_self_attn = self.masking_seq(gt_term)
        mask_encoder = torch.ones((x.shape[1], x.shape[1]), dtype=torch.uint8, device=x.device)
        
        out = self.word_emb(gt_term.long()) + self.pos_emb(seq) # bs, max_seq, dim
        for i in range(self.num_decode_layers):
            out = self.decoder[i](out, x, mask_queries, mask_self_attn, mask_encoder)
                
        return out

class TransMilPlipMoe(nn.Module):
    def __init__(self, vocab_size, max_seq_len, f_dim, d_model, vocab=None):
        super(TransMilPlipMoe, self).__init__()
        
        # Load base model & its weights
        self.pretrained_model = TransMilPlip(vocab_size=vocab_size, max_seq_len=max_seq_len, f_dim=f_dim, d_model=d_model)        
        model = deepcopy(self.pretrained_model)
        
        ckpt_weights = torch.load('/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir/logs_camel_brca_esca_rcc_nsclc_transplip/configs/camel_brca_esca_rcc_nsclc_transplip/fold0/epoch=03-val_loss=0.0739.ckpt')['state_dict']
        ckpt_weights = {k.replace('model.', ''):ckpt_weights[k] for k in ckpt_weights.keys()}
        model.load_state_dict(ckpt_weights)
        print("Load base model succesfully")

        # load expert models and their weights
        expert_paths = [
            '/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_task_specific/camel_fold0/configs_task_specific/camel_fold0/fold0/epoch=00-val_loss=0.0736.ckpt',
            '/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_task_specific/brca_fold0/configs_task_specific/brca_fold0/fold0/epoch=01-val_loss=0.0755.ckpt',
            '/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_task_specific/esca_fold0/configs_task_specific/esca_fold0/fold0/epoch=07-val_loss=0.0231.ckpt',
            '/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_task_specific/rcc_fold0/configs_task_specific/rcc_fold0/fold0/epoch=00-val_loss=0.0587.ckpt',
            '/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_task_specific/nsclc_fold0/configs_task_specific/nsclc_fold0/fold0/epoch=00-val_loss=0.0493.ckpt',
        ]

        finetuned_models = []
        for i, expert_path in enumerate(expert_paths):
            finetune_model = TransMilPlip(vocab_size=vocab_size, max_seq_len=max_seq_len, f_dim=f_dim, d_model=d_model)
            cpkt = torch.load(expert_path, map_location="cpu")['state_dict']
            cpkt = {k.replace('model.', ''):cpkt[k] for k in cpkt.keys()}
            finetune_model.load_state_dict(cpkt)
            finetuned_models.append(finetune_model)
            print("Load expert", i)

        sd = dict()
        base_sd = model.state_dict()
            
        for name in base_sd.keys():
            sd[name] = base_sd[name]

        # combine MOE weights 
        # theta = theta_base + lamda * tau_i
        # tau_i = theta_i - theta_base
        init_lambda = 0.3
        for m in finetuned_models:
            expert_sd = m.state_dict()
            for name in expert_sd.keys():
                if '_fc2' not in name: # ignore classification heads
                    sd[name] = sd[name] + (expert_sd[name] - base_sd[name]) * init_lambda

        model.load_state_dict(sd)
        # fix all parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # DictMOE for PWFF in Transformer Decoder
        for layer_idx in range(len(model.decoder)):
            model.decoder[layer_idx].pwff = DictMoE(
                hidden_size=512,
                base_model=self.pretrained_model.decoder[layer_idx].pwff,
                expert_models=[m.decoder[layer_idx].pwff for m in finetuned_models],
                init_lambda=init_lambda,
                fix_base_model_and_experts=True,
                router_hidden_layers=2,
                batch_first=True,
            )

        # delete head classifier of the base model
        model._fc2 = nn.Identity()
        self.model = model

        # classification heads
        self.classification_heads = []
        for finetuned_model in finetuned_models:
            self.classification_heads.append(finetuned_model._fc2.cuda())

        for m in self.classification_heads:
            for p in m.parameters():
                p.requires_grad_(False)
        
        ckpt_weights = torch.load("/home/compu/doanhbc/WSIs-classification/general_wsi_classifier/logs_dir_moe/logs_camel_brca_esca_rcc_nsclc_moe/configs_moe/dict_moe/fold0/epoch=02-val_loss=2.9321.ckpt")['state_dict']
        ckpt_weights = {k.replace('model.', ''):ckpt_weights[k] for k in ckpt_weights.keys()}

    def forward(self, **kwargs):
        task = int(kwargs['task'][0])
        out = self.model(**kwargs)
        logits = self.classification_heads[task](out['logits'])
        return logits
    
    def forward_test(self, **kwargs):
        task = int(kwargs['task'][0])
        out = self.model(**kwargs)
        logits = self.classification_heads[task](out['logits'])
        return logits

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMilPlipMoe(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)