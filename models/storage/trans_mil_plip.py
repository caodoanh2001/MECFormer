import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .nystrom_attention import NystromAttention
from .attention import MultiHeadAttention
from .plip_decoder import CLIPTextTransformer, plip_config, load_plip_weight

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
        
        logits = self._fc2(out)
        
        results_dict = {'logits': logits, 'Y_prob': F.log_softmax(logits, dim=-1), 'Y_hat': gt_term}
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
        
        logits = self._fc2(out)
        
        return F.log_softmax(logits, dim=-1)

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMilPlip(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)