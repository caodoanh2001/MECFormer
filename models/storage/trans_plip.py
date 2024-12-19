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
    
class TransPlip(nn.Module):
    def __init__(self, vocab_size, max_seq_len, f_dim, d_model, vocab=None):
        super(TransPlip, self).__init__()
        
        self.f_dim = f_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_encode_layers = 3
        self.vocab_size = vocab_size

        self._fc1 = nn.Sequential(nn.Linear(self.f_dim, self.d_model), nn.ReLU())
        self.encoder = nn.ModuleList()
        
        for _ in range(self.num_encode_layers):
            self.encoder.append(TransLayer(dim=self.d_model))

        decoder = CLIPTextTransformer(plip_config)
        self.decoder = load_plip_weight(decoder, ckpt_path='./models/tokenizers/pytorch_model.bin')
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, **kwargs):
        wsi_input = kwargs['data'].float()
        input_ids = kwargs['label']

        # Encoder
        x = self._fc1(wsi_input)
        for i in range(self.num_encode_layers):
            x = self.encoder[i](x)
        
        # Decoder
        out = self.decoder(input_ids=input_ids.int(), visual_tokens = x)
        out = self.lm_head(out)

        logits = F.log_softmax(out, dim=-1)
        results_dict = {'logits': logits, 'Y_prob': logits, 'Y_hat': input_ids}
        return results_dict
    
    def forward_test(self, **kwargs):
        wsi_input = kwargs['data'].float()
        input_ids = kwargs['label']

        # Encoder
        x = self._fc1(wsi_input)
        for i in range(self.num_encode_layers):
            x = self.encoder[i](x)

        # Decoder
        out = self.decoder(input_ids.int(), x, mask_encoder=None)
        out = self.lm_head(out)
        return F.log_softmax(out, dim=-1)

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()