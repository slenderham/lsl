import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.fc_q = nn.Linear(d_model, d_model, bias=False)
        self.fc_k = nn.Linear(d_model, d_model, bias=False)
        self.fc_v = nn.Linear(d_model, d_model, bias=False)
        self.fc_heads = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None, group_prob=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            seq_len=query.size()[-2]
            b = torch.eye(seq_len).to(query.device)
            scores = scores.masked_fill((mask|b) == 0, -1e9)

        if group_prob is not None:
            p_attn = F.softmax(scores, dim = -1)
            p_attn = p_attn*group_prob.unsqueeze(1)
        else:
            p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value, group_prob=None, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, seq_len, d_k)
        query, key, value = self.fc_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2), \
                            self.fc_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2), \
                            self.fc_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, group_prob=group_prob)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.fc_heads(x)

class GroupAttention(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(GroupAttention, self).__init__()
        self.d_model = d_model
        self.linear_key = nn.Linear(d_model, d_model, bias=False)
        self.linear_query = nn.Linear(d_model, d_model, bias=False)
        #self.linear_output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, eos_mask, prior):
        batch_size, seq_len = context.size()[:2]

        context = self.norm(context)

        a = torch.diag(torch.ones(seq_len - 1, dtype=torch.long), 1).to(context.device)
        b = torch.diag(torch.ones(seq_len, dtype=torch.long), 0).to(context.device)
        c = torch.diag(torch.ones(seq_len - 1, dtype=torch.long), -1).to(context.device)
        tri_matrix = torch.tril(torch.ones(seq_len, seq_len)).to(context.device)

        #mask = eos_mask & (a+c) | b
        mask = eos_mask & (a+c)
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5) # batch, len, len
        
        scores = scores.masked_fill(mask == 0, -1e9)
        neibor_attn = F.softmax(scores, dim=-1) # batch, len, len
        neibor_attn = torch.sqrt(neibor_attn*neibor_attn.transpose(-2,-1) + 1e-9)
        neibor_attn = prior + (1. - prior)*neibor_attn

        t = torch.log(neibor_attn + 1e-9).masked_fill(a==0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int()-b)==0, 0)     
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(b==0, 1e-9)
        
        return g_attn, neibor_attn

class Encoder(nn.Module):
    def __init__(self, layer, N, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, mask):
        break_probs = []
        group_prob = 0.
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask, group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)
        return self.proj(x), break_probs

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hidden_size, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(hidden_size//64, hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.group_attn = GroupAttention(hidden_size, dropout=0.0)
        self.hidden_size = hidden_size

    def forward(self, x, mask, group_prob):
        group_prob, break_prob = self.group_attn(x, mask, group_prob)
        x_norm = self.ln_1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, group_prob, mask)
        x_norm = self.ln_2(x)
        x = x + self.feed_forward(x_norm)
        return x, group_prob, break_prob

class ImagePositionalEmbedding(nn.Module):
    def __init__(self, height, width, hidden_size, bias=True, coord_type='sine'):
        super(ImagePositionalEmbedding, self).__init__()
        self.coord_type = coord_type

        x_coord_pos = torch.linspace(0, 1, height).reshape(1, height, 1).expand(1, height, width)
        x_coord_neg = torch.linspace(1, 0, height).reshape(1, height, 1).expand(1, height, width)
        y_coord_pos = torch.linspace(0, 1, width).reshape(1, 1, width).expand(1, height, width)
        y_coord_neg = torch.linspace(1, 0, width).reshape(1, 1, width).expand(1, height, width)

        if (coord_type=='cartesian'):
            self.register_buffer('coords', torch.cat([x_coord_pos, x_coord_neg, y_coord_pos, y_coord_neg], dim=0).unsqueeze(0))
            self.pos_emb = nn.Conv2d(4, hidden_size, 1, bias=bias)
        elif (coord_type=='polar'):
            coords = []
            for xx in [x_coord_neg, x_coord_pos]:
                for yy in [y_coord_neg, y_coord_pos]:
                    coords.append(torch.sqrt(xx**2+yy**2))
                    coords.append(torch.atan2(xx, yy))
            self.register_buffer('coords', torch.cat(coords, dim=0).unsqueeze(0))
            self.pos_emb = nn.Conv2d(8, hidden_size, 1, bias=bias)
        elif (coord_type=='sine'):
            div_term = 2*math.pi*torch.exp(-math.log(10000.0)*torch.arange(0., hidden_size, 2)/hidden_size).reshape(hidden_size//2, 1, 1)
            self.register_buffer('coords', torch.cat([torch.sin((x_coord_pos * div_term)[0::2]), torch.cos((x_coord_pos * div_term)[1::2]), \
                                                      torch.sin((y_coord_pos * div_term)[0::2]), torch.cos((y_coord_pos * div_term)[1::2])], dim=0).unsqueeze(0))
            self.pos_emb = nn.Identity()
        else:
            raise ValueError
        
    def forward(self, x):
        # add positional embedding to the feature vector
        return x+self.pos_emb(self.coords)

class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, scope):
        """
        Forward propagation.
        :param encoder_out: encoded slots, a tensor of dimension (batch_size, num_slots, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :param scope: scope for stick breaking which decreases every step, a tensor of dimension (batch_size, num_slots)
        :return: attention weighted encoding, weights
        """
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_slots, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_slots)
        alpha = self.softmax(att)  # (batch_size, num_slots)
        alpha_tilde = scope[:alpha.shape[0],:]*alpha
        scope[:alpha.shape[0],:] = scope[:alpha.shape[0],:]*(1-alpha)
        attention_weighted_encoding = (encoder_out * alpha_tilde.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha_tilde, scope

class TextPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(TextPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

