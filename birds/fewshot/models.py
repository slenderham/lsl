"""
Models
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as distributions
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from vision import ResNet18, ResNet50, ResNet10, BottleneckBlock, FPN, VGG16
import torchvision.models as models

def _cartesian_product(x, y):
    return torch.stack([torch.cat([x[i], y[j]], dim=0) for i in range(len(x)) for j in range(len(y))])

def normalize_feats(x):
    return F.normalize(x, p=2, dim=-1)

def pair_block_diag(x, block_size, num_blocks):
    pos_mask = (torch.block_diag(*([torch.ones(block_size, block_size)]*num_blocks))>0.5).to(x.device);
    pos_scores = torch.masked_select(x, pos_mask).reshape(block_size*block_size*num_blocks, 1);
    neg_scores = torch.masked_select(x, ~pos_mask).reshape(block_size*num_blocks, block_size*(num_blocks-1));
    neg_scores = torch.repeat_interleave(neg_scores, repeats=block_size, dim=0);
    assert(neg_scores.shape==(block_size**2*num_blocks, block_size*(num_blocks-1)))
    total_scores = torch.cat([pos_scores, neg_scores], dim=1);
    return total_scores, pos_scores.mean().item(), neg_scores.mean().item();

class ExWrapper(nn.Module):
    """
    Wrap around a model and allow training on examples
    i.e. tensor inputs of shape
    (batch_size, n_ex, *img_dims)
    """

    def __init__(self, model, freeze_model=False):
        super(ExWrapper, self).__init__()
        self.model = model
        self.freeze_model = freeze_model # whether or not to allow gradient to backprop through this step

    def forward(self, x, is_ex, **kwargs):
        batch_size = x.shape[0]
        if is_ex:
            n_ex = x.shape[1]
            img_dim = x.shape[2:]
            # Flatten out examples first
            x_flat = x.reshape(batch_size * n_ex, *img_dim)
        else:
            x_flat = x

        x_enc = self.model(x_flat, **kwargs)

        if is_ex:
            if (isinstance(x_enc, dict)):
                for k in x_enc.keys():
                    x_enc[k] = x_enc[k].reshape(batch_size, n_ex, *x_enc[k].shape[1:])
            else:
                x_enc = x_enc.reshape(batch_size, n_ex, *x_enc.shape[1:])

        if (self.freeze_model):
            if (isinstance(x_enc, dict)):
                for k in x_enc.keys():
                    x_enc[k] = x_enc[k].detach()
            else:
                x_enc = x_enc.detach()

        return x_enc

class Identity(nn.Module):
    def forward(self, x):
        return x

class ImageRep(nn.Module):
    r"""Two fully-connected layers to form a final image
    representation.

        VGG-16 -> FC -> ReLU -> FC

    Paper uses 512 hidden dimension.
    """

    def __init__(self, backbone=None, final_feat_dim = 4608, hidden_size=512, tune_backbone=True, normalize_feats=False):
        super(ImageRep, self).__init__()
        self.tune_backbone = tune_backbone
        self.normalize_feats = normalize_feats
        assert((not self.normalize_feats) or (self.normalize_feats and backbone is None)), "only normalize features if backbone is None"
        if backbone is None:
            self.backbone = Identity()
            self.backbone.final_feat_dim = final_feat_dim
        else:
            self.backbone = backbone
        if (hidden_size==None):
            self.model = nn.Identity()
        else:
            self.model = nn.Sequential(
                            nn.Linear(self.backbone.final_feat_dim, hidden_size), 
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x_enc = self.backbone(x)
        if (self.normalize_feats):
            x_enc = F.normalize(x_enc, dim=-1)
        if (not self.tune_backbone):
            x_enc = x_enc.detach()
        return self.model(x_enc)

class MLP(nn.Module):
    r"""
    MLP projection head
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MLP, self).__init__()
        assert(num_layers>=0), "At least 0 hidden_layers (a simple linear transform)"
        layers = []
        if (num_layers==0):
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            for _ in range(num_layers-1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TextRep(nn.Module):
    r"""Deterministic Bowman et. al. model to form
    text representation.

    Again, this uses 512 hidden dimensions.
    """

    def __init__(self, embedding_module, hidden_size, output_size = None, return_agg=False, bidirectional=False):
        super(TextRep, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.return_agg = return_agg
        self.gru = nn.GRU(self.embedding_dim, hidden_size, bidirectional=bidirectional)
        if output_size is not None:
            self.mlp = MLP(hidden_size, hidden_size, output_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist()
            if batch_size > 1 else length.data.tolist())

        hidden, final_hidden = self.gru(packed) 

        if self.return_agg:
            hidden, _ = rnn_utils.pad_packed_sequence(hidden)
            if self.bidirectional:
                hidden = (hidden[:,:,:self.hidden_size]+hidden[:,:,:self.hidden_size])/2 # for forward layer, get final for backward, get first
                hidden = hidden.sum(0)/sorted_lengths.float().unsqueeze(-1)
            else:
                hidden = final_hidden[0]
            if batch_size > 1:
                _, reversed_idx = torch.sort(sorted_idx)
                hidden = hidden[reversed_idx,:]
        else:
            hidden, _ = rnn_utils.pad_packed_sequence(hidden)
            if batch_size > 1:
                _, reversed_idx = torch.sort(sorted_idx)
                hidden = hidden[:,reversed_idx,:]
            hidden = hidden.transpose(0, 1)
            if self.bidirectional:
                hidden = (hidden[:,:,:self.hidden_size]+hidden[:,:,self.hidden_size:])/2

        if self.output_size is not None:
            hidden = self.mlp(hidden)

        return hidden

class TextRepTransformer(nn.Module):
    def __init__(self, embedding_module, hidden_size, return_agg=False, bidirectional=True):
        super(TextRepTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, dim_feedforward=4*hidden_size, dropout=0.0)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.pe = TextPositionalEncoding(hidden_size, dropout=0.0, max_len=16)
        self.return_agg = return_agg
        if not bidirectional:
            self.register_buffer("mask", torch.tril(torch.ones(16, 16))<0.5)
        else:
            self.mask = None

    def forward(self, seq, seq_len, src_key_padding_mask):
        batch_size = seq.size(0)

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)
        max_len = seq_len.max()

        # embed your sequences
        embed_seq = self.embedding(seq)
        embed_seq = self.pe(embed_seq)
        if (self.mask is not None):
            hidden = self.model(embed_seq, \
                            src_key_padding_mask=src_key_padding_mask,\
                            mask=self.mask[:max_len, :max_len])
        else:
            hidden = self.model(embed_seq, \
                            src_key_padding_mask=src_key_padding_mask)

        # reorder back to (B,L,D)
        hidden = hidden.transpose(0, 1)
        
        if (self.return_agg):
            hidden = hidden[torch.arange(batch_size), seq_len-1, :] # return last hidden state

        return hidden

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

class TextProposal(nn.Module):
    r"""Reverse proposal model, estimating:

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4 lambda)

    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(self, embedding_module, hidden_size):
        super(TextProposal, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.gru = nn.GRU(self.embedding_dim, hidden_size)
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size)
        self.hidden_size = hidden_size

    def forward(self, feats, seq, length):
        # feats is from example images
        batch_size = seq.size(0)

        if batch_size > 1:
            # BUGFIX? dont we need to sort feats too?
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            feats = feats[sorted_idx]

        feats = feats.unsqueeze(0)
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)
        packed_input = rnn_utils.pack_padded_sequence(embed_seq,
                                                      sorted_lengths)
        # shape = (seq_len, batch, hidden_dim)
        packed_output, _ = self.gru(packed_input, feats)
        output = rnn_utils.pad_packed_sequence(packed_output)
        output = output[0].contiguous()

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, self.hidden_size)
        outputs_2d = self.outputs2vocab(output_2d)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs

    def sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        """Generate from image features using greedy search."""
        with torch.no_grad():
            batch_size = feats.size(0)

            # initialize hidden states using image features
            states = feats.unsqueeze(0)

            # first input is SOS token
            inputs = np.array([sos_index for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in range(20):  # like in jacobs repo
                outputs, states = self.gru(inputs,
                                           states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)  # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)  # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)  # inputs: (L=1,B)
                inputs = self.embedding(inputs)  # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * pad_index

            for i in range(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths

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
        :return: attention weighted encoding, weights
        """
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_slots, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_slots)
        alpha = self.softmax(att)  # (batch_size, num_slots)
        alpha_tilde = scope[:alpha.shape[0],:]*alpha
        scope[:alpha.shape[0],:] = scope[:alpha.shape[0],:]*(1-alpha)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha_tilde, scope

class TextProposalWithAttn(nn.Module):
    r"""Reverse proposal model, estimating:

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4 lambda)

    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(self, embedding_module, encoder_dim, hidden_size):
        super(TextProposalWithAttn, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = embedding_module.num_embeddings
        self.gru = nn.GRUCell(self.embedding_dim+self.encoder_dim, hidden_size)
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size)
        self.hidden_size = hidden_size
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.attention = Attention(encoder_dim, hidden_size, hidden_size)

    def forward(self, feats, seq, length):
        # feats is from slots
        batch_size = seq.size(0)

        if batch_size > 1:
            # BUGFIX? dont we need to sort feats too?
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]
            feats = feats[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)

        # shape = (seq_len, batch, hidden_dim)

        output = torch.zeros(sorted_lengths[0], batch_size, self.vocab_size).to(seq.device)
        alphas = torch.zeros(sorted_lengths[0], batch_size, feats.shape[1]).to(seq.device)

        h = self.init_hidden_state(feats)

        scope = torch.ones(feats.shape[:2]).to(feats.device)

        for t in range(sorted_lengths[0]):
            batch_size_t = sum([l > t for l in sorted_lengths])
            attention_weighted_encoding, alpha, scope = self.attention(feats[:batch_size_t], h[:batch_size_t], scope)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h = self.gru(
                            torch.cat([embed_seq[t, :batch_size_t, :], attention_weighted_encoding], dim=1),
                            h[:batch_size_t]
                           )  # (batch_size_t, decoder_dim)
            preds = self.outputs2vocab(h)  # (batch_size_t, vocab_size)
            output[t, :batch_size_t, :] = preds
            alphas[t, :batch_size_t, :] = alpha

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)
        alphas = alphas.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]
            alphas = alphas[reversed_idx]

        return output, alphas

    def sample(self, feats, sos_index, eos_index, pad_index, greedy=False):
        """Generate from image features using greedy search."""
        with torch.no_grad():
            batch_size = feats.size(0)

            # initialize hidden states using image features
            states = feats.unsqueeze(0)

            # first input is SOS token
            inputs = np.array([sos_index for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in range(20):  # like in jacobs repo
                outputs, states = self.gru(inputs,
                                           states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)  # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)  # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)  # inputs: (L=1,B)
                inputs = self.embedding(inputs)  # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * pad_index

            for i in range(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        return h

class TextProposalTransformer(nn.Module):
    r"""Reverse proposal model, estimating:

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4 lambda)

    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(self, embedding_module, hidden_size):
        super(TextProposalTransformer, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=4*hidden_size, dropout=0.0)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.pe = TextPositionalEncoding(hidden_size, dropout=0.0, max_len=16)
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size)
        self.hidden_size = hidden_size
        self.register_buffer("mask", torch.tril(torch.ones(16, 16)).view(1, 1, 16, 16)<0.5)

    def forward(self, feats, seq, length, tgt_key_padding_mask):
        # feats is from example images
        batch_size = seq.size(0)
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)
        embed_seq = self.pe(embed_seq)
        max_len = length.max()
        
        # shape = (seq_len, batch, hidden_dim)
        output = self.model(embed_seq, feats, \
                            tgt_mask=self.mask[:max_len, :max_len], \
                            tgt_key_padding_mask=tgt_key_padding_mask)

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)
        output = self.outputs2vocab(output)

        return output

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None, num_iters = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        n_it = num_iters if num_iters is not None else self.iters
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = mu + torch.randn_like(mu)*sigma

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        attns = []

        for _ in range(n_it):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attns.append(attn)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(b*n_s, d),
                slots_prev.reshape(b*n_s, d)
            )

            slots = slots.reshape(b, n_s, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attns

class SANet(nn.Module):
    def __init__(self, im_size, num_slots=6, dim=64, iters = 3, eps = 1e-8, slot_model = 'slot_attn', use_relation = True):
        super(SANet, self).__init__()
        assert(slot_model in ['slot_attn', 'conv', 'pretrained'])
        self.slot_model = slot_model

        self.iters = iters
        self.num_slots = num_slots

        if (slot_model=='slot_attn'):
            im_size = (im_size+2-3)//2+1
            im_size = (im_size+2-3)//2+1
            self.encoder = nn.Sequential(
                ResNet18(flatten=False),
                nn.Conv2d(512, dim, 1),
                ImagePositionalEmbedding(im_size, im_size, dim)
            )

            if use_relation:
                self.slot_attn = RelationalSlotAttention(num_slots, dim, iters, eps, 2*dim)
            else:
                self.slot_attn = SlotAttention(num_slots, dim, iters, eps, 2*dim)
            self.final_feat_dim=dim
        elif (slot_model=='conv'):
            final_size = im_size
            for i in range(4):
                final_size = (final_size-3)//2+1
            self.encoder = nn.Sequential(
                nn.Conv2d(3, dim, 5, 2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 5, 2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True), 
                nn.Conv2d(dim, dim, 5, 2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True), 
                nn.Conv2d(dim, dim, 5, 2, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                ImagePositionalEmbedding(final_size, final_size, dim),
                nn.Flatten(),
                nn.Linear(final_size*final_size*dim, self.num_slots*dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_slots*dim, self.num_slots*dim)
            )
            self.final_feat_dim = dim
        elif (slot_model=='pretrained'):
            self.encoder = VGG16()
            self.final_feat_dim = 512*7*7

    def forward(self, img, **kwargs):
        visualize_attns=kwargs.get('visualize_attns', False)
        num_iters=kwargs.get('num_iters')
        num_slots=kwargs.get('num_slots')
        if (self.slot_model=='slot_attn'):
            x = self.encoder(img)
            n, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(n, h*w, c)
            x, attns = self.slot_attn(x, num_iters=num_iters, num_slots=num_slots) # --> N * num slots * feature size
            if visualize_attns:
                self._visualize_attns(img, attns, (num_iters if num_iters is not None else self.iters), (num_slots if num_slots is not None else self.num_slots))
        elif (self.slot_model=='pretrained'):
            x = self.encoder(img)
            if visualize_attns:
                plt.imshow(img[2].permute(1, 2, 0).detach().cpu())
        elif (self.slot_model=='conv'):
            x = self.encoder(img)
            if visualize_attns:
                plt.imshow(img[2].permute(1, 2, 0).detach().cpu())
        return x

    def _visualize_attns(self, img, attns, num_iters, num_slots):
        cmap = plt.get_cmap(name='Set3')
        bounds = list(range(12))  # values for each color
        norm = colors.BoundaryNorm(bounds, cmap.N)

        N, C, H, W = img.shape
        N, dim_q, dim_k = attns[0].shape # dim_q=the number of slots, dim_k=size of feature map
        H_k = W_k = math.isqrt(dim_k)
        # rand_idx = torch.randint(0, N, size=(1,)).item()
        rand_idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img[rand_idx].permute(1, 2, 0).detach().cpu())
        plt.show()
        fig1, axes1 = plt.subplots(num_iters, num_slots)
        fig2, axes2 = plt.subplots(num_iters, num_slots)
        for i in range(num_iters):
            for j in range(num_slots):
                axes1[i][j].imshow(torch.full(img[rand_idx].shape[:2], 1, dtype=torch.int), extent=(0, H, 0, W))
                axes1[i][j].imshow(torch.cat([img[rand_idx].permute(1, 2, 0).detach().cpu(),\
                    F.interpolate(attns[i][rand_idx][j].reshape(1, 1, H_k, W_k), size=(H, W), mode='bilinear').reshape(H, W, 1).detach().cpu()], dim=-1))
                axes2[i][j].imshow(F.interpolate(attns[i][rand_idx][j].reshape(1, 1, H_k, W_k), size=(H, W), mode='bilinear').reshape(H, W).detach().cpu(), vmin=0, vmax=1)
                axes1[i][j].axis('off')
                axes2[i][j].axis('off')
        # plt.show()

        # fig, axes = plt.subplots(1, num_iters)
        # for i in range(num_iters):
        #     masked_img = torch.zeros(H_k, W_k)
        #     for h in range(H_k):
        #         for w in range(W_k):
        #             masked_img[h, w] = torch.argmax(attns[i][rand_idx,:,h*W_k+w])
        #     axes[i].imshow(masked_img, cmap=cmap, norm=norm)

        # plt.show()


class ImagePositionalEmbedding(nn.Module):
    def __init__(self, height, width, hidden_size, coord_type='polar'):
        super(ImagePositionalEmbedding, self).__init__()
        self.coord_type = coord_type

        x_coord_pos = torch.linspace(0, 1, height).reshape(1, height, 1).expand(1, height, width)
        x_coord_neg = torch.linspace(1, 0, height).reshape(1, height, 1).expand(1, height, width)
        y_coord_pos = torch.linspace(0, 1, width).reshape(1, 1, width).expand(1, height, width)
        y_coord_neg = torch.linspace(1, 0, width).reshape(1, 1, width).expand(1, height, width)

        if (coord_type=='cartesian'):
            self.register_buffer('coords', torch.cat([x_coord_pos, x_coord_neg, y_coord_pos, y_coord_neg], dim=0).unsqueeze(0))
            self.pos_emb = nn.Conv2d(4, hidden_size, 1)
        elif (coord_type=='polar'):
            coords = [];
            for xx in [x_coord_neg, x_coord_pos]:
                for yy in [y_coord_neg, y_coord_pos]:
                    coords.append(torch.sqrt(xx**2+yy**2))
                    coords.append(torch.atan2(xx, yy))
            self.register_buffer('coords', torch.cat(coords, dim=0).unsqueeze(0))
            self.pos_emb = nn.Conv2d(8, hidden_size, 1)
        else:
            raise ValueError

    def forward(self, x):
        # add positional embedding to the feature vector
        return x+self.pos_emb(self.coords)

class RelationalSlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, gumbel_attention = False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.dim = dim
        self.scale = dim ** -0.5
        self.gumbel_attention = gumbel_attention

        self.slots_mu = nn.Parameter(torch.FloatTensor(1, 1, dim).uniform_(-1, 1)*self.scale)
        self.slots_sigma = nn.Parameter(torch.FloatTensor(1, 1, dim).uniform_(-1, 1)*self.scale)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim*2, dim)

        self.obj_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
         )

        self.rel_mlp = nn.Sequential(
            nn.Linear(dim*2, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_obj_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)        

    def _rel_msg(self, x):
        b, n_s, h = x.shape
        x_i = torch.unsqueeze(x, 2)  # b. n_s, 1, h
        x_i = x_i.expand(b, n_s, n_s, h)  # b. n_s*n_s, h: x1x1x1...x2x2x2...x3x3x3...
        x_j = torch.unsqueeze(x, 1)  # b, 1, n_s, h
        x_j = x_j.expand(b, n_s, n_s, h)  # b. n_s*n_s, h: x1x2x3...x1x2x3...x1x2x3...
        x_ij = torch.cat([x_i, x_j], dim=-1)
        x_ij = self.rel_mlp(x_ij)
        diag_mask = torch.eye(n_s, n_s).reshape(1, n_s, n_s, 1).expand(b, n_s, n_s, 1).to(x_ij.device) > 0.5
        x_ij = x_ij.masked_fill(diag_mask, 0)
        x_ij = x_ij.sum(dim=2)/(n_s-1)
        assert(x_ij.shape==(b, n_s, h)), f"x_rel's shape is {x_ij.shape}"
        return x_ij

    def forward(self, inputs, num_slots = None, num_iters = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        n_it = num_iters if num_iters is not None else self.iters
        
        obj_mu = self.slots_mu.expand(b, n_s, -1)
        obj_sigma = torch.exp(self.slots_sigma.expand(b, n_s, -1))
        obj_slots = obj_mu + torch.randn_like(obj_mu)*obj_sigma

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs) # batch, slot, dim

        attns = []

        for _ in range(n_it):
            # attention pooling from image source
            slots_prev = obj_slots
            obj_slots = self.norm_obj_slots(obj_slots)
            q = self.to_q(obj_slots)
            # q = torch.cat(q.split(dim_split, 2), 0) # head, batch, slot, dim

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # batch, slot, image loc
            if self.gumbel_attention:
                noise = distributions.Gumbel(0, 1).sample(dots.shape).to(dots.device)
                dots = dots + noise
            attn = dots.softmax(dim=1) + self.eps
            attns.append(attn)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn) # batch, slot, dim

            # aggregate message from edge slots
            rel_msg = self._rel_msg(slots_prev)

            # update object slots
            obj_slots = self.gru(
                torch.cat([updates, rel_msg], dim=-1).reshape(b*n_s, 2*d),
                slots_prev.reshape(b*n_s, d)
              ).reshape(b, n_s, d)
            obj_slots = obj_slots + self.obj_mlp(self.norm_pre_ff(obj_slots))

        return obj_slots, attns

class NonlinearSpreadOut(nn.Module):
    def __init__(self, margin=0.5):
        super(NonlinearSpreadOut, self).__init__();
        self.margin = margin;

    def forward(self, x):
        b, n, h = x.shape
        x_norm = F.normalize(x, dim=-1)
        gram = torch.matmul(x_norm, x_norm.t(-2, -1));
        mask = (torch.eye(n)<0.5).unsqueeze(0).expand(b, n, n)
        loss = F.relu(-self.margin+torch.masked_select(gram)).mean()
        return loss
        
"""
Similarity Scores
"""

class Scorer(nn.Module):
    def __init__(self):
        super(Scorer, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError

    def batchwise_score(self, x, y):
        raise NotImplementedError

class DotPScorer(Scorer):
    def __init__(self):
        super(DotPScorer, self).__init__()

    def score(self, x, y):
        return torch.sum(x * y, dim=1)

    def batchwise_score(self, y, x):
        # REVERSED
        bw_scores = torch.einsum('ijk,ik->ij', (x, y))
        return torch.sum(bw_scores, dim=1)

class CosineScorer(Scorer):
    def __init__(self, temperature):
        super(CosineScorer, self).__init__()
        self.temperature = temperature

    def score(self, x, y, input_is_normed=False, get_diag=True):
        if not input_is_normed:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        if get_diag:
            return torch.sum(x * y, dim=-1)/self.temperature
        else:
            # batch_size x num_obj_x x dim, batch_size x num_obj_y x dim --> batch_size x num_obj_x x num_obj_y
            return torch.matmul(x, y.transpose(-1, -2))/self.temperature
    
    def score_im_s(self, im, hint, input_is_normed=False):
        assert(len(im.shape)==3), "Image tensor should be of size (N, n_ex, hidden_size)"
        assert(len(hint.shape)==2), "Hint tensor should be of size (N, hidden_size)"
        N, n_ex, hidden_size = im.shape
        if not input_is_normed:
            im = F.normalize(im, p=2, dim=-1)
            hint = F.normalize(hint, p=2, dim=-1)
        im_flat = im.reshape(N*n_ex, hidden_size)
        scores = torch.matmul(im_flat, hint.t())
        assert(scores.shape[0]==N*n_ex and scores.shape[1]==N), "The size of scores should be of size NxNx(n_ex)"
        return scores/self.temperature

    def score_im_im(self, im, input_is_normed=False):
        assert(len(im.shape)==3), "Image tensor should be of size (N, n_ex, hidden_size)"
        N, n_ex, hidden_size = im.shape
        if not input_is_normed:
            im = F.normalize(im, p=2, dim=-1)
        im_flat = im.reshape(N*n_ex, hidden_size)
        scores = torch.matmul(im_flat, im_flat.t())
        # positive pairs are diagonal blocks of n_ex
        assert(scores.shape[0]==scores.shape[1]==N*n_ex), "The size of scores should be of size (N*n_ex)x(N*n_ex)"
        assert(torch.all(torch.eq(scores, scores.t()))), "The score matrix should be symmetric"
        return scores/self.temperature

class BilinearScorer(DotPScorer):
    def __init__(self, hidden_size, dropout=0.0, identity_debug=False):
        super(BilinearScorer, self).__init__()
        self.bilinear = nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.eye_(self.bilinear.weight)
        self.dropout_p = dropout
        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = lambda x: x
        if identity_debug:
            # Set this as identity matrix to make sure we get the same output
            # as DotPScorer
            self.bilinear.weight = nn.Parameter(
                torch.eye(hidden_size, dtype=torch.float32))
            self.bilinear.weight.requires_grad = False

    def score(self, x, y):
        wy = self.bilinear(y)
        wy = self.dropout(wy)
        return super(BilinearScorer, self).score(x, wy)

    def batchwise_score(self, x, y):
        """
        x: (batch_size, h)
        y: (batch_size, n_examples, h)
        """
        batch_size, n_examples, h = y.shape
        wy = self.bilinear(y.view(batch_size * n_examples,
                                  -1)).unsqueeze(1).view_as(y)
        wy = self.dropout(wy)
        # wy: (batch_size, n_examples, h)
        return super(BilinearScorer, self).batchwise_score(x, wy)

class SinkhornScorer(Scorer):
    def __init__(self, hidden_dim=None, iters=20, reg=0.1, cross_domain_weight=0.5, comparison='im_lang', im_blocks=[6, 30], im_dustbin=None, partial_mass=0, **kwargs):
        super(SinkhornScorer, self).__init__()
        assert(comparison in ['eval', 'im_im', 'im_lang'])
        self.cross_domain_weight = cross_domain_weight
        self.comparison = comparison
        self.im_blocks = im_blocks
        self.partial_mass = partial_mass
        self.temperature = kwargs['temperature']
        if (self.comparison=='im_lang'):
            if im_blocks is None:
                self.dustbin_scorer_im = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            else:
                self.dustbin_scorer_im = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)), 
                                                        nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))])
            self.dustbin_scorer_lang = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        elif (self.comparison=='im_im'):
            if im_blocks is None:
                self.dustbin_scorer_im = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            else:
                self.dustbin_scorer_im = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)), 
                                                        nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))])
        elif (self.comparison=='eval'):
            if im_dustbin is not None:
                self.dustbin_scorer_im = im_dustbin
            else:
                self.dustbin_scorer_im = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.base_scorer = CosineScorer(temperature=1)
        self.clip_dustbin = nn.Tanh()
        self.iters = iters
        self.reg = reg

    def forward(self, *args, **kwargs):
        if self.comparison=='im_lang':
            return self.forward_im_lang(*args, **kwargs)
        elif self.comparison=='im_im':
            return self.forward_im_im(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)
    
    def forward_eval(self, x, y):
        b, n_x, h = x.shape
        assert(y.shape[0]==b and y.shape[2]==h), f'{x.shape}, {y.shape}'
        n_y = y.shape[1]
        scores = self.base_scorer.score(x, y, get_diag=False)
        assert(scores.shape==(b, n_x, n_y)), f"scores's shape is wrong: {scores.shape}"
        # pad the score matrix where language is special token

        if self.im_blocks is not None:
            x_split = torch.split(x, self.im_blocks, dim=1)
            dustbin_im_x = torch.cat([self.dustbin_scorer_im[0](x_split[0]), self.dustbin_scorer_im[1](x_split[1])], dim=1)
            y_split = torch.split(y, self.im_blocks, dim=1)
            dustbin_im_y = torch.cat([self.dustbin_scorer_im[0](y_split[0]), self.dustbin_scorer_im[1](y_split[1])], dim=1)
        else:
            dustbin_im_x = self.dustbin_scorer_im(x)
            dustbin_im_y = self.dustbin_scorer_im(y)

        matching, scores = self.log_optimal_transport(scores, alpha_x=self.clip_dustbin(dustbin_im_x), \
                                                              alpha_y=self.clip_dustbin(dustbin_im_y), \
                                                              alpha_both=-100*torch.ones(1).to(scores.device), \
                                                              iters=self.iters)

        assert(matching.shape==(b, n_x+1, n_y+1)), f"{matching.shape}"
        
        return matching, scores

    def forward_im_im(self, x, n_shot):
        b, n_s, h = x.shape
        x = x.reshape(b, n_s, h)
        x_expand = torch.repeat_interleave(x, repeats=b, dim=0) # --> [x1], [x1], [x1], ... [x2], [x2], [x2], ... [xn], [xn], [xn
        y_expand = x.repeat(b, 1, 1)  # --> y1, y2, ... yn, y1, y2, ... yn, y1, y2, ... yn
        scores = self.base_scorer.score(x_expand, y_expand, get_diag=False)
        assert(scores.shape==((b)**2, n_s, n_s)), f"scores's shape is wrong: {scores.shape}"

        if self.im_blocks is not None:
            x_split = torch.split(x_expand, self.im_blocks, dim=1)
            dustbin_im_x = torch.cat([self.dustbin_scorer_im[0](x_split[0]), self.dustbin_scorer_im[1](x_split[1])], dim=1)
            y_split = torch.split(y_expand, self.im_blocks, dim=1)
            dustbin_im_y = torch.cat([self.dustbin_scorer_im[0](y_split[0]), self.dustbin_scorer_im[1](y_split[1])], dim=1)
        else:
            dustbin_im_x = self.dustbin_scorer_im(x_expand)
            dustbin_im_y = self.dustbin_scorer_im(y_expand)

        matching, scores = self.log_optimal_transport(scores, alpha_x=self.clip_dustbin(dustbin_im_x), \
                                                              alpha_y=self.clip_dustbin(dustbin_im_y), \
                                                              alpha_both=-100*torch.ones(1).to(scores.device), \
                                                              iters=self.iters)

        assert(matching.shape==((b)**2, n_s+1, n_s+1)), f"{matching.shape}"
        scores = scores.reshape(b, b)
        matching = matching.reshape(b, b, n_s+1, n_s+1)

        metric = {}
        mask = torch.eye(b)
        pos_mask = torch.as_tensor((mask-torch.eye(b))>.5).to(scores.device)
        neg_mask = torch.as_tensor(mask<.5).to(scores.device)
        pos_scores = scores.masked_select(pos_mask)
        neg_scores = scores.masked_select(neg_mask)

        assert(pos_scores.shape[0]==b);
        assert(neg_scores.shape[0]==(b-1)*b);

        neg_scores = neg_scores.reshape(b, b-1);
        all_scores_im_im = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1);

        loss = - F.log_softmax(all_scores_im_im, dim=1)[:,0].mean()

        # average R@1 scores for image and text retrieval
        metric['acc'] = (torch.argmax(all_scores_im_im, dim=1)==0).float().mean().item()
        metric['pos_score'] = pos_scores.mean().item()
        metric['neg_score'] = neg_scores.mean().item()
        return matching, loss, metric

    def forward_im_lang(self, x, y, y_mask=None):
        # x.shape = batch_size, num_obj_x, h 
        # y.shape = batch_size, num_obj_y, h 
        # y_mask.shape = batch_size, num_obj_y
        n = y.shape[0]
        assert(x.shape[0]==n)
        assert(y_mask is None or y_mask.shape==y.shape[:2])
        x_expand = torch.repeat_interleave(x, repeats=n, dim=0) # --> [x1], [x1], [x1], ... [x2], [x2], [x2], ... [xn], [xn], [xn
        y_expand = y.repeat(n, 1, 1)  # --> y1, y2, ... yn, y1, y2, ... yn, y1, y2, ... yn
        scores = self.base_scorer.score(x_expand, y_expand, get_diag=False)
        assert(scores.shape==(n**2, x.shape[1], y.shape[1])), f"scores's shape is wrong: {scores.shape}"
        # pad the score matrix where language is special token
        if y_mask is not None:
            y_mask = y_mask.unsqueeze(1).repeat(n, x.shape[1]+1, 1) # the similarity of each image to special language token is -inf
            y_mask = torch.cat([y_mask, (torch.ones(n**2, x.shape[1]+1, 1)<0.5).to(y_mask.device)], dim=2) # append dustbin dimension as FALSE

        if self.im_blocks is not None:
            x_split = torch.split(x_expand, self.im_blocks, dim=1)
            dustbin_im = torch.cat([self.dustbin_scorer_im[0](x_split[0]), self.dustbin_scorer_im[1](x_split[1])], dim=1)
        else:
            dustbin_im = self.dustbin_scorer_im(x_expand)

        matching, scores = self.log_optimal_transport(scores, alpha_x=self.clip_dustbin(dustbin_im), \
                                                              alpha_y=self.clip_dustbin(self.dustbin_scorer_lang(y_expand)), \
                                                              alpha_both=-100*torch.ones(1).to(scores.device), \
                                                              scores_mask=y_mask, iters=self.iters)

        assert(matching.shape==(n**2, x.shape[1]+1, y.shape[1]+1)), f"{matching.shape}"
        scores = scores.reshape(n, n)
        matching = matching.reshape(n, n, x.shape[1]+1, y.shape[1]+1)
        
        metric = {}
        # pos_mask = (torch.block_diag(*([torch.ones(n_ex, 1)]*n))>0.5).to(scores.device)
        pos_mask = (torch.eye(n)>0.5).to(scores.device)
        pos = scores.masked_select(pos_mask)
        neg_by_lang = scores.masked_select(~pos_mask).reshape(n, n-1)
        neg_by_im = (scores.t()).masked_select(~pos_mask.t()).reshape(n, n-1)
        scores_reshaped_by_lang = torch.cat([pos.reshape(n, 1), neg_by_lang], dim=1)
        scores_reshaped_by_im = torch.cat([pos.reshape(n, 1), neg_by_im], dim=1)
        loss = -self.cross_domain_weight*F.log_softmax(scores_reshaped_by_lang, dim=1)[:,0].mean()\
               -(1-self.cross_domain_weight)*F.log_softmax(scores_reshaped_by_im, dim=1)[:,0].mean()

        # average R@1 scores for image and text retrieval
        metric['acc_im_lang'] = (torch.argmax(scores_reshaped_by_lang, dim=1)==0).float().mean().item()
        metric['acc_lang_im'] = (torch.argmax(scores_reshaped_by_im, dim=1)==0).float().mean().item()
        metric['pos_score'] = pos.mean().item()
        assert(torch.isclose(neg_by_im.mean(), neg_by_lang.mean()))
        metric['neg_score'] = neg_by_lang.mean().item()
        return matching, loss, metric
    
    def log_optimal_transport(self, scores, iters: int, alpha_x=None, alpha_y=None, alpha_both=None, scores_mask: torch.BoolTensor=None, use_ipot=False):
        """https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
            Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms = (m*one).to(scores)
        if (scores_mask is not None):
            ns = ((~scores_mask).float().sum(dim=[1, 2]))/(m+1)-1 # -> batch size
        else:
            ns = (n*one).to(scores)
        
        assert((alpha_x is not None)==(alpha_y is not None)==(alpha_both is not None))
        if (alpha_x is not None):
            bins0 = alpha_x.reshape(b, m, 1)
            bins1 = alpha_y.reshape(b, 1, n)
            alpha = alpha_both.expand(b, 1, 1)
            couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)
        else:
            couplings = scores
        mask_val = -1e6

        if (scores_mask is not None):
            couplings = couplings.masked_fill(scores_mask, mask_val)
        
        if (alpha_x is not None):
            if scores_mask is not None:
                norm = - (ms + ns - 2*self.partial_mass).log().unsqueeze(-1) # --> batch size x 1
                log_mu = torch.cat([norm.expand(b, m), (ns-self.partial_mass).log()[:, None] + norm], dim=1) # batch size x num_obj_x+1
            else:
                norm = - (ms + ns - 2*self.partial_mass).log().view(1, 1).expand(b, 1)
                log_mu = torch.cat([norm.expand(b, m), (ns-self.partial_mass).log()[None, None] + norm], dim=1) # batch size x num_obj_x+1
            log_nu = torch.cat([norm.expand(b, n), (ms-self.partial_mass).log()[None, None] + norm], dim=1) # batch size x num_obj_y+1
        else:
            log_mu = -ms.log().reshape(1, 1).expand(b, m)
            log_nu = -ns.log().reshape(1, 1).expand(b, n)

        if (scores_mask is not None):
            log_nu = log_nu.masked_fill(scores_mask[:, 0, :], mask_val)

        if use_ipot:
            Z = self.log_ipot(couplings, log_mu, log_nu, scores_mask, iters)
        else:
            Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, scores_mask, iters)
        if (scores_mask is not None):
            Z = Z-norm.reshape(b, 1, 1)
        Z = Z.exp() 
        
        if (alpha_x is not None):
            final_scores = (scores*Z[:,:-1,:-1]).sum(dim=(1,2))/self.temperature
        else:
            final_scores = (scores*Z).sum(dim=(1,2))/self.temperature
        
        return Z, final_scores

    def log_ipot(self, Z, log_mu, log_nu, scores_mask, iters: int):
        v = log_nu
        T = log_mu.unsqueeze(2) + log_nu.unsqueeze(1)
        A = Z/self.reg
        if scores_mask is not None:
            T = T.masked_fill(scores_mask, -1e6)
            A = A.masked_fill(scores_mask, -1e6)
        for _ in range(self.iters):
            Q = A + T
            u = log_mu - torch.logsumexp(Q + v.unsqueeze(1), dim=2)
            if scores_mask is not None:
                v = log_nu - torch.logsumexp(Q + u.unsqueeze(2) + scores_mask[:,0:1,:].to(Q.dtype)*1e6, dim=1)
                v = v.masked_fill(scores_mask[:,0,:], -1e6)
            else:
                v = log_nu - torch.logsumexp(Q + u.unsqueeze(2), dim=1)
            T = Q + u.unsqueeze(2) + v.unsqueeze(1)
            if scores_mask is not None:
                T = T.masked_fill(scores_mask, -1e6)
        return T

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, scores_mask, iters: int):
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        if scores_mask is not None:
            v = v.masked_fill(scores_mask[:,0,:], -1e6)
        for i in range(iters):
            u += self.reg * (log_mu - torch.logsumexp(self.M(Z, u, v), dim=2))
            if scores_mask is not None:
                v += self.reg * (log_nu - torch.logsumexp(self.M(Z, u, v) + scores_mask[:,0:1,:].to(Z.dtype)*1e6, dim=1))
                v = v.masked_fill(scores_mask[:,0,:], -1e6)
            else:
                v += self.reg * (log_nu - torch.logsumexp(self.M(Z, u, v), dim=1))
        return self.M(Z, u, v)

    def M(self, Z, u, v):
        return (Z + u.unsqueeze(2) + v.unsqueeze(1)) / self.reg

class SetCriterion(nn.Module):
    """
            Taken from DETR, simplified by removing object detection losses
    """
    def __init__(self, num_classes, eos_coef, target_type, pos_cost_weight=1.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            pos_cost_weight: relative weight of the position loss
        """
        super().__init__()
        self.matcher = self.hungarian
        self.num_classes = num_classes
        self.pos_cost_weight = pos_cost_weight
        self.eos_coef = eos_coef
        assert(target_type in ['multihead_single_label', 'multilabel'])
        self.target_type = target_type
        if self.target_type=='multihead_single_label':
            self.empty_weights = []
            for i, n_c in enumerate(self.num_classes):
                empty_weight = torch.ones(n_c)
                empty_weight[-1] = self.eos_coef
                self.register_buffer(f'empty_weight{i}', empty_weight)
                self.empty_weights.append(empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)"""

        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        
        if self.target_type=='multilabel':
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)]).to(src_logits.device)
            target_classes = torch.zeros(src_logits.shape, device=src_logits.device)
            target_classes[idx] = target_classes_o
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes, weight=self.eos_coef+target_classes*(1-self.eos_coef))
            acc = (target_classes.long()==(src_logits>0).long()).float().mean()
            f1 = f1_score(target_classes.long().cpu().numpy().ravel(), (src_logits>0).long().cpu().numpy().ravel())
            metric = {'acc': acc.item(), 'f1': f1}
        else:
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets["labels"], indices)]).to(src_logits.device)
            default = torch.zeros(sum(self.num_classes)).to(src_logits.device)
            for n in np.cumsum(self.num_classes):
                default[n-1] = 1.0
            target_classes = default.reshape(1, 1, -1).expand(src_logits.shape)
            target_classes[idx] = target_classes_o
            src_logits_spl = torch.split(src_logits, self.num_classes, dim=-1)
            target_classes_spl = torch.split(target_classes, self.num_classes, dim=-1)
            target_classes_spl = [torch.argmax(t_c, dim=-1) for t_c in target_classes_spl]
            loss_ce = sum([F.cross_entropy(src_logits_spl[i].transpose(1, 2), target_classes_spl[i], self.empty_weights[i].to(src_logits.device))
                            for i, _ in enumerate(src_logits_spl)])/len(self.num_classes)
            metric = {'acc': (torch.stack([torch.argmax(logit, dim=-1) for logit in src_logits_spl], dim=-1)==torch.stack(target_classes_spl, dim=-1)).float().mean().item()}
        return loss_ce, metric

    def loss_position(self, outputs, targets, indices, num_boxes):
        """Huber Loss"""
        idx = self._get_src_permutation_idx(indices)
        src_pos = outputs['pred_poses']
        target_poses = torch.cat([t[J] for t, (_, J) in zip(targets["poses"], indices)]).to(outputs['pred_poses'].device)
        loss_l1 = F.smooth_l1_loss(src_pos[idx], target_poses)
        return loss_l1

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices 
        # index of sample in batch, repeated by the number of objects in that image
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # the index of selected slot in that batch
        src_idx = torch.cat([src for (src, _) in indices]) 
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: tensor batch_size x num_slot x num_class
             targets: list of tensor, such that len(targets) == batch_size. 
        """
        # Retrieve the matching between the outputs of the last layer and the targets

        assert("pred_logits" in outputs.keys() and "pred_poses" in outputs.keys())
        assert("labels" in targets.keys() and "poses" in targets.keys())

        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_objs = sum(len(t) for t in targets["labels"])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=outputs["pred_logits"].device)

        # Compute all the requested losses
        losses = {}
        loss_cls, metrics = self.loss_labels(outputs, targets, indices, num_objs)
        losses['class'] = loss_cls
        losses['position'] = self.loss_position(outputs, targets, indices, num_objs)
        return losses, metrics

    @ torch.no_grad()
    def hungarian(self, outputs, targets):
        """ 
        adapted from https://github.com/facebookresearch/detr/blob/master/models/matcher.py'
        Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": List of tensor of dim [batch_size, num_slots, num_classes] with the 
                                classification logits for each  classification problem
                 "pred_poses": Tensor of dim [batch_size, num_slots, 2] with the predicted position
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_objects, num_classes] containing the binary indicator of attributes
                        or List of tensors of dim [num_objects] containing the label for each cls problem
                 "poses": Tensor of dim [num_objects, 2] containing the target position
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        sizes = [len(t) for t in targets['poses']]
        n, num_slots, _ = outputs["pred_logits"].shape

        if self.target_type=='multilabel':
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            tgt_ids = torch.cat([v for v in targets['labels']]).to(out_prob.device) # [sum_i num_obj_i] x num_classes
            cost_class = torch.cdist(out_prob, tgt_ids, p=1)
        elif self.target_type=='multihead_single_label':
            out_prob = outputs["pred_logits"].flatten(0, 1)
            # split into chunks for different cls heads, then concat back together
            out_prob = torch.split(out_prob, self.num_classes, dim=-1)
            for o_p in out_prob:
                o_p = o_p.softmax(-1)
            tgt_ids = torch.cat([v for v in targets['labels']]).to(out_prob[0].device)
            target_classes_spl = torch.split(tgt_ids, self.num_classes, dim=-1)
            target_classes_spl = [torch.argmax(t_c, dim=-1) for t_c in target_classes_spl]
            cost_class = -sum([o_p[:,t_c] for o_p, t_c in zip(out_prob, target_classes_spl)])/len(out_prob)
        else:
            raise ValueError('Not a valid target type')

        
        out_pos = outputs["pred_poses"].flatten(0, 1)
        tgt_pos = torch.cat([v for v in targets['poses']]).to(out_pos.device) # [sum_i num_obj_i] x 2
        cost_pos = torch.cdist(out_pos, tgt_pos, p=1)

        cost = cost_class + self.pos_cost_weight*cost_pos
        cost = cost.reshape(n, num_slots, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class TransformerAgg(Scorer):
    def __init__(self, hidden_size):
        super(TransformerAgg, self).__init__()
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, dropout=0.0)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.base_scorer = SinkhornScorer(hidden_size, iters=10, reg=0.1, comparison='eval', temperature=0.1, im_blocks=None)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, n_shot):
        n_way, n_total, num_slot, h = x.shape
        x = x.flatten(0, 1).transpose(0, 1);
        assert(x.shape==(num_slot, n_way*n_total, h))
        x = self.model(x).transpose(0, 1)
        # x.shape = n_way*n_total, h
        x = x.reshape(n_way, n_total, num_slot, self.hidden_size)
        support = x[:, :n_shot].reshape(n_way, n_shot, num_slot, self.hidden_size).flatten(0, 1) # n_way*n_shot, n_s, h
        query = x[:, n_shot:].flatten(0, 1) # n_way*n_query, n_s, h
        support_expanded = support.unsqueeze(1).expand(n_way*n_shot, n_way*(n_total-n_shot), num_slot, h).flatten(0, 1)
        query_expanded = query.unsqueeze(0).expand(n_way*n_shot, n_way*(n_total-n_shot), num_slot, h).flatten(0, 1)
        scores = self.base_scorer(support_expanded, query_expanded)[1]
        scores = scores.reshape(n_way, n_shot, n_way*(n_total-n_shot)).mean(1).t()
        assert(scores.shape==(n_way*(n_total-n_shot), n_way))
        return scores


class ContrastiveLoss(Scorer):
    """
    Compute contrastive loss
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.sim = CosineScorer(temperature)

    def forward(self, im, s):
        # compute image-sentence score matrix

        N = im.shape[0]
        n_ex = im.shape[1]

        scores_im_lang = self.sim.score_im_s(im, s) #--> N x N x n_ex
        mask = torch.block_diag(*([torch.ones(n_ex, 1)]*N))
        pos_mask = mask > .5
        neg_mask = ~pos_mask
        if torch.cuda.is_available():
            pos_mask = pos_mask.cuda()
            neg_mask = neg_mask.cuda()

        positive_scores_im_lang = scores_im_lang.masked_select(pos_mask) # --> N X n_ex, positive pairs 
        negative_scores_im_lang = scores_im_lang.masked_select(neg_mask) # --> (N-1) x N x n_ex, 
        assert(positive_scores_im_lang.shape[0]==N*n_ex)
        assert(negative_scores_im_lang.shape[0]==N*(N-1)*n_ex)

        negative_scores_im_lang = negative_scores_im_lang.reshape(N*n_ex, N-1)

        all_scores_im_lang_by_lang = torch.cat([positive_scores_im_lang.unsqueeze(1), negative_scores_im_lang], dim=1)

        # normalize by hint dimension (push away negative hints from image) and/or normalize by image dimension (push away negative images from hint)
        loss = -F.log_softmax(all_scores_im_lang_by_lang, dim=1)[:,0].mean()

        metric = {}
        metric['pos_score'] = positive_scores_im_lang.mean().item()
        metric['neg_score'] = negative_scores_im_lang.mean().item()
        metric['part_acc'] = (torch.argmax(all_scores_im_lang_by_lang, dim=1)==0).float().mean().item()

        return loss, metric

class MLPMeanScore(Scorer):
    def __init__(self, input_size, output_size, rep_type, blocks):
        super(MLPMeanScore, self).__init__()
        self.mlp1 = MLP(input_size, output_size, output_size)
        assert(rep_type in ['rel', 'slot', 'whole'])
        self.rep_type = rep_type
        if self.rep_type!='whole':
            self.mlp2 = MLP(input_size, output_size, output_size)
            if self.rep_type=='rel':
                self.blocks = blocks
                self.mlp3 = MLP(input_size, output_size, output_size)

    def forward(self, x, y):
        if self.rep_type=='whole':
            assert(len(x.shape)==3), "x should be of shape batch size X n_ex X dim"
            assert(len(y.shape)==2), "y should be of shape batch size X dim"
            assert(x.shape[0]==y.shape[0] and x.shape[-1]==y.shape[-1])
            x = self.mlp1(x).mean(dim=1)
            y = self.mlp1(y)
        elif self.rep_type=='slot':
            assert(len(x.shape)==4), "x should be of shape batch size X n_ex X num_slots X dim"
            assert(len(y.shape)==3), "y should be of shape batch size X num_slots X dim"
            x = self.mlp1(x).mean(dim=2)
            y = self.mlp1(y).mean(dim=2)
            x = self.mlp2(x).mean(dim=1)
            y = self.mlp2(y)
        elif self.rep_type=='rel':
            assert(len(x.shape)==4), "x should be of shape batch size X n_ex X num_slots X dim"
            assert(len(y.shape)==3), "y should be of shape batch size X num_slots X dim"
            x = torch.split(x, self.blocks, dim=2)
            y = torch.split(y, self.blocks, dim=1)
            x_obj = self.mlp1(x[0]).mean(dim=2)
            y_obj = self.mlp1(y[0]).mean(dim=1)
            x_rel = self.mlp2(x[1]).mean(dim=2)
            y_rel = self.mlp2(y[1]).mean(dim=1)
            x = self.mlp3((x_obj+x_rel)/2).mean(dim=1)
            y = self.mlp3((y_obj+y_rel)/2)
        
        assert(x.shape==y.shape)
        return (x*y).sum(dim=1)
