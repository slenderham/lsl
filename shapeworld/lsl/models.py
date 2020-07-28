"""
Models
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import linear_sum_assignment

def _cartesian_product(x, y):
    return torch.stack([torch.cat([x[i], y[j]], dim=0) for i in range(len(x)) for j in range(len(y))]);

class ExWrapper(nn.Module):
    """
    Wrap around a model and allow training on examples
    i.e. tensor inputs of shape
    (batch_size, n_ex, *img_dims)
    """

    def __init__(self, model):
        super(ExWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 5:
            n_ex = x.shape[1]
            img_dim = x.shape[2:]
            # Flatten out examples first
            x_flat = x.reshape(batch_size * n_ex, *img_dim)
        else:
            x_flat = x

        x_enc = self.model(x_flat)

        if len(x.shape) == 5:
            x_enc = x_enc.reshape(batch_size, n_ex, *x_enc.shape[1:]);

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
        self.tune_backbone = tune_backbone;
        self.normalize_feats = normalize_feats;
        assert((not self.normalize_feats) or (self.normalize_feats and backbone is None)), "only normalize features if backbone is None";
        if backbone is None:
            self.backbone = Identity()
            self.backbone.final_feat_dim = final_feat_dim
        else:
            self.backbone = backbone
        if (hidden_size==None):
            self.model = nn.Identity();
        else:
            self.model = nn.Sequential(
                nn.Linear(self.backbone.final_feat_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        x_enc = self.backbone(x)
        if (self.normalize_feats):
            x_enc = F.normalize(x_enc, dim=-1);
        if (not self.tune_backbone):
            x_enc = x_enc.detach();
        return self.model(x_enc);

class MLP(nn.Module):
    r"""
    MLP projection head
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MLP, self).__init__();
        assert(num_layers>=0), "At least 0 hidden_layers (a simple linear transform)";
        layers = [];
        if (num_layers==0):
            layers.append(nn.Linear(input_size, output_size));
        else:
            layers.append(nn.Linear(input_size, hidden_size));
            layers.append(nn.ReLU());
            for _ in range(num_layers-1):
                layers.append(nn.Linear(hidden_size, hidden_size));
                layers.append(nn.ReLU());
            layers.append(nn.Linear(hidden_size, output_size));

        self.model = nn.Sequential(*layers);

    def forward(self, x):
        return self.model(x);

class TextRep(nn.Module):
    r"""Deterministic Bowman et. al. model to form
    text representation.

    Again, this uses 512 hidden dimensions.
    """

    def __init__(self, embedding_module, hidden_size):
        super(TextRep, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.gru = nn.GRU(self.embedding_dim, hidden_size);

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)

        packed_input = rnn_utils.pack_padded_sequence(embed_seq,
                                                      sorted_lengths)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist()
            if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...];

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

class TextRepTransformer(nn.Module):
    def __init__(self, embedding_module, hidden_size):
        super(TextRepTransformer, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.model = MultilayerTransformer(hidden_size, 4, 4);
        self.pe = TextPositionalEncoding(hidden_size, dropout=0.0, max_len=32);

    def forward(self, seq, padding_mask):
        batch_size = seq.size(0)

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = self.embedding(seq)
        embed_seq = self.pe(embed_seq)
        hidden = self.model(embed_seq, src_key_padding_mask=padding_mask)[0,...];

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

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n; lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4; lambda)

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

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.gru = nn.GRU(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        attns = [];

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attns.append(attn);
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots, _ = self.gru(
                updates.reshape(1, -1, d),
                slots_prev.reshape(1, -1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attns;

class SANet(nn.Module):
    def __init__(self, im_size, num_slots=6, dim=64, iters = 3, eps = 1e-8):
        super(SANet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3),
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3),
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim),
            ImagePositionalEmbedding(im_size-2*4, im_size-2*4, dim)
        );
        self.final_feat_dim=dim;
        self.iters = iters;
        self.num_slots = num_slots;

        self.slot_attn = SlotAttention(num_slots, dim, iters, eps, 2*dim)

    def forward(self, img, visualize_attns=True):
        x = self.encoder(img);
        n, c, h, w = x.shape;
        x = x.permute(0, 2, 3, 1).reshape(n, h*w, c);
        x, attns = self.slot_attn(x); # --> N * num slots * feature size
        if visualize_attns:
            self._visualize_attns(img, attns);
        return x;

    def _visualize_attns(self, img, attns):
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        norm = colors.BoundaryNorm(bounds, cmap.N)

        N, C, H, W = img.shape;
        N, dim_q, dim_k = attns[0].shape; # dim_q=the number of slots, dim_k=size of feature map
        H_k = W_k = math.isqrt(dim_k);
        rand_idx = torch.randint(0, N, size=(1,)).item();
        plt.imshow(img[rand_idx].permute(1, 2, 0).detach().cpu());
        fig, axes = plt.subplots(self.iters, self.num_slots);
        for i in range(self.iters):
            for j in range(self.num_slots):
                im = axes[i][j].imshow(F.interpolate(attns[i][rand_idx][j].reshape(1, 1, H_k, W_k), size=(H, W), mode='nearest').squeeze().detach().cpu());

        fig, axes = plt.subplots(1, self.iters);
        for i in range(self.iters):
            masked_img = torch.zeros(H_k, W_k);
            for h in range(H_k):
                for w in range(W_k):
                    masked_img[h, w] = torch.argmax(attns[i][rand_idx,:,h*W_k+w]);
            axes[i].imshow(masked_img, cmap=cmap, norm=norm);

        plt.show();

class ImagePositionalEmbedding(nn.Module):
    def __init__(self, height, width, hidden_size):
        super(ImagePositionalEmbedding, self).__init__();
        x_coord_pos = torch.linspace(0, 1, height).reshape(1, height, 1).expand(1, height, width);
        x_coord_neg = torch.linspace(1, 0, height).reshape(1, height, 1).expand(1, height, width);
        y_coord_pos = torch.linspace(0, 1, width).reshape(1, 1, width).expand(1, height, width);
        y_coord_neg = torch.linspace(1, 0, width).reshape(1, 1, width).expand(1, height, width);

        self.coords = torch.cat([
            x_coord_pos,
            x_coord_neg,
            y_coord_pos,
            y_coord_neg
        ], dim=0).unsqueeze(0);

        if torch.cuda.is_available():
            self.coords = self.coords.cuda();

        self.pos_emb = nn.Conv2d(4, hidden_size, 1);

    def forward(self, x):
        # add positional embedding to the feature vector
        return x+self.pos_emb(self.coords);

class MultilayerTransformer(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super(MultilayerTransformer, self).__init__();
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=4*dim, dropout=0.0);
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers);

    def forward(self, x):
        return self.transformer(x.transpose(0, 1)).transpose(0, 1);
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
        return torch.sum(x * y, dim=1);

    def batchwise_score(self, y, x):
        # REVERSED
        bw_scores = torch.einsum('ijk,ik->ij', (x, y))
        return torch.sum(bw_scores, dim=1);

class CosineScorer(Scorer):
    def __init__(self, temperature):
        super(CosineScorer, self).__init__()
        self.temperature = temperature

    def score(self, x, y, input_is_normed=False, get_diag=True):
        if not input_is_normed:
            x = F.normalize(x, p=2, dim=1);
            y = F.normalize(y, p=2, dim=1);
        if get_diag:
            return torch.sum(x * y, dim=1)/self.temperature;
        else:
            return torch.mm(x, y.t())/self.temperature;

    def score_sequence(self, x, y, input_is_normed=False):
         if not input_is_normed:
            x = F.normalize(x, p=2, dim=1);
            y = F.normalize(y, p=2, dim=1);
    
    def score_im_s(self, im, hint, input_is_normed=False):
        assert(len(im.shape)==3), "Image tensor should be of size (N, n_ex, hidden_size)";
        assert(len(hint.shape)==2), "Hint tensor should be of size (N, hidden_size)";
        N, n_ex, hidden_size = im.shape;
        if not input_is_normed:
            im = F.normalize(im, p=2, dim=-1);
            hint = F.normalize(hint, p=2, dim=-1);
        im_flat = im.reshape(N*n_ex, hidden_size);
        scores = torch.matmul(im_flat, hint.t());
        assert(scores.shape[0]==N*n_ex and scores.shape[1]==N), "The size of scores should be of size NxNx(n_ex)";
        return scores/self.temperature;

    def score_im_im(self, im, input_is_normed=False):
        assert(len(im.shape)==3), "Image tensor should be of size (N, n_ex, hidden_size)";
        N, n_ex, hidden_size = im.shape;
        if not input_is_normed:
            im = F.normalize(im, p=2, dim=-1);
        im_flat = im.reshape(N*n_ex, hidden_size);
        scores = torch.matmul(im_flat, im_flat.t());
        # positive pairs are diagonal blocks of n_ex
        assert(scores.shape[0]==scores.shape[1]==N*n_ex), "The size of scores should be of size (N*n_ex)x(N*n_ex)";
        assert(torch.all(torch.eq(scores, scores.t()))), "The score matrix should be symmetric";
        return scores/self.temperature;

class BilinearScorer(DotPScorer):
    def __init__(self, hidden_size, dropout=0.0, identity_debug=False):
        super(BilinearScorer, self).__init__()
        self.bilinear = nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.eye_(self.bilinear.weight);
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

class TransformerScorer(Scorer):
    def __init__(self, hidden_size, scorer, get_diag):
        # transformer layer for contextualized embedding of objects
        super(TransformerScorer, self).__init__();
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, dim_feedforward=2*hidden_size, dropout=0.0);
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=2);
        
        # aggregation of all objects after embedded
        self.agg_gate = nn.Linear(hidden_size, hidden_size);
        self.agg = nn.Linear(hidden_size, hidden_size);

        # final scorer on aggregations
        self.scorer = {
            'cosine': CosineScorer(temperature=scorer['temp']),
            'dotp': DotPScorer(),
        }[scorer['name']];

        self.get_diag = get_diag

    def score(self, x, y, y_mask):
        N, n_ex, num_obj_x, hidden_size = x.shape;
        x = x.reshape(N, n_ex*num_obj_x, hidden_size);
        assert(y.shape[0]==N and y.shape[2]==hidden_size)

        x_mask = torch.ones(N, n_ex*num_obj_x)<0.5; # --> N x n_ex*num_obj_x
        if y_mask==None:
            y_mask = torch.ones(y.shape[0], y.shape[1])<0.5 # --> N x num_obj_y

        if torch.cuda.is_available():
            x_mask = x_mask.cuda()
            y_mask = y_mask.cuda()

        if (self.get_diag):
            total_input = torch.cat([x, y], dim=1).transpose(0, 1); # --> (num_obj_x*n_ex+num_obj_y) x N x hidden size
            total_mask = torch.cat([x_mask, y_mask], dim=1); # --> N x (n_ex*num_obj_x + num_obj_y)
        else:
            total_input = self._cartesian_product(x, y).transpose(0, 1); # --> (num_obj_x*n_ex+num_obj_y) x N*N x hidden size
            total_mask = self._cartesian_product(x_mask, y_mask); # --> N*N x (num_obj_x*n_ex+num_obj_y)
            y_mask = y_mask.repeat(N, 1)

        x_y_enc = self.model(total_input); # --> (num_obj_x*n_ex+num_obj_y) x N*N x hidden size
        x_y_enc = torch.sigmoid(self.agg_gate(x_y_enc))*self.agg(x_y_enc);
        x_enc = x_y_enc[:n_ex*num_obj_x];
        y_enc = x_y_enc[n_ex*num_obj_x:];
        y_mask = (~y_mask).transpose(0, 1).unsqueeze(-1).float();
        y_enc = y_enc*y_mask;

        x_agged = torch.mean(x_enc, dim=0); # --> N*N x hidden size
        y_agged = torch.sum(y_enc, dim=0)/y_mask.sum(dim=0); # --> N*N x hidden size
        if (not self.get_diag):
            return self.scorer.score(x_agged, y_agged).reshape(N, N);
        else:
            return self.scorer.score(x_agged, y_agged);

class SinkhornScorer(Scorer):
    def __init__(self, base_scorer, iters=20):
        super(SinkhornScorer, self).__init__();
        self.base_scorer = base_scorer;
        assert(isinstance(self.base_scorer, Scorer)), "base_scorer should be a scorer itself"
        self.dustbin_weights = nn.Parameter(torch.ones(1))
        self.iters = iters

    def score(self, x, y):
        n = x.shape[0]; # x.shape = n, num_obj, h; y.shape = n, num_obj_y, h
        assert(y.shape[0]==n)
        x_expand = torch.repeat_interleave(x, repeats=n, dim=0);
        y_expand = torch.repeat(y, repeats=n, dim=0);
        scores = self.base_scorer(x, y);
        scores = self.log_optimal_transport(scores, self.dustbin_weights, self.iters);
        return scores;
    
    def log_optimal_transport(self, scores, alpha, iters: int):
        """https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
            Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters: int):
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

class SetCriterion(nn.Module):
    """
            Taken from DETR, simplified by removing object detection losses
    """
    def __init__(self, num_classes, eos_coef=0.1, pos_cost_weight=5.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            eos_coef: relative classification weight applied to the no-object category
            pos_cost_weight: relative weight of the position loss
        """
        super().__init__()
        self.matcher = self.hungarian
        self.eos_coef = eos_coef
        self.num_classes = num_classes
        self.pos_cost_weight = pos_cost_weight
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)"""
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        acc = (torch.argmax(src_logits[idx], dim=-1)==target_classes_o).float().mean()

        return loss_ce, acc

    def loss_position(self, outputs, targets, indices, num_boxes):
        """Huber Loss"""
        idx = self._get_src_permutation_idx(indices)
        src_pos = outputs['pred_poses'];
        target_poses = torch.cat([t['poses'][J] for t, (_, J) in zip(targets, indices)]).to(outputs.device)
        loss_l1 = F.smooth_l1_loss(src_pos[idx], target_poses);
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
        assert("labels" in targets.keys() and "poses" in targets.keys());

        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_classes = sum(len(t) for t in targets)
        num_classes = torch.as_tensor([num_classes], dtype=torch.float, device=outputs.device)

        # Compute all the requested losses
        losses = {};
        loss_cls, acc = self.loss_labels(outputs, targets, indices, num_classes);
        losses['class'] = loss_cls;
        losses['position'] = self.loss_position(outputs, targets, indices, num_classes);
        return losses, acc

    @ torch.no_grad()
    def hungarian(self, outputs, targets):
        """ 
        adapted from https://github.com/facebookresearch/detr/blob/master/models/matcher.py'
        Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_poses": Tensor of dim [batch_size, num_queries, 2] with the predicted position
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "poses": Tensor of dim [num_target_boxes, 2] containing the target position
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        n, num_slots, num_classes = output["pred_logits"].shape;
        sizes = [len(t) for t in target['labels']];

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_pos = outputs["pred_poses"].flatten(0, 1)

        tgt_ids = torch.cat([v for v in targets['labels']]).long()
        tgt_pos = torch.cat([v for v in targets['poses']])

        cost_class = -out_prob[:, tgt_ids]; # get the probability of each class
        cost_pos = F.smooth_l1_loss(out_pos, tgt_pos);

        cost = cost_class + self.pos_cost_weight*cost_pos;
        cost = cost.reshape(n, num_slots, -1).cpu();

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))];
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
