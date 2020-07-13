"""
Models
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils


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
            x_enc = x_enc.reshape(batch_size, n_ex, -1)

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
        return self.model(x_enc)

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
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

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

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden


class MultimodalDeepRep(nn.Module):
    def __init__(self):
        super(MultimodalDeepRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512 * 2), nn.ReLU(),
                                   nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalRep(nn.Module):
    r"""Concat Image and Text representations."""

    def __init__(self):
        super(MultimodalRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalSumExp(nn.Module):
    def forward(self, x, y):
        return x + y


class MultimodalLinearRep(nn.Module):
    def __init__(self):
        super(MultimodalLinearRep, self).__init__()
        self.model = nn.Linear(512 * 2, 512)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


class MultimodalWeightedRep(nn.Module):
    def __init__(self):
        super(MultimodalWeightedRep, self).__init__()
        self.model = nn.Sequential(nn.Linear(512 * 2, 512), nn.ReLU(),
                                   nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        w = self.model(xy)
        out = w * x + (1. - w) * y
        return out


class MultimodalSingleWeightRep(nn.Module):
    def __init__(self):
        super(MultimodalSingleWeightRep, self).__init__()
        self.w = nn.Parameter(torch.normal(torch.zeros(1), 1))

    def forward(self, x, y):
        w = torch.sigmoid(self.w)
        out = w * x + (1. - w) * y
        return out


class TextProposal(nn.Module):
    r"""Reverse proposal model, estimating:

        argmax_lambda log q(w_i|x_1, y_1, ..., x_n, y_n; lambda)

    approximation to the distribution of descriptions.

    Because they use only positive labels, it actually simplifies to

        argmax_lambda log q(w_i|x_1, ..., x_4; lambda)

    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
    """

    def __init__(self, embedding_module):
        super(TextProposal, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.gru = nn.GRU(self.embedding_dim, 512)
        self.outputs2vocab = nn.Linear(512, self.vocab_size)

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
        output_2d = output.view(batch_size * max_length, 512)
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


class EmbedImageRep(nn.Module):
    def __init__(self, z_dim):
        super(EmbedImageRep, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(nn.Linear(self.z_dim, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x):
        return self.model(x)


class EmbedTextRep(nn.Module):
    def __init__(self, z_dim):
        super(EmbedTextRep, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(nn.Linear(self.z_dim, 512), nn.ReLU(),
                                   nn.Linear(512, 512))

    def forward(self, x):
        return self.model(x)

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
        self.bias = torch.nn.Parameter(torch.zeros(1));

    def score(self, x, y):
        return torch.sum(x * y, dim=1) + self.bias;

    def batchwise_score(self, y, x):
        # REVERSED
        bw_scores = torch.einsum('ijk,ik->ij', (x, y))
        return torch.sum(bw_scores, dim=1) + self.bias;

class CosineScorer(Scorer):
    def __init__(self, temperature):
        super(CosineScorer, self).__init__()
        self.temperature = temperature

    def score(self, x, y, input_is_normed=False, get_diag=True):
        if not input_is_normed:
            x = F.normalize(x, p=2, dim=1);
            y = F.normalize(y, p=2, dim=1);
        if get_diag:
            return torch.sum(x * y, dim=1);
        else:
            return torch.mm(x, y.t())/self.temperature;
    
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


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, temperature=0.1, max_violation=False, loss_type="cpc", pairing="im+lang&im+im"):
        super(ContrastiveLoss, self).__init__()
        assert(loss_type in ["cpc", "margin"]), "please select cpc (logit) or margin loss"
        self.loss_type = loss_type;
        self.margin = margin
        self.sim = CosineScorer(temperature);
        self.max_violation = max_violation
        self.pairing = pairing

    def forward(self, im, s):
        # compute image-sentence score matrix

        loss = 0;
        N = im.shape[0];
        n_ex = im.shape[1];

        if (self.pairing in ["im+lang_by_im", "im+lang_by_lang", "im+lang_by_both"]):
            scores_im_lang = self.sim.score_im_s(im, s); #--> N x N x n_ex
            mask = np.kron(np.eye(N), np.ones((n_ex, 1))) #--> block diagonal of ones
            # remove the diagonal
            pos_mask = torch.as_tensor(mask > .5);
            neg_mask = torch.as_tensor(mask < .5);
            if torch.cuda.is_available():
                pos_mask = pos_mask.cuda();
                neg_mask = neg_mask.cuda();

            positive_scores_im_lang = scores_im_lang.masked_select(pos_mask); # --> N X n_ex, positive pairs 
            negative_scores_im_lang_by_lang = scores_im_lang.masked_select(neg_mask); # --> (N-1) x N x n_ex, 
            negative_scores_im_lang_by_im = scores_im_lang.t().masked_select(neg_mask.t());
            assert(positive_scores_im_lang.shape[0]==N*n_ex);
            assert(negative_scores_im_lang_by_lang.shape[0]==negative_scores_im_lang_by_im.shape[0]==N*(N-1)*n_ex);

            negative_scores_im_lang_by_im = negative_scores_im_lang_by_im.reshape(N, (N-1)*n_ex).t();
            negative_scores_im_lang_by_lang = negative_scores_im_lang_by_lang.reshape(N*n_ex, N-1);

            all_scores_im_lang_by_lang = torch.cat([positive_scores_im_lang.unsqueeze(1), negative_scores_im_lang_by_lang], dim=1);
            all_scores_im_lang_by_im = torch.repeat_interleave(negative_scores_im_lang_by_im, repeats=n_ex, dim=1);
            all_scores_im_lang_by_im = torch.cat([positive_scores_im_lang.unsqueeze(0), all_scores_im_lang_by_im], dim=0);

            # normalize by hint dimension (push away negative hints from image) and/or normalize by image dimension (push away negative images from hint)
            if ("_by_lang" in self.pairing):
                loss += -F.log_softmax(all_scores_im_lang_by_lang, dim=1)[:,0].mean();
            elif ("_by_im" in self.pairing):
                loss += -F.log_softmax(all_scores_im_lang_by_im, dim=0)[0,:].mean(); 
            elif ("_by_both" in self.pairing):
                loss += -0.5*F.log_softmax(all_scores_im_lang_by_lang, dim=1)[:,0].mean()\
                        -0.5*F.log_softmax(all_scores_im_lang_by_im, dim=0)[0,:].mean();
            # each is sum over N*n_ex terms

            positive_scores_im_lang = positive_scores_im_lang.mean()
            negative_scores_im_lang = negative_scores_im_lang_by_im.mean();
            assert(torch.allclose(negative_scores_im_lang_by_im.mean(), negative_scores_im_lang_by_lang.mean()));

            best_score_im_lang_by_img = torch.argmax(all_scores_im_lang_by_im, dim=0);
            best_score_im_lang_by_lang = torch.argmax(all_scores_im_lang_by_lang, dim=1);

            if ("_by_lang" in self.pairing):
                acc_im_lang = torch.as_tensor(best_score_im_lang_by_lang==0, dtype=torch.float).mean();
            elif ("_by_im" in self.pairing):
                acc_im_lang = torch.as_tensor(best_score_im_lang_by_img==0, dtype=torch.float).mean();
            elif ("_by_both" in self.pairing):
                acc_im_lang = (0.5*torch.as_tensor(best_score_im_lang_by_img==0, dtype=torch.float)\
                        +0.5*torch.as_tensor(best_score_im_lang_by_lang==0, dtype=torch.float)).mean();

        elif (self.pairing=="im+im"):
            scores_im_im = self.sim.score_im_im(im); # --> (N*n_ex) x (N*n_ex)
            mask = np.kron(np.eye(N), np.ones((n_ex, n_ex))) #--> block diagonal of ones
            # remove the diagonal
            pos_mask = torch.as_tensor(mask-np.eye(N*n_ex) > .5);
            neg_mask = torch.as_tensor(mask < .5);
            if torch.cuda.is_available():
                pos_mask = pos_mask.cuda();
                neg_mask = neg_mask.cuda();
            
            positive_scores_im_im = scores_im_im.masked_select(pos_mask); 
            assert(positive_scores_im_im.shape[0]==N*n_ex*(n_ex-1));
            negative_scores_im_im = scores_im_im.masked_select(neg_mask);
            assert(negative_scores_im_im.shape[0]==(N-1)*N*n_ex**2);

            negative_scores_im_im = negative_scores_im_im.reshape(N*n_ex, (N-1)*n_ex);
            all_scores_im_im = torch.repeat_interleave(negative_scores_im_im, repeats=n_ex-1, dim=0);
            all_scores_im_im = torch.cat([positive_scores_im_im.unsqueeze(1), all_scores_im_im], dim=1);

            normed_scores_im_im = F.log_softmax(all_scores_im_im, dim=1)[:,0].mean();

            positive_scores_im_im = positive_scores_im_im.mean();
            negative_scores_im_im = negative_scores_im_im.mean();

            loss += -normed_scores_im_im;

            best_score_im_im = torch.argmax(all_scores_im_im, dim=1);

            acc_im_im = torch.as_tensor(best_score_im_im==0, dtype=torch.float).mean();

        elif (self.pairing=="im+lang_im+im"):
            scores_im_lang = self.sim.score_im_s(im, s); #--> N x N x n_ex
            mask = np.kron(np.eye(N), np.ones((n_ex, 1))) #--> block diagonal of ones
            # remove the diagonal
            pos_mask = torch.as_tensor(mask > .5);
            neg_mask = torch.as_tensor(mask < .5);
            if torch.cuda.is_available():
                pos_mask = pos_mask.cuda();
                neg_mask = neg_mask.cuda();

            positive_scores_im_lang = scores_im_lang.masked_select(pos_mask); # --> N X n_ex, positive pairs 
            negative_scores_im_lang_by_lang = scores_im_lang.masked_select(neg_mask); # --> (N x n_ex) * N-1, 
            negative_scores_im_lang_by_lang = negative_scores_im_lang_by_lang.reshape(N*n_ex, N-1);

            scores_im_im = self.sim.score_im_im(im); # --> (N*n_ex) x (N*n_ex)
            mask = np.kron(np.eye(N), np.ones((n_ex, n_ex))) #--> block diagonal of ones
            # remove the diagonal
            pos_mask = torch.as_tensor(mask-np.eye(N*n_ex) > .5);
            neg_mask = torch.as_tensor(mask < .5);
            if torch.cuda.is_available():
                pos_mask = pos_mask.cuda();
                neg_mask = neg_mask.cuda();
            
            positive_scores_im_im = scores_im_im.masked_select(pos_mask); 
            assert(positive_scores_im_im.shape[0]==N*n_ex*(n_ex-1)); # --> N x n_ex x n_ex-1
            negative_scores_im_im = scores_im_im.masked_select(neg_mask);
            assert(negative_scores_im_im.shape[0]==(N-1)*N*n_ex**2); # --> (N x n_ex) * ((N-1) x n_ex)
            negative_scores_im_im = negative_scores_im_im.reshape(N*n_ex, (N-1)*n_ex);

            negative_scores_total = torch.cat([negative_scores_im_im, negative_scores_im_lang_by_lang], dim=1);
            negative_scores_total = torch.repeat_interleave(negative_scores_total, dim=0, repeats=n_ex);

            positive_scores_total = torch.cat([positive_scores_im_lang.reshape(N, n_ex), positive_scores_im_im.reshape(N, n_ex*(n_ex-1))], dim=1);
            positive_scores_total = positive_scores_total.reshape(N*n_ex*n_ex, 1);

            all_scores_total = torch.cat([positive_scores_total, negative_scores_total], dim=1);

            loss += -F.log_softmax(all_scores_total, dim=1)[:,0].mean();

            positive_scores_total = positive_scores_total.mean()
            negative_scores_total = negative_scores_total.mean()
            
            best_score_total = torch.argmax(all_scores_total, dim=1);
            acc_total = torch.as_tensor(best_score_total==0, dtype=torch.float).mean();


        if (self.loss_type=="margin"):
            raise NotImplementedError;
        
        if (self.pairing=="im+lang_by_lang" or self.pairing=="im+lang_by_im" or self.pairing=="im+lang_by_both"):
            return loss, \
                positive_scores_im_lang, \
                negative_scores_im_lang, \
                acc_im_lang;
        elif (self.pairing=="im+im"):
            return loss, \
                positive_scores_im_im, \
                negative_scores_im_im, \
                acc_im_im;
        else:
            return loss, \
                positive_scores_total, \
                negative_scores_total, \
                acc_total;
