import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from models import CosineScorer

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
