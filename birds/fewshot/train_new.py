"""
Training script
"""

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import constants
from data import lang_utils
from data.datamgr import SetDataManager
from io_utils import get_resume_file, model_dict, parse_args
from models import *
from vision import Conv4NP, ResNet18, Conv4
from utils import GradualWarmupScheduler, save_defaultdict_to_fs, save_checkpoint, AverageMeter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Output directory')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='Size of hidden representations')
    parser.add_argument('--num_slots', 
                        type=int,
                        default=6,
                        help='Number of slots')
    parser.add_argument('--comparison',
                        choices=['dotp', 'cosine', 'transformer'],
                        default='transformer',
                        help='How to compare support to query reps')
    parser.add_argument('--freeze_slots',
                        action='store_true',
                        help='If True, freeze slots.')
    parser.add_argument('--hypo_model',
                        choices=['uni_gru', 'bi_gru', 'uni_transformer', 'bi_transformer'],
                        default='bi_gru',
                        help='Which language model to use for ')
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Max number of training examples')
    parser.add_argument('--aux_task',
                        type=str,
                        choices=['caption_slot', 'caption_image', 'matching_slot', 'matching_image'],
                        default='matching',
                        help='Whether to predict caption or predict objects')
    parser.add_argument('--visualize_attns',
                        action='store_true',
                        help='If true, visualize attention masks of slots and matching/caption if applicable')
    parser.add_argument('--temperature',
                        default=0.5,
                        type=float,
                        help='Temperature parameter used in contrastive loss')
    parser.add_argument('--pt_epochs', type=int, default=150, help='Pretrain epochs')
    parser.add_argument('--ft_epochs', type=int, default=20, help='Finetune epochs')
    parser.add_argument('--debug_example', 
                        action="store_true",
                        help="If true, print out example images and hint")
    parser.add_argument('--skip_eval',
                        action="store_true",
                        help="If true, skip the zero shot evaluation and only save the pretrained features.")
    parser.add_argument('--data_dir',
                        default=None,
                        help='Specify custom data directory (must have shapeworld folder)')
    parser.add_argument('--pt_lr',
                        type=float,
                        default=0.001,
                        help='Pretrain Learning rate')
    parser.add_argument('--ft_lr',
                        type=float,
                        default=0.001,
                        help='Finetuning Learning rate')
    parser.add_argument('--optimizer',
                        choices=['adam', 'rmsprop', 'sgd'],
                        default='adam',
                        help='Optimizer to use')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument("--language_filter", 
                        default="all", 
                        choices=["all", "color", "nocolor"])
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='How often to log loss')
    parser.add_argument('--save_checkpoint',
                        action='store_true',
                        help='Save model')
    parser.add_argument('--load_checkpoint',
                        action='store_true',
                        help='Load model')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA training')
    parser.add_argument("--glove_init", action="store_true")
    parser.add_argument("--freeze_emb", action="store_true")
    parser.add_argument("--scramble_lang", action="store_true")
    parser.add_argument("--sample_class_lang", action="store_true")
    parser.add_argument("--scramble_all", action="store_true")
    parser.add_argument("--shuffle_lang", action="store_true")
    parser.add_argument("--scramble_lang_class", action="store_true")
    parser.add_argument("--n_caption", choices=list(range(1, 11)), type=int, default=1)
    parser.add_argument("--max_class", type=int, default=None)
    parser.add_argument("--max_img_per_class", type=int, default=None)
    parser.add_argument("--max_lang_per_class", type=int, default=None)
    parser.add_argument("--lang_emb_size", type=int, default=300)
    parser.add_argument("--lang_hidden_size", type=int, default=200)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--test_n_way", type=int, default=5, help="Specify to change n_way eval at test")
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    base_file = os.path.join(constants.DATA_DIR, "base.json")
    val_file = os.path.join(constants.DATA_DIR, "val.json")

    # Load language
    vocab = lang_utils.load_vocab(constants.LANG_DIR)
    special_idx = lang_utils.get_special_indices(vocab)
    train_i2w = lang_utils.load_idx_to_word(vocab)
    train_vocab_size = len(vocab) # get largest token index value as size of vocab

    n_query = max(1, int(16 * args.test_n_way / args.n_way))

    train_few_shot_args = dict(n_way=args.n_way, n_support=args.n_shot)
    base_datamgr = SetDataManager(
        "CUB", 84, n_query=n_query, **train_few_shot_args, args=args
    )
    print("Loading train data")

    base_loader = base_datamgr.get_data_loader(
        base_file,
        aug=True,
        lang_dir=constants.LANG_DIR,
        normalize=True,
        vocab=vocab,
        # Maximum training data restrictions only apply at train time
        max_class=args.max_class,
        max_img_per_class=args.max_img_per_class,
        max_lang_per_class=args.max_lang_per_class,
    )

    val_datamgr = SetDataManager(
        "CUB",
        84,
        n_query=n_query,
        n_way=args.test_n_way,
        n_support=args.n_shot,
        args=args,
    )
    print("Loading val data")
    val_loader = val_datamgr.get_data_loader(
        val_file, aug=False, lang_dir=constants.LANG_DIR, normalize=True, vocab=vocab,
    )
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor


    ''' vision '''
    # if _image in task name, get vector for each image with conv net else get set of vectors with slots
    image_model = 'conv' if '_image' in args.aux_task else 'slot_attn'
    backbone_model = SANet(im_size=84, num_slots=args.num_slots, dim=args.hidden_size, slot_model=image_model)
    image_part_model = ExWrapper(backbone_model).to(device)
    params_to_pretrain = list(image_part_model.parameters())
    models_to_save = [image_part_model]

    # abuse of variable name here. This is just to project to the correct dimension

    ''' scorer '''
    im_im_scorer_model = TransformerAgg(args.hidden_size).to(device)
    params_to_finetune = list(im_im_scorer_model.parameters())
    models_to_save.append(im_im_scorer_model)

    ''' aux task specific '''
    if args.aux_task=='caption_slot' or args.aux_task=='caption_image':
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        if args.glove_init:
            vecs = lang_utils.glove_init(vocab, emb_size=args.hidden_size)
        embedding_model = nn.Embedding(
            train_vocab_size, args.hidden_size, _weight=vecs if args.glove_init else None
        )
        if args.freeze_emb:
            embedding_model.weight.requires_grad = False
        
        if args.aux_task=='caption_slot':
            hint_model = TextProposalWithAttn(embedding_model, encoder_dim=args.hidden_size, hidden_size=args.hidden_size, )
        else:
            hint_model = TextProposal(embedding_model, hidden_size=args.hidden_size)
        hint_model = hint_model.to(device)
        params_to_pretrain.extend(hint_model.parameters())
        models_to_save.append(hint_model)

    elif args.aux_task=='matching_slot' or args.aux_task=='matching_image':
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        hint_model = {
                        'uni_gru': TextRep(embedding_model, hidden_size=args.hidden_size, bidirectional=False, return_agg=args.aux_task=='matching_image'),
                        'bi_gru': TextRep(embedding_model, hidden_size=args.hidden_size, bidirectional=True, return_agg=args.aux_task=='matching_image'),
                        'uni_transformer': TextRepTransformer(embedding_model, hidden_size=args.hidden_size, bidirectional=False, return_agg=args.aux_task=='matching_image'),
                        'bi_transformer': TextRepTransformer(embedding_model, hidden_size=args.hidden_size, bidirectional=True, return_agg=args.aux_task=='matching_image')
                     }[args.hypo_model]
        hint_model = hint_model.to(device)
        params_to_pretrain.extend(hint_model.parameters())
        models_to_save.append(hint_model)

    else:
        raise ValueError('invalid auxiliary task name')

    # loss
    if args.aux_task=='matching_slot':
        hype_loss = SinkhornScorer(idx_to_word=train_i2w, temperature=args.temperature).to(device)
        params_to_pretrain.extend(hype_loss.parameters())
        models_to_save.append(hype_loss)
    elif args.aux_task=='matching_image':
        hype_loss = ContrastiveLoss(temperature=args.temperature)
        params_to_pretrain.extend(hype_loss.parameters())
        models_to_save.append(hype_loss)

    # optimizer
    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }[args.optimizer]
    pretrain_optimizer = optfunc(params_to_pretrain, lr=args.pt_lr)
    finetune_optimizer = optfunc(params_to_finetune, lr=args.ft_lr)
    # models_to_save.append(optimizer)
    after_scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, 5000, 0.5)
    scheduler = GradualWarmupScheduler(pretrain_optimizer, 1.0, total_epoch=100*args.pt_epochs//10, after_scheduler=after_scheduler)
    print(sum([p.numel() for p in params_to_pretrain]))
    print(sum([p.numel() for p in params_to_finetune]))

    if args.load_checkpoint and os.path.exists(os.path.join(args.exp_dir, 'checkpoint.pth.tar')):
        ckpt_path = os.path.join(args.exp_dir, 'checkpoint.pth.tar')
        sds = torch.load(ckpt_path, map_location=device)
        for m in models_to_save:
            if (not isinstance(m, TransformerAgg)):
                print(m.load_state_dict(sds[repr(m)]))
        print("loaded checkpoint")

    def pretrain(epoch, n_steps=100):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.train()

        aux_loss_total = 0
        cls_acc = 0
        pbar = tqdm(total=n_steps)
        for batch_idx, (x, target, (lang, lang_length, lang_mask)) in enumerate(base_loader):
            # x size: [n_way, n_support + n_query, dim, w, h] 
            n_way = x.size(0)
            n_query = x.size(1) - args.n_shot
            x = x.to(device)
            target = target.to(device)

            # load hint, size [n_way, n_support + n_query, length] 
            max_hint_length = lang_length.max()
            hint_seq = lang[:, :, :max_hint_length]
            hint_mask = lang_mask[:, :, :max_hint_length]
            hint_seq = hint_seq.to(device)
            hint_length = lang_length.to(device)
            hint_mask = hint_mask.to(device)

            if args.debug_example:
                rand_idx = np.random.randint(0, x.size(1)) # sample a random index from current batch
                print([train_i2w[k.item()] for k in lang[rand_idx]]) # get hint in words
                print(target[rand_idx])
                fig, axes = plt.subplots(5)
                for i in range(4):
                    axes[i].imshow(x[rand_idx][i].permute(1, 2, 0)) # plot examples, transpose to put channel in the last dim
                    axes[i].axis('off')
                axes[4].imshow(x[rand_idx].permute(1, 2, 0))
                axes[4].axis('off')
                plt.show()

            # Learn representations of images
            # flatten the n_way and n_support+n_query dimensions
            image_slot = image_part_model(x, is_ex=True, visualize_attns=args.visualize_attns)
            
            if args.aux_task=='caption_slot' or args.aux_task=='caption_image':
                n_total = image_slot.shape[1]
                hint_seq = hint_seq.reshape(n_way * n_total, max_hint_length);
                hint_mask = hint_mask.reshape(n_way * n_total, -1)
                hint_length = hint_length.reshape(n_way * n_total)
                if (args.aux_task=='caption_slot'):
                    assert(len(image_slot.shape)==4), "The examples_full should have shape: n_way X (n_support+n_query) X num_slots X dim"
                    image_slot = image_slot.reshape(n_way * n_total, args.num_slots, args.hidden_size);
                    hypo_out, attns = hint_model(image_slot, hint_seq, hint_length)
                else:
                    assert(len(image_slot.shape)==3), "The examples_full should be of shape: batch_size X n_ex, X dim"
                    image_slot = image_slot.reshape(n_way * n_total, args.hidden_size);
                    hypo_out, attns = hint_model(image_slot, hint_seq, hint_length)

                seq_len = hint_seq.size(1)
                
                if (args.visualize_attns):
                    if (args.aux_task=='caption_slot'):
                        plt.subplot(111).imshow(attns[2].detach().t())
                    elif(args.aux_task=='caption_image'):
                        fig, axes = plt.subplots(2, 7)
                        for i, a in enumerate(attns[2].detach()):
                            axes[i//7][i%7].imshow(a.reshape(56, 56))
                    print([train_i2w[h.item()] for h in torch.argmax(hypo_out[2], dim=-1)])
                    print([train_i2w[h.item()] for h in hint_seq[2]])
                    plt.show()

                hypo_nofinal = hypo_out[:, :-1].contiguous()
                lang_nostart = hint_seq[:, 1:].contiguous()
                mask_nostart = hint_mask[:, 1:].contiguous()
                hyp_batch_size = n_way * n_total;

                hypo_nofinal_2d = hypo_nofinal.view(hyp_batch_size * (seq_len - 1), -1)
                lang_nostart_2d = lang_nostart.long().view(hyp_batch_size * (seq_len - 1))
                hypo_loss = F.cross_entropy(hypo_nofinal_2d, lang_nostart_2d, reduction="none")
                hypo_loss = hypo_loss.view(hyp_batch_size, (seq_len - 1))
                # Mask out sequences based on length
                hypo_loss.masked_fill_(mask_nostart, 0.0)
                # Sum over timesteps / divide by length
                hypo_loss_per_sentence = torch.div(
                    hypo_loss.sum(dim=1), (lang_length - 1).float()
                )
                hypo_loss = hypo_loss_per_sentence.mean()
                loss = hypo_loss

                hypo_pred = torch.argmax(hypo_nofinal_2d, dim=-1).masked_select(mask_nostart)
                hypo_gt = lang_nostart_2d.masked_select(mask_nostart)
                metric = {'acc': (hypo_pred==hypo_gt).float().mean().item()} 
                aux_loss_total += hypo_loss.item()
                cls_acc += metric['acc']
            elif args.aux_task=='matching_slot' or args.aux_task=='matching_image':
                n_total = image_slot.shape[1]
                hint_seq = hint_seq.flatten(0, 1);
                hint_mask = hint_mask.flatten(0, 1);
                hint_length = hint_length.flatten(0, 1);
                if ('transformer' in args.hypo_model):
                    hint_rep = hint_model(hint_seq, hint_length, hint_mask) 
                else:
                    hint_rep = hint_model(hint_seq, hint_length) 

                if (args.aux_task=='matching_slot'):
                    assert(len(image_slot.shape)==4), "The examples_full should have shape: batch_size X n_ex X (num_slots or ) X dim"
                    assert(hint_rep.shape==(n_way * n_total, max_hint_length, args.hidden_size))
                    matching, hypo_loss, metric = hype_loss(x=image_slot.flatten(0, 1), y=hint_rep, word_idx=hint_seq, \
                                    y_mask=((hint_seq==special_idx["sos_index"]) | \
                                            (hint_seq==special_idx["eos_index"]) | \
                                            (hint_seq==special_idx["pad_index"])));
                else:
                    assert(len(image_slot.shape)==3), "The examples_full should be of shape: batch_size X n_ex X dim"
                    assert(hint_rep.shape==(n_way * n_total, args.hidden_size))
                    hypo_loss, metric = hype_loss(im=image_slot, s=hint_rep)
                
                if args.visualize_attns:
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    im = ax.imshow(matching[2][0].detach().cpu(), vmin=0, vmax=1)
                    ylabels = list(range(args.num_slots))
                    ax.set_xticks(np.arange(len(hint_seq[0])))
                    ax.set_xticklabels([train_i2w[h.item()] for h in hint_seq[0]], rotation=45)
                    ax.set_yticks(np.arange(len(ylabels)))
                    ax.set_yticklabels(ylabels)
                    ax.set_aspect('auto')
                    fig.colorbar(im, ax=ax)
                    plt.show()

                loss = hypo_loss
                aux_loss_total += hypo_loss.item()
                cls_acc += metric['part_acc']
            else:
                raise ValueError("invalid auxiliary task name")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_pretrain, 1.0)
            pretrain_optimizer.step()
            scheduler.step()
            pretrain_optimizer.zero_grad()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f} Metric: {}'.format(
                    epoch, loss.item(), [(k, "{:.6f}".format(v)) for k,v in metric.items()]))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tAuxiliary Loss: {:.4f} Auxiliary Acc: {:.4f}'.format('(train)', epoch, aux_loss_total, cls_acc))
        return loss, metric

    def finetune(epoch, n_steps=100):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.train()
            if (args.freeze_slots and (isinstance(m, ExWrapper))):
                m.eval()

        main_acc = 0
        pred_loss_total = 0
        pbar = tqdm(total=n_steps)
        for batch_idx, (x, target, (lang, lang_length, lang_mask)) in enumerate(base_loader):
            # x size: [n_way, n_support + n_query, dim, w, h] 
            n_way = x.size(0)
            n_query = x.size(1) - args.n_shot
            x = x.to(device)
            target = target.to(device)

            # Learn representations of images and examples
            image_slot = image_part_model(x, is_ex=True, visualize_attns=args.visualize_attns)

            if "_image" in args.aux_task:
                image_slot = image_slot.reshape(n_way, n_query+args.n_shot, 1, args.hidden_size)

            score = im_im_scorer_model(image_slot, args.n_shot).squeeze() # this will be of size (n_way*n_query, n_way)
            y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).to(device)
            loss = F.cross_entropy(score, y_query)
            pred_loss_total += loss.item()
            main_acc += (torch.argmax(score)==y_query).float().mean().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_finetune, 1.0)
            finetune_optimizer.step()
            finetune_optimizer.zero_grad()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f}'.format(
                    epoch, loss.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tConcept Loss: {:.4f} Concept Acc: {:.4f}'.format('(train)', epoch, pred_loss_total, main_acc))
        return loss.item()

    def test(epoch):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.eval()

        concept_avg_meter = AverageMeter(raw=True)

        with torch.no_grad():
            for batch_idx, (x, target, (lang, lang_length, lang_mask)) in enumerate(val_loader):
                # x size: [n_way, n_support + n_query, dim, w, h] 
                n_way = x.size(0)
                n_query = x.size(1) - args.n_shot
                x = x.to(device)
                target = target.to(device)

                # Learn representations of images and examples
                image_slot = image_part_model(x, is_ex=True, visualize_attns=args.visualize_attns)

                if "_image" in args.aux_task:
                    image_slot = image_slot.reshape(n_way, n_query+args.n_shot, 1, args.hidden_size)

                score = im_im_scorer_model(image_slot, args.n_shot).squeeze() # this will be of size (n_way*n_query, n_way)
                y_hat = torch.argmax(score, -1).cpu().numpy()
                y_query = np.repeat(range(n_way), n_query).astype(np.uint8)
                accuracy = accuracy_score(y_query, y_hat)
                concept_avg_meter.update(accuracy, n_way*n_query, raw_scores=(y_hat==y_query))
        
        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}'.format(
            '(test)', epoch, concept_avg_meter.avg))

        return concept_avg_meter.avg, concept_avg_meter.raw_scores

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    for epoch in range(1, args.pt_epochs+1):
        train_loss, pt_metric = pretrain(epoch)
        for k, v in pt_metric.items():
            metrics[k].append(v)
        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            save_checkpoint({repr(m): m.state_dict() for m in models_to_save}, is_best=True, folder=args.exp_dir)

    if args.freeze_slots:
        # for m in models_to_save:
        #     if (isinstance(m, ExWrapper)):
        #         setattr(m, 'freeze_model', True)
        #         m.eval()
        for p in params_to_pretrain:
            p.requires_grad = False

    for epoch in range(1, args.ft_epochs+1):
        train_loss = finetune(epoch)
        test_acc, test_raw_scores = test(epoch)

        # Compute confidence intervals
        n_test = len(test_raw_scores)
        test_acc_ci = 1.96 * np.std(test_raw_scores) / np.sqrt(n_test)
        
        is_best_epoch = test_acc > best_epoch_acc
        if is_best_epoch:
            best_epoch = epoch
            best_epoch_acc = test_acc
            best_test_acc = test_acc
            best_test_acc_ci = test_acc_ci

        if args.save_checkpoint:
            save_checkpoint({repr(m): m.state_dict() for m in models_to_save}, is_best=is_best_epoch, folder=args.exp_dir)

        metrics['train_acc'].append(train_loss)
        metrics['test_acc'].append(test_acc)
        metrics['test_acc_ci'].append(test_acc_ci)

        metrics = dict(metrics)
        # Assign best accs
        metrics['best_epoch'] = best_epoch
        metrics['best_test_acc'] = best_test_acc
        metrics['best_test_acc_ci'] = best_test_acc_ci
        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))

    print('====> DONE')
    print('====> BEST EPOCH: {}'.format(best_epoch))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test)', best_epoch, best_test_acc))

