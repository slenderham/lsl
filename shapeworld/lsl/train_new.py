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

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
    save_checkpoint,
    featurize
)
from datasets import ShapeWorld, extract_features, extract_objects, extract_objects_and_positions
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, COLORS, SHAPES
from models import *
from vision import Conv4NP, ResNet18, Conv4
from utils import GradualWarmupScheduler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Output directory')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='Size of hidden representations')
    parser.add_argument('--num_vision_slots', 
                        type=int,
                        default=6,
                        help='Number of slots')
    parser.add_argument('--num_lang_slots', 
                        type=int,
                        default=4,
                        help='Number of slots')
    parser.add_argument('--img_spec_weight',
                        type=float,
                        default=0.,
                        help='Weight on image autoencoding')
    parser.add_argument('--lang_spec_weight',
                        type=float,
                        default=0.,
                        help='Weight on language autoencoding')
    parser.add_argument('--comparison',
                        choices=['dotp', 'cosine', 'transformer'],
                        default='transformer',
                        help='How to compare support to query reps')
    parser.add_argument('--freeze_slots',
                        action='store_true',
                        help='If True, freeze slots.')
    parser.add_argument('--use_relational_model',
                        action='store_true',
                        help='Use relational model on top of slots or only slots.')
    parser.add_argument('--hypo_model',
                        choices=['uni_gru', 'bi_gru', 'uni_transformer', 'bi_transformer', 'slot'],
                        default='bi_gru',
                        help='Which language model to use')
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Max number of training examples')
    parser.add_argument('--oracle_world_config',
                        action='store_true',
                        help='If true, let slots predict all objects and positions. Else use those from language.')
    parser.add_argument('--target_type',
                        type=str,
                        choices=['multihead_single_label', 'multilabel'],
                        default='multihead_single_label',
                        help='Whether to use one softmax for each attribute or sigmoid for all.')
    parser.add_argument('--representation',
                        type=str,
                        choices=['slot', 'whole'],
                        default='slot',
                        help='whether to use single vector or slots')
    parser.add_argument('--aux_task',
                        type=str,
                        choices=['imagenet_pretrain', 'caption', 'matching'],
                        default='matching',
                        help='Whether to predict caption or match objects to captions')
    parser.add_argument('--visualize_attns',
                        action='store_true',
                        help='If true, visualize attention masks of slots and matching/caption if applicable')
    parser.add_argument('--temperature',
                        default=0.5,
                        type=float,
                        help='Temperature parameter used in contrastive loss')
    parser.add_argument('--fixed_noise_colors',
                        default=None,
                        type=int,
                        help='Fix noise based on class, with a max of this many')
    parser.add_argument('--fixed_noise_colors_max_rgb',
                        default=0.2,
                        type=float,
                        help='Maximum color value a single color channel '
                            'can have for noise background')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Train batch size')
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
    parser.add_argument('--language_filter',
                        default=None,
                        type=str,
                        choices=['color', 'nocolor'],
                        help='Filter language')
    parser.add_argument('--shuffle_words',
                        action='store_true',
                        help='Shuffle words for each caption')
    parser.add_argument('--shuffle_captions',
                        action='store_true',
                        help='Shuffle captions for each class')
    parser.add_argument('--log_interval',
                        type=int,
                        default=10,
                        help='How often to log loss')
    parser.add_argument('--pos_weight',
                        type=float,
                        default=1.0,
                        help="Weight on the object position loss")
    parser.add_argument('--save_checkpoint',
                        action='store_true',
                        help='Save model')
    parser.add_argument('--load_checkpoint',
                        action='store_true',
                        help='Load model')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA training')
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    if args.aux_task=='set_pred' and not args.oracle_world_config:
        args.pos_weight = 0.0 # if not using oracle object info, can't use coordinates for supervision

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        print("No CUDA available so not using it")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if args.cuda else 'cpu')

    # train dataset will return (image, label, hint_input, hint_target, hint_length)
    preprocess = False
    train_dataset = ShapeWorld(
        split='train',
        vocab=None,
        augment=True,
        precomputed_features=None,
        max_size=args.max_train,
        preprocess=preprocess,
        noise=0.,
        class_noise_weight=0,
        fixed_noise_colors=args.fixed_noise_colors,
        fixed_noise_colors_max_rgb=args.fixed_noise_colors_max_rgb,
        noise_type='gaussian',
        data_dir=args.data_dir,
        language_filter=args.language_filter,
        shuffle_words=args.shuffle_words,
        shuffle_captions=args.shuffle_captions)
    train_vocab = train_dataset.vocab
    train_vocab_size = train_dataset.vocab_size
    train_max_length = train_dataset.max_length
    train_w2i, train_i2w, train_w2c = train_vocab['w2i'], train_vocab['i2w'], train_vocab['w2c']
    pad_index = train_w2i[PAD_TOKEN]
    sos_index = train_w2i[SOS_TOKEN]
    eos_index = train_w2i[EOS_TOKEN]

    test_class_noise_weight = 0.0
    test_noise = 0.0
    val_dataset = ShapeWorld(split='val',
                             precomputed_features=None,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             noise=test_noise,
                             class_noise_weight=0.0,
                             noise_type='gaussian',
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=None,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              noise=test_noise,
                              class_noise_weight=0.0,
                              noise_type='gaussian',
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=None,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type='gaussian',
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=None,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type='gaussian',
            data_dir=args.data_dir)
        has_same = True
    except RuntimeError:
        val_same_dataset = test_same_dataset = None
        has_same = False

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    if has_same:
        val_same_loader = torch.utils.data.DataLoader(
            val_same_dataset, batch_size=args.batch_size, shuffle=False)
        test_same_loader = torch.utils.data.DataLoader(
            test_same_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_same_loader = test_same_loader = None

    data_loader_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'val_same': val_same_loader if has_same else None,
        'test_same': test_same_loader if has_same else None,
    }
    if args.aux_task=='set_pred':
        labels_to_idx = train_dataset.label2idx

    ''' vision '''
    # if _image in task name, get vector for each image with conv net else get set of vectors with slots
    image_model = 'conv' if args.representation=='whole' else 'slot_attn'
    backbone_model = SANet(im_size=64, num_slots=args.num_vision_slots, dim=args.hidden_size, slot_model=image_model, use_relation=args.use_relational_model)
    image_part_model = ExWrapper(backbone_model).to(device)
    params_to_pretrain = list(image_part_model.parameters())
    models_to_save = [image_part_model]

    # if use relational model and use slots, add relational net structure
    # if use relational and not slots, return error
    # if not use relational and use slots, relation model is MLP to approximately balance number of params
    # if not use relational and not use slots, relational model is MLP as well
    # abuse of variable name here. This is just to project to the correct dimension

    ''' scorer '''
    if args.representation=='slot':
        im_im_scorer_model = TransformerAgg(args.hidden_size).to(device)
        simple_val_scorer = SinkhornScorer(hidden_dim=None, comparison='im_im', iters=100, reg=1).to(device)
    else:
        im_im_scorer_model = MLPMeanScore(args.hidden_size, args.hidden_size)
        simple_val_scorer = CosineScorer(temperature=1).to(device)
    params_to_finetune = list(im_im_scorer_model.parameters())
    models_to_save.append(im_im_scorer_model)

    ''' aux task specific '''
    if args.aux_task=='set_pred':
        image_cls_projection = MLP(64, args.hidden_size, len(labels_to_idx['color'])+len(labels_to_idx['shape'])).to(device) # add one for no object
        params_to_pretrain.extend(image_cls_projection.parameters())
        models_to_save.append(image_cls_projection)

        image_pos_projection = MLP(64, args.hidden_size, 2).to(device)
        params_to_pretrain.extend(image_pos_projection.parameters())
        models_to_save.append(image_pos_projection)

    elif args.aux_task=='caption':
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        if args.representation=='slot':
            hint_model = TextProposalWithAttn(embedding_model, encoder_dim=args.hidden_size, hidden_size=args.hidden_size)
        else:
            hint_model = TextProposal(embedding_model, hidden_size=args.hidden_size)
        hint_model = hint_model.to(device)
        params_to_pretrain.extend(hint_model.parameters())
        models_to_save.append(hint_model)

    elif args.aux_task=='matching':
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        hint_model = {
                        'uni_gru': TextRep(embedding_model, hidden_size=args.hidden_size, bidirectional=False, return_agg=args.representation=='whole'),
                        'bi_gru': TextRep(embedding_model, hidden_size=args.hidden_size, bidirectional=True, return_agg=args.representation=='whole'),
                        'uni_transformer': TextRepTransformer(embedding_model, hidden_size=args.hidden_size, bidirectional=False, return_agg=args.representation=='whole'),
                        'bi_transformer': TextRepTransformer(embedding_model, hidden_size=args.hidden_size, bidirectional=True, return_agg=args.representation=='whole'),
                        'slot': TextRepSlot(embedding_model, hidden_size=args.hidden_size, return_agg=args.representation=='whole', num_slots=args.num_lang_slots)
                     }[args.hypo_model]
        hint_model = hint_model.to(device)
        params_to_pretrain.extend(hint_model.parameters())
        models_to_save.append(hint_model)

    else:
        raise ValueError('invalid auxiliary task name')

    # loss
    if args.aux_task=='set_pred':
        hype_loss = SetCriterion(num_classes=[len(labels_to_idx['color']), len(labels_to_idx['shape'])], 
                            pos_cost_weight=args.pos_weight, 
                            eos_coef=0.5, 
                            target_type=args.target_type).to(device)
    elif args.aux_task=='matching' and args.representation=='slot':
        hype_loss = SinkhornScorer(hidden_dim=args.hidden_size, idx_to_word=train_i2w, temperature=args.temperature).to(device)
    elif args.aux_task=='matching' and args.representation=='whole':
        hype_loss = ContrastiveLoss(temperature=args.temperature)
    else:
        raise AssertionError('There are only three types of aux_tasks that require special loss')

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
    after_scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, 4000, 0.5)
    scheduler = GradualWarmupScheduler(pretrain_optimizer, 1.0, total_epoch=1000, after_scheduler=after_scheduler)
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
        for batch_idx in range(n_steps):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device)
            image = image.to(device)
            label = label.to(device)
            batch_size = len(image)
            n_ex = examples.shape[1]

            if args.aux_task=='set_pred':
                world = rest[-2] # this should be a list of lists
                world_len = rest[-1] # batch size x n_ex, for how many objects each image contains
                if args.oracle_world_config:
                    objs, poses = extract_objects_and_positions(world, world_len, labels_to_idx)
                else:
                    objs = extract_objects([[train_i2w[token.item()] for token in h if token.item()!=pad_index] for h in hint_seq])
                    _, poses = extract_objects_and_positions(world, world_len, labels_to_idx) # this will not be used, but need to be here for set criterion to work

            if args.debug_example:
                rand_idx = np.random.randint(0, args.batch_size) # sample a random index from current batch
                print([train_i2w[k.item()] for k in hint_seq[rand_idx]]) # get hint in words
                print(label[rand_idx])
                if (args.aux_task=='set_pred'):
                    for w in world[rand_idx]:
                        print(w) 
                    print(objs[(n_ex+1)*rand_idx:(n_ex+1)*(rand_idx+1)])
                    print(poses[(n_ex+1)*rand_idx:(n_ex+1)*(rand_idx+1)])
                print(examples[rand_idx][0].shape)
                fig, axes = plt.subplots(5)
                for i in range(4):
                    axes[i].imshow(examples[rand_idx][i].permute(1, 2, 0)) # plot examples, transpose to put channel in the last dim
                    axes[i].axis('off')
                axes[4].imshow(image[rand_idx].permute(1, 2, 0))
                axes[4].axis('off')
                plt.show()

            # Load hint
            hint_seq = hint_seq.to(device)
            hint_length = hint_length.to(device)
            max_hint_length = hint_length.max().item()
            # Cap max length if it doesn't fill out the tensor
            if max_hint_length != hint_seq.shape[1]:
                hint_seq = hint_seq[:, :max_hint_length]

            # Learn representations of images and examples
            image_slot = image_part_model(image, is_ex=False, visualize_attns=False) # --> N x n_slot x C
            examples_slot = image_part_model(examples, is_ex=True, visualize_attns=args.visualize_attns) # --> N x n_ex x n_slot x C

            if args.aux_task=='set_pred':
                slot_cls_score = image_cls_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1)
                slot_pos_pred = image_pos_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1)
    
                losses, metric = hype_loss({'pred_logits': slot_cls_score, 'pred_poses': slot_pos_pred}, {'labels': objs, 'poses': poses})

                # Hypothesis loss
                loss = losses['class'] + args.pos_weight*losses['position']

                aux_loss_total += losses['class'].item()
                pos_loss_total += losses['position'].item()
                cls_acc += metric['acc']
            elif args.aux_task=='caption':
                hint_seq = torch.repeat_interleave(hint_seq, repeats=n_ex, dim=0)  # repeat captions to match each image
                if (args.representation=='slot'):
                    assert(len(examples_slot.shape)==4), "The examples_full should have shape: batch_size X n_ex X (num_slots or ) X dim"
                    hypo_out, attns = hint_model(examples_slot.flatten(0, 1), hint_seq, \
                        torch.repeat_interleave(hint_length, repeats=n_ex, dim=0).to(device))
                else:
                    assert(len(examples_slot.shape)==3), "The examples_full should be of shape: batch_size X n_ex, X dim"
                    hypo_out = hint_model(examples_slot.flatten(0, 1), hint_seq, \
                        torch.repeat_interleave(hint_length, repeats=n_ex, dim=0).to(device))
                seq_len = hint_seq.size(1)

                if (args.visualize_attns):
                    if (args.representation=='slot'):
                        plt.subplot(111).imshow(attns[2].detach().t())
                    elif(args.representation=='whole'):
                        fig, axes = plt.subplots(2, 7)
                        for i, a in enumerate(attns[2].detach()):
                            axes[i//7][i%7].imshow(a.reshape(56, 56))
                    print([train_i2w[h.item()] for h in torch.argmax(hypo_out[2], dim=-1)])
                    print([train_i2w[h.item()] for h in hint_seq[2]])
                    plt.show()

                hypo_out = hypo_out[:, :-1].contiguous()
                hint_seq = hint_seq[:, 1:].contiguous()
                hyp_batch_size = batch_size*n_ex
                hypo_out_2d = hypo_out.view(hyp_batch_size * (seq_len - 1),
                                            train_vocab_size)
                hint_seq_2d = hint_seq.long().view(hyp_batch_size * (seq_len - 1))
                hypo_loss = F.cross_entropy(hypo_out_2d,
                                            hint_seq_2d,
                                            reduction='mean',
                                            ignore_index=pad_index) # switch to token-wise loss
                loss = hypo_loss
                non_pad_mask = hint_seq_2d!=pad_index
                hypo_pred = torch.argmax(hypo_out_2d, dim=-1).masked_select(non_pad_mask)
                hypo_gt = hint_seq_2d.masked_select(non_pad_mask)
                metric = {'acc': (hypo_pred==hypo_gt).float().mean().item()} 
                aux_loss_total += hypo_loss.item()
                cls_acc += metric['acc']
            elif args.aux_task=='matching':
                if ('gru' not in args.hypo_model):
                    hint_rep = hint_model(hint_seq, hint_length, hint_seq==pad_index) 
                else:
                    hint_rep = hint_model(hint_seq, hint_length) 

                if (args.representation=='slot'):
                    assert(len(examples_slot.shape)==4), "The examples_full should have shape: batch_size X n_ex X (num_slots or ) X dim"
                    assert(hint_rep.shape==(batch_size, max_hint_length if args.hypo_model!='slot' else args.num_lang_slots, args.hidden_size))
                    if args.hypo_model=='slot':
                        y_mask = None
                    else:
                        y_mask = ((hint_seq==pad_index) | (hint_seq==sos_index) | (hint_seq==eos_index))
                    matching, hypo_loss, metric = hype_loss(x=examples_slot.flatten(0, 1), y=hint_rep, word_idx=hint_seq, y_mask=y_mask)
                else:
                    assert(len(examples_slot.shape)==3), "The examples_full should be of shape: batch_size X n_ex X dim"
                    assert(hint_rep.shape==(batch_size, args.hidden_size))
                    hypo_loss, metric = hype_loss(im=examples_slot, s=hint_rep)
                
                if args.visualize_attns:
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    im = ax.imshow(matching[2][0].detach().cpu(), vmin=0, vmax=1)
                    ylabels = list(range(args.num_vision_slots))
                    # xlabels = list(range(args.num_lang_slots))
                    ylabels = ylabels + [str(y2)+' x '+str(y1) for y1 in range(args.num_vision_slots) for y2 in range(args.num_vision_slots) if y1!=y2]
                    ax.set_xticks(np.arange(len(hint_seq[0])))
                    ax.set_xticklabels([train_i2w[h.item()] for h in hint_seq[0]], rotation=90)
                    # ax.set_xticks(np.arange(len(xlabels)))
                    # ax.set_xticklabels(xlabels)
                    ax.set_yticks(np.arange(len(ylabels)))
                    ax.set_yticklabels(ylabels)
                    ax.set_aspect('auto')
                    plt.colorbar(im, ax=ax)
                    plt.show()

                loss = hypo_loss
                aux_loss_total += hypo_loss.item()
                cls_acc += (metric['acc_im_lang'] + metric['acc_lang_im'])/2
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

    def simple_eval(epoch, split):
        if (args.representation=='whole'):
            raise NotImplementedError

        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.eval()

        concept_avg_meter = AverageMeter()
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                n_ex = examples.shape[1]
                is_neg = label==0

                examples = examples[is_neg]
                image = image[is_neg]
                batch_size = len(image)

                # Learn representations of images and examples
                examples_slot = image_part_model(examples, is_ex=True, visualize_attns=False) # --> N x n_ex x n_slot x C
                image_slot = image_part_model(image, is_ex=False) # --> N x n_slot x C
                
                anchor = torch.repeat_interleave(examples_slot, repeats=n_ex, dim=1).flatten(0, 1)
                pos_ex = examples_slot.repeat(1, n_ex, 1, 1).flatten(0, 1)
                neg_ex = image_slot.unsqueeze(1).expand(batch_size, n_ex, -1, args.hidden_size).flatten(0, 1)
                
                pos_scores = simple_val_scorer(anchor, pos_ex) # --> batch_size*n_ex*n_ex
                pos_scores = torch.masked_selected(pos_scores, torch.eye(n_ex).unsqueeze(0)<0.5)
                pos_scores = pos_scores.reshape(batch_size, n_ex*(n_ex-1))
                pos_scores = torch.mean(pos_scores, -1)

                anchor = examples_slot.flatten(0, 1)
                neg_scores = simple_val_scorer(anchor, neg_ex).reshape(batch_size, n_ex) # --> batch_size*n_ex
                neg_scores = torch.mean(neg_scores, -1)
                concept_avg_meter.update((pos_scores>neg_scores).float().mean().item(), is_neg.float().sum().item())

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}'.format(
            '({})'.format(split), epoch, concept_avg_meter.avg))

        return concept_avg_meter.avg

    def finetune(epoch, n_steps=100):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.train()
            if (args.freeze_slots and (isinstance(m, ExWrapper))):
                m.eval()

        main_acc = 0
        pred_loss_total = 0
        pbar = tqdm(total=n_steps)
        for batch_idx in range(n_steps):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device)
            image = image.to(device)
            label = label.to(device)
            batch_size = len(image)
            n_ex = examples.shape[1]

            # Learn representations of images and examples
            image_slot = image_part_model(image, is_ex=False, visualize_attns=False) # --> N x n_slot x C
            examples_slot = image_part_model(examples, is_ex=True, visualize_attns=False) # --> N x n_ex x n_slot x C

            if args.representation=='whole':
                examples_slot = examples_slot.reshape(batch_size, n_ex, 1, args.hidden_size)
                image_slot = image_slot.reshape(batch_size, 1, args.hidden_size)

            score = im_im_scorer_model(examples_slot, image_slot).squeeze()
            loss = F.binary_cross_entropy_with_logits(score, label.float())
            pred_loss_total += loss.item()
            main_acc += ((score>0).long()==label).float().mean().item()

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
        return loss

    def test(epoch, split='train'):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.eval()

        concept_avg_meter = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                label_np = label.cpu().numpy().astype(np.uint8)
                batch_size = len(image)
                n_ex = examples.shape[1]
                # Learn representations of images and examples
                image_slot = image_part_model(image, is_ex=False, visualize_attns=False) # --> N x n_slot x C
                examples_slot = image_part_model(examples, is_ex=True, visualize_attns=False) # --> N x n_ex x n_slot x C

                if args.representation=='whole':
                    examples_slot = examples_slot.reshape(batch_size, n_ex, 1, args.hidden_size)
                    image_slot = image_slot.reshape(batch_size, 1, args.hidden_size)

                score = im_im_scorer_model(examples_slot, image_slot).squeeze()
                label_hat = score > 0
                label_hat = label_hat.cpu().numpy()
                accuracy = accuracy_score(label_np, label_hat)
                concept_avg_meter.update(accuracy, batch_size, raw_scores=(label_hat == label_np))

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}'.format(
            '({})'.format(split), epoch, concept_avg_meter.avg))

        return concept_avg_meter.avg, concept_avg_meter.raw_scores

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    best_simple_eval_acc = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))

    if args.aux_task=='imagenet_pretrain':
        simple_val_acc = simple_eval(1, 'val')
        if has_same:
            simpel_val_same_acc = simple_eval(1, 'val_same')
            metrics['simple_val_acc'].append((simple_val_acc+simpel_val_same_acc)/2)
        else:
            metrics['simple_val_acc'].append(simple_val_acc)

        save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))
        if args.save_checkpoint:
            save_checkpoint({repr(m): m.state_dict() for m in models_to_save}, is_best=True, folder=args.exp_dir)

    else:
        for epoch in range(1, args.pt_epochs+1):
            train_loss, pt_metric = pretrain(epoch)
            for k, v in pt_metric.items():
                metrics[k].append(v)

            simple_val_acc = simple_eval(epoch, 'val')
            if has_same:
                simpel_val_same_acc = simple_eval(epoch, 'val_same')
                metrics['simple_val_acc'].append((simple_val_acc+simpel_val_same_acc)/2)
            else:
                metrics['simple_val_acc'].append(simple_val_acc)

            save_defaultdict_to_fs(metrics, os.path.join(args.exp_dir, 'metrics.json'))

            is_best_epoch = metrics['simple_val_acc'][-1] > best_simple_eval_acc;
            if is_best_epoch:
                best_simple_eval_acc = metrics['simple_val_acc'][-1]

            if args.save_checkpoint:
                save_checkpoint({repr(m): m.state_dict() for m in models_to_save}, is_best=is_best_epoch, folder=args.exp_dir)
        

    if args.freeze_slots:
        # for m in models_to_save:
        #     if (isinstance(m, ExWrapper)):
        #         setattr(m, 'freeze_model', True)
        #         m.eval()
        for p in params_to_pretrain:
            p.requires_grad = False

    for epoch in range(1, args.ft_epochs+1):
        train_loss = finetune(epoch, 1)
        train_acc, _ = test(epoch, 'train')
        val_acc, _ = test(epoch, 'val')
        test_acc, test_raw_scores = test(epoch, 'test')
        if has_same:
            val_same_acc, _ = test(epoch, 'val_same')
            test_same_acc, test_same_raw_scores = test(epoch, 'test_same')
            all_test_raw_scores = test_raw_scores + test_same_raw_scores
        else:
            val_same_acc = val_acc
            test_same_acc = test_acc
            all_test_raw_scores = test_raw_scores

        # Compute confidence intervals
        n_test = len(all_test_raw_scores)
        test_acc_ci = 1.96 * np.std(all_test_raw_scores) / np.sqrt(n_test)
        
        epoch_acc = (val_acc + val_same_acc) / 2
        is_best_epoch = epoch_acc > best_epoch_acc
        if is_best_epoch:
            best_epoch = epoch
            best_epoch_acc = epoch_acc
            best_val_acc = val_acc
            best_val_same_acc = val_same_acc
            best_test_acc = test_acc
            best_test_same_acc = test_same_acc
            best_test_acc_ci = test_acc_ci

        if args.save_checkpoint:
            save_checkpoint({repr(m): m.state_dict() for m in models_to_save}, is_best=is_best_epoch, folder=args.exp_dir)

        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_same_acc'].append(val_same_acc)
        metrics['test_acc'].append(test_acc)
        metrics['test_same_acc'].append(test_same_acc)
        metrics['test_acc_ci'].append(test_acc_ci)

        metrics = dict(metrics)
        # Assign best accs
        metrics['best_epoch'] = best_epoch
        metrics['best_val_acc'] = best_val_acc
        metrics['best_val_same_acc'] = best_val_same_acc
        metrics['best_test_acc'] = best_test_acc
        metrics['best_test_same_acc'] = best_test_same_acc
        metrics['best_test_acc_ci'] = best_test_acc_ci
        metrics['has_same'] = has_same
        save_defaultdict_to_fs(metrics,
                               os.path.join(args.exp_dir, 'metrics.json'))

    print('====> DONE')
    print('====> BEST EPOCH: {}'.format(best_epoch))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val)', best_epoch, best_val_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_same)', best_epoch, best_val_same_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test)', best_epoch, best_test_acc))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_same)', best_epoch, best_test_same_acc))
    print('====>')
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_val_avg)', best_epoch, (best_val_acc + best_val_same_acc) / 2))
    print('====> {:>17}\tEpoch: {}\tAccuracy: {:.4f}'.format(
        '(best_test_avg)', best_epoch,
        (best_test_acc + best_test_same_acc) / 2))
