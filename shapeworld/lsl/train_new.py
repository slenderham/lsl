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
from models import ImageRep, TextRep, TextProposalWithAttn, ExWrapper, Identity, TextRepTransformer, TextProposalTransformer
from models import SANet
from models import DotPScorer, BilinearScorer, CosineScorer, MLP, SinkhornScorer, SetCriterion, TransformerAgg
from vision import Conv4NP, ResNet18, Conv4NP
from loss import ContrastiveLoss
from utils import GradualWarmupScheduler

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
                        choices=['dotp', 'cosine'],
                        default='dotp',
                        help='How to compare support to query reps')
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
    parser.add_argument('--aux_task',
                        type=str,
                        choices=['set_pred_partial', 'caption_slot', 'caption_image', 'matching'],
                        default='matching',
                        help='Whether to predict caption or predict objects')
    parser.add_argument('--visualize_attns',
                        action='store_true',
                        help='If true, visualize attention masks of slots and matching/caption if applicable')
    parser.add_argument('--noise',
                        type=float,
                        default=0.0,
                        help='Amount of noise to add to each example')
    parser.add_argument('--class_noise_weight',
                        type=float,
                        default=0.0,
                        help='How much of that noise should be class diagnostic?')
    parser.add_argument('--temperature',
                        default=0.5,
                        type=float,
                        help='Temperature parameter used in contrastive loss')
    parser.add_argument('--noise_at_test',
                        action='store_true',
                        help='Add instance-level noise at test time')
    parser.add_argument('--noise_type',
                        default='gaussian',
                        choices=['gaussian', 'uniform'],
                        help='Type of noise')
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
    parser.add_argument('--epochs', type=int, default=50, help='Train epochs')
    parser.add_argument('--debug_example', 
                        action="store_true",
                        help="If true, print out example images and hint");
    parser.add_argument('--skip_eval',
                        action="store_true",
                        help="If true, skip the zero shot evaluation and only save the pretrained features.")
    parser.add_argument('--data_dir',
                        default=None,
                        help='Specify custom data directory (must have shapeworld folder)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
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
    parser.add_argument('--hypo_lambda',
                        type=float,
                        default=10.0,
                        help='Weight on hypothesis hypothesis')
    parser.add_argument('--concept_lambda',
                        type=float,
                        default=1.0)
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
        args.pos_weight = 0.0; # if not using oracle object info, can't use coordinates for supervision

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if (not torch.cuda.is_available()):
        print("No CUDA available so not using it");
        device = torch.device('cpu');
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
        noise=args.noise,
        class_noise_weight=args.class_noise_weight,
        fixed_noise_colors=args.fixed_noise_colors,
        fixed_noise_colors_max_rgb=args.fixed_noise_colors_max_rgb,
        noise_type=args.noise_type,
        data_dir=args.data_dir,
        language_filter=args.language_filter,
        shuffle_words=args.shuffle_words,
        shuffle_captions=args.shuffle_captions)
    train_vocab = train_dataset.vocab
    train_vocab_size = train_dataset.vocab_size
    train_max_length = train_dataset.max_length
    train_w2i, train_i2w = train_vocab['w2i'], train_vocab['i2w']
    pad_index = train_w2i[PAD_TOKEN]
    sos_index = train_w2i[SOS_TOKEN]
    eos_index = train_w2i[EOS_TOKEN]

    test_class_noise_weight = 0.0
    if args.noise_at_test:
        test_noise = args.noise
    else:
        test_noise = 0.0
    val_dataset = ShapeWorld(split='val',
                             precomputed_features=None,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             noise=test_noise,
                             class_noise_weight=0.0,
                             noise_type=args.noise_type,
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=None,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              noise=test_noise,
                              class_noise_weight=0.0,
                              noise_type=args.noise_type,
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=None,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=None,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        has_same = True
    except RuntimeError:
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

    data_loader_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'val_same': val_same_loader if has_same else None,
        'test_same': test_same_loader if has_same else None,
    }
    if args.aux_task=='set_pred':
        labels_to_idx = train_dataset.label2idx

    # vision
    backbone_model = SANet(im_size=64, num_slots=args.num_slots, dim=64, slot_model=('slot_mlp' if args.aux_task=='caption_image' else 'slot_attn'));
    image_part_model = ExWrapper(backbone_model).to(device);
    params_to_optimize = list(image_part_model.parameters())
    models_to_save = [image_part_model];

    image_whole_model = TransformerAgg(64, 4).to(device);
    params_to_optimize.extend(image_whole_model.parameters());
    models_to_save.append(image_whole_model);

    # scorer
    im_im_scorer_model = {
        'dotp': DotPScorer(),
        'cosine': CosineScorer(temperature=1),
    }[args.comparison]
    im_im_scorer_model = im_im_scorer_model.to(device)
    params_to_optimize.extend(im_im_scorer_model.parameters())
    models_to_save.append(im_im_scorer_model)

    # projection
    if args.aux_task=='set_pred':
        image_cls_projection = MLP(64, args.hidden_size, len(labels_to_idx['color'])+len(labels_to_idx['shape'])).to(device); # add one for no object
        params_to_optimize.extend(image_cls_projection.parameters());
        models_to_save.append(image_cls_projection)

        image_pos_projection = MLP(64, args.hidden_size, 2).to(device);
        params_to_optimize.extend(image_pos_projection.parameters());
        models_to_save.append(image_pos_projection)
    elif args.aux_task=='caption_slot' or args.aux_task=='caption_image':
    # language
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        hint_model = TextProposalTransformer(embedding_model, hidden_size=args.hidden_size)
        hint_model = hint_model.to(device)
        params_to_optimize.extend(hint_model.parameters())
        models_to_save.append(hint_model)
    elif args.aux_task=='matching':
        embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
        hint_model = TextRepTransformer(embedding_model, hidden_size=args.hidden_size)
        hint_model = hint_model.to(device)
        params_to_optimize.extend(hint_model.parameters())
        models_to_save.append(hint_model)
        
        slot_to_lang_matching = MLP(64, args.hidden_size, args.hidden_size).to(device);
        params_to_optimize.extend(slot_to_lang_matching.parameters())
        models_to_save.append(slot_to_lang_matching)
    else:
        raise ValueError('invalid auxiliary task name')

    # loss
    if args.aux_task=='set_pred':
        hype_loss = SetCriterion(num_classes=[len(labels_to_idx['color']), len(labels_to_idx['shape'])], 
                            pos_cost_weight=args.pos_weight, 
                            eos_coef=0.5, 
                            target_type=args.target_type).to(device);
    elif args.aux_task=='matching':
        hype_loss = SinkhornScorer(num_embedding=train_vocab_size, temperature=args.temperature).to(device);
        params_to_optimize.extend(hype_loss.parameters())
        models_to_save.append(hype_loss)

        # hype_whole_loss = CosineScorer(temperature=args.temperature).to(device);
        # params_to_optimize.extend(hype_whole_loss.parameters())
        # models_to_save.append(hype_whole_loss)

    # optimizer
    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }[args.optimizer]
    optimizer = optfunc(params_to_optimize, lr=args.lr)
    # after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=49000)
    # scheduler = GradualWarmupScheduler(optimizer, 1.0, total_epoch=1000, after_scheduler=after_scheduler)

    print(sum([p.numel() for p in params_to_optimize]));

    if args.load_checkpoint and os.path.exists(os.path.join(args.exp_dir, 'checkpoint.pth.tar')):
        ckpt_path = os.path.join(args.exp_dir, 'checkpoint.pth.tar');
        sds = torch.load(ckpt_path, map_location=lambda storage, loc: storage);
        for m, sd in zip(models_to_save, sds):
            print(m)
            print(m.load_state_dict(sd));
        print("loaded checkpoint");

    # if args.visualize_attns and args.aux_task=='matching':
    #     for k, v in train_w2i.items():
    #         print(k, hype_loss.dustbin_scores_lang.weight[:, v]);


    def train(epoch, n_steps=100):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.train();

        pred_loss_total = 0;
        aux_loss_total = 0;
        cls_acc = 0;
        main_acc = 0;
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
                world = rest[-2]; # this should be a list of lists
                world_len = rest[-1]; # batch size x n_ex, for how many objects each image contains
                if args.oracle_world_config:
                    objs, poses = extract_objects_and_positions(world, world_len, labels_to_idx);
                else:
                    objs = extract_objects([[train_i2w[token.item()] for token in h if token.item()!=pad_index] for h in hint_seq]);
                    _, poses = extract_objects_and_positions(world, world_len, labels_to_idx); # this will not be used, but need to be here for set criterion to work

            if args.debug_example:
                rand_idx = np.random.randint(0, args.batch_size); # sample a random index from current batch
                print([train_i2w[k.item()] for k in hint_seq[rand_idx]]); # get hint in words
                print(label[rand_idx])
                if (args.aux_task=='set_pred'):
                    for w in world[rand_idx]:
                        print(w); 
                    print(objs[(n_ex+1)*rand_idx:(n_ex+1)*(rand_idx+1)]);
                    print(poses[(n_ex+1)*rand_idx:(n_ex+1)*(rand_idx+1)]);
                print(examples[rand_idx][0].shape)
                fig, axes = plt.subplots(5);
                for i in range(4):
                    axes[i].imshow(examples[rand_idx][i].permute(1, 2, 0)); # plot examples, transpose to put channel in the last dim
                    axes[i].axis('off');
                axes[4].imshow(image[rand_idx].permute(1, 2, 0));
                axes[4].axis('off')
                plt.show();

            # Load hint
            hint_seq = hint_seq.to(device)
            hint_length = hint_length.to(device)
            max_hint_length = hint_length.max().item()
            # Cap max length if it doesn't fill out the tensor
            if max_hint_length != hint_seq.shape[1]:
                hint_seq = hint_seq[:, :max_hint_length]

            # Learn representations of images and examples
            image_slot = image_part_model(image, visualize_attns=False); # --> N x n_slot x C
            examples_slot = image_part_model(examples, visualize_attns=args.visualize_attns); # --> N x n_ex x n_slot x C
            examples_whole, image_whole = image_whole_model(examples_slot, image_slot);
            score = im_im_scorer_model.score(examples_whole.mean(dim=(1,2)), image_whole.mean(dim=1)).squeeze();
            pred_loss = F.binary_cross_entropy_with_logits(score, label.float());
            pred_loss_total += pred_loss
            main_acc += ((score>0).long()==label).float().mean()

            loss = args.concept_lambda*pred_loss

            if args.aux_task=='set_pred':
                slot_cls_score = image_cls_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
                slot_pos_pred = image_pos_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
    
                losses, metric = hype_loss({'pred_logits': slot_cls_score, 'pred_poses': slot_pos_pred},
                                    {'labels': objs, 'poses': poses});

                # Hypothesis loss
                loss += args.hypo_lambda*(losses['class'] + args.pos_weight*losses['position'])

                aux_loss_total += losses['class'].item()
                pos_loss_total += losses['position'].item()
                cls_acc += metric['acc'];
            elif args.aux_task=='caption_slot' or args.aux_task=='caption_image':
                raise NotImplementedError("need to see how to extract attention");
                hint_seq = torch.repeat_interleave(hint_seq, repeats=n_ex, dim=0); 
                hypo_out, attns = hint_model(examples_slot.flatten(0, 1), hint_seq, torch.repeat_interleave(hint_length, repeats=n_ex, dim=0));   
                seq_len = hint_seq.size(1)
                
                if (args.visualize_attns):
                    if (args.aux_task=='caption_slot'):
                        plt.subplot(111).imshow(attns[2].detach().t());
                    elif(args.aux_task=='caption_image'):
                        fig, axes = plt.subplots(2, 7)
                        for i, a in enumerate(attns[2].detach()):
                            axes[i//7][i%7].imshow(a.reshape(56, 56));
                    print([train_i2w[h.item()] for h in torch.argmax(hypo_out[2], dim=-1)]);
                    print([train_i2w[h.item()] for h in hint_seq[2]]);
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
                                            ignore_index=pad_index); # switch to token-wise loss
                loss += args.hypo_lambda*hypo_loss;
                non_pad_mask = hint_seq_2d!=pad_index;
                hypo_pred = torch.argmax(hypo_out_2d, dim=-1).masked_select(non_pad_mask);
                hypo_gt = hint_seq_2d.masked_select(non_pad_mask);
                metric = {'acc': (hypo_pred==hypo_gt).float().mean()}; 
                aux_loss_total += hypo_loss.item()
                cls_acc += metric['acc'];
            elif args.aux_task=='matching':
                hint_rep = hint_model(hint_seq, hint_seq==pad_index); 
                examples_whole = slot_to_lang_matching(examples_whole).flatten(0, 1);
                matching, scores = hype_loss.score(x=examples_whole, y=hint_rep, word_idx=hint_seq, \
                                    y_mask=((hint_seq==pad_index) | (hint_seq==sos_index) | (hint_seq==eos_index)));
                if args.visualize_attns:
                    ax = plt.subplot(111)
                    ax.imshow(matching[2][0].detach(), vmin=0, vmax=1)
                    ax.set_xticks(np.arange(len(hint_seq[0])))
                    ax.set_xticklabels([train_i2w[h.item()] for h in hint_seq[0]], rotation=45)
                    plt.show()
                
                pos_mask = (torch.block_diag(*([torch.ones(n_ex, 1)]*batch_size))>0.5).to(device)
                pos = scores.masked_select(pos_mask).reshape(batch_size*n_ex, 1);
                neg = scores.masked_select(~pos_mask).reshape(batch_size*n_ex, batch_size-1);
                scores_reshaped = torch.cat([pos, neg], dim=1);
                hypo_loss = F.log_softmax(scores_reshaped, dim=1)[:,0].mean();
                loss += -args.hypo_lambda*hypo_loss;
                metric = {'part_acc': (torch.argmax(scores_reshaped, dim=1)==0).float().mean().item()}
                aux_loss_total += hypo_loss.item();
                metric['pos_score'] = pos.mean().item();
                metric['neg_score'] = neg.mean().item();

                cls_acc += metric['part_acc'];
            else:
                raise ValueError("invalid auxiliary task name")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f} Metric: {}'.format(
                    epoch, loss.item(), metric))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tConcept Loss: {:.4f} Concept Acc: {:.4f} Auxiliary Loss: {:.4f} Auxiliary Acc: {:.4f}'.format('(train)', epoch, pred_loss_total, main_acc, aux_loss_total, cls_acc));

        return loss

    def test(epoch, split='train'):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.eval();

        concept_avg_meter = AverageMeter(raw=True)
        aux_metric_meter = AverageMeter(raw=False)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                label_np = label.cpu().numpy().astype(np.uint8)
                batch_size = len(image)
                n_ex = examples.shape[1]

                if args.aux_task=='set_pred':
                    world = rest[-2]; # this should be a list of lists
                    world_len = rest[-1]; # batch size x n_ex, for how many objects each image contains
                    actual_world = []; # the batches get collated in the weirdest way possible maybe better off not batching at all
                    for i in range(batch_size):
                        actual_world.append([[{
                                    'color': world[j][k]['color'][i],
                                    'shape': world[j][k]['shape'][i],
                                    'pos': (world[j][k]['pos'][0][i], world[j][k]['pos'][1][i])
                                } for k in range(world_len[i][j])] for j in range(n_ex+1)
                        ])
                    world = actual_world
                    if args.oracle_world_config:
                        objs, poses = extract_objects_and_positions(world, world_len, labels_to_idx);
                    else:
                        objs = extract_objects([[train_i2w[token.item()] for token in h if token.item()!=pad_index] for h in hint_seq]);
                        _, poses = extract_objects_and_positions(world, world_len, labels_to_idx); # this will not be used, but need to be here for set criterion to work
                
                
                hint_seq = hint_seq.to(device)
                hint_length = hint_length.to(device)
                max_hint_length = hint_length.max().item()
                # Cap max length if it doesn't fill out the tensor
                if max_hint_length != hint_seq.shape[1]:
                    hint_seq = hint_seq[:, :max_hint_length]

                # Learn representations of images and examples
                image_slot = image_part_model(image, visualize_attns=False); # --> N x n_slot x C
                examples_slot = image_part_model(examples, visualize_attns=args.visualize_attns); # --> N x n_ex x n_slot x C
                examples_whole, image_whole = image_whole_model(examples_slot, image_slot);
                score = im_im_scorer_model.score(examples_whole.mean(dim=(1,2)), image_whole.mean(dim=1)).squeeze();
                label_hat = score > 0
                label_hat = label_hat.cpu().numpy()
                accuracy = accuracy_score(label_np, label_hat);
                concept_avg_meter.update(accuracy, batch_size, raw_scores=(label_hat == label_np))

                if args.aux_task=='set_pred':
                    slot_cls_score = image_cls_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
                    slot_pos_pred = image_pos_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
        
                    _, metric = hype_loss({'pred_logits': slot_cls_score, 'pred_poses': slot_pos_pred},
                                        {'labels': objs, 'poses': poses});

                    metric_meter.update(metric['acc'], batch_size, raw_scores=None)

                elif args.aux_task=='caption_slot' or args.aux_task=='caption_image':
                    hint_seq = torch.repeat_interleave(hint_seq, repeats=n_ex, dim=0); 
                    hypo_out, attns = hint_model(examples_slot.flatten(0, 1), hint_seq, torch.repeat_interleave(hint_length, repeats=n_ex, dim=0));   
                    seq_len = hint_seq.size(1)
                    hypo_out = hypo_out[:, :-1].contiguous()
                    hint_seq = hint_seq[:, 1:].contiguous()
                    hyp_batch_size = batch_size*n_ex

                    hypo_out_2d = hypo_out.view(hyp_batch_size * (seq_len - 1),
                                                train_vocab_size)
                    hint_seq_2d = hint_seq.long().view(hyp_batch_size * (seq_len - 1))
                    hypo_loss = F.cross_entropy(hypo_out_2d,
                                                hint_seq_2d,
                                                reduction='mean',
                                                ignore_index=pad_index); # switch to token-wise loss
                    non_pad_total = (hint_seq_2d!=pad_index).float().sum()-hyp_batch_size; # total number of tokens, len-1 for each sample
                    aux_metric_meter.update(hypo_loss.item(), non_pad_total.item(), raw_scores=None);
                elif args.aux_task=='matching':
                    hint_rep = hint_model(hint_seq, hint_seq==pad_index); 
                    examples_whole = slot_to_lang_matching(examples_whole).flatten(0, 1);
                    matching, scores = hype_loss.score(x=examples_whole, y=hint_rep, word_idx=hint_seq, \
                                        y_mask=((hint_seq==pad_index) | (hint_seq==sos_index) | (hint_seq==eos_index)));
                    pos_mask = (torch.block_diag(*([torch.ones(n_ex, 1)]*batch_size))>0.5).to(device)
                    pos = scores.masked_select(pos_mask).reshape(batch_size*n_ex, 1);
                    neg = scores.masked_select(~pos_mask).reshape(batch_size*n_ex, batch_size-1);
                    scores_reshaped = torch.cat([pos, neg], dim=1);
                    hypo_loss = F.log_softmax(scores_reshaped, dim=1)[:,0].mean();

                    aux_metric_meter.update(hypo_loss.item(), batch_size*n_ex, raw_scores=None);
                else:
                    raise ValueError("invalid auxiliary task name")

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}\tMetric: {:.4f}'.format(
            '({})'.format(split), epoch, concept_avg_meter.avg, aux_metric_meter.avg))

        return concept_avg_meter.avg, aux_metric_meter.avg, concept_avg_meter.raw_scores

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch);
        if args.skip_eval:
            if args.save_checkpoint:
                save_checkpoint([m.state_dict() for m in models_to_save], is_best=True, folder=args.exp_dir);
            continue
        train_acc, train_aux_metric, _ = test(epoch, 'train')
        val_acc, val_aux_metric, _ = test(epoch, 'val')
        test_acc, test_aux_metric, test_raw_scores = test(epoch, 'test')
        if has_same:
            val_same_acc, val_same_aux_metric, _ = test(epoch, 'val_same')
            test_same_acc, test_same_aux_metric, test_same_raw_scores = test(epoch, 'test_same')
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
            save_checkpoint([m.state_dict() for m in models_to_save], is_best=is_best_epoch, folder=args.exp_dir);

        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['val_same_acc'].append(val_same_acc)
        metrics['test_acc'].append(test_acc)
        metrics['test_same_acc'].append(test_same_acc)
        metrics['test_acc_ci'].append(test_acc_ci)
        metrics['train_aux_metric'].append(train_aux_metric)
        metrics['val_aux_metric'].append(val_aux_metric)
        metrics['test_aux_metric'].append(test_aux_metric)
        metrics['val_same_aux_metric'].append(val_same_aux_metric)
        metrics['test_same_aux_metric'].append(test_same_aux_metric)

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

