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
from models import ImageRep, TextRep, TextProposal, ExWrapper, Identity, TextRepTransformer, MultilayerTransformer
from models import SANet
from models import DotPScorer, BilinearScorer, CosineScorer, MLP, TransformerScorer, SinkhornScorer, SetCriterion
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

    labels_to_idx = train_dataset.label2idx;

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

    # vision
    backbone_model = SANet(im_size=64, num_slots=6, dim=64);
    image_part_model = ExWrapper(ImageRep(backbone_model, \
                                     hidden_size=None, \
                                     tune_backbone=True, \
                                     normalize_feats=False));
    image_part_model = image_part_model.to(device)
    
    # image_whole_model = ExWrapper(MultilayerTransformer(64, 2, 2));
    # image_whole_model = image_whole_model.to(device);

    params_to_optimize = list(image_part_model.parameters())
    # params_to_optimize = list(image_whole_model.parameters())

    # scorer
    im_im_scorer_model = DotPScorer()
    im_im_scorer_model = im_im_scorer_model.to(device)
    params_to_optimize.extend(im_im_scorer_model.parameters())

    # base_scorer_model = CosineScorer(temperature=args.temperature)
    # im_lang_part_scorer_model = SinkhornScorer(base_scorer, iters=20);
    # im_lang_part_scorer_model = im_lang_part_scorer_model.to(device)
    # params_to_optimize.extend(im_lang_part_scorer_model.parameters())

    # im_lang_whole_scorer_model = SinkhornScorer(base_scorer, iters=20);
    # im_lang_whole_scorer_model = im_lang_whole_scorer_model.to(device)
    # params_to_optimize.extend(im_lang_whole_scorer_model.parameters())

    # projection
    image_cls_projection = MLP(64, args.hidden_size, len(labels_to_idx)+1).to(device); # add one for no object
    params_to_optimize.extend(image_cls_projection.parameters());

    image_pos_projection = MLP(64, args.hidden_size, 2).to(device);
    params_to_optimize.extend(image_pos_projection.parameters());

    # language
    embedding_model = nn.Embedding(train_vocab_size, args.hidden_size)
    hint_model = TextRep(embedding_model, hidden_size=args.hidden_size)
    hint_model = hint_model.to(device)
    params_to_optimize.extend(hint_model.parameters())

    # loss
    set_loss = SetCriterion(num_classes=len(labels_to_idx), eos_coef=0.1, pos_cost_weight=args.pos_weight).to(device);

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
    models_to_save = [image_part_model, image_cls_projection, image_pos_projection, im_im_scorer_model, hint_model, optimizer];

    if args.load_checkpoint and os.path.exists(os.path.join(args.exp_dir, 'model_best.pth.tar')):
        ckpt_path = os.path.join(args.exp_dir, 'model_best.pth.tar');
        sds = torch.load(ckpt_path, map_location=torch.device('cpu'));
        for m, sd in zip(models_to_save, sds):
            m.load_state_dict(sd);
        print("loaded checkpoint");

    def train(epoch, n_steps=100):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.train();

        pred_loss_total = 0;
        cls_loss_total = 0;
        pos_loss_total = 0;
        cls_acc = 0;
        pbar = tqdm(total=n_steps)
        for batch_idx in range(n_steps):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device)
            image = image.to(device)
            label = label.to(device)
            batch_size = len(image)
            n_ex = examples.shape[1]
            world = rest[-1]; # this should be a list of lists
            objs, poses = extract_objects_and_positions(world, labels_to_idx);

            if True:
                rand_idx = np.random.randint(0, args.batch_size); # sample a random index from current batch
                print([train_i2w[k.item()] for k in hint_seq[rand_idx]]); # get hint in words
                print(label[rand_idx])
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
                return 0;

            # Load hint
            hint_seq = hint_seq.to(device)
            hint_length = hint_length.to(device)
            max_hint_length = hint_length.max().item()
            # Cap max length if it doesn't fill out the tensor
            if max_hint_length != hint_seq.shape[1]:
                hint_seq = hint_seq[:, :max_hint_length]

            hint_rep = hint_model(hint_seq, hint_length); # --> N x C

            # Learn representations of images and examples
            image_slot = image_part_model(image); # --> N x n_slot x C
            # image_whole = image_whole_model(image_slot); # --> N x n_slot x C

            examples_slot = image_part_model(examples); # --> N x n_ex x n_slot x C
            # examples_whole = image_whole_model(examples); # --> N x n_ex x n_slot x C

            score = im_im_scorer_model.score(examples_slot.mean(dim=[1,2]), image_slot.mean(dim=1));
            pred_loss = F.binary_cross_entropy_with_logits(score, label.float());
            pred_loss_total += pred_loss

            slot_cls_score = image_cls_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
            slot_pos_pred = image_pos_projection(torch.cat([examples_slot, image_slot.unsqueeze(1)], dim=1)).flatten(0,1);
 
            losses, acc = set_loss({'pred_logits': slot_cls_score, 'pred_poses': slot_pos_pred},
                                {'labels': objs, 'poses': poses});

            # Hypothesis loss
            loss = args.concept_lambda*pred_loss + args.hypo_lambda*(losses['class'] + args.pos_weight*losses['position'])

            cls_loss_total += losses['class'].item()
            pos_loss_total += losses['position'].item()
            cls_acc += acc;
            # cls_acc += (torch.argmax(slot_cls_score, dim=-1)[]==).float().mean();

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f}'.format(
                    epoch, loss.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tConcept Loss: {:.4f} Classification Loss: {:.4f} Position Loss: {:.4f} Classification Acc: {:.4f}'.format('(train)', epoch, pred_loss_total, cls_loss_total, pos_loss_total, cls_acc));

        return args.concept_lambda*pred_loss_total + args.hypo_lambda*(cls_loss_total + args.pos_weight*pos_loss_total)

    def test(epoch, split='train'):
        for m in models_to_save:
            if (isinstance(m, nn.Module)):
                m.eval();

        accuracy_meter = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                label_np = label.cpu().numpy().astype(np.uint8)
                batch_size = len(image)

                image_rep = image_part_model(image).mean(dim=1);
                examples_rep = image_part_model(examples).mean(dim=[1,2]);
           
                score = im_im_scorer_model.score(examples_rep, image_rep)

                label_hat = score > 0
                label_hat = label_hat.cpu().numpy()

                accuracy = accuracy_score(label_np, label_hat)
                accuracy_meter.update(accuracy,
                                      batch_size,
                                      raw_scores=(label_hat == label_np))

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}'.format(
            '({})'.format(split), epoch, accuracy_meter.avg))

        return accuracy_meter.avg, accuracy_meter.raw_scores

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_val_tre = 0
    best_val_tre_std = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    lowest_val_tre = 1e10
    lowest_val_tre_std = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch);
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
            save_checkpoint([m.state_dict() for m in models_to_save], is_best=is_best_epoch, folder=args.exp_dir);

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

