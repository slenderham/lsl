"""
Training script
"""

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from matplotlib import pyplot as plt

from utils import (
    AverageMeter,
    save_defaultdict_to_fs,
)
from datasets import ShapeWorld, extract_features
from datasets import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from models import ImageRep, TextRep, TextProposal, ExWrapper, MLP, Identity
from models import MultimodalRep
from models import DotPScorer, BilinearScorer, ContrastiveLoss, CosineScorer
from vision import Conv4NP, ResNet18, Conv4NP, Conv4NPPosAware

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='Output directory')
    parser.add_argument('--backbone',
                        choices=['vgg16_fixed', 'conv4', 'resnet18', 'conv4posaware'],
                        default='conv4',
                        help='Image model')
    parser.add_argument('--loss_type',
                        choices=['cpc', 'margin'],
                        default='cpc',
                        help='Form of loss function')
    parser.add_argument('--comparison',
                        choices=['dotp', 'bilinear', 'cosine'],
                        default='dotp',
                        help='How to compare support to query reps')
    parser.add_argument('--dropout',
                        default=0.0,
                        type=float,
                        help='Apply dropout to comparison layer')
    parser.add_argument('--temperature',
                        default=0.5,
                        type=float,
                        help='Temperature parameter used in contrastive loss')
    parser.add_argument('--debug_bilinear',
                        action='store_true',
                        help='If using bilinear term, use identity matrix')
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Max number of training examples')
    parser.add_argument('--noise',
                        type=float,
                        default=0.0,
                        help='Amount of noise to add to each example')
    parser.add_argument('--class_noise_weight',
                        type=float,
                        default=0.0,
                        help='How much of that noise should be class diagnostic?')
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
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='Size of hidden representations')
    parser.add_argument('--epochs', type=int, default=30, help='Train epochs')
    parser.add_argument('--debug_example', 
                        action="store_true",
                        help="If true, print out example images and hint");
    parser.add_argument('--save_feats',
                        action="store_true",
                        help="If true, store precomputed features. Otherwise, directly evaluate using chosen metric.")
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
    parser.add_argument('--save_checkpoint',
                        action='store_true',
                        help='Save model')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='Enables CUDA training')
    args = parser.parse_args()

    if args.dropout > 0.0 and args.comparison == 'dotp':
        raise NotImplementedError

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
    preprocess = args.backbone == 'resnet18'
    precomputed_features = None;
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
                             precomputed_features=precomputed_features,
                             vocab=train_vocab,
                             preprocess=preprocess,
                             noise=test_noise,
                             class_noise_weight=0.0,
                             noise_type=args.noise_type,
                             data_dir=args.data_dir)
    test_dataset = ShapeWorld(split='test',
                              precomputed_features=precomputed_features,
                              vocab=train_vocab,
                              preprocess=preprocess,
                              noise=test_noise,
                              class_noise_weight=0.0,
                              noise_type=args.noise_type,
                              data_dir=args.data_dir)
    try:
        val_same_dataset = ShapeWorld(
            split='val_same',
            precomputed_features=precomputed_features,
            vocab=train_vocab,
            preprocess=preprocess,
            noise=test_noise,
            class_noise_weight=0.0,
            noise_type=args.noise_type,
            data_dir=args.data_dir)
        test_same_dataset = ShapeWorld(
            split='test_same',
            precomputed_features=precomputed_features,
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

    """
    Vision Embedding
    """
    if args.backbone == 'vgg16_fixed':
        backbone_model = None
    elif args.backbone == 'conv4':
        backbone_model = Conv4NP()
    elif args.backbone == 'resnet18':
        backbone_model = ResNet18()
    else:
        raise NotImplementedError(args.backbone)

    image_model = ExWrapper(ImageRep(backbone_model, hidden_size=args.hidden_size));
    image_model = image_model.to(device)
    params_to_optimize = list(image_model.parameters())

    """
    Loss
    """
    
    criterion = ContrastiveLoss(loss_type=args.loss_type, temperature=args.temperature);

    """
    Scorer Model
    """

    if args.comparison == 'dotp':
        scorer_model = DotPScorer()
    elif args.comparison == 'bilinear':
        # FIXME: This won't work with --poe
        scorer_model = BilinearScorer(512,
                                      dropout=args.dropout,
                                      identity_debug=args.debug_bilinear)
    elif args.comparison == "cosine":
        scorer_model = CosineScorer(temperature=1);
    else:
        raise NotImplementedError
    scorer_model = scorer_model.to(device)
    params_to_optimize.extend(scorer_model.parameters())

    """
    Projection heads
    """

    image_projection = ExWrapper(MLP(args.hidden_size, args.hidden_size//2, args.hidden_size)).to(device);
    hint_projection = MLP(args.hidden_size, args.hidden_size//2, args.hidden_size).to(device);
    params_to_optimize.extend(image_projection.parameters());
    params_to_optimize.extend(hint_projection.parameters());

    """
    Language Model
    """

    embedding_model = nn.Embedding(train_vocab_size, args.hidden_size);
    hint_model = TextRep(embedding_model, hidden_size=args.hidden_size);
    hint_model = hint_model.to(device)
    params_to_optimize.extend(hint_model.parameters())

    # optimizer and loss
    optfunc = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'sgd': optim.SGD
    }[args.optimizer]

    optimizer = optfunc(params_to_optimize, lr=args.lr)
    print(sum([torch.numel(p) for p in params_to_optimize]));


    def train(epoch, n_steps=100):
        image_model.train()
        hint_model.train()

        loss_total = 0
        pos_score_total = 0
        neg_score_total = 0
        pbar = tqdm(total=n_steps)
        for batch_idx in range(n_steps):
            examples, image, label, hint_seq, hint_length, *rest = \
                train_dataset.sample_train(args.batch_size)

            examples = examples.to(device) # N x n_ex x C x H x W
            image = image.to(device) # N x C x H x W
            label = label.to(device) # N
            batch_size = len(image)
            n_ex = examples.shape[1]

            if args.debug_example:
                rand_idx = np.random.randint(0, args.batch_size); # sample a random index from current batch
                print([train_i2w[k.item()] for k in hint_seq[rand_idx]]); # get hint in words
                print(examples[rand_idx][0].shape)
                fig, axes = plt.subplots(4);
                for i in range(4):
                    axes[i].imshow(examples[rand_idx][i].transpose(0, 2)); # plot examples, transpose to put channel in the last dim
                plt.show();
                return 0;
            # Load hint
            hint_seq = hint_seq.to(device)
            hint_length = hint_length.to(device)
            max_hint_length = hint_length.max().item();
            # Cap max length if it doesn't fill out the tensor
            if max_hint_length != hint_seq.shape[1]:
                hint_seq = hint_seq[:, :max_hint_length]

            # Learn representations of images and examples
            image_rep = image_model(image); # N x H
            examples_rep = image_model(examples); # N x n_ex x H
            examples_rep_mean = torch.mean(examples_rep, dim=1); # N x H

            # Encode hints, minimize distance between hint and images/examples
            hint_rep = hint_model(hint_seq, hint_length) # N * H

            loss, pos_score, neg_score = criterion(image_projection(examples_rep), hint_projection(hint_rep));

            loss_total += loss.item()
            pos_score_total += pos_score.item();
            neg_score_total += neg_score.item();

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 20.0);
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch {} Loss: {:.6f} pos-score {:.6f} neg-score {:.6f}'.format(
                    epoch, loss.item(), pos_score.item(), neg_score.item()))
                pbar.refresh()

            pbar.update()
        pbar.close()
        print('====> {:>12}\tEpoch: {:>3}\tLoss: {:.4f}\tPositive Score {:.4f}\tNegative Score {:.4f}'.format(
            '(train)', epoch, loss_total, pos_score_total, neg_score_total))

        return loss_total

    def test(epoch, split='train'):
        image_model.eval()
        scorer_model.eval()

        accuracy_meter = AverageMeter(raw=True)
        data_loader = data_loader_dict[split]

        with torch.no_grad():
            for examples, image, label, hint_seq, hint_length, *rest in data_loader:
                examples = examples.to(device)
                image = image.to(device)
                label = label.to(device)
                label_np = label.cpu().numpy().astype(np.uint8)
                batch_size = len(image)

                image_rep = image_model(image)
                examples_rep = image_model(examples)
                examples_rep_mean = torch.mean(examples_rep, dim=1)

                score = scorer_model.score(examples_rep_mean, image_rep);

                label_hat = torch.as_tensor(score > 0, dtype=torch.uint8);
                label_hat = label_hat.cpu().numpy();

                accuracy = accuracy_score(label_np, label_hat)
                accuracy_meter.update(accuracy,
                                    batch_size,
                                    raw_scores=(label_hat == label_np))

        print('====> {:>12}\tEpoch: {:>3}\tAccuracy: {:.4f}'.format(
            '({})'.format(split), epoch, accuracy_meter.avg))

        return accuracy_meter.avg, accuracy_meter.raw_scores

    def featurize():
        image_model.eval()
        N_FEATS = args.hidden_size
        # DATA_DIR = '/Users/wangchong/Downloads/hard_sw'
        DATA_DIR = '/data/cw9951/easy_sw'

        with torch.no_grad():
            for split in ("train", "val", "test", "val_same", "test_same"):
                print(split);
                data_loader = data_loader_dict[split]

                ex = np.load("{}/shapeworld/{}/examples.npz".format(DATA_DIR, split))['arr_0']
                ex = np.transpose(ex, (0, 1, 4, 2, 3))
                n_inp = ex.shape[0]
                n_ex = ex.shape[1]
                ex_feats = np.zeros((n_inp, n_ex, N_FEATS))
                for i in range(0, n_inp, args.batch_size):
                    if i % 1000 == 0:
                        print(i);
                    batch = ex[i:i+args.batch_size, ...]
                    n_batch = batch.shape[0]
                    batch = torch.from_numpy(batch).float().to(device);
                    feats = image_model(batch).cpu().numpy();
                    ex_feats[i:i+args.batch_size, ...] = feats
                np.savez("{}/shapeworld/{}/examples.feats.npz".format(DATA_DIR, split), ex_feats);

                inp = np.load("{}/shapeworld/{}/inputs.npz".format(DATA_DIR, split))['arr_0']
                inp = np.transpose(inp, (0, 3, 1, 2))
                n_inp = inp.shape[0]
                inp_feats = np.zeros((n_inp, N_FEATS))
                for i in range(0, n_inp, args.batch_size):
                    if i % 1000 == 0:
                        print(i)
                    batch = inp[i:i+args.batch_size, ...]
                    batch = torch.from_numpy(batch).float().to(device)
                    feats = image_model(batch).cpu().numpy()
                    feats = feats.reshape((-1, N_FEATS))
                    inp_feats[i:i+args.batch_size, :] = feats
                np.savez("{}/shapeworld/{}/inputs.feats.npz".format(DATA_DIR, split), inp_feats)

    best_epoch = 0
    best_epoch_acc = 0
    best_val_acc = 0
    best_val_same_acc = 0
    best_test_acc = 0
    best_test_same_acc = 0
    best_test_acc_ci = 0
    metrics = defaultdict(lambda: [])

    save_defaultdict_to_fs(vars(args), os.path.join(args.exp_dir, 'args.json'))
    total_epoch = 1 if args.debug_example else args.epochs;
    for epoch in range(1, total_epoch + 1):
        train_loss = train(epoch);
        if args.save_feats:
            continue;
        train_acc, _ = test(epoch, 'train')
        val_acc, _ = test(epoch, 'val')
        # Evaluate tre on validation set
        #  val_tre, val_tre_std = eval_tre(epoch, 'val')

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
        is_best_epoch = epoch_acc > best_val_acc
        if is_best_epoch:
            best_epoch = epoch
            best_epoch_acc = epoch_acc
            best_val_acc = val_acc
            best_val_same_acc = val_same_acc
            best_test_acc = test_acc
            best_test_same_acc = test_same_acc
            best_test_acc_ci = test_acc_ci

        if args.save_checkpoint:
            raise NotImplementedError

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

    if (args.save_feats):
        featurize();
    else:
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
