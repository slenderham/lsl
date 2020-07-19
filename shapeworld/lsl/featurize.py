import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torchvision import  models, transforms
from vision import Flatten
from models import ExWrapper, ImageRep

def VGG16():
    model = models.vgg16_bn(pretrained=True).features;
    model.add_module('avgpool', nn.AvgPool2d(3, 2));
    model.add_module('flatten', Flatten());
    model.final_feat_dim = 4608;
    return model;

image_model = ExWrapper(ImageRep(VGG16(), hidden_size=None));
image_model.eval()

N_FEATS = 4608
DATA_DIR = '/data/cw9951/hard_sw'

preprocess = True;
batch_size = 16;
if (torch.cuda.is_available()):
    device = torch.device('cuda');
else:
    device = torch.device('cpu');

image_model.to(device);
if preprocess:
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

with torch.no_grad():
    for split in ("train", "val", "test", "val_same", "test_same"):
        print(split);

        ex = np.load("{}/shapeworld/{}/examples.npz".format(DATA_DIR, split))['arr_0']
        ex = np.transpose(ex, (0, 1, 4, 2, 3))
        n_inp = ex.shape[0]
        n_ex = ex.shape[1]
        ex_feats = np.zeros((n_inp, n_ex, N_FEATS))
        for i in range(0, n_inp, batch_size):
            if i % 200 == 0:
                print(i);
            batch = ex[i:i+batch_size, ...]
            n_batch = batch.shape[0]
            batch = torch.from_numpy(batch).float().to(device)
            if preprocess:
                batch = batch.reshape(n_batch*n_ex, batch.shape[2], batch.shape[3], batch.shape[4]).cpu();
                batch = torch.stack([preprocess_transform(b) for b in batch]).to(device)
                batch = batch.reshape(n_batch, n_ex, batch.shape[1], 224, 224);
            feats = image_model(batch).cpu().numpy();
            ex_feats[i:i+batch_size, ...] = feats
        np.savez("{}/shapeworld/{}/examples.vggfeats.npz".format(DATA_DIR, split), ex_feats);

        inp = np.load("{}/shapeworld/{}/inputs.npz".format(DATA_DIR, split))['arr_0']
        inp = np.transpose(inp, (0, 3, 1, 2))
        n_inp = inp.shape[0]
        inp_feats = np.zeros((n_inp, N_FEATS))
        for i in range(0, n_inp, batch_size):
            if i % 200 == 0:
                print(i)
            batch = inp[i:i+batch_size, ...]
            batch = torch.from_numpy(batch).float().cpu()
            if preprocess:
                batch = torch.stack([preprocess_transform(b) for b in batch]).to(device);
            feats = image_model(batch).cpu().numpy()
            feats = feats.reshape((-1, N_FEATS))
            inp_feats[i:i+batch_size, :] = feats
        np.savez("{}/shapeworld/{}/inputs.vggfeats.npz".format(DATA_DIR, split), inp_feats)
