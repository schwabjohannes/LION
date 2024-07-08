#%% Noise2Inverse train

#%% Imports

# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# basic python imports
from tqdm import tqdm
import pathlib
import copy

# LION imports
import LION.CTtools.ct_utils as ct
from LION.models.learned_regularizer.NETT import NETT_ConvNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk, ATA_max_eigenvalue, operator_norm
import tomosipo as ts


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

N_iter = 30
regularization_parameter = 0.02
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/js2771/trained_models/NETT_tests/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("NETT_ConvNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("NETT_ConvNet_check_*.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#%% Load model
NETT_model, NETT_param, NETT_data = NETT_ConvNet.load(final_result_fname)

sig_1 = operator_norm(NETT_model.op, num_iter=10)
w = 2/sig_1**2
R = 1 / NETT_model.op(np.ones(NETT_model.op.domain_shape))
R = torch.tensor(np.minimum(R, 1 / ts.epsilon)).to(device)
C = 1 / NETT_model.op.T(np.ones(NETT_model.op.range_shape))
C = torch.tensor(np.minimum(C, 1 / ts.epsilon)).to(device)

print(w)
# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):
    err =[]
    B, C, W, H = sinogram.shape
    x = sinogram.new_zeros(B, 1, *NETT_model.geo.image_shape[1:])
    for i in range(N_iter):
        x -= w * NETT_model.AT((NETT_model.A(x)-sinogram))
        err.append(torch.sum((x[0,0]-target_reconstruction[0,0])**2).detach().cpu())
    err = torch.stack(err)
    
    plt.figure()
    plt.plot(err)
    plt.savefig("error_"+str(index)+".png")
    plt.close()


    x_land = x.clone()
    x_fdk = fdk(NETT_model.op,sinogram[0])
    

    B, C, W, H = sinogram.shape
    x_old = sinogram.new_zeros(B, 1, *NETT_model.geo.image_shape[1:])
    for i in range(N_iter):
        x = x_old - w * NETT_model.AT(NETT_model.A(x_old)-sinogram)
        reg_value = torch.sum(NETT_model.regularizer(x))
        g = torch.autograd.grad(reg_value, x, retain_graph=True)[0].data
        x_old = x.clone()-regularization_parameter*g

    vmi = torch.min(x_land[0,0])
    vma = torch.max(x_land[0,0])


    plt.figure()
    plt.subplot(221)
    plt.imshow(x_land[0,0].detach().cpu().numpy(),vmin = vmi, vmax = vma)
    plt.title('Landweber')
    plt.subplot(222)
    plt.imshow(x_old[0,0].detach().cpu().numpy(),vmin = vmi, vmax = vma)
    plt.title('NETT')
    plt.subplot(223)
    plt.imshow(x_fdk[0].cpu().numpy(),vmin = vmi, vmax = vma)
    plt.title('FDK')
    plt.subplot(224)
    plt.imshow(target_reconstruction[0,0].cpu().numpy(),vmin = vmi, vmax = vma)
    plt.title('target')
    plt.savefig("Recos_"+str(index)+".png")
    plt.close()
    # do whatever you want with this.
