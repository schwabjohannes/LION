#%% This example shows how to train NETT for sparse angle, noisy measurements.


#%% Imports
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils


import pathlib

from LION.models.learned_regularizer.NETT import ANETT_ConvNet
from LION.utils.parameter import LIONParameter
from LION.optimizers.supervised_learning import ANETTSolver
import LION.experiments.ct_experiments as ct_experiments
from skimage.metrics import structural_similarity as ssim


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


#%%
# % Chose device:
device = torch.device("cuda:2")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/js2771/trained_models/ANETT_tests/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("ANETT_ConvNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ANETT_ConvNet_check_*.pt")
validation_fname = savefolder.joinpath("ANETT_ConvNet_min_val.pt")


##%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

# smaller dataset for example. Remove this for full dataset
#indices = torch.arange(100)
#lidc_dataset = data_utils.Subset(lidc_dataset, indices)
#lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

def anett_loss(latent,reconstruction,target,q,noisy):
    loss = torch.sqrt(torch.sum((reconstruction-target)**2)) + (1-noisy)*torch.sum(torch.abs(latent)**q)**(1/q)
    return loss

#%% Define DataLoader
# Use the same amount of training
batch_size = 4
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, 1, shuffle=False)
lidc_testing = DataLoader(experiment.get_testing_dataset(), 1, shuffle=False)
#%% Model
# Default model is already from the paper.
model = ANETT_ConvNet(experiment.geo).to(device)

#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = anett_loss
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = anett_loss
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)


#%% Train
# create solver
solver = ANETTSolver(model, optimiser, loss_fcn, verbose=True)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname)
solver.set_testing(lidc_testing, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(savefolder, checkpoint_fname, 10, load_checkpoint=False)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
#solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname)

# test

#solver.test()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")
