'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in src/models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from scipy.linalg import qr
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os, glob
import argparse
from tqdm.auto import tqdm
import gc

from utils import *

from gbf import GBFRDataset
from DoubleGyre_Loader import DG_Dataset
from HYCOM_Loader import HYCOM_Dataset
from Lorenz_Loader import Lorenz_Dataset
from TNT_Loader import TNT_Dataset
from Kolmogorov_Loader import Kolmogorov_Dataset

from models.s4.s4model import S4Model, setup_optimizer
from models.lstm.lstmmodel import LSTM

parser = argparse.ArgumentParser(description='PyTorch Sensor Reconstruction Training')
# Model
parser.add_argument('--model', default='s4', choices=['s4','tcn','lstm'],type=str, help='Model')
# Training
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--patience', default=0, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
parser.add_argument('--tol', default=10, type=int, help='Tolerance for early stopping')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
# Dataset
parser.add_argument('--dataset', default='dg', choices=['dg', 'gbf', 'hycom', 'lorenz', 'tnt', 'kol'], type=str, help='Dataset')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=2, type=int, help='Number of layers')
parser.add_argument('--n_decoder', default=1, type=int, help='Number of decoder hidden layers')
parser.add_argument('--d_model', default=256, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--d_filter', default=0, type=int, help='Apply n-th order S4-Butterworth on inputs if >0')
parser.add_argument('--loss_len', default=100, type=int, help='Loss length')
parser.add_argument('--act', default='sigmoid', choices=['sigmoid', 'glu', 'tanh'], type=str, help='Activation function')
# S4 parameters
parser.add_argument('--norm', default='batchnorm', choices=['batchnorm', 'layernorm'], type=str, help='Normalization')
parser.add_argument('--feedback', action='store_true', help='Feedback control for S4 model')
parser.add_argument('--mode', default='diag', choices=['diag', 's4'], type=str, help='variants of S4 model')
parser.add_argument('--fixC', action='store_true', help='Fix C matrix')
parser.add_argument('--fixB', action='store_true', help='Fix B matrix')
parser.add_argument('--s4_dropout', default=0.0, type=float, help='S4 Dropout')
# General
parser.add_argument('--test', '-t', action='store_true', help='Testing only')
parser.add_argument('--seed', default=0, type=int, help='Model Seed')
parser.add_argument('--plot', '-p', action='store_true', help='Plot figures')
# Data parameters
parser.add_argument('--pos', default=None, type=str, help='Position Encoding')
parser.add_argument('--seq_t', default=4, type=int, help='Sequence time length')
parser.add_argument('--dt', default=0.01, type=float, help='time step')
parser.add_argument('--noise', default=0.0, type=float, help='Standard deviation of measurement noise')
# Lorenz parameter
parser.add_argument('--lorenz_step', default=1, type=int, help='random walk step size for Lorenz')
# HYCOM parameter
parser.add_argument('--hycom_region', default=None, type=str, help='hycom regional path')
# Kolmogorov parameter
parser.add_argument('--kol_step', default=1, type=int, help='kolmogorov random traj step size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_err = 1e5  # best test accuracy
counter = 0 # counter for early stopping
start_epoch = 0  # start from epoch 0 or last checkpoint epoch  

rng_seed = 373 if args.test else 123 + args.seed #
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

# print arguments
print('==> Options:',args)

# Data
print(f'==> Preparing {args.dataset} data..')

if args.dataset == 'gbf':
    trainset = GBFRDataset(seed=100)
    trainset, valset = split_train_val(trainset, val_split=0.1)

    testset = GBFRDataset(P=P, seed=200, size=200)

    d_input = 1
    d_output = 400

elif args.dataset == 'dg':
    transform_train = [
        # transforms.Lambda(lambda x: AddUnifNoise(x, low=[-args.noise*5,0], high=[args.noise*5,0])),
        transforms.Lambda(lambda x: AddGausNoise(x, std=[args.noise*5,0.0])),
        # transforms.Lambda(lambda x: Normalize(x, [0.0, 0.0], [5.098, 1.0])),
    ]
    transform_test = transform_train.copy()
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    dataset = DG_Dataset(size=2560, seed=100, dt=args.dt, seq_t=args.seq_t, transform=transform_train)
    trainset, _ = split_train_val(dataset, val_split=0.2)
    dataset = DG_Dataset(size=2560, seed=100, dt=args.dt, seq_t=args.seq_t, transform=transform_test)
    _, valset = split_train_val(dataset, val_split=0.2)

    testset = DG_Dataset(size=512, seed=200, dt=args.dt, seq_t=args.seq_t, transform=transform_test)

    d_input = 2
    d_output = dataset.width * dataset.height

elif args.dataset == 'hycom':
    transform_train = [
        transforms.Lambda(lambda x: AddUnifNoise(x, low=[-args.noise*11,0], high=[args.noise*11,0])),
        # transforms.Lambda(lambda x: Normalize(x, [15.208, 0.0], [11.122, 1.0])),
    ]
    transform_test = transform_train.copy()
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    dataset = HYCOM_Dataset(size=12800, seed=100, dt=args.dt, seq_t=args.seq_t, transform=transform_train, region=args.hycom_region)
    trainset, _ = split_train_val(dataset, val_split=0.2)
    dataset = HYCOM_Dataset(size=12800, seed=100, dt=args.dt, seq_t=args.seq_t, transform=transform_test, region=args.hycom_region)
    _, valset = split_train_val(dataset, val_split=0.2)

    testset = HYCOM_Dataset(size=2560, seed=200, dt=args.dt, seq_t=args.seq_t, train=False, transform=transform_test, region=args.hycom_region)

    d_input = 2
    d_output = dataset.nx

elif args.dataset == 'lorenz':
    trainset = Lorenz_Dataset(size=2560, seed=100, dt=args.dt, seq_t=args.seq_t, noise=args.noise, rw_step=args.lorenz_step)
    trainset, valset = split_train_val(trainset, val_split=0.2)
    # print(len(trainset), len(valset))
    testset = Lorenz_Dataset(size=512, seed=200, dt=args.dt, seq_t=args.seq_t, noise=args.noise, train=False, rw_step=args.lorenz_step)

    d_input = 2
    d_output = 128

elif args.dataset == 'tnt':
    filename = 'paths/TNT_{}'.format(args.seq_t)
    if glob.glob(filename):
        dataset = torch.load(filename + '_train.pt')
        testset = torch.load(filename + '_test.pt')
    else:
        dataset = TNT_Dataset(size=2000, seed=100, T=args.seq_t, loss_len=args.loss_len)
        testset = TNT_Dataset(size=400, seed=200, train=False, T=args.seq_t, loss_len=args.loss_len)
        torch.save(dataset, filename + '_train.pt')
        torch.save(testset, filename + '_test.pt')
    trainset, valset = split_train_val(dataset, val_split=0.2)

    d_input = 5
    d_output = 220**2*3

elif args.dataset == 'kol':
    dataset = Kolmogorov_Dataset()
    trainset, valset = split_train_val(dataset, val_split=0.2)
    testset = Kolmogorov_Dataset(train=False)

    # dataset = Kolmogorov_Dataset(size=5120, seed=100, T=args.seq_t, step_size=args.kol_step)
    # trainset, valset = split_train_val(dataset, val_split=0.2)
    # testset = Kolmogorov_Dataset(size=1024, seed=200, T=args.seq_t, step_size=args.kol_step,train=False)

    d_input = 3
    d_output = 128*128

else: raise NotImplementedError

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print(len(trainloader), len(valloader))

# Model
print('==> Building model..')
if args.model == 's4':
    # s4_lr = (min(0.001, args.lr), {})
    s4_lr = (min(0.001, 0.1*args.lr), {})
    if args.fixC: s4_lr[1]['C'] = 0.0
    if args.fixB: s4_lr[1]['B'] = 0.0
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_decoder=args.n_decoder,
        dropout=args.dropout,
        prenorm=args.prenorm,
        pos_encode = args.pos,
        transposed=True, 
        mode=args.mode, 
        d_filter=args.d_filter,
        init='diag-lin' if args.mode == 'diag' else 'legs', 
        s4_lr=s4_lr, 
        s4_dropout=args.s4_dropout,
        normalization=args.norm,
        feedback=args.feedback,
        act_fn=args.act,
    )

    criterion = torch.nn.MSELoss()
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience
    )  
elif args.model == 'lstm':
    model = LSTM(d_input, d_output, args.d_model, n_layers=args.n_layers, n_decoder=args.n_decoder, dropout=args.dropout, act_fn=args.act)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=args.epochs) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('Total number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

model_name = args.model
if model_name == 's4':
    # if args.mode == 'diag':
    #     model_name += 'D'
    if args.fixC:
        model_name += 'r'
    if args.d_filter:
        model_name += 'f' + str(args.d_filter)
save_folder = './checkpoint/{}/{}/{}_{}'.format(model_name, args.dataset, args.seq_t, args.dt) + \
     ('_{}'.format(args.pos) if args.pos is not None else '') + \
     ('_noise{}'.format(args.noise) if args.noise>0 else '') + \
     '/'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_path = save_folder + args.dataset + \
            ('_{}'.format(args.loss_len) if args.loss_len != 100 else '') + \
            ('_{}'.format(args.lorenz_step) if args.dataset == 'lorenz' and args.lorenz_step > 1 else '') + \
            ('_{}'.format(args.hycom_region) if args.dataset == 'hycom' and args.hycom_region else '') 
print('==> Logging at:', save_path + '_ckpt{}.pth'.format(args.seed if args.seed > 0 else ''))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '_ckpt{}.pth'.format(args.seed if args.seed > 0 else ''))
    model.load_state_dict(checkpoint['model'])
    best_err = checkpoint['loss']
    start_epoch = checkpoint['epoch']

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(loss_len=100):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader),disable=True)
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[:,-loss_len:,:], targets[:,-loss_len:,:])
        # loss = criterion(outputs[:,-loss_len:,:], targets)
        # if args.model == 's4' and args.mode == 'diag':
        #     energy = model.energy_loss()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f ' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1))
        )
    train_loss = train_loss/len(trainloader)
    return train_loss

def eval(epoch, dataloader, loss_len=100, mode='val', robust=False, plot=False):
    if robust:
        loss_len = 1
    model.eval()
    eval_loss = 0
    rel_err = [] #0
    count = 0
    diff = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader),disable=True)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            if robust:
                if args.dataset == 'dg':
                    inputs[:,-loss_len,0] = inputs[:,-loss_len,0] - torch.sign(inputs[:,-loss_len,0]) * 5 + torch.randn(inputs.size(0),device=inputs.device) * 1
                    # inputs[:,-loss_len,0] = inputs[:,-loss_len,0] + torch.randn(inputs.size(0),device=inputs.device) * 10
                    # inputs[:,-1,0] = torch.zeros(inputs.size(0),device=inputs.device)
                elif args.dataset == 'hycom':
                    # inputs[:,-loss_len,0] = torch.ones(inputs.size(0),device=inputs.device) * (-15.208/11.122)
                    inputs[:,-loss_len,0] = inputs[:,-loss_len,0] - torch.sign(inputs[:,-loss_len,0]) + torch.randn(inputs.size(0),device=inputs.device) * 0.2
                elif args.dataset == 'tnt':
                    inputs[:,-loss_len,:3] = torch.tensor([136.,3.,25.],device=inputs.device).repeat(inputs.size(0),1)/255
            outputs = model(inputs)
            # loss = criterion(outputs[:,-loss_len:,:], targets)
            loss = criterion(outputs[:,-loss_len:,:], targets[:,-loss_len:,:])
            eval_loss += loss.item()
            # rel_err += torch.sum(torch.norm(outputs[:,-loss_len:,:]-targets[:,-loss_len:,:],dim=2)/torch.norm(targets[:,-loss_len:,:],dim=2)).item()
            rel_err.append((torch.norm(outputs-targets,dim=2)/torch.norm(targets,dim=2)).detach().to('cpu').numpy())
            count += targets.shape[0]*loss_len
            if plot:
                diff += torch.abs(targets[:,-1,:]-outputs[:,-1,:]).flatten().tolist()

            # pbar.set_description(
            #     'Batch Idx: (%d/%d) | Loss: %.6f | Relative Error: %.3f' %
            #     (batch_idx, len(dataloader), eval_loss/(batch_idx+1), rel_err[:,-loss_len:].sum()/count)
            # )
    eval_loss = eval_loss/len(dataloader)
    # rel_err = rel_err / count
    rel_err = np.vstack(rel_err)
    rel_err_plot = rel_err.copy()
    rel_err = rel_err[:,-loss_len:].sum() / count

    if plot:
        plt.figure(figsize=(5,2))
        plot_hist(diff)
        plt.savefig('{}_diff_hist{}{}.png'.format(save_path, '_robust' if robust else '', '_'+mode), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(5,2))
        plt.plot(np.mean(rel_err_plot,axis=0))
        plt.fill_between(np.arange(rel_err_plot.shape[1]),np.min(rel_err_plot,axis=0),np.max(rel_err_plot,axis=0),alpha=0.2)
        plt.xlabel('Time')
        plt.ylabel('Relative Error')
        plt.tight_layout()
        plt.savefig('{}_err{}{}.png'.format(save_path,'_robust' if robust else '','_'+mode), bbox_inches='tight')
        plt.close()

        sample = [0, 5, 10] #if not robust else [0, 5, 10, 15, 20, 15, 30, 35]
        if args.dataset == 'tnt':
            sample = np.arange(7)
        if args.dataset == 'lorenz':
            vmin = torch.min(torch.min(outputs), torch.min(targets))
            vmax = torch.max(torch.max(outputs), torch.max(targets))
            for j in range(len(sample)):
                plt.figure()
                plt.subplot(1,2,1)
                dataset.plot_data(outputs[sample[j],:,:].cpu().numpy(),vmin=vmin,vmax=vmax)
                dataset.plot_traj(inputs[sample[j],:,-1].cpu().numpy())
                plt.subplot(1,2,2)
                dataset.plot_data(targets[sample[j],:,:].cpu().numpy(),vmin=vmin,vmax=vmax)
                dataset.plot_traj(inputs[sample[j],:,-1].cpu().numpy())
                plt.tight_layout()
                plt.savefig('{}_{}_estimates.png'.format(save_path,j), bbox_inches='tight')
                plt.close()

                plt.figure()
                dataset.plot_data(abs(outputs[sample[j],:,:]-targets[sample[j],:,:]).cpu().numpy(), err_cm=True, cbar=True)
                dataset.plot_traj(inputs[sample[j],:,-1].cpu().numpy(),c='w')
                plt.tight_layout()
                plt.savefig('{}_{}_error.png'.format(save_path,j), bbox_inches='tight')
                plt.close()
        else:
            for j in range(len(sample)):
                plt.figure(figsize=(5,2))
                plt.plot(inputs[sample[j],:,:dataset.d_measure].cpu().numpy())
                plt.xlabel('Time')
                plt.ylabel('Measurement')
                plt.tight_layout()
                plt.savefig('{}_{}{}_input{}.png'.format(save_path,j,'_robust' if robust else '','_'+mode), bbox_inches='tight')
                plt.close()

                t_eval = [-1] #[-1,-5,-10] if not robust else [-1]
                for i in range(len(t_eval)):
                    plt.figure()
                    dataset.plot_data(outputs[sample[j],t_eval[i],:].cpu().numpy())
                    plt.tight_layout()
                    plt.savefig('{}_{}_{}_estimate{}{}.png'.format(save_path,j,inputs.size(1)+t_eval[i],'_robust' if robust else '','_'+mode), bbox_inches='tight')
                    plt.close()

                    plt.figure()
                    dataset.plot_data(targets[sample[j],t_eval[i],:].cpu().numpy())
                    dataset.plot_traj(inputs[sample[j],:,dataset.d_measure-d_input:].cpu().numpy(), t=t_eval[i])
                    plt.tight_layout()
                    plt.savefig('{}_{}_{}_target{}{}.png'.format(save_path,j,inputs.size(1)+t_eval[i],'_robust' if robust else '','_'+mode), bbox_inches='tight')
                    plt.close()

                    plt.figure()
                    dataset.plot_data(abs(outputs[sample[j],t_eval[i],:]-targets[sample[j],t_eval[i],:]).cpu().numpy(), err_cm=True, cbar=True)
                    plt.tight_layout()
                    plt.savefig('{}_{}_{}_error{}{}.png'.format(save_path,j,inputs.size(1)+t_eval[i],'_robust' if robust else '','_'+mode), bbox_inches='tight')
                    plt.close()

                    if args.dataset == 'kol':
                        plt.figure()
                        dataset.plot_psd(outputs[sample[j],t_eval[i],:].cpu().numpy())
                        plt.tight_layout()
                        plt.savefig('{}_{}_{}_psd_estimate{}{}.png'.format(save_path,j,inputs.size(1)+t_eval[i],'_robust' if robust else '','_'+mode), bbox_inches='tight')
                        plt.close()

                        plt.figure()
                        dataset.plot_psd(targets[sample[j],t_eval[i],:].cpu().numpy())
                        plt.tight_layout()
                        plt.savefig('{}_{}_{}_psd_target{}{}.png'.format(save_path,j,inputs.size(1)+t_eval[i],'_robust' if robust else '','_'+mode), bbox_inches='tight')
                        plt.close()

    return eval_loss, rel_err, diff

if args.test:
    # plot initial S4 parameters
    if args.plot and args.model == 's4':
        for i in range(len(model.s4_layers)):
            plt.figure()
            model.dynamics_plot(i)
            plt.savefig('{}_layer{}_dynamics_init.png'.format(save_path, i), bbox_inches='tight')
            plt.close()

    if args.plot and args.model == 's4' and args.d_filter:
        bode_folder = save_folder + 'bode/'
        if not os.path.exists(bode_folder):
            os.makedirs(bode_folder)
        for j in range(model.s4_layers[i].d_model):
            plt.figure(figsize=(3,3))
            model.bode_plot(0,j)
            plt.savefig('{}{}_init.png'.format(bode_folder, j), bbox_inches='tight')
            plt.close()

    # Load checkpoint.
    print('==> Loading from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '_ckpt{}.pth'.format(args.seed if args.seed > 0 else ''), map_location=device)
    model.load_state_dict(checkpoint['model'])
    best_err = checkpoint['loss']
    epoch = checkpoint['epoch']

    if args.model == 's4' and args.mode == 'diag':
        energy = model.energy_loss().item()
        print('Model energy:', energy)

    # plot trained S4 parameters
    if args.plot and args.model == 's4':
        for i in range(len(model.s4_layers)):
            plt.figure()
            model.dynamics_plot(i)
            plt.savefig('{}_layer{}_dynamics_trained.png'.format(save_path, i), bbox_inches='tight')
            plt.close()

    # bode plots of S4D-BW
    if args.plot and args.model == 's4' and args.d_filter:
        bode_folder = save_folder + 'bode/'
        if not os.path.exists(bode_folder):
            os.makedirs(bode_folder)
        for j in range(model.s4_layers[i].d_model):
            plt.figure(figsize=(3,3))
            model.bode_plot(0,j)
            plt.savefig('{}{}.png'.format(bode_folder, j), bbox_inches='tight')
            plt.close()

    print('Validation:')
    val_loss, val_err, diff = eval(epoch, valloader, loss_len=args.loss_len, mode='val', plot=args.plot)
    print('Epoch: %d | Val loss (RMSE): %.6f | Val relative error: %.4f' % (epoch, np.sqrt(val_loss), val_err))
    print('Absolute point error: (%.4f, %.4f)' % (np.mean(diff), np.std(diff)))

    print('Testing:')
    val_loss, val_err, diff = eval(epoch, testloader, loss_len=args.loss_len, mode='test', plot=args.plot)
    print('Epoch: %d | Val loss (RMSE): %.6f | Val relative error: %.4f' % (epoch, np.sqrt(val_loss), val_err))
    print('Absolute point error: (%.4f, %.4f)' % (np.mean(diff), np.std(diff)))

    print('Robust Testing:')
    val_loss, val_err, diff = eval(epoch, testloader, mode='test', robust=True, plot=args.plot)
    print('Epoch: %d | Val loss (RMSE): %.6f | Val relative error: %.4f' % (epoch, np.sqrt(val_loss), val_err))
    print('Absolute point error: (%.4f, %.4f)' % (np.mean(diff), np.std(diff)))

else:
    energy = 0
    pbar = tqdm(range(start_epoch, args.epochs))
    for epoch in pbar:
        if epoch == start_epoch:
            pbar.set_description('Epoch: %d' % (epoch))
        else:
            pbar.set_description('Epoch: %d | Train loss: %.6f | Val loss: %.6f | Val relative error: %.4f | Model energy: %.4f' % (epoch, train_loss, val_loss, val_err, energy))
        train_loss = train(loss_len=args.loss_len)
        val_loss, val_err, _ = eval(epoch, valloader, loss_len=args.loss_len)
        if args.model == 's4' and args.mode == 'diag':
            energy = model.energy_loss().item()
        if args.model != 'lstm':
            scheduler.step()

        # Save checkpoint.
        if val_loss < best_err:
            state = {
                'model': model.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, save_path + '_ckpt{}.pth'.format(args.seed if args.seed > 0 else ''))
            best_err = val_loss
            counter = 0
        else:
            counter += 1

        if args.tol > 0 and counter == args.tol:
            print('Terminated Early.')
            break
