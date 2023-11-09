import os
import time
import argparse


import torch
from torch import cuda, nn, utils, optim
from torch.autograd import Variable
from glob import glob
import torchvision as tv

from model import Model, Encoder, Decoder
from dataset_test import Dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the experiment.')
    parser.add_argument('--data_source', required=True, help='path to source images')
    parser.add_argument('--weights', required=True, help='Path to the weights folder')
    parser.add_argument('--data_size', type=int, default=None, help='limit the size of dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    opt = parser.parse_args()
    print(opt.__dict__)

    gpu_ids = list(map(int, opt.gpu_ids.split(",")))
    device = torch.device('cuda:{}'.format(gpu_ids[0])) #if gpu_ids else torch.device('cpu')

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    experiment_dir = os.path.join(opt.output_dir, opt.name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        os.makedirs(os.path.join(experiment_dir, 'images'))
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'))

    print('Fetching models from ', experiment_dir)
    checkpoints = glob(os.path.join(opt.weights, '*.pth'))
    encoder_path = [s for s in checkpoints if "encoder" in s]
    decoder_path = [s for s in checkpoints if "decoder" in s]

    dataset = Dataset(opt.data_source, opt.load_size)
    dataloader = utils.data.DataLoader(dataset=dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       drop_last=True)

    print('Number of batches:', len(dataloader))

    model = Model() # .to(device)
    model.eval()
    model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    print('Loading encoder weights from ', encoder_path)
    encoder = Encoder()
    encoder.load_state_dict(torch.load(encoder_path[0]))

    print('Loading decoder weights from ', encoder_path)
    decoder = Decoder()
    decoder.load_state_dict(torch.load(decoder_path[0]))

    params_gen = list(model.module.encoder.parameters()) + list(model.module.decoder.parameters())
    optimizer_gen = optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    params_dis_a = model.module.dis_a.parameters()
    optimizer_dis_a = optim.Adam(params_dis_a, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    params_dis_b = model.module.dis_b.parameters()
    optimizer_dis_b = optim.Adam(params_dis_b, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    criterion_gan = nn.MSELoss().to(device)
    criterion_rec = nn.L1Loss().to(device)

    s_a = torch.normal(0, 1, size=(1, 2 * 256)).repeat(opt.batch_size, 1).to(device)
    s_b = torch.normal(0, 1, size=(1, 2 * 256)).repeat(opt.batch_size, 1).to(device)

    shape = [-1, 2, 256, 1, 1]  # [-1, 2, channel, 1, 1]
    s_a = s_a.view(shape)
    s_b = s_b.view(shape)

    losses_rec_aa = []
    losses_rec_bb = []
    losses_rec_aba = []
    losses_rec_bab = []

    losses_adv = []
    losses = []

    losses_dis_a = []
    losses_dis_b = []

    ones = torch.ones((8 * opt.batch_size, 1)).to(device)
    zeros = torch.zeros((8 * opt.batch_size, 1)).to(device)

    for iters, data in enumerate(dataloader):
        dataA = dataA.to(device)
        x_a = Variable(dataA).to(device)