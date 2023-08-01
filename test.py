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
    parser.add_argument('--data_target', required=True, help='path to target images')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    opt = parser.parse_args()
    print(opt.__dict__)

    gpu_ids = list(map(int, opt.gpu_ids.split(",")))
    if gpu_ids[0] == -1: 
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu_ids[0])) 


    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    experiment_dir = os.path.join(opt.output_dir, opt.name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        os.makedirs(os.path.join(experiment_dir, 'images'))
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'))

    print('Fetching models from ', experiment_dir)
    checkpoints = glob(os.path.join(opt.weights, '*.pth'))
    encoder_path = [s for s in checkpoints if "encoder_180" in s]
    decoder_path = [s for s in checkpoints if "decoder_180" in s]
    disc_a = [s for s in checkpoints if "discriminator_a_180" in s]
    disc_b = [s for s in checkpoints if "discriminator_b_180" in s]

    dataset = Dataset(opt.data_source, opt.data_target)
    dataloader = utils.data.DataLoader(dataset=dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       drop_last=True)

    print('Number of batches:', len(dataloader))

    model = Model() # .to(device)
    model.to(device)
    multi_gpu = False
    if multi_gpu:
        model = nn.DataParallel(model, gpu_ids)
        print('Loading encoder weights from ', encoder_path[0])
        model.module.encoder.load_state_dict(torch.load(encoder_path[0]))
        print('Loading decoder weights from ', decoder_path[0])
        model.module.decoder.load_state_dict(torch.load(decoder_path[0]))
        model.module.dis_a.load_state_dict(torch.load(disc_a[0]))
        model.module.dis_b.load_state_dict(torch.load(disc_b[0]))
        # model.to(device)
    else:
        print('Loading encoder weights from ', encoder_path[0])
        model.encoder.load_state_dict(torch.load(encoder_path[0]))
        print('Loading decoder weights from ', decoder_path[0])
        model.decoder.load_state_dict(torch.load(decoder_path[0]))
        model.dis_a.load_state_dict(torch.load(disc_a[0]))
        model.dis_b.load_state_dict(torch.load(disc_b[0]))
        # model.to(device)
    model.eval()

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
    with torch.no_grad():
        for iters, (dataA, dataB) in enumerate(dataloader):
            dataA = dataA.to(device)
            dataB = dataB.to(device)
            x_a = Variable(dataA).to(device)
            x_b = Variable(dataB).to(device)
            output = model(x_a, x_b, s_a, s_b)
            x_aa, x_bb, x_ab, x_ba, x_aba, x_bab, pred_real_a, pred_real_b, pred_fake_ab, pred_fake_ba = output
            res = torch.cat((x_a.detach().cpu(),
                        x_aa.detach().cpu(),
                        x_ab.detach().cpu(),
                        x_aba.detach().cpu(),
                        x_b.detach().cpu(),
                        x_bb.detach().cpu(),
                        x_ba.detach().cpu(),
                        x_bab.detach().cpu()))
            # tv.utils.save_image(res, os.path.join(experiment_dir, 'images', 'grid_epoch_{}.png'.format(iters)))
            tv.utils.save_image(x_a.detach().cpu(), os.path.join(experiment_dir, 'source.png'))
            tv.utils.save_image(x_ab.detach().cpu(), os.path.join(experiment_dir, 'trans.png'))
            print('hold')