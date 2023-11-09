import os
import time
import argparse

import torch
from torch import cuda, nn, utils, optim
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from glob import glob
import torchvision as tv
from tqdm import tqdm

from model import Model
from dataset import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the experiment.')
    parser.add_argument('--epochs', type=int, default=201, help='number of training epochs')
    parser.add_argument('--resume', type=str, help='Path to folder containing the checkpoints')
    parser.add_argument('--data_source', required=True, help='path to source images')
    parser.add_argument('--data_target', required=True, help='path to target images')
    parser.add_argument('--eval_only', action='store_true', help='evaluate model')
    parser.add_argument('--data_size', type=int, default=None, help='limit the size of dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--save_freq', type=int, default=20, help='frequency of saving checkpoints')


    opt = parser.parse_args()
    print(opt.__dict__)

    gpu_ids = list(map(int, opt.gpu_ids.split(",")))
    if len(gpu_ids) > 1:
        multi_gpu = True
    else:
        multi_gpu = False
    if gpu_ids[0] == -1: 
        device = torch.device('cpu')
    else:
        if multi_gpu:
            device = torch.device('cuda') # use all gpus
        else:
            device = torch.device('cuda:{}'.format(gpu_ids[0])) 
    print("[INFO] Current device: {} and MULTI_GPU = {}".format(device, multi_gpu))
    os.makedirs(opt.output_dir, exist_ok=True)
    experiment_dir = os.path.join(opt.output_dir, opt.name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'transformed_dataset'), exist_ok=True)

    print('Storing results to ', experiment_dir)

    dataset = Dataset(opt.data_source, opt.data_target, opt.load_size, opt.crop_size, opt.data_size)
    dataloader = utils.data.DataLoader(dataset=dataset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=True,
                                       drop_last=True)

    print('Number of batches:', len(dataloader))
    model = Model() # .to(device)
    if multi_gpu:
        model = nn.DataParallel(model, gpu_ids)
        model.to(device)

        params_gen = list(model.module.encoder.parameters()) + list(model.module.decoder.parameters())
        optimizer_gen = optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        params_dis_a = model.module.dis_a.parameters()
        optimizer_dis_a = optim.Adam(params_dis_a, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        params_dis_b = model.module.dis_b.parameters()
        optimizer_dis_b = optim.Adam(params_dis_b, lr=opt.lr, betas=(opt.beta1, opt.beta2))
    else:
        model.to(device)
        params_gen = list(model.encoder.parameters()) + list(model.decoder.parameters())
        optimizer_gen = optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        params_dis_a = model.dis_a.parameters()
        optimizer_dis_a = optim.Adam(params_dis_a, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        params_dis_b = model.dis_b.parameters()
        optimizer_dis_b = optim.Adam(params_dis_b, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    if opt.resume:
        default_ckpt = 180
        if multi_gpu:
            model.module.load(opt.resume, default_ckpt)
        else:
            model.load(opt.resume, default_ckpt)

    criterion_gan = nn.MSELoss().to(device)
    criterion_rec = nn.L1Loss().to(device)
    if opt.resume or opt.eval_only:
        s_a = torch.load(os.path.join(experiment_dir, 'checkpoints', 's_a_24'))[0, :, :, :, :]
        s_b = torch.load(os.path.join(experiment_dir, 'checkpoints', 's_b_24'))[0, :, :, :, :]
        shape = [-1, 2, 256, 1, 1]  # [-1, 2, channel, 1, 1]
        s_a = s_a.view(shape).to(device)
        s_b = s_b.view(shape).to(device)
    else:
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
    if opt.eval_only and opt.resume:
        print("[INFO] Evaluating model on the source dataset.")
        model.eval()
        for iters, (dataA, dataB) in enumerate(tqdm(dataloader)):
            dataA = dataA.to(device)
            dataB = dataB.to(device)

            x_a = Variable(dataA).to(device)
            x_b = Variable(dataB).to(device)
            output = model(x_a, x_b, s_a, s_b)
            x_aa, x_bb, x_ab, x_ba, x_aba, x_bab, pred_real_a, pred_real_b, pred_fake_ab, pred_fake_ba = output
            # res = torch.cat((x_a.detach().cpu(),
            #                 x_ab.detach().cpu()))
            image_grid = tv.utils.make_grid(x_ab.detach(), nrow=opt.batch_size, normalize=True)
            tv.utils.save_image(image_grid, os.path.join(experiment_dir, 
                                                         'transformed_dataset', 
                                                         '{:04}.png'.format(iters)))
    else:
        for epoch in range(opt.epochs):
            epoch_start = time.time()

            for iters, (dataA, dataB) in enumerate(tqdm(dataloader)):
                start = time.time()
                # if idx == 1: break
                dataA = dataA.to(device)
                dataB = dataB.to(device)

                x_a = Variable(dataA).to(device)
                x_b = Variable(dataB).to(device)

                #  Train Encoders and Generators
                optimizer_gen.zero_grad()

                output = model(x_a, x_b, s_a, s_b)
                x_aa, x_bb, x_ab, x_ba, x_aba, x_bab, pred_real_a, pred_real_b, pred_fake_ab, pred_fake_ba = output

                loss_rec_aa = criterion_rec(x_aa, x_a)
                loss_rec_bb = criterion_rec(x_bb, x_b)
                loss_rec_aba = criterion_rec(x_aba, x_a)
                loss_rec_bab = criterion_rec(x_bab, x_b)

                loss_adv_a = criterion_gan(pred_fake_ba, ones)
                loss_adv_b = criterion_gan(pred_fake_ab, ones)

                loss_rec = 10*(loss_rec_aa + loss_rec_bb + loss_rec_aba + loss_rec_bab)
                loss_adv = 1*(loss_adv_a + loss_adv_b)

                loss = loss_rec + loss_adv
                loss.backward(retain_graph=True)
                #         loss.backward()
                optimizer_gen.step()

                losses_rec_aa.append(loss_rec_aa.item())
                losses_rec_bb.append(loss_rec_bb.item())
                losses_rec_aba.append(loss_rec_aba.item())
                losses_rec_bab.append(loss_rec_bab.item())

                losses.append(loss.item())
                losses_adv.append(loss_adv.item())

                # Train Discriminator A
                optimizer_dis_a.zero_grad()
                loss_a_real = criterion_gan(pred_real_a, ones)
                loss_a_fake = criterion_gan(pred_fake_ba, zeros)
                
                loss_dis_a = loss_a_real + loss_a_fake  # * 0.5
                loss_dis_a.backward()
                optimizer_dis_a.step()
                losses_dis_a.append(loss_dis_a.item())

                # Train Discriminator B
                optimizer_dis_b.zero_grad()
                loss_b_real = criterion_gan(pred_real_b, ones)
                loss_b_fake = criterion_gan(pred_fake_ab, zeros)

                loss_dis_b = 1*(loss_b_real + loss_b_fake)  # * 0.5

                loss_dis_b.backward()
                optimizer_dis_b.step()
                losses_dis_b.append(loss_dis_b.item())

                if not (iters % 10):
                    print('[%d/%d;%d/%d]: gen: %.3f, rec_aa: %.3f, rec_bb: %.3f, gen_adv: %.3f, dis_a: %.3f, dis_b: %.3f'
                        % (iters, len(dataloader),
                            (epoch), opt.epochs,
                            torch.mean(torch.FloatTensor(losses[:])),
                            torch.mean(torch.FloatTensor(losses_rec_aa[:])),
                            torch.mean(torch.FloatTensor(losses_rec_bb[:])),
                            torch.mean(torch.FloatTensor(losses_adv[:])),
                            torch.mean(torch.FloatTensor(losses_dis_a[:])),
                            torch.mean(torch.FloatTensor(losses_dis_b[:])),
                            ))


            if not (epoch % opt.save_freq):
                model.module.save(os.path.join(experiment_dir, 'checkpoints'), postfix=epoch)
                torch.save(s_a, os.path.join(experiment_dir, 'checkpoints', 's_a_{}'.format(opt.batch_size)))
                torch.save(s_b, os.path.join(experiment_dir, 'checkpoints', 's_b_{}'.format(opt.batch_size)))

            print('Time: ', time.time() - epoch_start)

            # Store image results

            res = torch.cat((x_a.detach().cpu(),
                            x_aa.detach().cpu(),
                            x_ab.detach().cpu(),
                            x_aba.detach().cpu(),
                            x_b.detach().cpu(),
                            x_bb.detach().cpu(),
                            x_ba.detach().cpu(),
                            x_bab.detach().cpu()))

            image_grid = tv.utils.make_grid(res, nrow=opt.batch_size, normalize=True)
            tv.utils.save_image(image_grid, os.path.join(experiment_dir, 'images', 'grid_epoch_' + str(epoch) + '.png'))
        