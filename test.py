import os
import time
import argparse
import torch
from torch import cuda, nn, utils, optim
from torch.autograd import Variable
from glob import glob
import torchvision as tv
from tqdm import tqdm

from model import Model, Encoder, Decoder
from dataset_test import Dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the experiment.')
    parser.add_argument('--data_source', required=True, help='path to source images')
    parser.add_argument('--weights', required=True, help='Path to the weights folder')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--output_dir', required=True, help='path to output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

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

    print('Fetching models from ', experiment_dir)
    checkpoints = glob(os.path.join(opt.weights, '*.pth'))
    encoder_path = [s for s in checkpoints if "encoder_180" in s]
    decoder_path = [s for s in checkpoints if "decoder_180" in s]
    disc_a = [s for s in checkpoints if "discriminator_a_180" in s]
    disc_b = [s for s in checkpoints if "discriminator_b_180" in s]

    dataset = Dataset(opt.data_source)
    dataloader = utils.data.DataLoader(dataset=dataset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=False,
                                       drop_last=False)

    print('Number of batches:', len(dataloader))

    model = Model() 
    if multi_gpu:
        model.to(device)
        model = nn.DataParallel(model, gpu_ids)
        print('Loading encoder weights from ', encoder_path[0])
        model.module.encoder.load_state_dict(torch.load(encoder_path[0]))
        print('Loading decoder weights from ', decoder_path[0])
        model.module.decoder.load_state_dict(torch.load(decoder_path[0]))
        model.module.dis_a.load_state_dict(torch.load(disc_a[0]))
        model.module.dis_b.load_state_dict(torch.load(disc_b[0]))
    else:
        model.to(device)
        print('Loading encoder weights from ', encoder_path[0])
        model.encoder.load_state_dict(torch.load(encoder_path[0]))
        print('Loading decoder weights from ', decoder_path[0])
        model.decoder.load_state_dict(torch.load(decoder_path[0]))
        model.dis_a.load_state_dict(torch.load(disc_a[0]))
        model.dis_b.load_state_dict(torch.load(disc_b[0]))

    model.eval()
    s_a = torch.load(os.path.join(experiment_dir, 'checkpoints', 's_a_24'))[0, :, :, :, :]
    s_b = torch.load(os.path.join(experiment_dir, 'checkpoints', 's_b_24'))[0, :, :, :, :]
    shape = [-1, 2, 256, 1, 1]  # [-1, 2, channel, 1, 1]
    s_a = s_a.view(shape).to(device)
    s_b = s_b.view(shape).to(device)


    for iters, (data, filepaths) in enumerate(tqdm(dataloader)):
        data = data.to(device)
        x_a = Variable(data).to(device)
        c_a = model.encoder(x_a) 
        x_ab = model.decoder(c_a, s_b)
        for image, filepath in zip(range(x_ab.shape[0]), filepaths): # iterate along batch dimension
            filename = os.path.basename(filepath)
            trans_image = tv.utils.make_grid(x_ab[image, :, :, :].detach().cpu(), nrow=1, normalize=True)
            tv.utils.save_image(trans_image, os.path.join(experiment_dir,
                                                          'transformed_dataset',
                                                          filename))