import argparse
import os
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from torch.utils import data
import numpy as np
from dataset import DeepFashionDataset


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def generate(args, g_ema, device, mean_latent, loader):
    loader = sample_data(loader)
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            data = next(loader)

            input_image = data['input_image'].float().to(device)
            real_img = data['target_image'].float().to(device)
            pose = data['target_pose'].float().to(device)
            sil = data['target_sil'].float().to(device)

            source_sil = data['input_sil'].float().to(device)
            complete_coor = data['complete_coor'].float().to(device)

            if args.size == 256:
                complete_coor = torch.nn.functional.interpolate(complete_coor, size=(256,256), mode='bilinear')

            appearance = torch.cat([input_image, source_sil, complete_coor], 1)

            sample, _ = g_ema(appearance=appearance, pose=pose)

            RP = data['target_right_pad']
            LP = data['target_left_pad']

            utils.save_image(
                sample[:, :, :, int(RP[0].item()):args.size-int(LP[0].item())],
                os.path.join(args.save_path, data['save_name'][0]),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate reposing results")

    parser.add_argument("path", type=str, help="path to dataset")
    parser.add_argument("--size", type=int, default=512, help="output image size of the generator")
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--pretrained_model", type=str, default="posewithstyle.pt", help="pose with style pretrained model")
    parser.add_argument("--save_path", type=str, default="output", help="path to save output .data/output")

    args = parser.parse_args()

    args.latent = 2048
    args.n_mlp = 8

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.pretrained_model)
    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    dataset = DeepFashionDataset(args.path, 'test', args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        sampler=data.SequentialSampler(dataset),
        drop_last=False,
    )

    print ('Testing %d images...'%len(dataset))
    args.pics = len(dataset)

    generate(args, g_ema, device, mean_latent, loader)
