import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import time

from dataset import DeepFashionDataset
from model import Generator, Discriminator, VGGLoss

try:
    import wandb
except ImportError:
    wandb = None


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def getFace(images, FT, LP, RP):
    """
    images: are images where we want to get the faces
    FT: transform to get the aligned face
    LP: left pad added to the imgae
    RP: right pad added to the image
    """
    faces = []
    b, h, w, c = images.shape
    for b in range(images.shape[0]):
        if not (abs(FT[b]).sum() == 0): # all 3x3 elements are zero
            # only apply the loss to image with detected faces
            # need to do this per image because images are of different shape
            current_im = images[b][:, :, int(RP[b].item()):w-int(LP[b].item())].unsqueeze(0)
            theta = FT[b].unsqueeze(0)[:, :2] #bx2x3
            grid = torch.nn.functional.affine_grid(theta, (1, 3, 112, 96))
            current_face = torch.nn.functional.grid_sample(current_im, grid)
            faces.append(current_face)
    if len(faces) == 0:
        return None
    return torch.cat(faces, 0)


def train(args, loader, sampler, generator, discriminator, g_optim, d_optim, g_ema, device):
    pbar = range(args.epoch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_epoch, dynamic_ncols=True, smoothing=0.01)
        pbar.set_description('Epoch Counter')

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    g_L1_loss_val = 0
    g_vgg_loss_val = 0
    g_l1 = torch.tensor(0.0, device=device)
    g_vgg = torch.tensor(0.0, device=device)
    g_cos = torch.tensor(0.0, device=device)
    loss_dict = {}

    criterionL1 = torch.nn.L1Loss()
    criterionVGG = VGGLoss(device).to(device)
    if args.faceloss:
        criterionCOS = nn.CosineSimilarity()

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        epoch = idx + args.start_epoch

        if epoch > args.epoch:
            print("Done!")
            break

        if args.distributed:
            sampler.set_epoch(epoch)

        batch_time = AverageMeter()

        #####################################
        ############ START EPOCH ############
        #####################################
        for i, data in enumerate(loader):
            batch_start_time = time.time()

            input_image = data['input_image'].float().to(device)
            real_img = data['target_image'].float().to(device)
            pose = data['target_pose'].float().to(device)
            sil = data['target_sil'].float().to(device)

            LeftPad = data['target_left_pad'].float().to(device)
            RightPad = data['target_right_pad'].float().to(device)

            if args.faceloss:
                FT = data['TargetFaceTransform'].float().to(device)
                real_face = getFace(real_img, FT, LeftPad, RightPad)

            if args.finetune:
                # only mask padding
                sil = torch.zeros((sil.shape)).float().to(device)
                for b in range(sil.shape[0]):
                    w = sil.shape[3]
                    sil[b][:, :, int(RightPad[b].item()):w-int(LeftPad[b].item())] = 1 # mask out the padding
            # else only focus on the foreground - initial step of training

            real_img = real_img * sil

            # appearance = human foregound + fg mask (pass coor for warping)
            source_sil = data['input_sil'].float().to(device)
            complete_coor = data['complete_coor'].float().to(device)
            if args.size == 256:
                complete_coor = torch.nn.functional.interpolate(complete_coor, size=(256, 256), mode='bilinear')
            if args.finetune:
                appearance = torch.cat([input_image, source_sil, complete_coor], 1)
            else:
                appearance = torch.cat([input_image * source_sil, source_sil, complete_coor], 1)


            ############ Optimize Discriminator ############
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            fake_img, _ = generator(appearance=appearance, pose=pose)
            fake_img = fake_img * sil

            fake_pred = discriminator(fake_img, pose=pose)
            real_pred = discriminator(real_img, pose=pose)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()


            d_regularize = i % args.d_reg_every == 0

            if d_regularize:
                real_img.requires_grad = True

                real_pred = discriminator(real_img, pose=pose)
                r1_loss = d_r1_loss(real_pred, real_img)

                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

                d_optim.step()

            loss_dict["r1"] = r1_loss


            ############## Optimize Generator ##############
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_img, _ = generator(appearance=appearance, pose=pose)
            fake_img = fake_img * sil

            fake_pred = discriminator(fake_img, pose=pose)
            g_loss = g_nonsaturating_loss(fake_pred)

            loss_dict["g"] = g_loss

            ## reconstruction loss: L1 and VGG loss + face identity loss
            g_l1 = criterionL1(fake_img, real_img)
            g_loss += g_l1
            g_vgg = criterionVGG(fake_img, real_img)
            g_loss += g_vgg

            loss_dict["g_L1"] = g_l1
            loss_dict["g_vgg"] = g_vgg

            if args.faceloss and (real_face is not None):
                fake_face = getFace(fake_img, FT, LeftPad, RightPad)
                features_real_face = sphereface_net(real_face)
                features_fake_face = sphereface_net(fake_face)
                g_cos = 1. - criterionCOS(features_real_face, features_fake_face).mean()
                g_loss += g_cos

            loss_dict["g_cos"] = g_cos

            generator.zero_grad()
            g_loss.backward()
            g_optim.step()


            ############ Optimization Done ############
            accumulate(g_ema, g_module, accum)

            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced["d"].mean().item()
            g_loss_val = loss_reduced["g"].mean().item()
            g_L1_loss_val = loss_reduced["g_L1"].mean().item()
            g_cos_loss_val = loss_reduced["g_cos"].mean().item()
            g_vgg_loss_val = loss_reduced["g_vgg"].mean().item()
            r1_val = loss_reduced["r1"].mean().item()
            real_score_val = loss_reduced["real_score"].mean().item()
            fake_score_val = loss_reduced["fake_score"].mean().item()

            batch_time.update(time.time() - batch_start_time)

            if i % 100 == 0:
                print('Epoch: [{0}/{1}] Iter: [{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(epoch, args.epoch, i, len(loader), batch_time=batch_time)
                        +
                        f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; g_L1: {g_L1_loss_val:.4f}; g_vgg: {g_vgg_loss_val:.4f}; g_cos: {g_cos_loss_val:.4f}; r1: {r1_val:.4f}; "
                    )

            if get_rank() == 0:
                if wandb and args.wandb:
                    wandb.log(
                        {
                            "Generator": g_loss_val,
                            "Discriminator": d_loss_val,
                            "R1": r1_val,
                            "Real Score": real_score_val,
                            "Fake Score": fake_score_val,
                            "Generator_L1": g_L1_loss_val,
                            "Generator_vgg": g_vgg_loss_val,
                            "Generator_facecos": g_cos_loss_val,
                        }
                    )

                if i % 5000 == 0:
                    with torch.no_grad():
                        g_ema.eval()
                        sample, _ = g_ema(appearance=appearance[:args.n_sample], pose=pose[:args.n_sample])
                        sample = sample * sil
                        utils.save_image(
                            sample,
                            os.path.join('sample', args.name, f"epoch_{str(epoch)}_iter_{str(i)}.png"),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                if i % 5000 == 0:
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "args": args,
                        },
                        os.path.join('checkpoint', args.name, f"epoch_{str(epoch)}_iter_{str(i)}.pt"),
                    )

        ###################################
        ############ END EPOCH ############
        ###################################
        if get_rank() == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                os.path.join('checkpoint', args.name, f"epoch_{str(epoch)}.pt"),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Pose with Style trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--name", type=str, help="name of experiment")
    parser.add_argument("--epoch", type=int, default=50, help="total training epochs")
    parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--workers", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=4, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=512, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--faceloss", action="store_true", help="add face loss when faces are detected")
    parser.add_argument("--finetune", action="store_true", help="finetune to handle background- second step of training.")


    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        print ('Distributed Training Mode.')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if get_rank() == 0:
        if not os.path.exists(os.path.join('checkpoint', args.name)):
            os.makedirs(os.path.join('checkpoint', args.name))
        if not os.path.exists(os.path.join('sample', args.name)):
            os.makedirs(os.path.join('sample', args.name))

    args.latent = 2048
    args.n_mlp = 8

    args.start_epoch = 0

    if args.finetune and (args.ckpt is None):
        print ('to finetune the model, please specify --ckpt.')
        import sys
        sys.exit()

    # define models
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.faceloss:
        import sphereface
        sphereface_net = getattr(sphereface, 'sphere20a')()
        sphereface_net.load_state_dict(torch.load(os.path.join(args.path, 'resources', 'sphere20a_20171020.pth')))
        sphereface_net.to(device)
        sphereface_net.eval()
        sphereface_net.feature = True


    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_epoch = int(os.path.splitext(ckpt_name)[0].split('_')[1])+1 # asuming saving as epoch_1_iter_1000.pt  or epoch_1.pt
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    dataset = DeepFashionDataset(args.path, 'train', args.size)
    sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=False,
    )

    if get_rank() == 0 and (wandb is not None) and args.wandb:
        wandb.init(project=args.name)

    train(args, loader, sampler, generator, discriminator, g_optim, d_optim, g_ema, device)
