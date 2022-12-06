import argparse
import os
import torch
from torchvision import utils
from tqdm import tqdm
from torch.utils import data
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
from dataset import DeepFashionDataset
from model import Generator
from util.dp2coor import getSymXYcoordinates
from util.coordinate_completion_model import define_G as define_CCM


def tensors2square(im, pose, sil):
    width = im.shape[2]
    diff = args.size - width
    left = int((args.size-width)/2)
    right = diff - left
    im = torch.nn.functional.pad(input=im, pad=(right, left, 0, 0), mode='constant', value=0)
    pose = torch.nn.functional.pad(input=pose, pad=(right, left, 0, 0), mode='constant', value=0)
    sil = torch.nn.functional.pad(input=sil, pad=(right, left, 0, 0), mode='constant', value=0)
    return im, pose, sil

def tensor2square(x):
    width = x.shape[2]
    diff = args.size - width
    left = int((args.size-width)/2)
    right = diff - left
    x = torch.nn.functional.pad(input=x, pad=(right, left, 0, 0), mode='constant', value=0)
    return x

def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()

        path = args.input_path
        input_name = args.input_name
        target_name = args.target_name
        part = args.part

        # input
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_image = Image.open(os.path.join(path, input_name+'.png')).convert('RGB')
        iw, ih = input_image.size
        input_image = transform(input_image).float().to(device)
        input_pose = np.array(Image.open(os.path.join(path, input_name+'_iuv.png')))
        input_sil = np.array(Image.open(os.path.join(path, input_name+'_sil.png')))/255
        # get partial coordinates from dense pose
        dp_uv_lookup_256_np = np.load('util/dp_uv_lookup_256.npy')
        input_uv_coor, input_uv_mask, input_uv_symm_mask  = getSymXYcoordinates(input_pose, resolution = 512)
        # union sil with densepose masks
        input_sil = 1-((1-input_sil) * (input_pose[:, :, 0] == 0).astype('float'))
        input_sil = torch.from_numpy(input_sil).float().unsqueeze(0)
        input_pose = torch.from_numpy(input_pose).permute(2, 0, 1)

        # target
        target_image = Image.open(os.path.join(path, target_name+'.png')).convert('RGB')
        tw, th = target_image.size
        target_image = transform(target_image).float().to(device)
        target_pose = np.array(Image.open(os.path.join(path, target_name+'_iuv.png')))
        target_sil = np.array(Image.open(os.path.join(path, target_name+'_sil.png')))/255
        # get partial coordinates from dense pose
        target_uv_coor, target_uv_mask, target_uv_symm_mask  = getSymXYcoordinates(target_pose, resolution = 512)
        # union sil with densepose masks
        target_sil = 1-((1-target_sil) * (target_pose[:, :, 0] == 0).astype('float'))
        target_sil = torch.from_numpy(target_sil).float().unsqueeze(0)
        target_pose = torch.from_numpy(target_pose).permute(2, 0, 1)

        # convert to square by centering
        input_image, input_pose, input_sil = tensors2square(input_image, input_pose, input_sil)
        target_image, target_pose, target_sil = tensors2square(target_image, target_pose, target_sil)

        # add batch dimension
        input_image = input_image.unsqueeze(0).float().to(device)
        input_pose = input_pose.unsqueeze(0).float().to(device)
        input_sil = input_sil.unsqueeze(0).float().to(device)
        target_image = target_image.unsqueeze(0).float().to(device)
        target_pose = target_pose.unsqueeze(0).float().to(device)
        target_sil = target_sil.unsqueeze(0).float().to(device)

        # complete partial coordinates
        coor_completion_generator = define_CCM().cuda()
        CCM_checkpoint = torch.load(args.CCM_pretrained_model)
        coor_completion_generator.load_state_dict(CCM_checkpoint["g"])
        coor_completion_generator.eval()
        for param in coor_completion_generator.parameters():
            coor_completion_generator.requires_grad = False

        # uv coor preprocessing (put image in center)
        # input
        ishift = int((ih-iw)/2) # center shift
        input_uv_coor[:,:,0] = input_uv_coor[:,:,0] + ishift # put in center
        input_uv_coor = ((2*input_uv_coor/(ih-1))-1)
        input_uv_coor = input_uv_coor*np.expand_dims(input_uv_mask,2) + (-10*(1-np.expand_dims(input_uv_mask,2)))
        # target
        tshift = int((th-tw)/2) # center shift
        target_uv_coor[:,:,0] = target_uv_coor[:,:,0] + tshift # put in center
        target_uv_coor = ((2*target_uv_coor/(th-1))-1)
        target_uv_coor = target_uv_coor*np.expand_dims(target_uv_mask,2) + (-10*(1-np.expand_dims(target_uv_mask,2)))

        # coordinate completion
        # input
        uv_coor_pytorch = torch.from_numpy(input_uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
        uv_mask_pytorch = torch.from_numpy(input_uv_mask).unsqueeze(0).unsqueeze(0).float() #1xchw
        with torch.no_grad():
            coor_completion_generator.eval()
            input_complete_coor = coor_completion_generator(uv_coor_pytorch.cuda(), uv_mask_pytorch.cuda())
        # target
        uv_coor_pytorch = torch.from_numpy(target_uv_coor).float().permute(2, 0, 1).unsqueeze(0) # from h,w,c to 1,c,h,w
        uv_mask_pytorch = torch.from_numpy(target_uv_mask).unsqueeze(0).unsqueeze(0).float() #1xchw
        with torch.no_grad():
            coor_completion_generator.eval()
            target_complete_coor = coor_completion_generator(uv_coor_pytorch.cuda(), uv_mask_pytorch.cuda())


        # garment transfer
        appearance = torch.cat([input_image, input_sil, input_complete_coor, target_image, target_sil, target_complete_coor], 1)
        output, part_mask = g_ema(appearance=appearance, pose=input_pose)

        # visualize the transfered part
        zeros = torch.zeros(part_mask.shape).to(part_mask)
        ones255 = torch.ones(part_mask.shape).to(part_mask)*255
        part_red = torch.cat([part_mask*255, zeros, zeros], 1)
        part_img = part_red * ((input_pose[:, 0, :, :] != 0)) + torch.cat([ones255, ones255, ones255], 1)*(1-part_mask)

        utils.save_image(
            output[:, :, :, int(ishift):args.size-int(ishift)],
            os.path.join(args.save_path, input_name+'_and_'+target_name+'_'+args.part+'_vis.png'),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            part_img[:, :, :, int(ishift):args.size-int(ishift)],
            os.path.join(args.save_path, input_name+'_and_'+target_name+'_'+args.part+'.png'),
            nrow=1,
            normalize=True,
            range=(0, 255),
        )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="inference")

    parser.add_argument("--input_path", type=str, help="path to the input dataset")
    parser.add_argument("--input_name", type=str, default="fashionWOMENSkirtsid0000177102_1front", help="input file name")
    parser.add_argument("--target_name", type=str, default="fashionWOMENBlouses_Shirtsid0000635004_1front", help="target file name")
    parser.add_argument("--part", type=str, default="upper_body", help="body part to transfer upper_body, lower_body, and face")
    parser.add_argument("--size", type=int, default=512, help="output image size of the generator")
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument("--pretrained_model", type=str, default="posewithstyle.pt", help="pose with style pretrained model")
    parser.add_argument("--CCM_pretrained_model", type=str, default="CCM_epoch50.pt", help="pretrained coordinate completion model")
    parser.add_argument("--save_path", type=str, default="./data/output", help="path to save output .data/output")

    args = parser.parse_args()

    args.latent = 2048
    args.n_mlp = 8

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, garment_transfer=True, part=args.part).to(device)
    checkpoint = torch.load(args.pretrained_model)
    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
