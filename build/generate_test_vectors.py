import torch
import numpy as np
import sys
import os
from torch.nn.functional import interpolate

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from models import *
import datasets 
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test vectors generators for FINN verification steps')
    parser.add_argument('-m', '--model', type=str, choices=['resnet18', 'unet'], help='Model name', required=True)
    parser.add_argument('-i', '--input_size', type=int, nargs=4, help='Model input size in the NCWH format', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Output folder path', required=True)
    parser.add_argument('-c', '--ckpt', type=str, help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default="random")
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    if args.model == 'resnet18':
        net = quant_resnet.quantresnet18()
    elif args.model == 'unet' :
        encoder = quant_resnet.quantresnet18(weight_bit_width = 4, act_bit_width = 4)
        net = quant_unet.Unet(encoder, weight_bit_width = 4, act_bit_width = 4)
    else:
        raise ValueError('Unknown model name')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.ckpt is not None:
        with open(args.ckpt, 'rb') as f:
                checkpoint = torch.load(f, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint)

    net.eval()
    net = net.to(device)

    if args.dataset == 'cityscapes':
        test_dataset = datasets.CityScapesDataset(args.path, 'val')
        ind = np.random.randint(len(test_dataset))
        inputs = test_dataset[ind][0]
        inputs = interpolate(inputs[None, :, :, :], tuple(args.input_size)[2:4])
        scale = (2**(8-1) - 1)/(torch.max(torch.abs(inputs)))
        inputs = torch.round(scale*inputs)
    else:
        input_shape=(tuple(args.input_size))
        inputs = torch.randint(0, 2**8 - 1, input_shape)
    
    inputs.type(torch.int8)

    inputs = inputs.to(device).float()
    outputs = net(inputs)
   
    np.save(args.output_path + args.model + "_" + args.dataset + "_inputs.npy", inputs.cpu().detach().numpy())
    np.save(args.output_path + args.model + "_" + args.dataset + "_outputs.npy", outputs.cpu().detach().numpy())


if __name__ == '__main__':
    main()