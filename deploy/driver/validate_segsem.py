import argparse
import pickle
import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay
import torch

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def IoU(x, y, smooth=1):
    intersection = (x * y).abs().sum(dim=[1, 2])
    union = torch.sum(y.abs() + x.abs(), dim=[1, 2]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_mask(target, num_classes=19):
    mask = (target >= 0) & (target < num_classes)
    return mask.float()


def mIoU(output, target):
    l = list()
    mask = get_mask(target)
    transformed_output = output.permute(0, 2, 3, 1).argmax(dim=3)
    for c in range(output.shape[1]):
        x = (transformed_output == c).float() * mask
        y = (target == c).float()
        l.append(IoU(x, y))
    return torch.mean(torch.stack(l)).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Functional validation of accelerator on a single input"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="alveo"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="../bitfile/finn-accel.xclbin"
    )
    parser.add_argument(
        "--test_input", type=str, help='path to .npy file containing the input tensor' 
    )
    parser.add_argument(
        "--test_output", type=str, help='path to .npy file containing the output tensor' 
    )
  
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform
    test_in_path = args.test_input
    test_out_path = args.test_output

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
        runtime_weight_dir="runtime_weights/",
    )
    fp_factor = 0.6633201241493225
    
    miou = 0
    global_loss = 0
    data = np.load(test_in_path).astype(np.uint8)
    data = np.transpose(data, (0,2,3,1))
    outputs = driver.execute(data)
    outputs = torch.from_numpy(np.array(outputs))
    outputs = outputs.permute(0, 3, 1, 2)
    exp = np.load(test_out_path)/fp_factor

    print(np.allclose(exp, outputs))



