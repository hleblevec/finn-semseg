import torch
import os
import sys
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate


class CityScapesDataset(torch.utils.data.Dataset):
    __CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
    __CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]

    __IGNORE_CLASS_LABEL = 19
    __CITYSCAPES_CLASSES_TO_LABELS = {
        0: __IGNORE_CLASS_LABEL,
        1: __IGNORE_CLASS_LABEL,
        2: __IGNORE_CLASS_LABEL,
        3: __IGNORE_CLASS_LABEL,
        4: __IGNORE_CLASS_LABEL,
        5: __IGNORE_CLASS_LABEL,
        6: __IGNORE_CLASS_LABEL,
        7: 0,
        8: 1,
        9: __IGNORE_CLASS_LABEL,
        10: __IGNORE_CLASS_LABEL,
        11: 2,
        12: 3,
        13: 4,
        14: __IGNORE_CLASS_LABEL,
        15: __IGNORE_CLASS_LABEL,
        16: __IGNORE_CLASS_LABEL,
        17: 5,
        18: __IGNORE_CLASS_LABEL,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: __IGNORE_CLASS_LABEL,
        30: __IGNORE_CLASS_LABEL,
        31: 16,
        32: 17,
        33: 18,
        -1: __IGNORE_CLASS_LABEL
    }

    def __init__(self, path, subset):
        super(CityScapesDataset, self).__init__()

        if not os.path.isdir(path):
            sys.stderr.write('Error : specified dataset path does not exist\n')
            raise ValueError
        if subset != 'train' and subset != 'test' and subset != 'val':
            sys.stderr.write('Error : specified dataset subset is not "train", "test" or "val"\n')
            raise ValueError

        self.subset = subset

        images_path = os.path.join(path, 'leftImg8bit', subset)
        images_subfolders = [os.path.join(images_path, f) for f in os.listdir(images_path)]

        self.images_path = list()
        self.labels_path = list()
        for f in images_subfolders:
            im_dir = os.listdir(f)
            for i in im_dir:
                self.images_path.append(os.path.join(f, i))
                self.labels_path.append(os.path.join(f.replace('leftImg8bit', 'gtFine'),
                                                     i.replace('leftImg8bit', 'gtFine_labelIds')))

        self.length = len(self.images_path)

        self.common_data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])

        self.data_augmentation = transforms.Compose([
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)
        ])

    def __len__(self):
        return self.length

    def load_img(self, path):
        img = Image.open(path)
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())
                                ).view(img.size[1], img.size[0], len(img.getbands())).float()

    def random_resized_crop(self, img, size, scale):
        s = random.uniform(scale[0], scale[1])
        crop = (int(img.shape[-2] * s), int(img.shape[-1] * s))
        anchor = (random.randint(0, img.shape[-2] - crop[0] - 1), random.randint(0, img.shape[-1] - crop[1] - 1))
        img = img[:, anchor[0]:anchor[0] + crop[0], anchor[1]:anchor[1] + crop[1]]
        return interpolate(img[None, :, :, :], size)[0]

    def __getitem__(self, item):
        image_path, label_path = self.images_path[item], self.labels_path[item]
        image = self.load_img(image_path)

        label = self.load_img(label_path)
        new_label = label.clone()
        for k, v in self.__CITYSCAPES_CLASSES_TO_LABELS.items():
            new_label[label == k] = v
        label = new_label

        image = image / 255
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        if self.subset == 'train':
            cat = torch.cat((image, label), dim=0)
            cat = self.random_resized_crop(cat, (512, 1024), (0.25, 1.0))
            cat = self.common_data_augmentation(cat)
            image = cat[:-1, :, :]
            label = cat[-1, :, :]
            image = image[None, :, :, :]
            if random.random() >= 0.5:
                k = 1 + random.randint(0, 4) * 2
                sigma = 0.15 + random.random() * 1.15
                image = F.gaussian_blur(img=image, kernel_size=k, sigma=sigma)
            image = self.data_augmentation(image)
            image = image[0]
        else:
            label = label[0]

        F.normalize(image, self.__CITYSCAPES_MEAN, self.__CITYSCAPES_STD)
        return image, label


def load_cityscapes(path, batch_size=128, test_batch_size=256):
    train_dataset = CityScapesDataset(path, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = CityScapesDataset(path, 'val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return {'train': train_loader, 'test': test_loader}


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     train = CityScapesDataset('../data', 'train')

#     while True:
#         data, target = train[0]
#         data = data.permute(1, 2, 0)
#         fig, ax = plt.subplots(2, 1)
#         ax[0].imshow(data)
#         ax[1].imshow(target)
#         plt.show()
