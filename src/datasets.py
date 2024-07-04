import os

import torchvision
import torch
from torchvision.datasets import ImageFolder
from src.common_canvas import CommonCanvasDataset

HORRIBLE_CC_FILEPATH = "/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/0/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/1/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/10/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/11/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/12/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/13/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/14/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/15/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/16/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/17/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/18/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/19/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/2/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/20/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/21/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/22/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/23/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/24/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/25/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/26/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/27/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/28/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/29/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/3/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/30/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/31/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/32/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/33/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/34/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/35/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/36/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/37/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/38/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/39/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/4/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/40/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/41/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/42/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/43/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/44/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/45/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/46/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/47/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/48/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/49/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/5/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/50/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/51/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/52/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/53/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/54/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/55/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/56/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/57/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/58/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/59/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/6/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/60/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/61/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/62/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/63/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/64/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/65/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/66/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/67/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/68/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/69/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/7/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/70/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/71/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/72/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/73/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/74/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/75/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/76/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/77/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/78/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/79/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/8/;/home/mila/i/ibrahiad/scratch/shared_with_ben/common_canvas/CommonCatalog/v0-beta/blip2-CC-commercial-derives-100M-512maxres/v0/9/"


def res_filter(x, input_size):
    h, w = x[0].size
    return w >= input_size and h >= input_size


def transforms(size, crop_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x if x.size(0) == 3 else torch.cat([x, x, x], 0)),
        torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                      antialias=True),
        torchvision.transforms.RandomCrop(crop_size),
    ])
    return transforms


def coco_dataset(data_path: str = "../../../coco2017/", subset: str = "train", size: int = 256, crop_size: int = 256):
    dataset_path = os.path.join(data_path, subset)
    transform = transforms(size, crop_size)
    return ImageFolder(dataset_path, transform=transform)


CC_CACHE = None
def common_canvas(data_path: str = HORRIBLE_CC_FILEPATH, subset: str = "train", size: int = 256, crop_size: int = 256):
    transform = transforms(size, crop_size)
    global CC_CACHE
    try:
        dataset = CC_CACHE if CC_CACHE is not None else CommonCanvasDataset(data_path, transform=transform)
    except:
        dataset = CC_CACHE if CC_CACHE is not None else CommonCanvasDataset(data_path, transform=transform)

    CC_CACHE = dataset
    if subset == "train":
        size = len(dataset)
        subset = torch.utils.data.Subset(dataset=dataset, indices=list(range(size - 1000)))
    elif subset == "val":
        size = len(dataset)
        subset = torch.utils.data.Subset(dataset=dataset, indices=list(range(size - 1000, size)))
    else:
        subset = dataset
    return subset


if __name__ == '__main__':

    HORRIBLE_CC_FILEPATH = "../../CC/0/;../../CC/01/"
    ds = common_canvas(data_path=HORRIBLE_CC_FILEPATH, subset="train")#, transforms=transforms(size=256, crop_size=256))
    print(len(ds))

    ds = common_canvas(data_path=HORRIBLE_CC_FILEPATH, subset="val")#, transforms=transforms(size=256, crop_size=256))
    print(len(ds))

    ds = common_canvas(data_path=HORRIBLE_CC_FILEPATH, subset="full")#, transforms=transforms(size=256, crop_size=256))
    print(len(ds))

