from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.utils import make_grid
import tqdm


def plot(x):
    img = x.detach().clone()
    if img.min() < 0:
        img = (img+1.)/2.
    if len(img.shape) == 2:
        if isinstance(img, np.ndarray):
            plt.imshow(img)
        else:
            plt.imshow(img.cpu().expand(*img.shape, 3))
    elif img.shape[-1] == 3:
        if isinstance(img, np.ndarray):
            plt.imshow(img)
        else:
            plt.imshow(img.cpu())
    elif img.shape[-1] == 1:
        plt.imshow(img.cpu().expand(*img.shape[:-1], 3))
    elif img.shape[-3] == 3:
        if len(img.shape) == 4:
            plt.imshow(make_grid((img).cpu(), padding=1).permute(1,2,0))
        else:
            plt.imshow(img.cpu().permute(1,2,0))
    else:
        raise NotImplementedError("No idea how to process tensor of shape ", img.shape)
    plt.show()

def get_grid(start, stop, h, w):
    x_grid = torch.linspace(start,stop, w)
    y_grid = torch.linspace(start,stop, h)
    ii, jj = torch.meshgrid((x_grid, y_grid))
    return torch.stack((ii,jj), dim=-1)

manor_data = {
    "dir" : Path("data/manor"),
    "img" : [
        "manor_minus2ev.jpg",
        "manor_normal.jpg",
        "manor_plus2ev.jpg",
        "manor_plus4ev.jpg"]
}


def load_data(data, transforms=T.ToTensor()):
    photos = []
    for i in data["img"]:
        img = Image.open(data["dir"] / i)
        img = transforms(img)
        photos.append(img)
    photos = torch.stack(photos, dim=0)
    photos.requires_grad = False
    return photos


class PositionalEncoding(torch.nn.Module):
    """Apply positional encoding to the input.
    """

    def __init__(
        self,
        num_encoding_functions: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0, num_encoding_functions - 1, num_encoding_functions
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions
            )
        self.register_buffer("frequency_bands", frequency_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input = [x] if self.include_input else []
        xshape = list(x.shape)
        x = x[..., None].expand(*xshape, self.num_encoding_functions)
        x = self.frequency_bands * x
        x = x.view(*xshape[:-1], -1)
        encoding = torch.cat(input + [torch.sin(x), torch.cos(x)], dim=-1)
        return encoding

    def output_size(self):
        return 2 * 2 * self.num_encoding_functions + (2 if self.include_input else 0)


class SimpleModule(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU(), init=None):
        super(SimpleModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        if init is not None:
            init(self.linear.weight)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))

class HDRNet(torch.nn.Module):
    def __init__(self, basic_net, offsets, crf):
        super(HDRNet, self).__init__()
        self.crf = crf
        self.offsets = torch.nn.Parameter(offsets.clone())
        self.basic_net = basic_net

    def forward(self, x, img_id):
        x = self.basic_net(x)
        x = x + self.offsets[img_id]
        return self.crf(x)

class HDRNetComplex(torch.nn.Module):
    def __init__(self, basic_net, total_images):
        super(HDRNetComplex, self).__init__()
        self.offsets = torch.nn.Parameter(torch.zeros([total_images,1]))
        self.scale = torch.nn.Parameter(torch.ones([total_images, 3]))
        self.basic_net = basic_net

    def forward(self, x, img_id):
        x = self.basic_net(x)
        x = x * self.scale[img_id] + self.offsets[img_id]
        return torch.sigmoid(x)
