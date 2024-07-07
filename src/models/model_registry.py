import torch
from diffusers import UNet2DModel

from diffusers import UNet2DConditionModel
from torch import nn
from torchtools.utils import Diffuzz2


class TupleWrapper(nn.Module):

    def __init__(self, module: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)[0]


def unet_872M():
    model = UNet2DConditionModel(
        sample_size=128//8,  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16,  # the number of output channels
        resnet_time_scale_shift='scale_shift',
    )
    resnet_time_scale_shift = 'scale_shift',

    return TupleWrapper(model)


def unet_225M():
    model = UNet2DConditionModel(
        sample_size=128//8,  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(320 // 2, 640 // 2, 1280 // 2, 1280 // 2)

    )
    resnet_time_scale_shift = 'scale_shift',

    return TupleWrapper(model)

def unet_160M():
    model = UNet2DConditionModel(
        sample_size=128//8,  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(128, 256, 512, 512)

    )
    resnet_time_scale_shift = 'scale_shift',

    return TupleWrapper(model)

def unet_40M():
    model = UNet2DConditionModel(
        sample_size=128//8,  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(64, 128, 256, 256)

    )
    resnet_time_scale_shift = 'scale_shift',

    return TupleWrapper(model)


if __name__ == '__main__':

    model = unet_160M()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params // 1000000, "M Params")
    zeros = torch.zeros((2, 16, 256//8, 256//8))
    timestep = torch.zeros((2))
    encoder_hidden_state = torch.zeros(2, 77, 1280)
    x = model(zeros, timestep, encoder_hidden_state, return_dict=False)
    print(x[0].size())
    #sample: torch.FloatTensor
    #timestep: Union[torch.Tensor, float, int]
    #encoder_hidden_states: torch.Tensor

    diffuzz = Diffuzz2(device='cpu', scaler=64 / 64, clamp_range=(0, 1 - 1e-7),)


    result = diffuzz.sample(model, {
        'encoder_hidden_states': encoder_hidden_state, 'return_dict': False
    }, zeros.shape, unconditional_inputs={
        'encoder_hidden_states': encoder_hidden_state, 'return_dict': False
    }, cfg=1.5, sample_mode='e', t_scaler=(256//8) / 256)

    print(result)
    x = list(result)
    print(x)
