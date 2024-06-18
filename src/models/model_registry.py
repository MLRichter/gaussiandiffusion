import torch
from diffusers import UNet2DModel

from diffusers import UNet2DConditionModel



def unet_872M():
    model = UNet2DConditionModel(
        sample_size=(16, 256 // 8, 256 // 8),  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16,  # the number of output channels
    )
    return model


def unet_225M():
    model = UNet2DConditionModel(
        sample_size=(16, 256 // 8, 256 // 8),  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(320 // 2, 640 // 2, 1280 // 2, 1280 // 2)

    )
    return model

def unet_160M():
    model = UNet2DConditionModel(
        sample_size=(16, 256 // 8, 256 // 8),  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(128, 256, 512, 512)

    )
    return model

def unet_40M():
    model = UNet2DConditionModel(
        sample_size=(16, 256 // 8, 256 // 8),  # the target image resolution
        in_channels=16,  # the number of input channels, 3 for RGB images
        out_channels=16, block_out_channels=(64, 128, 256, 256)

    )
    return model


if __name__ == '__main__':
    model = unet_160M()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params // 1000000, "M Params")
    zeros = torch.zeros((2, 16, 256//8, 256//8))
    timestep = torch.zeros((2))
    encoder_hidden_state = torch.zeros(2, 77, 1280)
    x = model(zeros, timestep, encoder_hidden_state, return_dict=False)
    print(x[0].size())