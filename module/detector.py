from math import sqrt
import torch
from torch import nn


def convblock(in_channels, out_channels):
    def conv1d(in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(in_channels, out_channels):
        return nn.Sequential(
        conv1d(in_channels, out_channels),
        nn.LeakyReLU(inplace=True),
        nn.InstanceNorm1d(out_channels, affine=True),
        )

    return forward(in_channels, out_channels)


class base_discriminator(nn.Module):
    def _name(self):
        return "base_discriminator"

    def _conv1d(self, in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.conv1 = convblock(self.channels_size, self.hidden_size)
        self.conv2 = convblock(self.hidden_size, self.hidden_size)
        self.conv3 = convblock(self.hidden_size, self.hidden_size)
        self.conv_out = nn.Sequential(
            self._conv1d(self.hidden_size, 1),
        )

    def __init__(self, data_depth, hidden_size, channels_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._build_models()
        self.name = self._name()

    def forward(self, image):
        x = self.conv1(image)
        x_list = [x]
        x_1 = self.conv2(torch.cat(x_list, dim=1))
        x_list = [x_1]
        x_2 = self.conv3(torch.cat(x_list, dim=1))
        # x_out = self.conv_out(x_10)
        x_out = self.conv_out(x_2)
        return torch.mean(x_out.view(x_out.size(0), -1), dim=1).unsqueeze(1)


class redundant_discriminator(nn.Module):
    def _name(self):
        return "redundant_discriminator"

    def _conv1d(self, in_channels, out_channels):
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        self.conv1 = convblock(self.channels_size, self.hidden_size)
        self.conv2 = convblock(self.hidden_size, self.hidden_size)
        self.conv3 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv4 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv5 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv6 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv7 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv8 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv9 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv10 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv11 = convblock(self.hidden_size*2, self.hidden_size)
        self.conv_out = nn.Sequential(
            self._conv1d(self.hidden_size, 1),
        )

    def __init__(self, data_depth, hidden_size, channels_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.channels_size = channels_size
        self._build_models()
        self.name = self._name()

    def forward(self, image):
        x = self.conv1(image)
        x_list = [x]
        x_1 = self.conv2(torch.cat(x_list, dim=1))
        x_list.append(x_1)
        x_2 = self.conv3(torch.cat(x_list, dim=1))
        x_list = [x, x_2]
        x_3 = self.conv4(torch.cat(x_list, dim=1))
        x_list = [x, x_3]
        x_4 = self.conv5(torch.cat(x_list, dim=1))
        x_list = [x, x_4]
        x_5 = self.conv6(torch.cat(x_list, dim=1))
        x_list = [x, x_5]
        x_6 = self.conv7(torch.cat(x_list, dim=1))
        x_list = [x, x_6]
        x_7 = self.conv8(torch.cat(x_list, dim=1))
        x_list = [x, x_7]
        x_8 = self.conv9(torch.cat(x_list, dim=1))
        x_list = [x, x_8]
        x_9 = self.conv10(torch.cat(x_list, dim=1))
        x_list = [x, x_9]
        x_10 = self.conv11(torch.cat(x_list, dim=1))
        x_out = self.conv_out(x_10)
        return torch.mean(x_out.view(x_out.size(0), -1), dim=1).unsqueeze(1)


if __name__ == "__main__":

    model = base_discriminator(2, 32, 1)
    a=torch.randn(4,16000)
    a=a.unsqueeze(1)
    print(a.shape)
    out = model(a)
    print(out.shape)

    pass