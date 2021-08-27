import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        self.noise_dim = args.noise_dim
        self.channels = args.channels
        self.noise_filter = args.noise_filter_discriminator

        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(self.channels, self.noise_filter,
                                              kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.layer2 = nn.Sequential(nn.Conv2d(self.noise_filter, self.noise_filter * 2,
                                              kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 2),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.layer3 = nn.Sequential(nn.Conv2d(self.noise_filter * 2, self.noise_filter * 4,
                                              kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 4),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.layer4 = nn.Sequential(nn.Conv2d(self.noise_filter * 4, self.noise_filter * 8,
                                              kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 8),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.layer5 = nn.Sequential(nn.Conv2d(self.noise_filter * 8, 1,
                                              kernel_size=2, stride=1, padding=0, bias=False),
                                    nn.Sigmoid())

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        output = x.view(inputs.shape[0], -1)
        return output