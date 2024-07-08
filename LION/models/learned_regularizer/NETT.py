# This file is part of LION library
# License : BSD-3
#
# Author  : Johannes Schwab
# Modifications: -
# =============================================================================


import torch
import torch.nn as nn
from LION.models import LIONmodel
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
from ts_algorithms import fdk

# Implementation of:

# Li, Housen, et al.
# "NETT: Solving Inverse Problems with Deep Neural Networks."
# Inverse Problems 36.6 (2020):  065005.
# DOI: 10.1088/1361-6420/ab6d57



class ConvBlock(nn.Module):
    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3):
        super().__init__()
        # input parsing:

        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            layer_list.append(
                nn.Conv2d(
                    channels[ii], channels[ii + 1], kernel_size, padding=1, bias=False
                )
            )
            layer_list.append(nn.BatchNorm2d(channels[ii + 1]))
            if ii < layers - 1 or relu_last:
                if relu_type == "ReLU":
                    layer_list.append(torch.nn.ReLU())
                elif relu_type == "LeakyReLU":
                    layer_list.append(torch.nn.LeakyReLU())
                elif relu_type == 'ELU':
                    layer_list.append(torch.nn.ELU())
                elif relu_type != "None":
                    raise ValueError("Wrong ReLu type " + relu_type)
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Downscaling with transpose conv"""

    def __init__(self, channels, stride=2, relu_type="ReLU"):
        super().__init__()
        kernel_size = 3
        layer_list = []
        layer_list.append(
            nn.ConvTranspose2d(
                channels[0],
                channels[1],
                kernel_size,
                padding=1,
                output_padding=1,
                stride=stride,
                bias=False,
            )
        )
        layer_list.append(nn.BatchNorm2d(channels[1]))
        if relu_type == "ReLU":
            layer_list.append(nn.ReLU())
        elif relu_type == "LeakyReLU":
            layer_list.append(nn.LeakyReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)

class NETT_ConvNet(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None
    ):

        if geometry_parameters is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry_parameters)

        self._make_operator()

        # Down blocks
        self.block_1_down = ConvBlock(
            self.model_parameters.down_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_1 = Down()
        self.block_2_down = ConvBlock(
            self.model_parameters.down_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_2 = Down()
        self.block_3_down = ConvBlock(
            self.model_parameters.down_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_3 = Down()
        self.block_4_down = ConvBlock(
            self.model_parameters.down_4_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_4 = Down()

        # "latent space"
        self.block_bottom = ConvBlock(
            self.model_parameters.latent_channels,
            relu_type=self.model_parameters.activation,
        )

        # Up blocks
        self.up_1 = Up(
            [
                self.model_parameters.latent_channels[-1],
                self.model_parameters.up_1_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_1_up = ConvBlock(
            self.model_parameters.up_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_2 = Up(
            [
                self.model_parameters.up_1_channels[-1],
                self.model_parameters.up_2_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_2_up = ConvBlock(
            self.model_parameters.up_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_3 = Up(
            [
                self.model_parameters.up_2_channels[-1],
                self.model_parameters.up_3_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_3_up = ConvBlock(
            self.model_parameters.up_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_4 = Up(
            [
                self.model_parameters.up_3_channels[-1],
                self.model_parameters.up_4_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_4_up = ConvBlock(
            self.model_parameters.up_4_channels,
            relu_type=self.model_parameters.activation,
        )

        self.block_last = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.last_block[0],
                self.model_parameters.last_block[1],
                self.model_parameters.last_block[2],
                padding=0,
            )
        )
        self.trained = False

    @staticmethod
    def default_parameters():
        NETT_ConvNet_params = LIONParameter()
        NETT_ConvNet_params.down_1_channels = [1, 64, 64, 64]
        NETT_ConvNet_params.down_2_channels = [64, 128, 128]
        NETT_ConvNet_params.down_3_channels = [128, 256, 256]
        NETT_ConvNet_params.down_4_channels = [256, 512, 512]

        NETT_ConvNet_params.latent_channels = [512, 1024, 1024]

        NETT_ConvNet_params.up_1_channels = [1024, 512, 512]
        NETT_ConvNet_params.up_2_channels = [512, 256, 256]
        NETT_ConvNet_params.up_3_channels = [256, 128, 128]
        NETT_ConvNet_params.up_4_channels = [128, 64, 64]

        NETT_ConvNet_params.last_block = [64, 1, 1]

        NETT_ConvNet_params.activation = "ELU"

        return NETT_ConvNet_params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(" Li, Housen, et al.")
            print(
                '"NETT: Solving inverse problems with deep neural networks."'
            )
            print("\x1B[3m Inverse Problems\x1B[0m")
            print(" 36.6 (2020):  065005.")
        elif cite_format == "bib":
            string = """
            @article{li2020nett,
            title={NETT: Solving inverse problems with deep neural networks},
            author={Li, Housen and Schwab, Johannes and Antholzer, Stephan and Haltmeier, Markus},
            journal={Inverse Problems},
            volume={36},
            number={6},
            pages={065005},
            year={2020},
            publisher={IOP Publishing}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, x, y):
        B, C, W, H = x.shape
        batch_rand = torch.randint(0,2,(1,))
        if batch_rand == 1:
            image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
            for i in range(B):
                aux = fdk(self.op, x[i, 0])
                image[i] = aux
        else:
            image = y.clone().detach().requires_grad_(True)

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)
        
        return image+res
    
    def regularizer(self, x):
        B, C, W, H = x.shape
        
        image = x

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)
        res = torch.sum(res**2)

        return res
        

class ANETT_ConvNet(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None, q_value = 1.2
    ):

        if geometry_parameters is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry_parameters)

        self._make_operator()
        self.q = q_value

        # Down blocks
        self.block_1_down = ConvBlock(
            self.model_parameters.down_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_1 = Down()
        self.block_2_down = ConvBlock(
            self.model_parameters.down_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_2 = Down()
        self.block_3_down = ConvBlock(
            self.model_parameters.down_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_3 = Down()
        self.block_4_down = ConvBlock(
            self.model_parameters.down_4_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_4 = Down()

        # "latent space"
        self.block_bottom = ConvBlock(
            self.model_parameters.latent_channels,
            relu_type=self.model_parameters.activation,
        )

        # Up blocks
        self.up_1 = Up(
            [
                self.model_parameters.latent_channels[-1],
                self.model_parameters.up_1_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_1_up = ConvBlock(
            self.model_parameters.up_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_2 = Up(
            [
                self.model_parameters.up_1_channels[-1],
                self.model_parameters.up_2_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_2_up = ConvBlock(
            self.model_parameters.up_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_3 = Up(
            [
                self.model_parameters.up_2_channels[-1],
                self.model_parameters.up_3_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_3_up = ConvBlock(
            self.model_parameters.up_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_4 = Up(
            [
                self.model_parameters.up_3_channels[-1],
                self.model_parameters.up_4_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_4_up = ConvBlock(
            self.model_parameters.up_4_channels,
            relu_type=self.model_parameters.activation,
        )

        self.block_last = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.last_block[0],
                self.model_parameters.last_block[1],
                self.model_parameters.last_block[2],
                padding=0,
            )
        )
        self.trained = False

    @staticmethod
    def default_parameters():
        NETT_ConvNet_params = LIONParameter()
        NETT_ConvNet_params.down_1_channels = [1, 64, 64, 64]
        NETT_ConvNet_params.down_2_channels = [64, 128, 128]
        NETT_ConvNet_params.down_3_channels = [128, 256, 256]
        NETT_ConvNet_params.down_4_channels = [256, 512, 512]

        NETT_ConvNet_params.latent_channels = [512, 1024, 1024]

        NETT_ConvNet_params.up_1_channels = [1024, 512, 512]
        NETT_ConvNet_params.up_2_channels = [512, 256, 256]
        NETT_ConvNet_params.up_3_channels = [256, 128, 128]
        NETT_ConvNet_params.up_4_channels = [128, 64, 64]

        NETT_ConvNet_params.last_block = [64, 1, 1]

        NETT_ConvNet_params.activation = "ELU"

        return NETT_ConvNet_params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(" Li, Housen, et al.")
            print(
                '"NETT: Solving inverse problems with deep neural networks."'
            )
            print("\x1B[3m Inverse Problems\x1B[0m")
            print(" 36.6 (2020):  065005.")
        elif cite_format == "bib":
            string = """
            @article{li2020nett,
            title={NETT: Solving inverse problems with deep neural networks},
            author={Li, Housen and Schwab, Johannes and Antholzer, Stephan and Haltmeier, Markus},
            journal={Inverse Problems},
            volume={36},
            number={6},
            pages={065005},
            year={2020},
            publisher={IOP Publishing}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, x, y):
        B, C, W, H = x.shape
        batch_rand = torch.randint(0,2,(1,))
        if batch_rand == 1:
            image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
            for i in range(B):
                aux = fdk(self.op, x[i, 0])
                image[i] = aux
        else:
            image = y.clone().detach().requires_grad_(True)

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)
        
        return block_4_res, image+res, batch_rand
    
    def regularizer(self, x):
        B, C, W, H = x.shape
        
        image = x

        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)
        res = torch.sqrt(torch.sum(res**2)) + torch.sum(torch.abs(block_4_res)**self.q)**(1/self.q)

        return res
