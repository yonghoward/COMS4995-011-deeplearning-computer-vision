from keras.applications.nasnet import decode_predictions
from tools import KBest
# from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dense
# from keras.models import Model
# from keras.applications.vgg16 import VGG16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import numpy as np

import glob, os


# Down Block 1: increase channels with 2 convolutional layers.
class downConvBlock1(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(downConvBlock1, self).__init__()
    #
    self.input_dim = input_dim
    self.output_dim = output_dim
    #
    self.conv1 = nn.Conv2d(
      self.input_dim, 
      self.output_dim, 
      kernel_size=(1,1), 
      stride=(1,1),
      padding=(1,1))
    self.conv2 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3), 
      stride=(1,1),
      padding=(1,1))
    self.conv2 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(5,5), 
      stride=(1,1),
      padding=(1,1))
      
  def forward(self, x):
    x = self.conv1(x)
    x = nn.ReLU(x)
    x = self.conv2(x)
    x = nn.ReLU(x)
    x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)
    return x


# Down Block 2: increase channels with 3 convolutional layers.
class downConvBlock2(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(downConvBlock2, self).__init__()
    #
    self.input_dim = input_dim
    self.output_dim = output_dim
    #
    self.conv1 = nn.Conv2d(
      self.input_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))
    self.conv2 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))
    self.conv3 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))

  def forward(self, x):
    x = self.conv1(x)
    x = nn.ReLU(x)
    x = self.conv2(x)
    x = nn.ReLU(x)
    x = self.conv3(x)
    x = nn.ReLU(x)
    x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)
    return x


# Up Block 1: decrease channels with 3 convolutional layers
class upConvBlock1(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(upConvBlock1, self).__init__()
    #
    self.input_dim = input_dim
    self.output_dim = output_dim
    #
    self.conv1 = nn.Conv2d(
      self.input_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))
    self.conv2 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))
    self.conv3 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))

  def forward(self, x):
    x = self.conv1(x)
    x = nn.ReLU(x)
    x = self.conv2(x)
    x = nn.ReLU(x)
    x = self.conv3(x)
    x = nn.ReLU(x)
    x = nn.UpsamplingBilinear2d((2,2))(x)
    return x


# Up Block 2: decrease channels with 2 convolutional layers
# Up Block 1: decrease channels with 3 convolutional layers
class upConvBlock2(nn.Module):
  def __init__(self, input_dim, output_dim, needUpSample):
    super(upConvBlock2, self).__init__()
    #
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.needUpSample = needUpSample
    #
    self.conv1 = nn.Conv2d(
      self.input_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))
    self.conv2 = nn.Conv2d(
      self.output_dim,
      self.output_dim,
      kernel_size=(3,3),
      stride=(1,1),
      padding=(1,1))

  def forward(self, x):
    x = self.conv1(x)
    x = nn.ReLU(x)
    x = self.conv2(x)
    if self.needUpSample:
      x = nn.ReLU(x)
      x = nn.UpsamplingBilinear2d((2,2))(x)
    else:
      x = nn.Conv2d(self.input_dim, 
        self.output_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1))(x)
    return x


# Linear layer
class linearEncoder(nn.Module):
  def __init__(self, size, neck_dim=512, k=256):
    super(linearEncoder, self).__init__()
    self.size = size # (nChannels, H, W)
    self.input_dim = size[1] * size[2]
    self.neck_dim = neck_dim
    self.k = k
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(self.size[1], self.size[2]))
    self.linear = nn.Linear(self.input_dim, self.neck_dim, bias=False)

  def forward(self, x):
    x = self.avgpool(x)
    x = self.linear(x)
    code = torch.zeros(x.shape[0], self.size[0], self.neck_dim)
    for val in range(self.size[0]):
      c = KBest(x, self.k, symmetric=True)
      code[:, val, :] = c
    return code


# Grouped linear encoder group
class groupedLinearEncoder(nn.Module):
    def __init__(self, size, neck_dim, k):
        super(groupedLinearEncoder, self).__init__()
        #
        self.size = size   # (num_channels, H, W)
        self.in_dim = size[1] * size[2]
        self.num_filts = size[0]
        self.neck_dim = neck_dim
        self.k = k
        #
        self.LinearLayers = nn.ModuleList()
        for l in range(self.num_filts):
            lin = nn.Linear(self.in_dim,self.neck_dim,bias=False)
            self.LinearLayers.append(lin)
    #
    def forward(self,x):
        batch_size = x.shape[0]
        device = x.device
        code = torch.zeros(batch_size,self.num_filts,self.neck_dim).to(device)
        for l in range(self.num_filts):
            c = self.LinearLayers[l](x[:,l,:,:].view(batch_size,-1))
            c = KBest(c,self.k,symmetric=True)
            code[:,l,:] = c
            #
        return code


# Linear decoder
class linearDecoder(nn.Module):
  def __init__(self, size, neck_dim=512, k=256):
    super(linearDecoder, self).__init__()
    self.size = size # (nChannels, H, W)
    self.neck_dim = neck_dim
    self.k = k

  def forward(self, z):
    x_hat = torch.zeros(z.shape[0], self.size[0], self.size[1], self.size[2])
    for l in range(self.size[0]):
      x_hat[:, l, :, :] = F.linear(
        z[:, l, :], weight=self.weights[l].weight.t()).view(
          z.shape[0], 1, self.size[1], self.size[2]
        ).squeeze(1)
      
    return x_hat


# Grouped linear decoders.
class groupedLinearDecoder(nn.Module):
    def __init__(self, size, weights):
        super(groupedLinearDecoder, self).__init__()
        #
        self.size = size   # (num_channels, H, W)
        self.num_filts = size[0]
        self.weights = weights
        #
    def forward(self,code):
        batch_size = code.shape[0]
        device = code.device
        x_hat = torch.zeros(batch_size,self.num_filts, self.size[1],self.size[2]).to(device)
        for l in range(self.num_filts):
            x_hat[:,l,:,:] = F.linear(
                code[:,l,:],weight=self.weights[l].weight.t()).view(
                batch_size,1,self.size[1],self.size[2]
            ).squeeze(1)
        return x_hat


class Encoder(nn.Module):
  def __init__(self, image_size, sizes, num_codes, strides, neck_dim, k=256):
    super(Encoder, self).__init__()
    #
    self.image_size = image_size # (nChannels, H, W)
    self.num_codes = num_codes
    self.strides = strides
    self.sizes = sizes
    self.neck_dim = neck_dim
    self.k =k
    #
    self.model = nn.Sequential(
      downConvBlock1(self.sizes[0], self.sizes[1]),
      downConvBlock1(self.sizes[1], self.sizes[2]),
      downConvBlock2(self.sizes[2], self.sizes[3]),
      downConvBlock2(self.sizes[3], self.sizes[4]),
      downConvBlock2(self.sizes[4], self.sizes[4])
    )
    self.LinearLayers = groupedLinearEncoder(
      (self.num_codes,
          int(image_size[1]/(np.prod(self.strides))),
          int(image_size[2]/(np.prod(self.strides)))),
      self.neck_dim, self.k
    )

  def forward(self, x):
    code = self.model(x)
    code = self.linearLayers(code)
    return code


class Decoder(nn.Module):
  def __init__(self, image_size, sizes, num_codes, strides, weights, k=256):
    super(Decoder, self).__init__()
    #
    self.image_size = image_size
    self.num_codes = num_codes
    self.strides = strides
    self.sizes = sizes[::-1]
    self.weights = weights
    #
    self.LinearLayers = groupedLinearDecoder(
      (self.num_codes,
        int(image_size[2]/(np.prod(self.strides))), int(image_size[1]/(np.prod(self.strides)))),
        self.weights)
    self.model = nn.Sequential(
      upConvBlock1(self.sizes[0], self.sizes[1]),
      upConvBlock1(self.sizes[1], self.sizes[2]),
      upConvBlock1(self.sizes[2], self.sizes[3]),
      upConvBlock2(self.sizes[3], self.sizes[4], True),
      upConvBlock2(self.sizes[4], self.sizes[4], False)
    )

  def forward(self, z):
    x_hat_tmp = self.linearLayers(z)
    x_hat = x_hat_tmp[:, 0:self.sizes[0], :, :]
    x_hat = self.model(x_hat)
    return x_hat


class Autoencoder(nn.Module):
  def __init__(self, image_size, sizes, num_codes, neck_dim, strides, k=256):
    super(Autoencoder, self).__init__()
    #
    self.image_size = image_size
    self.sizes = sizes
    self.strides = strides
    self.num_codes = num_codes
    self.neck_dim = neck_dim
    self.k = k
    #
    self.encoder = Encoder(self.image_size, self.sizes, self.num_codes
                            , self.strides, self.neck_dim, self.k)
    self.decoder = Decoder(self.image_size, self.sizes, self.num_codes
                            , self.strides, self.encoder.LinearLayers.LinearLayers
                            , self.k)


  def forward(self,x):
        code = self.encoder(x)
        x_hat = self.decoder(code)
        return x_hat, code

