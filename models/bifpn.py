import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5), nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
            P7_0 -------------------------- P7_2 -------->
            P6_0 ---------- P6_1 ---------- P6_2 -------->
            P5_0 ---------- P5_1 ---------- P5_2 -------->
            P4_0 ---------- P4_1 ---------- P4_2 -------->
            P3_0 -------------------------- P3_2 -------->
        """

        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))
        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


#class ConvModule(nn.Module):
#    def __init__(self, num_channels):
#        super(ConvModule, self).__init__()
#        self.module = nn.Sequential(
#                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
#                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0),
#                nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5),
#                nn.ReLU()
#                )
#    def forward(self, inputs):
#        return self.module(inputs)
#class BiFPN(nn.Module):
#    def __init__(self, num_channels, eps=0.0001):
#        super(BiFPN, self).__init__()
#        self.eps = eps 
#
#        self.conv6_up = ConvModule(num_channels)
#        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#        self.conv5_up = ConvModule(num_channels)
#        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#        self.conv5_up = ConvModule(num_channels)
#        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#        self.conv4_up = ConvModule(num_channels)
#        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#        self.conv3_up = ConvModule(num_channels)
#        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#        self.conv4_down = ConvModule(num_channels)
#        self.conv5_down = ConvModule(num_channels)
#        self.conv6_down = ConvModule(num_channels)
#        self.conv7_down = ConvModule(num_channels)
#
#        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
#        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
#        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
#        self.p7_downsample = nn.MaxPool2d(kernel_size=2)
#
#
#        
#        # weights
#        self.w1 = nn.Parameter(torch.ones(2, 5))
#        self.relu1 = nn.ReLU()
#        self.w2 = nn.Parameter(torch.ones(3, 3))
#        self.relu2 = nn.ReLU()
#
#    def forward(self, inputs):
#        p3_in, p4_in, p5_in, p6_in, p7_in = inputs 
#        w1 = self.relu1(self.w1)
#        w1 /= torch.sum(w1, dim=0) + self.eps 
#
#        w2 = self.relu2(self.w2)
#        w2 /= torch.sum(w2, dim=0) + self.eps
#        P3_in, P4_in, P5_in, P6_in, P7_in = inputs 
#        p6_up = self.conv6_up(w1[0, 1]*p6_in + w1[1, 1]*self.p6_upsample(p7_in))
#        p5_up = self.conv5_up(w1[0, 2]*p5_in + w1[1, 2]*self.p5_upsample(p6_up))
#        p4_up = self.conv5_up(w1[0, 3]*p4_in + w1[1, 3]*self.p4_upsample(p5_up))
#        p3_out = self.conv3_up(w1[0, 4]*p3_in + w1[1, 4]*self.p3_upsample(p4_up))
#
#        p4_out = self.conv4_down(w2[0, 0]*P4_in+w2[1, 0]*p4_up + w2[2, 0]*self.p4_downsample(p3_out))
#        p5_out = self.conv5_down(w2[0, 1]*P5_in+w2[1, 1]*p5_up + w2[2, 1]*self.p5_downsample(p4_out))
#        p6_out = self.conv6_down(w2[0, 2]*P6_in+w2[1, 2]*p6_up + w2[2, 2]*self.p6_downsample(p5_out))
#        p7_out = self.conv7_down(w1[0, 0]*P7_in + w1[1, 0]*self.p7_downsample(p6_out))
#        return p3_out, p4_out, p5_out, p6_out, p7_out 
#
#
#
#if __name__=='__main__':
#    p3 = torch.randn(5, 32, 64, 64)
#    p4 = torch.randn(5, 32, 32, 32)
#    p5 = torch.randn(5, 32, 16, 16)
#    p6 = torch.randn(5, 32, 8, 8)
#    p7 = torch.randn(5, 32, 4, 4)
#    inputs = [p3, p4, p5, p6, p7]
#    model = BiFPN(32)
#    output = model(inputs)
#    for out in output:
#        print(out.size())
#
#
#
