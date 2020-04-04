import torch
from efficientnet_pytorch import EfficientNet

class ModifyEfficientNet(EfficientNet):
    def get_list_features(self):
        list_feature = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if(block._depthwise_conv.stride == [2, 2]):
                list_feature.append(block._bn2.num_features)
        return list_feature[1:]
    def forward(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        P = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if(block._depthwise_conv.stride == [2, 2]):
                P.append(x)
        return P[1:]
        

if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b0')
    model = ModifyEfficientNet.from_pretrained('efficientnet-b0')
    inputs = torch.randn(4, 3, 640, 640)
    P = model(inputs)
    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))
    # print('model: ', model)