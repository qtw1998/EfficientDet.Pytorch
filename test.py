import torch
from models import EfficientDet
from utils import EFFICIENTDET
from models.efficientnet import ModifyEfficientNet
MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}

class ModifyDet(EfficientDet):
        def __init__(self,
                 num_classes,
                 network='efficientdet-d0',
                 D_bifpn=3,
                 W_bifpn=88,
                 D_class=3,
                 is_training=True,
                 threshold=0.01,
                 iou_threshold=0.5):
            super(ModifyDet, self).__init__(num_classes=num_classes,
                    network=network,
                    D_bifpn=D_bifpn,
                    W_bifpn=W_bifpn,
                    D_class=D_class,
                    is_training=is_training,
                    threshold=threshold,
                    iou_threshold=iou_threshold)
            self.backbone = ModifyEfficientNet.from_name(MODEL_MAP[network])
if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    for t in range(1, 2):
        model = ModifyEfficientNet.from_name('efficientnet-b{}'.format(t))
        outs = model(inputs)
        print('Output: ', )
        for out in outs:
            print(out.size(), end='')
        
    for i in range(0, 3):
        network = 'efficientdet-d{}'.format(int(i))
        inputs = torch.randn(5, 3, EFFICIENTDET[network]['input_size'], EFFICIENTDET[network]['input_size'])
        num_class = 10

        model = EfficientDet(num_classes=num_class,
                        network=network,
                        W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                        D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                        D_class=EFFICIENTDET[network]['D_class'],
                        is_training=False
                        )
        model.backbone = ModifyEfficientNet.from_name(MODEL_MAP[network])
        if(torch.cuda.is_available()):
            model = model.cuda()
            inputs = inputs.cuda()
        output = model(inputs)
        for out in output:
            print(out.size())
        print('Done: efficientdet-d{}'.format(int(i)))
        
    
#     model = EfficientDet(num_classes=20, is_training=False, network='efficientdet-d1')

#     output = model(inputs)
#     for out in output:
#         print(out.size())
