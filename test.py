import torch
from models import EfficientDet
from utils import EFFICIENTDET
from models.efficientnet import EfficientNet

from models.efficientnet import ModifyEfficientNet
if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    for t in range(1, 8):
        model = ModifyEfficientNet.from_pretrained('efficientnet-b{}'.format(t))
        outs = model(inputs)
        print('Output: ', )
        for out in outs:
            print(out.size(), end='')
        
    pass 
    # for i in range(0, 1):
    #     network = 'efficientdet-d{}'.format(int(i))
    #     inputs = torch.randn(5, 3, EFFICIENTDET[network]['input_size'], EFFICIENTDET[network]['input_size'])
    #     # Test EfficientNet
    #     model = EfficientNet.from_pretrained('efficientnet-b{}'.format(int(i)))
    #     if(torch.cuda.is_available()):
    #         model = model.cuda()
    #         inputs = inputs.cuda()
    #     P = model(inputs)
    #     print('Done: efficientnet-b{}'.format(int(i)))
    #     num_class = 10

    #     model = EfficientDet(num_classes=num_class,
    #                     network=network,
    #                     W_bifpn=EFFICIENTDET[network]['W_bifpn'],
    #                     D_bifpn=EFFICIENTDET[network]['D_bifpn'],
    #                     D_class=EFFICIENTDET[network]['D_class'],
    #                     is_training=False
    #                     )
    #     if(torch.cuda.is_available()):
    #         model = model.cuda()
    #         inputs = inputs.cuda()
    #     output = model(inputs)
    #     for out in output:
    #         print(out.size())
    #     print('Done: efficientdet-d{}'.format(int(i)))
        
    
#     model = EfficientDet(num_classes=20, is_training=False, network='efficientdet-d1')

#     output = model(inputs)
#     for out in output:
#         print(out.size())
