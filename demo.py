import os 
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from models import EfficientDet
from torchvision import transforms
import numpy as np
import skimage
from datasets import VOC_CLASSES, demoNormalizer, demoResizer
from timeit import default_timer as timer
import argparse
import copy
from utils import vis_bbox, EFFICIENTDET


parser = argparse.ArgumentParser(description='EfficientDet')

parser.add_argument('-n', '--network', default='efficientdet-d0',
                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('-s', '--score', default=True,
                    action="store_true", help='Show score')
parser.add_argument('-t', '--threshold', default=0.6,
                    type=float, help='Visualization threshold')
parser.add_argument('-it', '--iou_threshold', default=0.6,
                    type=float, help='Visualization threshold')
parser.add_argument('-w', '--weight', default='./weights/voc0712.pth',
                    type=str, help='Weight model path')
parser.add_argument('-c', '--cam',
                    action="store_true", help='Use camera')
parser.add_argument('-f', '--file_name', default='pic.jpg',
                    help='Image path')
parser.add_argument('--num_class', default=21, type=int,
                    help='Number of class used in model')
args = parser.parse_args()


class Detect(object):
    """
        dir_name: Folder or image_file
    """

    def __init__(self, weights, num_class=21, network='efficientdet-d0', size_image=(512, 512)):
        super(Detect,  self).__init__()
        self.weight = weights
        self.size_image = size_image
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        # self.transform = get_augumentation(phase='test')
        self.transform = transforms.Compose([demoNormalizer(), demoResizer()])

        checkpoint = []
        gpu = None
        if(self.weight is not None):
            if os.path.isfile(args.weight):
                print("=> loading checkpoint '{}'".format(args.weight))
                if gpu is None:
                    checkpoint = torch.load(args.weight)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.weight, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        args.start_epoch = checkpoint['epoch'] + 1
        print('params: ', params)
        del params
        self.model = EfficientDet(num_classes=args.num_class,
                         network=args.network,
                         W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                         D_class=EFFICIENTDET[args.network]['D_class'],
                         is_training=False,
                         threshold = args.threshold,
                         iou_threshold= args.iou_threshold
                         )
        if(self.weight is not None):
            self.model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def process(self, file_name=None, img=None, show=False):
        if file_name is not None:
            img = cv2.imread(file_name)
        origin_img = copy.deepcopy(img)
        # augmentation = self.transform(image=img)
        # img = augmentation['image']
        img = self.transform(img)
        img = img.to(self.device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img)
            bboxes = list()
            labels = list()
            bbox_scores = list()
            colors = list()
            for j in range(scores.shape[0]):
                bbox = transformed_anchors[[j], :][0].data.cpu().numpy()
                x1 = int(bbox[0]*origin_img.shape[1]/self.size_image[1])
                y1 = int(bbox[1]*origin_img.shape[0]/self.size_image[0])
                x2 = int(bbox[2]*origin_img.shape[1]/self.size_image[1])
                y2 = int(bbox[3]*origin_img.shape[0]/self.size_image[0])
                bboxes.append([x1, y1, x2, y2])
                label_name = VOC_CLASSES[int(classification[[j]])]
                labels.append(label_name)

                if(args.cam):
                    cv2.rectangle(origin_img, (x1, y1),
                                  (x2, y2), (179, 255, 179), 2, 1)
                if args.score:
                    score = np.around(
                        scores[[j]].cpu().numpy(), decimals=2) * 100
                    if(args.cam):
                        labelSize, baseLine = cv2.getTextSize('{} {}'.format(
                            label_name, int(score)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(
                            origin_img, (x1, y1-labelSize[1]), (x1+labelSize[0], y1+baseLine), (223, 128, 255), cv2.FILLED)
                        cv2.putText(
                            origin_img, '{} {}'.format(label_name, int(score)),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2
                        )
                    bbox_scores.append(int(score))
                else:
                    if(args.cam):
                        labelSize, baseLine = cv2.getTextSize('{}'.format(
                            label_name), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(
                            origin_img, (x1, y1-labelSize[1]), (x1+labelSize[0], y1+baseLine), (0, 102, 255), cv2.FILLED)
                        cv2.putText(
                            origin_img, '{} {}'.format(label_name, int(score)),
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 2
                        )
            if show:
                fig, ax = vis_bbox(img=origin_img, bbox=bboxes,
                                   label=labels, score=bbox_scores)
                print('Write file at here {}'.format("./docs/demo.png"))
                fig.savefig('./docs/demo.png')
                plt.show()
            else:
                return origin_img

    def camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera")
            exit(-1)
        count_tfps = 1
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            res, img = cap.read()
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = curr_fps
                curr_fps = 0
            if res:
                show_image = self.process(img=img)
                cv2.putText(
                    show_image, "FPS: " + str(fps), (10,  20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (204, 51, 51), 2
                )

                cv2.imshow("Detection", show_image)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            else:
                print("Unable to read image")
                exit(-1)
            count_tfps += 1
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detect = Detect(weights=args.weight)
    print('cam: ', args.cam)
    if args.cam:
        detect.camera()
    else:
        detect.process(file_name=args.file_name, show=True)
