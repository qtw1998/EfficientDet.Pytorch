import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ( # always index 0
    'car', 'bus', 'person', 'bike', 'truck', 'motor',
    'train', 'rider', 'traffic sign', 'traffic light')
# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join('/home/toandm2', "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            # difficult = int(obj.find('difficult').text) == 1
            difficult = 0
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2020', 'train')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', '%s.xml')
        self._imgpath = osp.join('%s', 'images', '100k', image_sets[0][1], '%s.jpg')
        self.ids = list()
        # print('here: ', image_sets)
        for (year, name) in image_sets:
            rootpath = self.root
            # print('rootpath: ', rootpath)
            for line in open(osp.join(rootpath,  name + '.txt')):
                # print('line strip ', line.strip())
                self.ids.append((rootpath, osp.join('Annotations', name, line.strip())))
                
                # print(line.strip())

    def __getitem__(self, index):
        try: 
            img_id = self.ids[index]
            # print('img_id: ', img_id)
            target = ET.parse(self._annopath % img_id).getroot()
        except:
            print('throw at ', index)
            return self[index]
        img_id = (img_id[0], img_id[1].split('/')[-1])
        # print('img_id ', img_id)
        # print('path image ', self._imgpath%img_id)
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        target = np.array(target)
        sample = {'img': img, 'annot': target}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

        bbox = target[:, :4]
        labels = target[:, 4]

        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
        return {'image': img, 'bboxes': bbox, 'category_id': labels}

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(VOC_CLASSES)

    def label_to_name(self, label):
        return VOC_CLASSES[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt
