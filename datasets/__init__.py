from .voc0712 import VOCDetection, VOC_CLASSES
from .augmentation import get_augumentation, detection_collate, Resizer, Normalizer, Augmenter, collater, demoNormalizer, demoResizer
from .coco import CocoDataset