import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class COCOSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # Image preprocessing
        if self.transforms is not None:
            img = self.transforms(img)

        # Get segmentation masks
        masks = []
        for annotation in coco_annotation:
            mask = coco.annToMask(annotation)
            masks.append(mask)

        # Convert masks to tensor
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks])

        return img, masks

    def __len__(self):
        return len(self.ids)
