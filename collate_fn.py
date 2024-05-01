from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import os

class CustomCocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        annotations = [{'bbox': ann['bbox'], 'label': ann['category_id']} for ann in coco_annotation]
        return img, annotations

    def __len__(self):
        return len(self.ids)

def custom_collate(batch):
    """Custom collate function for batching images and annotations."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Collate images normally since they are all the same size after transformation
    images = default_collate(images)
    
    # Annotations need to be handled as a list of dictionaries
    return images, targets

