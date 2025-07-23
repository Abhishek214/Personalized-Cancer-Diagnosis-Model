import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, root_dir, set="train2017", transform=None):
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', f'instances_{self.set_name}.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # Load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # Also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # Some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # Parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # Transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape

        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Enhanced augmentation for document detection"""
    
    def __init__(self, 
                 flip_x=0.5,
                 brightness_range=0.15,
                 contrast_range=0.15,
                 rotation_range=8,
                 noise_prob=0.2):
        self.flip_x = flip_x
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.noise_prob = noise_prob

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        
        # Horizontal flip
        if np.random.rand() < self.flip_x:
            image, annots = self._flip_horizontal(image, annots)
        
        # Brightness adjustment
        if np.random.rand() < 0.4:
            image = self._adjust_brightness(image)
        
        # Contrast adjustment
        if np.random.rand() < 0.4:
            image = self._adjust_contrast(image)
        
        # Add noise
        if np.random.rand() < self.noise_prob:
            image = self._add_noise(image)
        
        # Small rotation for document orientation variations
        if np.random.rand() < 0.25:
            image, annots = self._rotate_image(image, annots)
        
        return {'img': image, 'annot': annots}
    
    def _flip_horizontal(self, image, annots):
        image = image[:, ::-1, :]
        rows, cols, channels = image.shape
        
        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()
        
        annots[:, 0] = cols - x2
        annots[:, 2] = cols - x1
        
        return image, annots
    
    def _adjust_brightness(self, image):
        factor = 1 + np.random.uniform(-self.brightness_range, self.brightness_range)
        image = np.clip(image * factor, 0, 1)
        return image
    
    def _adjust_contrast(self, image):
        factor = 1 + np.random.uniform(-self.contrast_range, self.contrast_range)
        mean = np.mean(image)
        image = np.clip((image - mean) * factor + mean, 0, 1)
        return image
    
    def _add_noise(self, image):
        noise = np.random.normal(0, 0.015, image.shape)
        image = np.clip(image + noise, 0, 1)
        return image
    
    def _rotate_image(self, image, annots):
        """Small rotation for document orientation variations"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Transform bounding boxes
        if len(annots) > 0:
            # Convert bbox to corners
            corners = np.zeros((len(annots), 4, 2))
            for i, bbox in enumerate(annots):
                x1, y1, x2, y2 = bbox[:4]
                corners[i] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            # Transform corners
            corners_flat = corners.reshape(-1, 2)
            corners_homo = np.column_stack([corners_flat, np.ones(len(corners_flat))])
            rotated_corners_flat = corners_homo.dot(M.T)
            rotated_corners = rotated_corners_flat.reshape(-1, 4, 2)
            
            # Convert back to bboxes
            new_annots = []
            for i, corner in enumerate(rotated_corners):
                x_coords = corner[:, 0]
                y_coords = corner[:, 1]
                x1, y1 = np.min(x_coords), np.min(y_coords)
                x2, y2 = np.max(x_coords), np.max(y_coords)
                
                # Check if bbox is still valid and within image bounds
                if (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and 
                    x2 < w and y2 < h and (x2-x1) > 5 and (y2-y1) > 5):
                    new_annots.append([x1, y1, x2, y2, annots[i, 4]])
            
            annots = np.array(new_annots) if new_annots else np.zeros((0, 5))
        
        return rotated, annots


class Normalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {
            'img': ((image.astype(np.float32) - self.mean) / self.std),
            'annot': annots
        }