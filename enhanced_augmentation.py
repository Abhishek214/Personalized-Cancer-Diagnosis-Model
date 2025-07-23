import cv2
import numpy as np
import torch

class EnhancedAugmenter(object):
    """Enhanced augmentation for document detection"""
    
    def __init__(self, 
                 flip_prob=0.5,
                 brightness_range=0.2,
                 contrast_range=0.2,
                 rotation_range=15,
                 scale_range=(0.8, 1.2),
                 noise_prob=0.3):
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_prob = noise_prob
    
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        
        # Horizontal flip
        if np.random.rand() < self.flip_prob:
            image, annots = self._flip_horizontal(image, annots)
        
        # Brightness adjustment
        if np.random.rand() < 0.5:
            image = self._adjust_brightness(image)
        
        # Contrast adjustment
        if np.random.rand() < 0.5:
            image = self._adjust_contrast(image)
        
        # Add noise
        if np.random.rand() < self.noise_prob:
            image = self._add_noise(image)
        
        # Small rotation (good for document orientation variations)
        if np.random.rand() < 0.3:
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
        noise = np.random.normal(0, 0.02, image.shape)
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
            corners = self._get_bbox_corners(annots)
            rotated_corners = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 4, 2)
            annots = self._corners_to_bbox(rotated_corners)
            
            # Filter out boxes that went outside image bounds
            valid_mask = (annots[:, 0] >= 0) & (annots[:, 1] >= 0) & \
                        (annots[:, 2] < w) & (annots[:, 3] < h) & \
                        (annots[:, 2] > annots[:, 0]) & (annots[:, 3] > annots[:, 1])
            annots = annots[valid_mask]
        
        return rotated, annots
    
    def _get_bbox_corners(self, bboxes):
        """Convert bboxes to corner points"""
        corners = np.zeros((len(bboxes), 4, 2))
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            corners[i] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return corners
    
    def _corners_to_bbox(self, corners):
        """Convert corner points back to bboxes"""
        bboxes = np.zeros((len(corners), 5))
        for i, corner in enumerate(corners):
            x_coords = corner[:, 0]
            y_coords = corner[:, 1]
            bboxes[i, 0] = np.min(x_coords)  # x1
            bboxes[i, 1] = np.min(y_coords)  # y1
            bboxes[i, 2] = np.max(x_coords)  # x2
            bboxes[i, 3] = np.max(y_coords)  # y2
            bboxes[i, 4] = corners[0, 4] if hasattr(corners[0], '__len__') and len(corners[0]) > 4 else 0
        return bboxes