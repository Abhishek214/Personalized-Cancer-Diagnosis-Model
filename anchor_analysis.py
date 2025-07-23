import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import json

def analyze_dataset_for_anchors(annotation_file):
    """Analyze dataset to determine optimal anchor configurations"""
    
    coco = COCO(annotation_file)
    
    # Get all bounding boxes
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for ann in anns:
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w > 0 and h > 0:  # Valid boxes only
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            aspect_ratios.append(w / h)
    
    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)
    
    # Size analysis
    print("=== SIZE ANALYSIS ===")
    print(f"Width - Min: {widths.min():.1f}, Max: {widths.max():.1f}, Mean: {widths.mean():.1f}")
    print(f"Height - Min: {heights.min():.1f}, Max: {heights.max():.1f}, Mean: {heights.mean():.1f}")
    print(f"Area - Min: {areas.min():.1f}, Max: {areas.max():.1f}, Mean: {areas.mean():.1f}")
    
    # COCO size categories (adjust input size accordingly)
    input_size = 768  # Your training input size
    small_threshold = 32 * 32 * (input_size/512)**2
    medium_threshold = 96 * 96 * (input_size/512)**2
    
    small_objects = np.sum(areas < small_threshold)
    medium_objects = np.sum((areas >= small_threshold) & (areas < medium_threshold))
    large_objects = np.sum(areas >= medium_threshold)
    
    print(f"\nObject size distribution:")
    print(f"Small (<{small_threshold:.0f}px²): {small_objects} ({small_objects/len(areas)*100:.1f}%)")
    print(f"Medium: {medium_objects} ({medium_objects/len(areas)*100:.1f}%)")
    print(f"Large (>{medium_threshold:.0f}px²): {large_objects} ({large_objects/len(areas)*100:.1f}%)")
    
    # Aspect ratio analysis
    print(f"\n=== ASPECT RATIO ANALYSIS ===")
    print(f"Aspect ratios - Min: {aspect_ratios.min():.2f}, Max: {aspect_ratios.max():.2f}, Mean: {aspect_ratios.mean():.2f}")
    
    # Common percentiles for anchor design
    ar_percentiles = [10, 25, 50, 75, 90]
    ar_values = np.percentile(aspect_ratios, ar_percentiles)
    
    print("Aspect ratio percentiles:")
    for p, v in zip(ar_percentiles, ar_values):
        print(f"  {p}th: {v:.2f}")
    
    # Recommended anchor ratios
    print(f"\n=== RECOMMENDED ANCHOR RATIOS ===")
    # Use percentiles to define ratios
    narrow = ar_values[0]  # 10th percentile
    medium_narrow = ar_values[1]  # 25th percentile  
    square = 1.0
    medium_wide = ar_values[3]  # 75th percentile
    wide = ar_values[4]  # 90th percentile
    
    recommended_ratios = [narrow, medium_narrow, square, medium_wide, wide]
    print("Suggested ratios:", [f"({1/r:.1f}, {r:.1f})" if r < 1 else f"({r:.1f}, {1/r:.1f})" for r in recommended_ratios])
    
    # Scale analysis based on object sizes
    print(f"\n=== RECOMMENDED ANCHOR SCALES ===")
    
    # Base anchor size (typically 4 for EfficientDet)
    base_anchor = 4
    
    # Calculate scales to cover size distribution
    sqrt_areas = np.sqrt(areas)
    size_percentiles = [25, 50, 75, 90]
    size_values = np.percentile(sqrt_areas, size_percentiles)
    
    # Convert to scales relative to base anchor
    scales = []
    for size in size_values:
        # Scale needed to match this size at the finest feature level
        scale = size / (base_anchor * 8)  # 8 is stride of P3 level
        scales.append(scale)
    
    print("Object size percentiles (sqrt of area):", [f"{v:.1f}" for v in size_values])
    print("Suggested scales:", [f"2**{np.log2(s):.2f}" for s in scales])
    
    # Plot distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Width/Height distribution
    ax1.hist([widths, heights], bins=50, alpha=0.7, label=['Width', 'Height'])
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Width/Height Distribution')
    ax1.legend()
    
    # Area distribution
    ax2.hist(areas, bins=50, alpha=0.7)
    ax2.axvline(small_threshold, color='r', linestyle='--', label='Small/Medium')
    ax2.axvline(medium_threshold, color='r', linestyle='--', label='Medium/Large')
    ax2.set_xlabel('Area (px²)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Area Distribution')
    ax2.legend()
    
    # Aspect ratio distribution
    ax3.hist(aspect_ratios, bins=50, alpha=0.7)
    ax3.set_xlabel('Aspect Ratio (W/H)')
    ax3.set_ylabel('Frequency') 
    ax3.set_title('Aspect Ratio Distribution')
    
    # Box scatter plot
    ax4.scatter(widths, heights, alpha=0.5, s=1)
    ax4.set_xlabel('Width')
    ax4.set_ylabel('Height')
    ax4.set_title('Width vs Height Scatter')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'recommended_ratios': recommended_ratios,
        'recommended_scales': scales,
        'size_distribution': {'small': small_objects, 'medium': medium_objects, 'large': large_objects},
        'aspect_ratio_stats': {'mean': aspect_ratios.mean(), 'std': aspect_ratios.std()}
    }

# Usage
if __name__ == "__main__":
    # Analyze your dataset
    results = analyze_dataset_for_anchors('datasets/abhil/annotations/instances_train2017.json')
    
    # Generate config
    ratios_str = str([(1/r, r) if r < 1 else (r, 1/r) for r in results['recommended_ratios']])
    scales_str = str([f"2 ** {np.log2(s):.2f}" for s in results['recommended_scales']])
    
    print(f"\n=== FINAL CONFIG ===")
    print(f"anchors_ratios: '{ratios_str}'")
    print(f"anchors_scales: '{scales_str}'")