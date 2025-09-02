"""
Generate Real Example Images of Endoscopy Corruptions
This script creates actual corrupted images showing different severity levels
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from endoscopycorruptions import corrupt

# List of all endoscopy corruptions
CORRUPTIONS = [
    'brightness', 'darkness', 'contrast', 'fog',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'iso_noise',
    'lens_distortion', 'resolution_change', 'specular_reflection', 'color_changes'
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

def find_sample_image():
    """Find a sample image from the kidney stone dataset"""
    data_dir = "data"
    
    # Priority search order for best representative images
    search_patterns = [
        # Look for Michel Daudon dataset first
        "Michel Daudon*/*/train/*/",
        "Michel Daudon*/*/test/*/", 
        # Then Jonathan El-Beze dataset
        "Jonathan El-Beze*/*/train/*/",
        "Jonathan El-Beze*/*/test/*/",
        # Any other dataset structure
        "*/*/train/*/",
        "*/*/test/*/",
        # Fallback to any image
        "*/"
    ]
    
    print(f"🔍 Searching for kidney stone images in '{data_dir}' directory...")
    
    # Search through the directory structure
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                
                # Prefer images with descriptive names (stones, normal, etc.)
                if any(keyword in file.lower() for keyword in ['stone', 'normal', 'kidney']):
                    print(f"✅ Found prioritized image: {image_path}")
                    return image_path
                
                # Store first valid image as backup
                if 'first_found' not in locals():
                    first_found = image_path
    
    # Use first found image if no prioritized image found
    if 'first_found' in locals():
        print(f"✅ Using first found image: {first_found}")
        return first_found
    
    # If no dataset image found, create a synthetic one
    print("⚠️  No real kidney stone images found in dataset!")
    print("📁 Searched in:", data_dir)
    print("🔧 Creating synthetic endoscopy-like image for demonstration...")
    return create_synthetic_endoscopy_image()

def create_synthetic_endoscopy_image():
    """Create a synthetic endoscopy-like image for demonstration"""
    # Create a circular endoscopy view with some features
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create circular endoscopy field of view
    center = (128, 128)
    radius = 120
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Add tissue-like background
    img[mask] = [180, 120, 100]  # Pinkish tissue color
    
    # Add some kidney stone-like features
    # Stone 1
    stone_mask1 = (x - 100)**2 + (y - 90)**2 <= 15**2
    img[stone_mask1 & mask] = [240, 240, 200]  # Yellowish stone
    
    # Stone 2
    stone_mask2 = (x - 150)**2 + (y - 140)**2 <= 20**2
    img[stone_mask2 & mask] = [200, 180, 160]  # Brownish stone
    
    # Add some texture and noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save synthetic image
    synthetic_path = "synthetic_endoscopy_sample.jpg"
    Image.fromarray(img).save(synthetic_path)
    print(f"Created synthetic image: {synthetic_path}")
    
    return synthetic_path

def generate_corruption_grid(image_path, corruption_type, output_dir="corruption_examples"):
    """Generate a grid showing one corruption type at all severity levels"""
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    original_array = np.array(original_img)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{corruption_type.upper()} Corruption - Severity Levels 1-5', fontsize=16, fontweight='bold')
    
    # Show original in first position
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('ORIGINAL', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Generate corrupted versions
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, severity in enumerate(SEVERITY_LEVELS):
        row, col = positions[i]
        
        try:
            # Apply corruption
            corrupted_array = corrupt(original_array, corruption_name=corruption_type, severity=severity)
            corrupted_img = Image.fromarray(corrupted_array.astype(np.uint8))
            
            # Display
            axes[row, col].imshow(corrupted_img)
            axes[row, col].set_title(f'Severity {severity}', fontweight='bold')
            axes[row, col].axis('off')
            
            # Save individual corrupted image
            individual_path = os.path.join(output_dir, f"{corruption_type}_severity_{severity}.jpg")
            corrupted_img.save(individual_path)
            
        except Exception as e:
            print(f"Error applying {corruption_type} at severity {severity}: {e}")
            axes[row, col].text(0.5, 0.5, f'Error\nSeverity {severity}', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # Save grid
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"{corruption_type}_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated {corruption_type} examples: {grid_path}")

def generate_category_comparison(image_path, output_dir="corruption_examples"):
    """Generate comparison showing one example from each corruption category"""
    
    categories = {
        'Lighting': ['brightness', 'darkness', 'contrast', 'fog'],
        'Movement': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'Noise': ['gaussian_noise', 'impulse_noise', 'shot_noise', 'iso_noise'],
        'Endoscopy': ['lens_distortion', 'resolution_change', 'specular_reflection', 'color_changes']
    }
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    original_array = np.array(original_img)
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Endoscopy Corruption Categories (Severity 3)', fontsize=18, fontweight='bold')
    
    for cat_idx, (category, corruptions) in enumerate(categories.items()):
        # Category label
        axes[cat_idx, 0].text(0.5, 0.5, f'{category}\nCorruptions', 
                             ha='center', va='center', fontsize=14, fontweight='bold',
                             transform=axes[cat_idx, 0].transAxes)
        axes[cat_idx, 0].axis('off')
        
        # Generate examples for each corruption in category
        for corr_idx, corruption in enumerate(corruptions):
            try:
                corrupted_array = corrupt(original_array, corruption_name=corruption, severity=3)
                corrupted_img = Image.fromarray(corrupted_array.astype(np.uint8))
                
                axes[cat_idx, corr_idx + 1].imshow(corrupted_img)
                axes[cat_idx, corr_idx + 1].set_title(corruption.replace('_', ' ').title(), fontsize=10)
                axes[cat_idx, corr_idx + 1].axis('off')
                
            except Exception as e:
                print(f"Error with {corruption}: {e}")
                axes[cat_idx, corr_idx + 1].text(0.5, 0.5, f'Error\n{corruption}', 
                                               ha='center', va='center', 
                                               transform=axes[cat_idx, corr_idx + 1].transAxes)
                axes[cat_idx, corr_idx + 1].axis('off')
    
    plt.tight_layout()
    category_path = os.path.join(output_dir, "corruption_categories_overview.png")
    plt.savefig(category_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated category comparison: {category_path}")

def generate_severity_progression(image_path, selected_corruptions, output_dir="corruption_examples"):
    """Generate severity progression for selected corruptions"""
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    original_array = np.array(original_img)
    
    # Create figure
    fig, axes = plt.subplots(len(selected_corruptions), 6, figsize=(18, 3*len(selected_corruptions)))
    fig.suptitle('Severity Progression for Key Corruptions', fontsize=16, fontweight='bold')
    
    for corr_idx, corruption in enumerate(selected_corruptions):
        # Show original
        axes[corr_idx, 0].imshow(original_img)
        axes[corr_idx, 0].set_title('Original')
        axes[corr_idx, 0].axis('off')
        
        # Show severity progression
        for severity in SEVERITY_LEVELS:
            try:
                corrupted_array = corrupt(original_array, corruption_name=corruption, severity=severity)
                corrupted_img = Image.fromarray(corrupted_array.astype(np.uint8))
                
                axes[corr_idx, severity].imshow(corrupted_img)
                axes[corr_idx, severity].set_title(f'Severity {severity}')
                axes[corr_idx, severity].axis('off')
                
            except Exception as e:
                print(f"Error with {corruption} severity {severity}: {e}")
        
        # Add corruption name as y-label
        axes[corr_idx, 0].set_ylabel(corruption.replace('_', ' ').title(), 
                                   rotation=0, ha='right', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    progression_path = os.path.join(output_dir, "severity_progression.png")
    plt.savefig(progression_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated severity progression: {progression_path}")

def main():
    """Generate all corruption example images"""
    
    print("🖼️  Generating Real Endoscopy Corruption Examples")
    print("=" * 60)
    
    # Create output directory
    output_dir = "corruption_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find sample image
    print("📂 Finding sample image...")
    image_path = find_sample_image()
    if not image_path:
        print("❌ No sample image found!")
        return
    
    # Determine if this is a real or synthetic image
    if "synthetic" in image_path.lower() or "temp_" in image_path:
        image_type = "Synthetic demo image"
    else:
        image_type = "Real kidney stone image"
    
    print(f"✅ Using {image_type}: {image_path}")
    
    # Load and inspect the image
    try:
        test_img = plt.imread(image_path)
        print(f"   Image dimensions: {test_img.shape}")
        print(f"   Data type: {test_img.dtype}")
    except Exception as e:
        print(f"   Warning: Could not inspect image - {e}")
    
    # Generate category comparison (overview)
    print("\n📊 Generating corruption categories overview...")
    generate_category_comparison(image_path, output_dir)
    
    # Generate severity progression for key corruptions
    print("\n📈 Generating severity progression examples...")
    key_corruptions = ['brightness', 'gaussian_noise', 'motion_blur', 'specular_reflection']
    generate_severity_progression(image_path, key_corruptions, output_dir)
    
    # Generate detailed grids for a few representative corruptions
    print("\n🔍 Generating detailed corruption grids...")
    representative_corruptions = [
        'brightness', 'darkness', 'gaussian_noise', 'motion_blur', 
        'specular_reflection', 'lens_distortion'
    ]
    
    for corruption in representative_corruptions:
        print(f"   Generating {corruption} grid...")
        generate_corruption_grid(image_path, corruption, output_dir)
    
    print(f"\n🎉 All corruption examples generated in '{output_dir}/' directory!")
    print("\nGenerated files:")
    print("  📊 corruption_categories_overview.png - Overview of all 16 corruptions")
    print("  📈 severity_progression.png - Severity levels for key corruptions")
    print("  🔍 {corruption}_grid.png - Individual grids for each corruption type")
    print("  📷 Individual corrupted images for each severity level")
    
    # Display file count
    files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n📁 Total files generated: {len(files)}")

if __name__ == "__main__":
    main() 