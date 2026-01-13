#!/usr/bin/env python3
"""
Example usage of the Stitcher C++ class from Python

This demonstrates how to use the GPU-accelerated Stitcher class
to create RGB-D panoramas from fisheye camera images.
"""
import torch
import numpy as np
from my_stereo_pkg import _core_cpp

def create_example_calibration(camera_index, num_cameras, baseline=0.1):
    """
    Create example calibration for a camera
    
    Args:
        camera_index: Index of camera (0-based)
        num_cameras: Total number of cameras
        baseline: Distance between cameras in meters
    
    Returns:
        _core_cpp.Calibration object
    """
    calib = _core_cpp.Calibration()
    
    # Double sphere camera parameters
    calib.fl = (500.0, 500.0)  # Focal length (fx, fy)
    calib.principal = (320.0, 240.0)  # Principal point (cx, cy)
    calib.xi = 1.2  # First distortion parameter
    calib.alpha = 0.5  # Second distortion parameter
    calib.matching_scale = 1.0  # Scale factor
    
    # Extrinsic matrix (rotation + translation)
    # Create identity matrix and add translation
    rt = torch.eye(4, dtype=torch.float32)
    
    # Arrange cameras in a circle or line
    angle = (camera_index / num_cameras) * 2 * np.pi
    rt[0, 3] = baseline * camera_index  # X offset (linear arrangement)
    # Or for circular arrangement:
    # rt[0, 3] = baseline * np.cos(angle)
    # rt[1, 3] = baseline * np.sin(angle)
    
    calib.rt = rt
    
    return calib


def main():
    """
    Main example demonstrating Stitcher usage
    """
    print("=" * 60)
    print("Stitcher C++ Class - Python Example")
    print("=" * 60)
    
    # Configuration
    num_cameras = 4
    image_width = 640
    image_height = 480
    pano_width = 1280
    pano_height = 640
    min_dist = 0.1  # meters
    max_dist = 10.0  # meters
    
    # Check CUDA availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    print(f"📷 Number of cameras: {num_cameras}")
    print(f"🖼️  Input resolution: {image_width}x{image_height}")
    print(f"🗺️  Panorama resolution: {pano_width}x{pano_height}")
    
    # Create calibrations for all cameras
    print(f"\n🔧 Creating calibrations for {num_cameras} cameras...")
    calibrations = []
    for i in range(num_cameras):
        calib = create_example_calibration(i, num_cameras, baseline=0.15)
        calib.rt = calib.rt.to(device)  # Move to GPU
        calibrations.append(calib)
    print("   ✅ Calibrations created")
    
    # Create reprojection viewpoint (world origin)
    reprojection_viewpoint = torch.zeros(3, dtype=torch.float32, device=device)
    
    # Create masks (all pixels valid)
    masks = torch.ones(num_cameras, image_height, image_width, 
                      dtype=torch.float32, device=device)
    
    # Initialize Stitcher
    print(f"\n🎨 Initializing Stitcher...")
    stitcher = _core_cpp.Stitcher(
        calibrations,
        reprojection_viewpoint,
        masks,
        min_dist,
        max_dist,
        image_width,      # matching_cols
        image_height,     # matching_rows
        image_width,      # rgb_to_stitch_cols
        image_height,     # rgb_to_stitch_rows
        pano_width,       # panorama_cols
        pano_height,      # panorama_rows
        device,
        smoothing_radius=15,
        inpainting_iterations=32
    )
    print("   ✅ Stitcher initialized")
    
    # Create example input data (random for demonstration)
    print(f"\n📸 Creating example input images and depth maps...")
    images = []
    distance_maps = []
    
    for i in range(num_cameras):
        # Create gradient image for visualization
        img = torch.zeros(image_height, image_width, 3, dtype=torch.uint8, device=device)
        img[:, :, 0] = 100 + i * 30  # Red channel varies by camera
        img[:, :, 1] = torch.arange(image_width, device=device).repeat(image_height, 1) // 3
        img[:, :, 2] = 150
        images.append(img)
        
        # Create example distance map (gradient)
        dist = torch.ones(image_height, image_width, dtype=torch.float32, device=device)
        dist = dist * (min_dist + (max_dist - min_dist) * 0.5)  # Mid-range distance
        # Add some variation
        y_grad = torch.arange(image_height, device=device).unsqueeze(1).float() / image_height
        dist = dist + y_grad * 2.0
        distance_maps.append(dist)
    
    print(f"   ✅ Created {num_cameras} test images and depth maps")
    
    # Perform stitching
    print(f"\n🔄 Stitching panorama...")
    rgb_panorama, depth_panorama = stitcher.stitch(images, distance_maps)
    
    print(f"   ✅ Stitching complete!")
    print(f"\n📊 Output:")
    print(f"   RGB Panorama: {rgb_panorama.shape} ({rgb_panorama.dtype})")
    print(f"   Depth Panorama: {depth_panorama.shape} ({depth_panorama.dtype})")
    print(f"   RGB range: [{rgb_panorama.min().item()}, {rgb_panorama.max().item()}]")
    print(f"   Depth range: [{depth_panorama.min().item():.2f}, {depth_panorama.max().item():.2f}] meters")
    
    # Optional: Save results (requires torchvision or opencv-python)
    try:
        import cv2
        
        # Convert to numpy for saving
        rgb_np = rgb_panorama.cpu().numpy()
        depth_np = depth_panorama.cpu().numpy()
        
        # Save RGB (convert from RGB to BGR for OpenCV)
        cv2.imwrite('/tmp/panorama_rgb.png', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        
        # Save depth as normalized grayscale
        depth_normalized = ((depth_np - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
        cv2.imwrite('/tmp/panorama_depth.png', depth_normalized)
        
        print(f"\n💾 Results saved:")
        print(f"   RGB: /tmp/panorama_rgb.png")
        print(f"   Depth: /tmp/panorama_depth.png")
        
    except ImportError:
        print(f"\n⚠️  Install opencv-python to save results")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
