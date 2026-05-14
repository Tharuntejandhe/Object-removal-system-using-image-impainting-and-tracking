import os
import cv2
import glob
import numpy as np
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="OpenCV Video Inpainting Baseline")
    parser.add_argument("--frames_dir", default="frames", help="Original frames")
    parser.add_argument("--masks_dir", default="masks_raw", help="SAM 2 generated masks")
    parser.add_argument("--clean_masks_dir", default="masks_clean", help="Dilated masks")
    parser.add_argument("--out_dir", default="inpainted_frames", help="Inpainted output frames")
    parser.add_argument("--dilation", type=int, default=10, help="Pixels to dilate mask by")
    args = parser.parse_args()

    # Ensure directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_masks_dir = os.path.join(base_dir, args.clean_masks_dir)
    out_dir = os.path.join(base_dir, args.out_dir)
    frames_dir = os.path.join(base_dir, args.frames_dir)
    masks_dir = os.path.join(base_dir, args.masks_dir)

    os.makedirs(clean_masks_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        print(f"Error: No frames found in {frames_dir}")
        return

    print("Starting OpenCV inpainting process...")
    kernel = np.ones((args.dilation, args.dilation), np.uint8)

    for frame_path in tqdm(frames, desc="Inpainting frames"):
        basename = os.path.basename(frame_path)
        mask_path = os.path.join(masks_dir, basename)
        
        # Load frame
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # If no mask, just copy the frame
            cv2.imwrite(os.path.join(out_dir, basename), img)
            continue
            
        # Dilate mask slightly to cover edges
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(os.path.join(clean_masks_dir, basename), dilated_mask)
        
        # Inpaint
        inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Save
        cv2.imwrite(os.path.join(out_dir, basename), inpainted)

    print("Inpainting complete.")

if __name__ == "__main__":
    main()
