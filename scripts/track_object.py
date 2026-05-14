import os
import json
import torch
import numpy as np
import cv2
import argparse
import glob

# Import SAM 2 video predictor
from sam2.build_sam import build_sam2_video_predictor

def main():
    parser = argparse.ArgumentParser(description="SAM 2 Video Object Tracking")
    parser.add_argument("--config", default="config/selection.json", help="Path to selection.json")
    parser.add_argument("--model", default="models/sam2_hiera_tiny.pt", help="Path to SAM 2 model")
    parser.add_argument("--model_cfg", default="sam2_hiera_t.yaml", help="SAM 2 model config name")
    parser.add_argument("--frames_dir", default="frames", help="Directory with extracted frames")
    parser.add_argument("--masks_dir", default="masks_raw", help="Output directory for masks")
    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load selection
    if not os.path.exists(args.config):
        print(f"Error: Selection file not found at {args.config}")
        return

    with open(args.config, 'r') as f:
        selection = json.load(f)

    frame_path = selection.get("frame")
    if not frame_path or not os.path.exists(frame_path):
        print(f"Error: Selected frame not found at {frame_path}")
        return

    # SAM 2 predictor.init_state(video_path) expects a directory of JPEGs/PNGs.
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frames:
        print(f"Error: No frames found in {args.frames_dir}")
        return

    try:
        frame_idx = [os.path.basename(f) for f in frames].index(os.path.basename(frame_path))
    except ValueError:
        print(f"Error: Frame {frame_path} not found in the loaded frames list.")
        return

    print(f"Selection is on frame index {frame_idx} ({os.path.basename(frame_path)})")

    # Load SAM 2 model
    if not os.path.exists(args.model):
        print(f"Error: SAM 2 model weights not found at {args.model}")
        print("Make sure the download completed successfully.")
        return

    print(f"Loading SAM 2 model {args.model_cfg} from {args.model}...")
    try:
        predictor = build_sam2_video_predictor(args.model_cfg, args.model, device=device)
    except Exception as e:
        print(f"Failed to load SAM 2 model: {e}")
        return

    print(f"Initializing tracking state for video frames in {args.frames_dir}...")
    inference_state = predictor.init_state(
        video_path=args.frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    )
    predictor.reset_state(inference_state)

    # Add prompt
    obj_id = 1
    if selection["mode"] == "box":
        box = selection["box"]
        box_coords = np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float32)
        print(f"Adding box prompt to frame {frame_idx}: {box_coords}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_coords,
        )
    elif selection["mode"] == "point":
        point = selection["point"]
        point_coords = np.array([[point["x"], point["y"]]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        print(f"Adding point prompt to frame {frame_idx}: {point_coords}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_coords,
            labels=labels,
        )

    video_segments = {}
    print("Propagating tracking FORWARD through the video. This may take a while...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print("Propagating tracking BACKWARD through the video. This may take a while...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save masks
    print(f"Saving masks to {args.masks_dir}...")
    os.makedirs(args.masks_dir, exist_ok=True)

    for idx, obj_masks in video_segments.items():
        if obj_id in obj_masks:
            mask = obj_masks[obj_id]
            mask_uint8 = (mask[0] * 255).astype(np.uint8)
            frame_name = os.path.basename(frames[idx])
            mask_path = os.path.join(args.masks_dir, frame_name)
            cv2.imwrite(mask_path, mask_uint8)

    print("Tracking complete!")

if __name__ == "__main__":
    main()
