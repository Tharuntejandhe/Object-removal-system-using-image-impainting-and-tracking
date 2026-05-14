import os
import cv2
import json
import argparse
import numpy as np
import glob

# Global variables to store selection state
drawing = False
ix, iy = -1, -1
current_box = None
current_point = None
img_copy = None

def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, current_box, img_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_copy.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        temp_img = img_copy.copy()
        cv2.rectangle(temp_img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)
        cv2.imshow('Video', temp_img)

def draw_point(event, x, y, flags, param):
    global current_point, img_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_point = (x, y)
        temp_img = img_copy.copy()
        cv2.circle(temp_img, current_point, 5, (0, 0, 255), -1)
        cv2.imshow('Video', temp_img)

def main():
    global img_copy, current_box, current_point

    parser = argparse.ArgumentParser(description="Object Selection Tool")
    parser.add_argument("--frames_dir", help="Path to directory with extracted frames")
    parser.add_argument("--frame", help="Path to a single input frame")
    parser.add_argument("--mode", required=True, choices=['box', 'point'], help="Selection mode")
    args = parser.parse_args()

    if not args.frames_dir and not args.frame:
        print("Error: Must provide either --frames_dir or --frame")
        return

    frames_list = []
    if args.frames_dir:
        frames_list = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
        if not frames_list:
            print(f"Error: No PNG frames found in {args.frames_dir}")
            return
    else:
        if not os.path.exists(args.frame):
            print(f"Error: Frame not found at {args.frame}")
            return
        frames_list = [args.frame]

    # Ensure directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, 'config')
    output_dir = os.path.join(base_dir, 'output')
    masks_dir = os.path.join(base_dir, 'masks_raw')
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    cv2.namedWindow('Video')
    
    frame_idx = 0
    # If only 1 frame, start paused
    paused = len(frames_list) == 1 

    print("=====================================================")
    if not paused:
        print("Controls:")
        print("  [SPACE] : Play / Pause video")
        print("  [ , ]   : Step back 1 frame (when paused)")
        print("  [ . ]   : Step forward 1 frame (when paused)")
    print("  [ R ]   : Reset selection (when paused)")
    print("  [ENTER] : Save selection and exit")
    print("  [ ESC ] : Cancel and exit")
    print("=====================================================")

    # Initial callback setup
    if paused:
        if args.mode == 'box':
            cv2.setMouseCallback('Video', draw_box)
        else:
            cv2.setMouseCallback('Video', draw_point)

    while True:
        current_frame_path = frames_list[frame_idx]
        img = cv2.imread(current_frame_path)
        if img is None:
            break

        if not paused:
            img_copy = img.copy()
            cv2.putText(img_copy, f"PLAYING (Space to Pause) - {os.path.basename(current_frame_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', img_copy)
            
            key = cv2.waitKey(33) & 0xFF
            if key == 32: # SPACE
                paused = True
                current_box = None
                current_point = None
                if args.mode == 'box':
                    cv2.setMouseCallback('Video', draw_box)
                else:
                    cv2.setMouseCallback('Video', draw_point)
            elif key == 27 or key == ord('q'):
                break
            else:
                frame_idx = (frame_idx + 1) % len(frames_list)
        else:
            # Paused mode - allow drawing
            if current_box is None and current_point is None and not drawing:
                img_copy = img.copy()
                cv2.putText(img_copy, f"PAUSED - Draw Selection! - {os.path.basename(current_frame_path)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Video', img_copy)

            key = cv2.waitKey(20) & 0xFF
            if key == 32 and len(frames_list) > 1: # SPACE to play again
                paused = False
                cv2.setMouseCallback('Video', lambda *args: None) # Remove callback
            elif key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                current_box = None
                current_point = None
            elif key == ord(','): # Step back
                frame_idx = max(0, frame_idx - 1)
                current_box = None
                current_point = None
            elif key == ord('.'): # Step forward
                frame_idx = min(len(frames_list) - 1, frame_idx + 1)
                current_box = None
                current_point = None
            elif key == 13 or key == ord('s'): # ENTER or S
                if args.mode == 'box' and current_box is not None:
                    selection_data = {
                        "mode": "box",
                        "frame": current_frame_path,
                        "box": {
                            "x1": current_box[0],
                            "y1": current_box[1],
                            "x2": current_box[2],
                            "y2": current_box[3]
                        }
                    }
                    config_path = os.path.join(config_dir, 'selection.json')
                    with open(config_path, 'w') as f:
                        json.dump(selection_data, f, indent=2)
                    
                    preview_img = img.copy()
                    cv2.rectangle(preview_img, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir, 'selection_preview.png'), preview_img)
                    
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.rectangle(mask, (current_box[0], current_box[1]), (current_box[2], current_box[3]), 255, -1)
                    frame_basename = os.path.basename(current_frame_path)
                    cv2.imwrite(os.path.join(masks_dir, frame_basename), mask)
                    
                    print(f"Selection saved to {config_path}")
                    break

                elif args.mode == 'point' and current_point is not None:
                    selection_data = {
                        "mode": "point",
                        "frame": current_frame_path,
                        "point": {
                            "x": current_point[0],
                            "y": current_point[1]
                        }
                    }
                    config_path = os.path.join(config_dir, 'selection.json')
                    with open(config_path, 'w') as f:
                        json.dump(selection_data, f, indent=2)
                    
                    preview_img = img.copy()
                    cv2.circle(preview_img, current_point, 5, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(output_dir, 'selection_preview.png'), preview_img)
                    
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    cv2.circle(mask, current_point, 40, 255, -1)
                    frame_basename = os.path.basename(current_frame_path)
                    cv2.imwrite(os.path.join(masks_dir, frame_basename), mask)
                    
                    print(f"Selection saved to {config_path}")
                    break
                else:
                    print("No selection made yet. Draw your selection first.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    