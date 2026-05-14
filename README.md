# Video Object Removal Pipeline

A professional video object removal pipeline. Currently implements the base structure and media extraction (frames and audio), object tracking with SAM 2, and inpainting.

## Methodology

This pipeline operates in several distinct stages:
1. **Pre-processing**: The input video is split into individual frames and its audio track is extracted using FFmpeg.
2. **User Input & Selection**: The user selects the object to be removed from a single frame using either a bounding box or point prompts.
3. **Automated Tracking (SAM 2)**: Utilizing Meta's Segment Anything Model 2 (SAM 2), the selected object is tracked forward and backward across all frames. This generates highly accurate binary segmentation masks for the object in every frame.
4. **Mask Refinement**: The generated masks are automatically dilated slightly to ensure the entire object and its boundaries are covered, preventing edge artifacts.
5. **Inpainting**: The pipeline applies OpenCV's inpainting algorithms (like Telea or Navier-Stokes) frame-by-frame. It uses the dilated masks to remove the object and realistically fill the gap by interpolating surrounding background pixels.
6. **Post-processing**: The inpainted frames are seamlessly compiled back into a video format, and the original audio track is reattached.

## Installation

1. Create and activate a conda environment (recommended):
   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```

2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `ffmpeg`:
   - **macOS**: `brew install ffmpeg`
   - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
   - **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or use `winget install ffmpeg`
   
## Validation

Before running the full pipeline, verify your dependencies are correctly installed:
```bash
python run_pipeline.py --check
```
This checks Python version, required imports, ffmpeg, and PyTorch/CUDA availability.

## How to Use Guide

### 1. Quick Start (GUI)

The easiest way to use this pipeline is via the desktop graphical interface.

```bash
python gui.py
```

This will open a window where you can select your video and run all 4 steps with the click of a button! The GUI will also automatically clean up thousands of temporary files once the final video is generated to save your disk space.

---

### 2. Command Line Usage

1. Place your input video inside the `input/` folder, for example `input/input.mp4`.
2. Run the base pipeline:
   ```bash
   python run_pipeline.py --video input/input.mp4
   ```

### Debugging

If you encounter errors, you can run the pipeline in debug mode to see full tracebacks:
```bash
python run_pipeline.py --video input/input.mp4 --debug
```

### Outputs

After extraction, your files will be saved here:
- **Frames**: `frames/` directory (saved as `000001.jpg`, `000002.jpg`, etc.)
- **Audio**: `audio/audio.aac`
- **Logs**: `logs/pipeline.log`

### Common Errors

- `ffmpeg is not installed or not in PATH`: Make sure `ffmpeg` is properly installed on your system. Run `ffmpeg -version` in your terminal to verify.
- `Place your video at input/input.mp4`: The script cannot find the video file. Ensure the file path provided via `--video` exists.
- `Zero frames were extracted`: The input video might be corrupted, or ffmpeg could not decode it. Check the logs for ffmpeg errors.

## Object Selection

After extraction, you can manually select the object you want to remove from the first frame.

```bash
python scripts/select_object.py --frames_dir frames/ --mode box
```

Or for point selection:
```bash
python scripts/select_object.py --frames_dir frames/ --mode point
```

- **box mode** is better for selecting the full object manually.
- **point mode** will be useful later for SAM 2 to track objects automatically.
- Your selection is saved to `config/selection.json`.
- A preview is saved to `output/selection_preview.jpg`.
- A test mask is saved to `masks_raw/000001.jpg`.

## Object Tracking (SAM 2)

Once you have saved your selection, run the tracking step:

```bash
python run_pipeline.py --track
```

This will automatically:
1. Load the SAM 2 model.
2. Inject your selection box on the specific frame.
3. Track the object automatically through the entire video (both forward and backward).
4. Save exactly one black-and-white mask per frame to the `masks_raw/` directory.

> **Note**: Tracking requires the SAM 2 weights (`sam2_hiera_tiny.pt`). The script will automatically download them to the `models/` folder if they are not present.

## Inpainting

The final step is to erase the object using the generated masks.

```bash
python run_pipeline.py --inpaint
```

This will:
1. Slightly expand the masks so they perfectly cover the object edges.
2. Run OpenCV's inpainting algorithm frame-by-frame to remove the object.
3. Save the clean frames to `inpainted_frames/`.
4. Merge the final frames with the original audio into `output/final_video.mp4`.

**Congratulations! Your video is ready in the `output/` folder.**
