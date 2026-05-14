import os
import sys
import argparse
import traceback
import logging
import platform
import shutil

from scripts.utils import create_folders, check_ffmpeg_installed, setup_logging
from scripts.extract_media import get_video_metadata, extract_frames, extract_audio

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check all system dependencies and print results."""
    print("="*50)
    print("Dependency Check")
    print("="*50)
    
    # Python version
    print(f"Python version: {platform.python_version()}")
    
    # ffmpeg and ffprobe
    ffmpeg_exists = shutil.which("ffmpeg") is not None
    ffprobe_exists = shutil.which("ffprobe") is not None
    print(f"ffmpeg installed: {ffmpeg_exists}")
    print(f"ffprobe installed: {ffprobe_exists}")
    
    # Check imports
    imports_to_check = {
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "PyYAML": "yaml",
        "tqdm": "tqdm"
    }
    
    for name, module in imports_to_check.items():
        try:
            __import__(module)
            print(f"{name} import: OK")
        except ImportError:
            print(f"{name} import: FAILED")
            
    # Check torch/cuda
    try:
        import torch
        print(f"PyTorch installed: OK (Version: {torch.__version__})")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed yet. This is okay for extraction stage.")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Video Object Removal Pipeline")
    parser.add_argument("--video", help="Path to input video file (e.g., input/input.mp4)")
    parser.add_argument("--check", action="store_true", help="Run dependency check and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with full tracebacks")
    parser.add_argument("--track", action="store_true", help="Run SAM 2 tracking step")
    parser.add_argument("--inpaint", action="store_true", help="Run OpenCV inpainting step")
    parser.add_argument("--cleanup", action="store_true", help="Clean up intermediate frames after inpainting")
    
    args = parser.parse_args()

    if args.check:
        check_dependencies()
        return

    if args.track:
        print("="*50)
        print("Starting SAM 2 tracking phase...")
        print("="*50)
        import subprocess
        result = subprocess.run([sys.executable, "scripts/track_object.py"])
        if result.returncode != 0:
            print("Tracking phase failed.")
        return

    if args.inpaint:
        print("="*50)
        print("Starting OpenCV inpainting phase...")
        print("="*50)
        import subprocess
        result = subprocess.run([sys.executable, "scripts/inpaint_video.py"])
        if result.returncode != 0:
            print("Inpainting phase failed.")
            return
            
        print("Merging into final video...")
        from scripts.utils import merge_audio_video
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        frames_dir = os.path.join(base_dir, "inpainted_frames")
        audio_path = os.path.join(base_dir, "audio", "audio.aac")
        output_path = os.path.join(base_dir, "output", "final_video.mp4")
        success = merge_audio_video(frames_dir, audio_path, output_path, fps=30)
        
        if success and args.cleanup:
            from scripts.utils import clean_temp_directories
            clean_temp_directories(base_dir)
            
        return

    if not args.video:
        parser.print_help()
        print("\nError: --video, --track, or --inpaint argument is required unless running --check.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "pipeline.log")
    setup_logging(log_file, debug_mode=args.debug)
    
    try:
        logger.info("="*50)
        logger.info("Started pipeline")
        logger.info(f"Video path: {args.video}")
        logger.info("="*50)
        
        video_path = args.video
        
        if not os.path.exists(video_path):
            logger.error("Place your video at input/input.mp4 or pass --video path/to/video.mp4")
            return

        logger.info("Setting up directory structure...")
        folders = [
            "input",
            "frames",
            "masks_raw",
            "masks_clean",
            "inpainted_frames",
            "audio",
            "output",
            "models",
            "scripts",
            "config",
            "logs"
        ]
        create_folders(base_dir, folders)
        
        # Clean up old temporary files before starting a new extraction
        from scripts.utils import clean_temp_directories
        clean_temp_directories(base_dir)
        
        logger.info("Checking system dependencies...")
        if not check_ffmpeg_installed():
            logger.error("Please install ffmpeg and ensure it is in your system PATH.")
            return
            
        logger.info("Extracting video metadata...")
        metadata = get_video_metadata(video_path)
        if metadata:
            logger.info("Video Metadata:")
            logger.info(f"  FPS: {metadata['fps']:.2f}")
            logger.info(f"  Width: {metadata['width']}")
            logger.info(f"  Height: {metadata['height']}")
            logger.info(f"  Duration: {metadata['duration']:.2f} seconds")
            if metadata.get('total_frames'):
                logger.info(f"  Total Frames (estimated): {metadata['total_frames']}")
        else:
            logger.warning("Could not extract video metadata.")
            
        frames_dir = os.path.join(base_dir, "frames")
        success = extract_frames(video_path, frames_dir)
        if not success:
            logger.error("Pipeline stopped due to frame extraction failure.")
            return
        
        audio_path = os.path.join(base_dir, "audio", "audio.aac")
        extract_audio(video_path, audio_path)
        
        logger.info("="*50)
        logger.info("Base extraction pipeline finished successfully.")
        logger.info("="*50)

    except Exception as e:
        if args.debug:
            logger.error(f"Pipeline crashed:\n{traceback.format_exc()}")
        else:
            logger.error(f"Pipeline error: {str(e)}")
            logger.error("Run with --debug for full traceback.")

if __name__ == "__main__":
    main()
