import os
import subprocess
import shutil
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_file, debug_mode=False):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_folders(base_dir, folders):
    """Create a list of folders under the given base directory."""
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        logger.debug(f"Ensured folder exists: {folder_path}")

def run_shell_command(command, check=True):
    """Run a shell command safely and return its output."""
    logger.debug(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=check, text=True, capture_output=True)
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e.stderr}")
        if check:
            raise
        return e.stdout, e.stderr, e.returncode
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        if check:
            raise
        return "", f"Command not found: {command[0]}", 1

def check_ffmpeg_installed():
    """Check if ffmpeg and ffprobe are installed and available in the system PATH."""
    ffmpeg_exists = shutil.which("ffmpeg") is not None
    ffprobe_exists = shutil.which("ffprobe") is not None
    
    if ffmpeg_exists:
        logger.info("ffmpeg check result: Installed")
    else:
        logger.error("ffmpeg check result: Not installed or not in PATH.")
        
    if ffprobe_exists:
        logger.info("ffprobe check result: Installed")
    else:
        logger.error("ffprobe check result: Not installed or not in PATH.")
        
    return ffmpeg_exists and ffprobe_exists

def merge_audio_video(frames_dir, audio_path, output_path, fps=30):
    """Merge frames and audio into final video using ffmpeg."""
    logger.info("Merging audio and frames into final video...")
    
    input_pattern = os.path.join(frames_dir, "%06d.jpg")
    has_audio = os.path.exists(audio_path)
    
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", input_pattern
    ]
    
    if has_audio:
        command.extend(["-i", audio_path])
        command.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?"])
        
    command.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
        "-y"
    ])
    
    _, stderr, returncode = run_shell_command(command, check=False)
    if returncode == 0:
        logger.info(f"Successfully saved final video to {output_path}")
        return True
    else:
        logger.error(f"FFmpeg merge failed: {stderr}")
        return False

def clean_temp_directories(base_dir):
    """Deletes temporary extracted frames and masks to save disk space."""
    import glob
    logger.info("Cleaning up temporary directories to save space...")
    
    folders_to_clean = ["frames", "masks_raw", "masks_clean", "inpainted_frames"]
    
    for folder in folders_to_clean:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, "*"))
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Could not remove {f}: {e}")
            logger.info(f"Cleaned {folder_path}")
            
    logger.info("Cleanup complete.")
