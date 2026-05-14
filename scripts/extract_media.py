import os
import json
import logging
from .utils import run_shell_command

logger = logging.getLogger(__name__)

def get_video_metadata(video_path):
    """Get video metadata using ffprobe."""
    if not os.path.exists(video_path):
        return None
        
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-of", "json",
        video_path
    ]
    
    stdout, stderr, returncode = run_shell_command(command, check=False)
    if returncode == 0:
        try:
            data = json.loads(stdout)
            stream = data.get("streams", [{}])[0]
            
            # calculate fps from r_frame_rate (e.g. "30000/1001" or "30/1")
            fps_str = stream.get("r_frame_rate", "0/1")
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
            
            metadata = {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": float(stream.get("duration", 0)),
                "total_frames": int(stream.get("nb_frames", 0)) if stream.get("nb_frames") else None
            }
            logger.info("Successfully extracted video metadata.")
            return metadata
        except Exception as e:
            logger.error(f"Error parsing metadata: {e}")
            return None
    else:
        logger.error(f"ffprobe failed: {stderr}")
        return None

def extract_frames(video_path, output_dir):
    """Extract frames from input video using ffmpeg."""
    if not os.path.exists(video_path):
        logger.error(f"Video not found at {video_path}")
        return False
        
    logger.info(f"Extracting frames to {output_dir}...")
    output_pattern = os.path.join(output_dir, "%06d.jpg")
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-qscale:v", "2",
        output_pattern,
        "-y"
    ]
    
    _, stderr, returncode = run_shell_command(command, check=False)
    if returncode == 0:
        logger.info("Frame extraction command complete.")
        
        # Frame count validation
        extracted_frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        num_frames = len(extracted_frames)
        logger.info(f"Total extracted frames: {num_frames}")
        
        if num_frames == 0:
            logger.error("Zero frames were extracted. Please check the video file.")
            return False
            
        return True
    else:
        logger.error(f"Frame extraction failed: {stderr}")
        return False

def extract_audio(video_path, output_path):
    """Extract audio from input video using ffmpeg."""
    if not os.path.exists(video_path):
        logger.error(f"Video not found at {video_path}")
        return False
        
    logger.info(f"Extracting audio to {output_path}...")
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        output_path,
        "-y"
    ]
    
    _, stderr, returncode = run_shell_command(command, check=False)
    if returncode == 0:
        logger.info("Audio extraction complete.")
        return True
    else:
        logger.warning("No audio stream found or audio extraction failed. Continuing without audio.")
        return False
