"""Video source handling for server-side MP4 files."""

import os
import cv2
import time
from typing import List, Optional, Generator
from pathlib import Path
import numpy as np

from app.config import config


class VideoSource:
    """Handles reading frames from video files."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self._open()

    def _open(self):
        """Open the video file."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns None at end of video."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def frames(self, loop: bool = True) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames at the video's FPS."""
        frame_interval = 1.0 / self.fps

        while True:
            start_time = time.time()

            frame = self.read_frame()
            if frame is None:
                if loop:
                    self.reset()
                    continue
                else:
                    break

            yield frame

            # Maintain FPS timing
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def reset(self):
        """Reset video to beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """Close the video file."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


def get_available_videos() -> List[dict]:
    """Get list of available video files in the videos directory."""
    videos_dir = Path(config.videos_dir)

    if not videos_dir.exists():
        videos_dir.mkdir(parents=True, exist_ok=True)
        return []

    videos = []
    extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    for f in sorted(videos_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            # Get video info
            try:
                cap = cv2.VideoCapture(str(f))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frames / fps if fps > 0 else 0
                    cap.release()

                    videos.append({
                        "name": f.name,
                        "path": str(f),
                        "duration": round(duration, 1),
                        "fps": round(fps, 1),
                    })
            except Exception:
                pass

    return videos


def get_video_path(filename: str) -> Optional[str]:
    """Get full path for a video file, validating it exists."""
    videos_dir = Path(config.videos_dir)
    video_path = videos_dir / filename

    # Security: ensure path is within videos directory
    try:
        video_path = video_path.resolve()
        videos_dir = videos_dir.resolve()
        if not str(video_path).startswith(str(videos_dir)):
            return None
    except Exception:
        return None

    if video_path.exists() and video_path.is_file():
        return str(video_path)

    return None
