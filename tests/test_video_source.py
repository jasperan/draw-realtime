"""Tests for app/video_source.py - video discovery and reading logic."""

from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# get_available_videos
# ---------------------------------------------------------------------------

class TestGetAvailableVideos:
    """Tests for the get_available_videos() function."""

    def test_empty_directory(self, tmp_path):
        """Returns empty list when videos dir is empty."""
        from app.video_source import get_available_videos

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(tmp_path / "videos")
            result = get_available_videos()
            assert result == []

    def test_creates_directory_if_missing(self, tmp_path):
        """Creates the videos directory if it doesn't exist."""
        from app.video_source import get_available_videos

        videos_path = tmp_path / "nonexistent_videos"
        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_path)
            result = get_available_videos()
            assert result == []
            assert videos_path.exists()

    def test_ignores_non_video_files(self, tmp_path):
        """Only returns files with video extensions."""
        from app.video_source import get_available_videos

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "readme.txt").write_text("not a video")
        (videos_dir / "image.png").write_bytes(b'\x89PNG')
        (videos_dir / "data.json").write_text("{}")

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            # Even if cv2 can't open them, non-video extensions should be skipped
            result = get_available_videos()
            assert result == []

    def test_lists_video_files(self, tmp_path, create_dummy_video):
        """Returns metadata for valid video files."""
        from app.video_source import get_available_videos

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        create_dummy_video("clip.mp4", directory=videos_dir)

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            result = get_available_videos()
            assert len(result) == 1
            assert result[0]["name"] == "clip.mp4"
            assert "duration" in result[0]
            assert "fps" in result[0]


# ---------------------------------------------------------------------------
# get_video_path
# ---------------------------------------------------------------------------

class TestGetVideoPath:
    """Tests for the get_video_path() function."""

    def test_returns_path_for_existing_file(self, tmp_path, create_dummy_video):
        """Returns full path when video exists."""
        from app.video_source import get_video_path

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        create_dummy_video("test.mp4", directory=videos_dir)

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            path = get_video_path("test.mp4")
            assert path is not None
            assert path.endswith("test.mp4")

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when video doesn't exist."""
        from app.video_source import get_video_path

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            path = get_video_path("nonexistent.mp4")
            assert path is None

    def test_blocks_directory_traversal(self, tmp_path):
        """Rejects path traversal attempts."""
        from app.video_source import get_video_path

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            assert get_video_path("../../etc/passwd") is None
            assert get_video_path("../secret.mp4") is None

    def test_blocks_sibling_directory_prefix_escape(self, tmp_path):
        """A sibling directory with the same prefix is not inside videos_dir."""
        from app.video_source import get_video_path

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        sibling_dir = tmp_path / "videos_evil"
        sibling_dir.mkdir()
        escaped_video = sibling_dir / "secret.mp4"
        escaped_video.write_bytes(b"\x00" * 1024)

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            assert get_video_path(str(escaped_video)) is None

    def test_blocks_symlink_loop(self, tmp_path):
        """Invalid symlink loops are treated as missing videos, not server errors."""
        from app.video_source import get_video_path

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "loop_a.mp4").symlink_to("loop_b.mp4")
        (videos_dir / "loop_b.mp4").symlink_to("loop_a.mp4")

        with patch("app.video_source.config") as mock_config:
            mock_config.videos_dir = str(videos_dir)
            assert get_video_path("loop_a.mp4") is None


# ---------------------------------------------------------------------------
# VideoSource class
# ---------------------------------------------------------------------------

class TestVideoSource:
    """Tests for the VideoSource class."""

    def test_open_valid_video(self, create_dummy_video):
        """VideoSource can open a valid video file."""
        from app.video_source import VideoSource

        video_path = create_dummy_video("valid.mp4")
        vs = VideoSource(str(video_path))
        assert vs.fps > 0
        assert vs.frame_count > 0
        assert vs.width > 0
        assert vs.height > 0
        vs.close()

    def test_open_nonexistent_video_raises(self):
        """Raises FileNotFoundError for missing file."""
        from app.video_source import VideoSource

        with pytest.raises(FileNotFoundError):
            VideoSource("/nonexistent/path/video.mp4")

    def test_read_frame_returns_rgb(self, create_dummy_video):
        """read_frame returns an RGB numpy array."""
        from app.video_source import VideoSource

        video_path = create_dummy_video("rgb_test.mp4")
        vs = VideoSource(str(video_path))
        frame = vs.read_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        vs.close()

    def test_read_frame_returns_none_at_end(self, create_dummy_video):
        """read_frame returns None when video is exhausted."""
        from app.video_source import VideoSource

        video_path = create_dummy_video("short.mp4")
        vs = VideoSource(str(video_path))
        # Read all frames
        while vs.read_frame() is not None:
            pass
        assert vs.read_frame() is None
        vs.close()

    def test_reset_rewinds_video(self, create_dummy_video):
        """reset() allows reading frames again."""
        from app.video_source import VideoSource

        video_path = create_dummy_video("reset_test.mp4")
        vs = VideoSource(str(video_path))
        frame1 = vs.read_frame()
        assert frame1 is not None
        vs.reset()
        frame2 = vs.read_frame()
        assert frame2 is not None
        vs.close()

    def test_close_releases_resources(self, create_dummy_video):
        """close() sets cap to None."""
        from app.video_source import VideoSource

        video_path = create_dummy_video("close_test.mp4")
        vs = VideoSource(str(video_path))
        vs.close()
        assert vs.cap is None
