"""Tests for app/video_processor.py - job management and processing logic."""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_status_values(self):
        from app.video_processor import JobStatus

        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# ProcessingJob
# ---------------------------------------------------------------------------

class TestProcessingJob:
    def test_to_dict(self):
        from app.video_processor import ProcessingJob, JobStatus

        job = ProcessingJob(
            job_id="abc123",
            input_path="/videos/in.mp4",
            output_path="/outputs/out.mp4",
            prompt="test prompt",
            model="sd-turbo",
            total_frames=100,
        )
        d = job.to_dict()
        assert d["job_id"] == "abc123"
        assert d["status"] == "pending"
        assert d["total_frames"] == 100
        assert d["input_path"] == "in.mp4"
        assert d["output_filename"] == "out.mp4"

    def test_default_values(self):
        from app.video_processor import ProcessingJob, JobStatus

        job = ProcessingJob(
            job_id="x",
            input_path="in.mp4",
            output_path="out.mp4",
            prompt="p",
            model="m",
        )
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.error is None
        assert job.created_at > 0


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------

class TestVideoProcessor:
    @pytest.fixture
    def processor(self, tmp_path):
        """Create a VideoProcessor with temp directories."""
        from app.video_processor import VideoProcessor

        with patch("app.video_processor.config") as mock_config:
            mock_config.outputs_dir = str(tmp_path / "outputs")
            p = VideoProcessor()
            # Override directories to use tmp_path
            p.outputs_dir = tmp_path / "outputs"
            p.uploads_dir = tmp_path / "uploads"
            p.preview_dir = tmp_path / "previews"
            p.outputs_dir.mkdir(exist_ok=True)
            p.uploads_dir.mkdir(exist_ok=True)
            p.preview_dir.mkdir(exist_ok=True)
            return p

    def test_create_job(self, processor, create_dummy_video):
        """create_job creates a valid ProcessingJob."""
        video = create_dummy_video("input.mp4")

        job = processor.create_job(
            input_path=str(video),
            prompt="test",
            model="sd-turbo",
        )
        assert job.job_id in processor.jobs
        assert job.total_frames > 0
        assert job.prompt == "test"
        assert job.model == "sd-turbo"

    def test_create_job_invalid_video(self, processor, tmp_path):
        """create_job raises ValueError for invalid video."""
        bad_path = tmp_path / "bad.mp4"
        bad_path.write_bytes(b"not a video")

        with pytest.raises(ValueError, match="Cannot open video"):
            processor.create_job(
                input_path=str(bad_path),
                prompt="test",
                model="sd-turbo",
            )

    def test_get_job(self, processor, create_dummy_video):
        """get_job returns existing job or None."""
        video = create_dummy_video("input.mp4")
        job = processor.create_job(str(video), "test", "sd-turbo")

        assert processor.get_job(job.job_id) is job
        assert processor.get_job("nonexistent") is None

    def test_get_all_jobs(self, processor, create_dummy_video):
        """get_all_jobs returns list of dicts."""
        video = create_dummy_video("input.mp4")
        processor.create_job(str(video), "test1", "sd-turbo")
        processor.create_job(str(video), "test2", "sd-turbo")

        jobs = processor.get_all_jobs()
        assert len(jobs) == 2
        assert all(isinstance(j, dict) for j in jobs)

    def test_cleanup_old_jobs(self, processor, create_dummy_video):
        """cleanup_old_jobs removes old completed/failed jobs."""
        from app.video_processor import JobStatus

        video = create_dummy_video("input.mp4")
        job = processor.create_job(str(video), "old", "sd-turbo")

        # Simulate completed job
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time() - 25 * 3600  # 25 hours ago

        processor.cleanup_old_jobs(max_age_hours=24)
        assert processor.get_job(job.job_id) is None

    def test_cleanup_keeps_recent_jobs(self, processor, create_dummy_video):
        """cleanup_old_jobs keeps recent completed jobs."""
        from app.video_processor import JobStatus

        video = create_dummy_video("input.mp4")
        job = processor.create_job(str(video), "recent", "sd-turbo")

        job.status = JobStatus.COMPLETED
        job.completed_at = time.time() - 1 * 3600  # 1 hour ago

        processor.cleanup_old_jobs(max_age_hours=24)
        assert processor.get_job(job.job_id) is not None

    def test_cleanup_ignores_processing_jobs(self, processor, create_dummy_video):
        """cleanup_old_jobs doesn't remove in-progress jobs."""
        from app.video_processor import JobStatus

        video = create_dummy_video("input.mp4")
        job = processor.create_job(str(video), "processing", "sd-turbo")
        job.status = JobStatus.PROCESSING

        processor.cleanup_old_jobs(max_age_hours=0)
        assert processor.get_job(job.job_id) is not None


# ---------------------------------------------------------------------------
# get_processor singleton
# ---------------------------------------------------------------------------

def test_get_processor_singleton():
    """get_processor returns the same instance."""
    from app.video_processor import get_processor
    import app.video_processor as vp

    # Reset global state
    vp._processor = None

    with patch("app.video_processor.config") as mock_config:
        mock_config.outputs_dir = "/tmp/test_outputs"
        p1 = get_processor()
        p2 = get_processor()
        assert p1 is p2

    # Clean up
    vp._processor = None
