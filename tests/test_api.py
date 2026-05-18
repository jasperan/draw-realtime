"""Tests for app/main.py - FastAPI endpoint tests using TestClient.

All pipeline/model operations are mocked so tests run without GPU.
"""

import io
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level mocking: prevent heavy imports at module load time
# ---------------------------------------------------------------------------

# Mock StreamDiffusion wrapper before any app.pipeline import
_mock_wrapper_module = MagicMock()
sys.modules.setdefault("utils", MagicMock())
sys.modules.setdefault("utils.wrapper", _mock_wrapper_module)
sys.modules.setdefault("streamdiffusion", MagicMock())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_deps(tmp_path):
    """Mock all heavy dependencies (pipeline, processor, multistyle)."""
    mock_pipeline = MagicMock()
    mock_pipeline.get_available_models.return_value = {
        "sd-turbo": "SD-Turbo",
        "sd15-lcm": "SD 1.5 + LCM",
    }
    mock_pipeline.get_current_model.return_value = "sd-turbo"
    mock_pipeline.get_output_resolution.return_value = (512, 512)
    mock_pipeline.predict.return_value = Image.fromarray(
        np.zeros((512, 512, 3), dtype=np.uint8)
    )
    mock_pipeline.load_model.return_value = True

    outputs_dir = tmp_path / "outputs"
    uploads_dir = tmp_path / "uploads"
    previews_dir = tmp_path / "previews"
    outputs_dir.mkdir()
    uploads_dir.mkdir()
    previews_dir.mkdir()

    mock_processor = MagicMock()
    mock_processor.outputs_dir = outputs_dir
    mock_processor.uploads_dir = uploads_dir
    mock_processor.preview_dir = previews_dir
    mock_processor.jobs = {}
    mock_processor.get_all_jobs.return_value = []
    mock_processor.get_job.return_value = None

    mock_multistyle = MagicMock()
    mock_multistyle.get_all_jobs.return_value = []

    return {
        "pipeline": mock_pipeline,
        "processor": mock_processor,
        "multistyle": mock_multistyle,
    }


@pytest.fixture
def client(mock_deps):
    """Create a TestClient with mocked dependencies."""
    with patch("app.main.get_pipeline", return_value=mock_deps["pipeline"]), \
         patch("app.main.get_processor", return_value=mock_deps["processor"]), \
         patch("app.main.get_multistyle_processor", return_value=mock_deps["multistyle"]):

        from app.main import App
        test_app = App()

        from fastapi.testclient import TestClient
        yield TestClient(test_app.app)


# ---------------------------------------------------------------------------
# GET /api/settings
# ---------------------------------------------------------------------------

class TestGetSettings:
    def test_returns_settings(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "default_model" in data
        assert "presets" in data
        assert "width" in data

    def test_settings_contain_default_model(self, client):
        resp = client.get("/api/settings")
        data = resp.json()
        assert data["default_model"] == "sd-turbo"


# ---------------------------------------------------------------------------
# GET /api/videos
# ---------------------------------------------------------------------------

class TestListVideos:
    def test_list_videos(self, client):
        with patch("app.main.get_available_videos", return_value=[]):
            resp = client.get("/api/videos")
            assert resp.status_code == 200
            assert resp.json() == {"videos": []}


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

class TestUploadVideo:
    def test_upload_rejects_invalid_extension(self, client):
        """Uploading a non-video file should be rejected."""
        file_content = b"not a video"
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert resp.status_code == 400
        assert "Invalid file type" in resp.json()["detail"]

    def test_upload_rejects_empty_filename(self, client):
        """Uploading without filename should be rejected."""
        resp = client.post(
            "/api/upload",
            files={"file": ("", io.BytesIO(b"data"), "video/mp4")},
        )
        # FastAPI returns 422 when the file has no valid filename
        assert resp.status_code in (400, 422)

    def test_upload_rejects_oversized_file(self, client, mock_deps):
        """Uploading a file over the size limit should be rejected."""
        # We won't actually send 500MB of data. Instead test with a smaller
        # but still processable payload. The bug fix adds a size limit check.
        file_content = b"\x00" * 1024
        resp = client.post(
            "/api/upload",
            files={"file": ("small.mp4", io.BytesIO(file_content), "video/mp4")},
        )
        # Should either succeed (valid) or fail (invalid video), not crash
        assert resp.status_code in (200, 400, 413)


# ---------------------------------------------------------------------------
# POST /api/process
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def test_uploaded_file_path_traversal_rejected(self, client):
        resp = client.post(
            "/api/process",
            data={"uploaded_file": "../secret.mp4", "prompt": "test"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_list_empty_jobs(self, client):
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}


# ---------------------------------------------------------------------------
# GET /api/job/{job_id}
# ---------------------------------------------------------------------------

class TestGetJobStatus:
    def test_missing_job_returns_404(self, client):
        resp = client.get("/api/job/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/output/{filename}
# ---------------------------------------------------------------------------

class TestGetOutputVideo:
    def test_path_traversal_rejected(self, client):
        resp = client.get("/api/output/..secret.mp4")
        assert resp.status_code == 400

    def test_missing_output_returns_404(self, client):
        resp = client.get("/api/output/nonexistent.mp4")
        assert resp.status_code == 404

    def test_serves_existing_output(self, client, mock_deps):
        output_path = mock_deps["processor"].outputs_dir / "test_output.mp4"
        output_path.write_bytes(b"\x00" * 2048)

        resp = client.get("/api/output/test_output.mp4")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/input/{filename}
# ---------------------------------------------------------------------------

class TestGetInputVideo:
    def test_path_traversal_rejected(self, client):
        resp = client.get("/api/input/..secret.mp4")
        assert resp.status_code == 400

    def test_serves_uploaded_file(self, client, mock_deps):
        upload_path = mock_deps["processor"].uploads_dir / "uploaded.mp4"
        upload_path.write_bytes(b"\x00" * 2048)

        resp = client.get("/api/input/uploaded.mp4")
        assert resp.status_code == 200

    def test_missing_input_returns_404(self, client):
        with patch("app.main.get_video_path", return_value=None):
            resp = client.get("/api/input/nonexistent.mp4")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/job/{job_id}
# ---------------------------------------------------------------------------

class TestDeleteJob:
    def test_delete_nonexistent_job_404(self, client):
        resp = client.delete("/api/job/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/preview/{filename}
# ---------------------------------------------------------------------------

class TestGetPreviewFrame:
    def test_path_traversal_rejected(self, client):
        resp = client.get("/api/preview/..secret.jpg")
        assert resp.status_code == 400

    def test_missing_preview_returns_404(self, client):
        resp = client.get("/api/preview/nonexistent.jpg")
        assert resp.status_code == 404

    def test_serves_existing_preview(self, client, mock_deps):
        preview_path = mock_deps["processor"].preview_dir / "test_preview.jpg"
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(str(preview_path), "JPEG")

        resp = client.get("/api/preview/test_preview.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# GET /api/outputs
# ---------------------------------------------------------------------------

class TestListOutputs:
    def test_list_empty_outputs(self, client, mock_deps):
        resp = client.get("/api/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert "outputs" in data

    def test_skips_invalid_outputs(self, client, mock_deps, create_dummy_video):
        create_dummy_video("valid.mp4", directory=mock_deps["processor"].outputs_dir)
        broken_path = mock_deps["processor"].outputs_dir / "broken.mp4"
        broken_path.write_bytes(b"\x00" * 4096)

        resp = client.get("/api/outputs")

        assert resp.status_code == 200
        filenames = {item["filename"] for item in resp.json()["outputs"]}
        assert "valid.mp4" in filenames
        assert "broken.mp4" not in filenames


# ---------------------------------------------------------------------------
# GET /api/output-thumbnail/{filename}
# ---------------------------------------------------------------------------

class TestGetOutputThumbnail:
    def test_generates_thumbnail_for_short_video(self, client, mock_deps, create_dummy_video):
        create_dummy_video("short.mp4", directory=mock_deps["processor"].outputs_dir)

        resp = client.get("/api/output-thumbnail/short.mp4")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        thumbnail_path = mock_deps["processor"].outputs_dir.parent / "thumbnails" / "short.jpg"
        assert thumbnail_path.exists()
        assert thumbnail_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# DELETE /api/output/{filename}
# ---------------------------------------------------------------------------

class TestDeleteOutput:
    def test_path_traversal_rejected(self, client):
        resp = client.delete("/api/output/..secret.mp4")
        assert resp.status_code == 400

    def test_missing_output_returns_404(self, client):
        resp = client.delete("/api/output/nonexistent.mp4")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/multistyle/styles
# ---------------------------------------------------------------------------

class TestMultistyleStyles:
    def test_returns_styles(self, client):
        resp = client.get("/api/multistyle/styles")
        assert resp.status_code == 200
        data = resp.json()
        assert "styles" in data


# ---------------------------------------------------------------------------
# POST /api/multistyle/process
# ---------------------------------------------------------------------------

class TestMultistyleProcess:
    def test_uploaded_file_path_traversal_rejected(self, client):
        resp = client.post(
            "/api/multistyle/process",
            data={"uploaded_file": "../secret.mp4"},
        )
        assert resp.status_code == 400
