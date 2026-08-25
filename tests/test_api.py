import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_endpoint_missing_file():
    # Verify validation error when no file is provided
    response = client.post("/analyze")
    assert response.status_code == 422

def test_analyze_endpoint_mock_audio():
    # Create a tiny dummy wav/audio byte stream for structural verification
    dummy_audio = bRIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00
    response = client.post(
        "/analyze",
        files={"file": ("test.wav", dummy_audio, "audio/wav")},
        data={"contact_id": "test-uuid-123"}
    )
    # Check that the API contract schema matches successfully
    assert response.status_code == 200
    data = response.json()
    assert data["contact_id"] == "test-uuid-123"
    assert "gender" in data
    assert "age_bracket" in data
    assert "audio_quality" in data
    assert "processing_ms" in data