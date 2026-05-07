from fastapi.testclient import TestClient
import io
import json

from main import app 

client = TestClient(app)

def test_upload_invalid_file_type():
    """Test that the API rejects non-CSV/JSON files."""
    # create a dummy text file in memory
    file_content = b"This is a random text file."
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/api/v1/compliance/profile-dataset/", files=files)
    
    assert response.status_code == 415
    assert "Unsupported file format" in response.json()["detail"] or "Unsupported media type" in response.json()["detail"]

def test_profile_valid_json_upload():
    """Test that uploading a valid JSON dataset triggers profiling correctly."""
    mock_dataset = [
        {"transaction_id": "1", "amount": 100, "currency": "USD"},
        {"transaction_id": "2", "amount": 200, "currency": "EUR"}
    ]
    file_content = json.dumps(mock_dataset).encode("utf-8")
    files = {"file": ("data.json", io.BytesIO(file_content), "application/json")}
    
    response = client.post("/api/v1/compliance/profile-dataset/", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "inferred_schema" in data
    assert "amount" in data["inferred_schema"]
    assert data["inferred_schema"]["amount"]["type"] in ["integer", "float"]

def test_profile_valid_csv_upload():
    """Test that uploading a valid CSV dataset works."""
    csv_content = b"transaction_id,amount,currency\n1,100.50,USD\n2,200.00,EUR\n"
    files = {"file": ("data.csv", io.BytesIO(csv_content), "text/csv")}
    
    response = client.post("/api/v1/compliance/profile-dataset/", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "inferred_schema" in data
    assert "amount" in data["inferred_schema"]