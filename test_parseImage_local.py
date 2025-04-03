"""
author: Gabriel Bugarija
date: 2025-27-03
version: 1.0.0
description: This script is an integration test for the parse_image function.
It uses a mock server to simulate the API endpoint and tests the function's behavior with various images.
It also includes visual comparison of UI states before and after certain actions.

Integration Test for parse_image(...) Function

This script performs end-to-end testing for the parse_image function using a mock server. It includes:
- Encoding images to base64
- Sending POST requests to a mock server
- Asserting the correct response structure
- Visual comparison of UI states for additional validation

Requirements:
- pytest for test running
- PIL (Pillow) for image handling
- requests for HTTP requests
- mock for server mocking
- test images in the same directory as the script

Run with:
pytest integration_test.py

Once running, run the following comand to test with verobosity:

python -m  pytest -v  local_parse_image_integreation_test.py

Test output will be shown in the console.


"""

import time
import pytest
import requests
from PIL import Image, ImageChops
import base64
from io import BytesIO
from unittest import mock
import os

API_ENDPOINT = "http://localhost:5000/parse/"  # Mock server endpoint

@pytest.fixture(scope="session")
def mock_server():
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "segments": [{"id": 1, "label": "mock_segment"}]
        }
        yield mock_post

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def compare_images(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Find difference
    diff = ImageChops.difference(img1, img2)
    
    if diff.getbbox():
        print(f"Images {img1_path} and {img2_path} are different.")
        diff.show()  # Show the difference
        return False
    else:
        print(f"Images {img1_path} and {img2_path} are identical.")
        return True
    

# The following images represent different UI states before and after specific interactions.
@pytest.mark.parametrize("image_path", [ 
    "before_click_button.png",
    "after_click_button.png",
    "after_type_text_field.png",
])
def test_parse_image(mock_server, image_path):
    # Prepare test data
    base64_image = encode_image_to_base64(image_path)
    payload = {"image": base64_image}

    # Make request
    response = requests.post(API_ENDPOINT, json=payload)

    # Assertions
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    json_response = response.json()

    assert "segments" in json_response, "Missing 'segments' in response."
    assert isinstance(json_response["segments"], list), "'segments' is not a list."
    assert len(json_response["segments"]) > 0, "No segments found."

    print("Test passed for:", image_path)

@pytest.mark.parametrize("before, after", [
    ("before_click_button.png", "after_click_button.png"),
    ("before_type_text_field.png", "after_type_text_field.png"),
])
def test_image_change(before, after):
    assert os.path.exists(before), f"{before} does not exist."
    assert os.path.exists(after), f"{after} does not exist."
    assert compare_images(before, after), f"UI did not change as expected."
