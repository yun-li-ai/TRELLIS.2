# Modifications Copyright (c) Applied Intuition, Inc.


import requests
import uuid

HOST_PORT = 8000

def call_create_3d_model_service(body):
    url = f"http://localhost:{HOST_PORT}/create-3d-model-from-paths"

    response = requests.post(url, json=body)

    # Check the response
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Failed to call service:", response.status_code, response.text)


# Run the function
if __name__ == "__main__":
    body = {
        "image_paths": [
            # Note, those should be the path in the container, if you want to test with your own image you will need to
            # 1. Manually copy some test image into the container
            # 2. Change the path to the path in the container
            # 3. Run   `python request_test.py` from the host machine

            # The following is an existing image from the TRELLIS2 codebase so you do not need to copy it. 
            "/app/assets/example_image/T.png"
        ],
        "job_id": str(uuid.uuid4())[:26],
        "seed": 42,
        "randomize_seed": False,
        "ss_guidance_strength": 7.5,
        "ss_sampling_steps": 12,
        "slat_guidance_strength": 3.0,
        "slat_sampling_steps": 12,
    }

    call_create_3d_model_service(body)
