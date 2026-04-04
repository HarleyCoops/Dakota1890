"""Deploy a dedicated Hugging Face inference endpoint for the Dakota adapter."""

import os
import time
import argparse
from huggingface_hub import create_inference_endpoint, get_inference_endpoint, get_token

MODEL_ID = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890"
ENDPOINT_NAME = "dakota-qwen3-30b-inference"

def deploy_endpoint(framework="pytorch", accelerator="gpu", instance_size="large", vendor="aws", region="us-east-1"):
    """
    Deploy a dedicated inference endpoint for the Dakota model.
    
    Note: "large" is a placeholder. For 30B, we specifically need high VRAM.
    The SDK maps 'large', 'medium', 'xlarge' to specific GPU types depending on the cloud provider.
    For 30B, we typically need 'x2' or 'x4' A10Gs or a single A100.
    """
    print(f"Deploying Dedicated Endpoint for {MODEL_ID}...")
    
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("Please login using 'huggingface-cli login' first.")

    try:
        # Check if exists first
        try:
            endpoint = get_inference_endpoint(ENDPOINT_NAME, token=token)
            print(f"Endpoint '{ENDPOINT_NAME}' already exists with status: {endpoint.status}")
            if endpoint.status == "paused":
                print("Resuming existing endpoint...")
                endpoint.resume()
            return endpoint.url
        except:
            pass # Does not exist, create new

        # Create Endpoint
        # We request a specific instance type for 30B models.
        # A100 is often safest, or 4x A10G.
        # Using "A100" if available or falling back to high-memory config.
        
        print("Requesting resources (this may take a few minutes)...")
        
        # NOTE: Valid instance_size values usually: 'small', 'medium', 'large', 'xlarge', '2xlarge', '4xlarge'
        # For 30B:
        # - '2xlarge' is often 1x A100 or 4x A10G on AWS.
        
        endpoint = create_inference_endpoint(
            name=ENDPOINT_NAME,
            repository=MODEL_ID,
            framework=framework,
            task="text-generation",
            accelerator=accelerator,
            vendor=vendor,
            region=region,
            type="protected", # token-based access
            instance_size="2xlarge", # Aiming for A100/multi-GPU
            instance_type="nvidia-a100", # Explicitly requesting A100 if supported by SDK version
            namespace="HarleyCooper",
            token=token
        )
        
        print(f"Endpoint created: {endpoint.name}")
        print(f"URL: {endpoint.url}")
        print("Waiting for initialization (typically 5-10 mins)...")
        
        endpoint.wait(timeout=900) # Wait up to 15 mins
        
        if endpoint.status == "running":
            print(f"\nSUCCESS! Endpoint is live.")
            print(f"Use this URL: {endpoint.url}")
            print("\nExample usage:")
            print(f'python hf_inference_standalone.py --endpoint-url "{endpoint.url}" --interactive')
            return endpoint.url
        else:
            print(f"\nEndpoint status is '{endpoint.status}'. Check HF Dashboard logs.")
            return None

    except Exception as e:
        print(f"\nDeployment failed: {e}")
        print("Tip: You may need to request quota increase for A100 GPUs via HF settings.")

if __name__ == "__main__":
    deploy_endpoint()

