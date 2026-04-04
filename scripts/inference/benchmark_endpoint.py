import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

# Ensure UTF-8 stdout for Dakota characters on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add project root to path to import hf_inference_standalone
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    from hf_inference_standalone import DakotaInferenceClient
except ImportError:
    print("Error: Could not import DakotaInferenceClient from hf_inference_standalone.py")
    print(f"Ensure hf_inference_standalone.py is in {project_root}")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Please install it:")
    print("pip install datasets")
    sys.exit(1)

# Default endpoint from your successful test
DEFAULT_ENDPOINT = "https://sodh94mt9pt8xuzf.us-east4.gcp.endpoints.huggingface.cloud"

def main():
    parser = argparse.ArgumentParser(description="Benchmark Dakota model endpoint against QA dataset")
    parser.add_argument("--samples", type=int, default=1, help="Number of random samples to test")
    parser.add_argument("--endpoint-url", type=str, default=DEFAULT_ENDPOINT, help="Inference Endpoint URL")
    parser.add_argument(
        "--output",
        type=str,
        default="eval/benchmarks/benchmark_results.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument("--dataset", type=str, default="HarleyCooper/dakota-bilingual-qa", help="Dataset ID")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to sample from (train/validation)")
    parser.add_argument("--timeout", type=int, default=900, help="Client timeout in seconds")
    
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.split})...")
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Random sampling
    total_rows = len(ds)
    num_samples = min(args.samples, total_rows)
    print(f"Selecting {num_samples} random samples from {total_rows} total rows...")
    
    # Create random indices
    indices = random.sample(range(total_rows), num_samples)
    samples = ds.select(indices)

    # Initialize client
    print(f"Initializing client for endpoint: {args.endpoint_url}")
    print(f"Timeout set to: {args.timeout}s")
    client = DakotaInferenceClient(endpoint_url=args.endpoint_url, timeout=args.timeout)

    results = []
    print("\nStarting inference loop...")
    print("=" * 60)

    for i, item in enumerate(samples):
        question = item['question']
        expected_answer = item['answer']
        source_lang = item.get('source_language', 'unknown')
        
        print(f"[{i+1}/{num_samples}] Q: {question[:60]}...")
        
        try:
            # Generate response
            output = client.generate(
                prompt=question,
                max_new_tokens=96,  # keep responses shorter to avoid timeouts
                temperature=0.3,
                top_p=0.9
            )
            
            if "error" in output:
                model_response = f"ERROR: {output['error']}"
                print(f"  -> Failed: {output['error']}")
            else:
                model_response = output['response']
                print(f"  -> Generated: {len(model_response)} chars")

            # Record result
            result_entry = {
                "pair_id": item.get('pair_id'),
                "source_language": source_lang,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": model_response,
                "endpoint": args.endpoint_url
            }
            results.append(result_entry)

            # Write incrementally to save progress
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  -> Exception: {e}")

    print("\n" + "=" * 60)
    print(f"Benchmark complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
