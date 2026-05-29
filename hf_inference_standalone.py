#!/usr/bin/env python3
"""Standalone Hugging Face inference client for the Dakota Qwen3.6 adapter."""

import os
import argparse
import json
import sys
from typing import Optional, Dict, Any
from huggingface_hub import InferenceClient, login
from huggingface_hub.utils import HfHubHTTPError

MODEL_ID = "HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO"
BASE_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

DEFAULT_SYSTEM_PROMPT = "Answer Dakota grammar tasks concisely. Return only the answer."


def format_chat_messages(system_prompt: str, user_message: str) -> list:
    """
    Format messages in chat format for Qwen models.
    
    IMPORTANT: This matches the exact format used during RL training.
    The verifiers framework (used during training) automatically uses the model's
    built-in chat template via tokenizer.apply_chat_template(). This function
    creates the same message structure that the chat template expects.
    
    The Qwen3 model has a built-in chat template that formats:
    - system messages
    - user messages  
    - assistant responses
    
    During training, the environment used message_type="chat" which tells verifiers
    to use this same template. So we use tokenizer.apply_chat_template() to ensure
    inference matches training exactly.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


class DakotaInferenceClient:
    """Client for running inference on Dakota Grammar RL model using HF infrastructure."""
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        base_model_id: str = BASE_MODEL_ID,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 120,
        allow_login: bool = True,
    ):
        """
        Initialize the inference client.
        
        Args:
            model_id: HuggingFace model ID
            base_model_id: Base model repo used for tokenizer chat templating
            endpoint_url: Optional Inference Endpoint URL (if using dedicated endpoints)
            token: Optional HF token (will use login if not provided)
            timeout: Client-side timeout in seconds
        """
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        
        # Get token from environment or use login (skip login for public endpoints)
        if token:
            self.token = token
        elif os.getenv("HF_TOKEN"):
            self.token = os.getenv("HF_TOKEN")
        elif os.getenv("HUGGINGFACE_TOKEN"):
            self.token = os.getenv("HUGGINGFACE_TOKEN")
        elif endpoint_url:
            # Public endpoint usage: no token and no interactive login.
            self.token = None
        elif not allow_login:
            self.token = None
        else:
            # Try to login interactively (Inference API path)
            print("No HF token found. Attempting to login...")
            try:
                login()
                self.token = None  # Will use cached token
            except Exception as e:
                print(f"Login failed: {e}")
                print("Please set HF_TOKEN environment variable or run: huggingface-cli login")
                raise
        
        # Initialize client
        if endpoint_url:
            # Using Inference Endpoint
            # Note: In newer huggingface_hub, model param accepts endpoint URLs too
            self.client = InferenceClient(
                model=endpoint_url,
                token=self.token,
                timeout=self.timeout,
            )
            self.mode = "endpoint"
        else:
            # Using Inference API
            # Note: Some models may not be available on Inference API
            # and may require Inference Endpoints instead
            try:
                self.client = InferenceClient(
                    model=model_id,
                    token=self.token,
                    timeout=self.timeout,
                )
                self.mode = "api"
            except Exception as e:
                # If model isn't available on Inference API, provide helpful error
                raise Exception(
                    f"Model '{model_id}' is not available on Hugging Face Inference API.\n"
                    f"This could mean:\n"
                    f"  1. The model needs to be enabled for Inference API (check model settings)\n"
                    f"  2. The model requires Inference Endpoints instead (use --endpoint-url)\n"
                    f"  3. Use the HuggingFace Space instead (see huggingface_space/)\n"
                    f"  4. Use local inference with scripts/inference/test_model_inference.py (requires GPU)\n"
                    f"\nOriginal error: {str(e)}"
                ) from e
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (uses default if not provided)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with response and metadata
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Format messages
        messages = format_chat_messages(system_prompt, prompt)
        
        # Prepare parameters
        parameters = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }
        
        try:
            if self.mode == "endpoint":
                # Endpoint handlers expose text-generation; build chat prompt via tokenizer and call text_generation.
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=self.token)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                response = self.client.text_generation(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    return_full_text=False,
                    **kwargs
                )
                generated_text = response
            else:
                # Inference API - use text generation with chat template
                # Note: Most models on Inference API use text_generation, not chat_completion
                # CRITICAL: This uses the model's built-in chat template, which is
                # the SAME template used during RL training by the verifiers framework.
                # The verifiers framework automatically calls apply_chat_template()
                # when message_type="chat" is set (as it was in DakotaGrammarEnv).
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=self.token)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                
                # Call Inference API
                try:
                    response = self.client.text_generation(
                        formatted_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        return_full_text=False,
                        **kwargs
                    )
                    generated_text = response
                except Exception as api_error:
                    # Re-raise with more context
                    raise Exception(
                        f"Inference API call failed: {str(api_error)}\n"
                        f"Model: {self.model_id}\n"
                        f"Note: The model may not be available on Inference API yet, "
                        f"or may require Inference Endpoints instead."
                    ) from api_error
            
            return {
                "response": generated_text.strip(),
                "prompt": prompt,
                "parameters": parameters,
                "mode": self.mode
            }
        
        except HfHubHTTPError as e:
            # Some hub exceptions lack a direct status_code attribute; pull from response when available.
            resp = getattr(e, "response", None)
            status_code = getattr(resp, "status_code", None)
            error_msg = f"HTTP Error: {getattr(e, 'message', str(e))}"
            if resp is not None:
                try:
                    error_detail = resp.json()
                    error_msg += f" | Details: {error_detail}"
                except Exception:
                    try:
                        error_msg += f" | Response: {resp.text[:200]}"
                    except Exception:
                        pass
            return {
                "error": error_msg,
                "status_code": status_code,
                "prompt": prompt
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "error": f"Generation failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": error_details,
                "prompt": prompt
            }


def main():
    """Main CLI interface."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run inference on Dakota Grammar RL model using HF infrastructure"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="Input prompt for the model (required unless --interactive is set)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=BASE_MODEL_ID,
        help=f"Base model ID for tokenization (default: {BASE_MODEL_ID})"
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Inference Endpoint URL (optional, uses Inference API if not provided)"
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip interactive login even if no HF_TOKEN is set (useful for public endpoints)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, uses login if not provided)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Client timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (optional)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error messages and tracebacks"
    )
    
    args = parser.parse_args()
    
    # Check prompt requirement
    if not args.interactive and not args.prompt:
        parser.error("--prompt is required unless --interactive is set")

    # Initialize client
    try:
        client = DakotaInferenceClient(
            model_id=args.model_id,
            base_model_id=args.base_model_id,
            endpoint_url=args.endpoint_url,
            token=args.token,
            timeout=args.timeout,
            allow_login=not args.no_login,
        )
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return 1
    
    if args.interactive:
        # Interactive mode
        print(f"Dakota Grammar RL Inference ({client.mode} mode)")
        print(f"Model: {args.model_id}")
        print("Enter prompts (type 'quit' to exit, 'clear' to clear history)")
        print("=" * 70)
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                
                result = client.generate(
                    prompt=prompt,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nResponse: {result['response']}")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        return 0
    
    # Single prompt mode
    result = client.generate(
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
        if "error" in result:
            return 1
    else:
        if "error" in result:
            print(f"Error: {result['error']}")
            if "error_type" in result:
                print(f"   Error Type: {result['error_type']}")
            if "traceback" in result and args.verbose:
                print(f"\nTraceback:\n{result['traceback']}")
            return 1
        else:
            print(result['response'])
    
    return 0


if __name__ == "__main__":
    exit(main())

