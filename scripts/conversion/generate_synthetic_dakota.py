"""Generate Dakota synthetic Q&A pairs from extracted dictionary entries via Gemini."""

import argparse
import json
import logging
from typing import Dict, Generator, List
from pathlib import Path
from datetime import datetime
import os
import re
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

# Set up our logging system to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.5-flash')
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
DEFAULT_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "medium")


class DakotaQAGenerator:
    def __init__(
        self,
        extracted_dict_dir: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_level: str | None = DEFAULT_THINKING_LEVEL,
        thinking_budget: int | None = None,
    ):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Default to the current stable Flash model; can be overridden via CLI or GEMINI_MODEL.
        logger.info("Using Gemini model: %s", model_name)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.extracted_dict_dir = Path(extracted_dict_dir)
        
        if not self.extracted_dict_dir.exists():
            raise FileNotFoundError(f"Dictionary directory not found: {extracted_dict_dir}")
        
        # Rate limiting configuration
        self.base_delay = 1.0  # Base delay between requests in seconds
        self.max_retries = 5
        self.max_retry_delay = 300  # Maximum retry delay (5 minutes)
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level if thinking_level and thinking_level != "none" else None
        self.thinking_budget = thinking_budget
        if self.thinking_budget is not None:
            logger.info("Using Gemini thinking budget: %s", self.thinking_budget)
        elif self.thinking_level:
            logger.info("Using Gemini thinking level: %s", self.thinking_level)
        else:
            logger.info("Using Gemini default thinking behavior")

    def _thinking_config(self) -> types.ThinkingConfig | None:
        if self.thinking_budget is not None:
            return types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if self.thinking_level:
            return types.ThinkingConfig(thinking_level=self.thinking_level)
        return None

    def _extract_response_text(self, response) -> str:
        """Return text from a google-genai response, including part fallback."""
        if getattr(response, "text", None):
            return response.text.strip()

        candidates = getattr(response, "candidates", None) or []
        parts_text: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    parts_text.append(text)
        return "\n".join(parts_text).strip()

    def load_dakota_entries(self) -> Generator[Dict, None, None]:
        """Load Dakota dictionary entries from extracted JSON files."""
        json_files = sorted(self.extracted_dict_dir.glob("page_*.json"))
        logger.info(f"Found {len(json_files)} dictionary JSON files")
        
        if json_files:
            first_file = json_files[0].name
            last_file = json_files[-1].name
            logger.info(f"Extraction source: {first_file} through {last_file} ({len(json_files)} files total)")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get('entries', [])
                    for entry in entries:
                        # Ensure entry has required fields
                        if entry.get('headword') and entry.get('definition_primary'):
                            # Add source file info to entry for tracking
                            entry['_source_file'] = json_file.name
                            yield entry
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue

    def create_dakota_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Dakota language. Using the following Dakota-English dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of the language.
        
        Guidelines:
        1. Create questions that test translation from English to Dakota and vice versa
        2. Focus on how Dakota words express different aspects of English concepts
        3. Test understanding of grammatical classifications (verbs, nouns, adjectives) and subtle meaning differences
        4. Create scenarios that demonstrate when to use each Dakota variation
        5. Generate questions about word relationships and patterns
        6. Include cultural context where relevant
        7. Highlight Dakota special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) in examples
        8. Test understanding of inflected forms and word derivations
        
        Dictionary entries:
        """
        
        for entry in entries:
            # Format entry for context
            entry_str = json.dumps(entry, ensure_ascii=False, indent=2)
            context += f"\n{entry_str}"
            
        return context

    def create_reverse_dakota_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Dakota language. Using the following Dakota-English dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of Dakota from the Dakota-speaker's perspective.
        
        Guidelines:
        1. Create questions that test translation from Dakota to English
        2. Focus on proper usage of Dakota words in different contexts
        3. Test understanding of parts of speech and grammatical rules
        4. Create scenarios for practical usage
        5. Generate questions about cultural significance where relevant
        6. Include questions about related words and concepts (see_also, compare_with)
        7. Test knowledge of inflected forms and derivations
        8. Emphasize preservation of Dakota orthography (special characters)
        
        Dictionary entries:
        """
        
        for entry in entries:
            entry_str = json.dumps(entry, ensure_ascii=False, indent=2)
            context += f"\n{entry_str}"
            
        return context

    def _extract_retry_delay(self, error_message: str) -> float:
        """Extract retry delay from 429 error message."""
        # Look for "Please retry in X.XXXXXXs" pattern
        match = re.search(r'Please retry in ([\d.]+)s', error_message)
        if match:
            return float(match.group(1))
        
        # Look for retry_delay { seconds: X } pattern
        match = re.search(r'retry_delay.*?seconds[:\s]+(\d+)', error_message)
        if match:
            return float(match.group(1))
        
        # Default to exponential backoff if no delay found
        return None

    def generate_qa_pairs(self, entries: List[Dict], is_english_perspective: bool, context_size: int = 5) -> Generator[Dict, None, None]:
        """Generate Q&A pairs from Dakota dictionary entries with retry logic."""
        
        if not entries:
            return
        
        context = self.create_dakota_context_prompt(entries) if is_english_perspective else self.create_reverse_dakota_context_prompt(entries)
        
        prompt = """Based on these dictionary entries, generate 5 diverse 
        question-answer pairs. Format your response EXACTLY as shown below, maintaining
        valid JSON structure:

        [
            {
                "question": "What is the Dakota word for X?",
                "answer": "The Dakota word for X is Y."
            }
        ]

        Ensure your response is a valid JSON array containing exactly 5 question-answer pairs.
        Do not include any additional text or formatting.
        
        Important: Preserve Dakota orthography exactly (including special characters like ć, š, ŋ, etc.)"""
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending request to Google API with {len(entries)} entries... (attempt {attempt + 1}/{self.max_retries})")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=context + "\n" + prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=self.max_output_tokens,
                        temperature=0.4,
                        thinking_config=self._thinking_config(),
                    ),
                )
                logger.info("Received response from Google API.")
                response_text = self._extract_response_text(response)
                if not response_text:
                    finish_reason = None
                    if getattr(response, "candidates", None):
                        finish_reason = getattr(response.candidates[0], "finish_reason", None)
                    logger.warning(f"Empty Gemini response text; finish_reason={finish_reason}")
                    return
                
                # Extract JSON array from response
                if not response_text.startswith('['):
                    start_idx = response_text.find('[')
                    if start_idx >= 0:
                        response_text = response_text[start_idx:]
                    else:
                        logger.warning("No JSON array found in response")
                        return
                
                if not response_text.endswith(']'):
                    end_idx = response_text.rfind(']')
                    if end_idx >= 0:
                        response_text = response_text[:end_idx+1]
                    else:
                        logger.warning("No closing bracket found in response")
                        return
                
                qa_pairs = json.loads(response_text)
                for qa_pair in qa_pairs:
                    if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answer' in qa_pair:
                        qa_pair['source_language'] = 'english' if is_english_perspective else 'dakota'
                        yield qa_pair
                    else:
                        logger.warning("Skipping invalid QA pair format")
                
                # Success - add small delay to avoid hitting rate limits
                time.sleep(self.base_delay)
                return
                
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON response: {str(e)}")
                logger.debug(f"Response text: {response_text[:500]}")
                # Don't retry JSON decode errors
                return
                
            except Exception as e:
                error_str = str(e)
                
                # Check for 429 rate limit error
                if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                    retry_delay = self._extract_retry_delay(error_str)
                    
                    if retry_delay is None:
                        # Use exponential backoff
                        retry_delay = min(self.base_delay * (2 ** attempt), self.max_retry_delay)
                    else:
                        # Add a small buffer to the retry delay
                        retry_delay = min(retry_delay + 2.0, self.max_retry_delay)
                    
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Rate limit exceeded. Waiting {retry_delay:.2f} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {self.max_retries} attempts. Please wait and try again later.")
                        logger.error(f"Error details: {error_str[:500]}")
                        return
                else:
                    # Other errors - log and retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        retry_delay = min(self.base_delay * (2 ** attempt), self.max_retry_delay)
                        logger.warning(f"Error generating Q&A pairs (attempt {attempt + 1}/{self.max_retries}): {error_str[:200]}")
                        logger.info(f"Retrying in {retry_delay:.2f} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to generate Q&A pairs after {self.max_retries} attempts: {error_str[:500]}")
                        return
        
        logger.error("Failed to generate Q&A pairs after all retry attempts.")

    def _load_existing_pairs(self, output_path: Path) -> tuple[int, int]:
        """Load existing pairs from output file and return counts."""
        english_count = 0
        dakota_count = 0
        
        if not output_path.exists():
            return english_count, dakota_count
        
        logger.info(f"Checking existing pairs in {output_path}...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pair = json.loads(line)
                        source_lang = pair.get('source_language', '')
                        if source_lang == 'english':
                            english_count += 1
                        elif source_lang == 'dakota':
                            dakota_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading existing file: {e}. Starting fresh.")
            return 0, 0
        
        logger.info(f"Found {english_count} English-perspective pairs and {dakota_count} Dakota-perspective pairs")
        return english_count, dakota_count

    def generate_training_set(self, output_file: str, pairs_per_language: int = 75000, context_size: int = 5):
        """
        Generate Q&A pairs from Dakota dictionary entries.
        
        Args:
            output_file: Path to output JSONL file
            pairs_per_language: Target number of pairs per language direction
            context_size: Number of dictionary entries to include per API call
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_path.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Check for existing pairs and resume
        existing_english, existing_dakota = self._load_existing_pairs(output_path)
        resume_mode = existing_english > 0 or existing_dakota > 0
        
        if resume_mode:
            logger.info(f"Resuming generation: {existing_english} English pairs, {existing_dakota} Dakota pairs")
            logger.info(f"Output file: {output_path}")
            logger.info(f"New pairs will be appended to existing file")
        
        # Load all entries into memory
        logger.info("Loading Dakota dictionary entries...")
        all_entries = list(self.load_dakota_entries())
        logger.info(f"Loaded {len(all_entries)} dictionary entries")
        
        if not all_entries:
            raise ValueError("No dictionary entries found. Check that data/extracted/*.json files exist.")
        
        total_pairs = pairs_per_language * 2
        pair_count = existing_english + existing_dakota
        checkpoint_count = pair_count // 1000
        start_time = time.time()
        
        logger.info(f"Target: {total_pairs} Q&A pairs ({pairs_per_language} per language)...")
        logger.info(f"Current progress: {pair_count}/{total_pairs} ({pair_count/total_pairs*100:.2f}%)")
        
        # Determine file mode
        file_mode = 'a' if resume_mode else 'w'
        
        with open(output_path, file_mode, encoding='utf-8') as f:
            # Generate English-perspective Q&A pairs
            english_count = existing_english
            if english_count < pairs_per_language:
                logger.info(f"Generating English-perspective Q&A pairs... ({english_count}/{pairs_per_language} complete)")
                entries_buffer = []
                
                for entry in tqdm(all_entries, desc="Processing entries for English perspective"):
                    if english_count >= pairs_per_language:
                        break
                    
                    entries_buffer.append(entry)
                    
                    if len(entries_buffer) >= context_size:
                        # Generate Q&A pairs - handle empty generator (rate limit or errors)
                        qa_generated = False
                        for qa_pair in self.generate_qa_pairs(entries_buffer, is_english_perspective=True, context_size=context_size):
                            qa_generated = True
                            if english_count >= pairs_per_language:
                                break
                            qa_pair['generated_at'] = datetime.now().isoformat()
                            qa_pair['pair_id'] = pair_count + 1
                            # Add source page info from entries used
                            source_pages = sorted(set(e.get('page_number', 'unknown') for e in entries_buffer if e.get('page_number')))
                            qa_pair['source_pages'] = source_pages
                            qa_pair['source_files'] = sorted(set(e.get('_source_file', 'unknown') for e in entries_buffer if e.get('_source_file')))
                            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure data is written immediately
                            pair_count += 1
                            english_count += 1
                            
                            if pair_count % 1000 == 0:
                                self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                                checkpoint_count += 1
                        
                        # If no Q&A pairs were generated (rate limit hit), wait longer before continuing
                        if not qa_generated:
                            logger.warning("No Q&A pairs generated for this batch. Waiting 30 seconds before continuing...")
                            time.sleep(30)
                            # Don't clear buffer - retry with same entries
                            continue
                        
                        # Keep last 2 entries for context continuity
                        entries_buffer = entries_buffer[-2:]
            else:
                logger.info(f"English-perspective pairs complete ({english_count}/{pairs_per_language})")
            
            # Generate Dakota-perspective Q&A pairs
            dakota_count = existing_dakota
            if dakota_count < pairs_per_language:
                logger.info(f"Generating Dakota-perspective Q&A pairs... ({dakota_count}/{pairs_per_language} complete)")
                entries_buffer = []
                
                for entry in tqdm(all_entries, desc="Processing entries for Dakota perspective"):
                    if dakota_count >= pairs_per_language:
                        break
                    
                    entries_buffer.append(entry)
                    
                    if len(entries_buffer) >= context_size:
                        # Generate Q&A pairs - handle empty generator (rate limit or errors)
                        qa_generated = False
                        for qa_pair in self.generate_qa_pairs(entries_buffer, is_english_perspective=False, context_size=context_size):
                            qa_generated = True
                            if dakota_count >= pairs_per_language:
                                break
                            qa_pair['generated_at'] = datetime.now().isoformat()
                            qa_pair['pair_id'] = pair_count + 1
                            # Add source page info from entries used
                            source_pages = sorted(set(e.get('page_number', 'unknown') for e in entries_buffer if e.get('page_number')))
                            qa_pair['source_pages'] = source_pages
                            qa_pair['source_files'] = sorted(set(e.get('_source_file', 'unknown') for e in entries_buffer if e.get('_source_file')))
                            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure data is written immediately
                            pair_count += 1
                            dakota_count += 1
                            
                            if pair_count % 1000 == 0:
                                self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                                checkpoint_count += 1
                        
                        # If no Q&A pairs were generated (rate limit hit), wait longer before continuing
                        if not qa_generated:
                            logger.warning("No Q&A pairs generated for this batch. Waiting 30 seconds before continuing...")
                            time.sleep(30)
                            # Don't clear buffer - retry with same entries
                            continue
                        
                        # Keep last 2 entries for context continuity
                        entries_buffer = entries_buffer[-2:]
            else:
                logger.info(f"Dakota-perspective pairs complete ({dakota_count}/{pairs_per_language})")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed. Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Generated {pair_count} Q&A pairs ({english_count} English-perspective, {dakota_count} Dakota-perspective)")

    def _create_checkpoint(self, checkpoint_dir: Path, checkpoint_count: int, pair_count: int, total_pairs: int):
        """Create a checkpoint file to track progress."""
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_count}.jsonl"
        with open(checkpoint_file, 'w', encoding='utf-8') as cf:
            cf.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'pairs_generated': pair_count,
                'target_pairs': total_pairs,
                'percent_complete': (pair_count / total_pairs) * 100 if total_pairs > 0 else 0
            }, ensure_ascii=False) + '\n')

def main() -> None:
    """CLI for generating Dakota synthetic QA with controllable sample sizes."""
    parser = argparse.ArgumentParser(
        description="Generate Dakota synthetic question/answer pairs from extracted dictionary entries."
    )
    parser.add_argument(
        "--extracted-dir",
        default="data/extracted",
        help="Directory containing extracted Dakota page JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default="data/bilingual_training_set.jsonl",
        help="Destination JSONL path for the generated QA pairs.",
    )
    parser.add_argument(
        "--pairs-per-language",
        type=int,
        default=75000,
        help="Target number of QA pairs per direction (English->Dakota and Dakota->English).",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Number of dictionary entries to send per Gemini request.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model to use for Q&A generation (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Gemini max output tokens per request (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["none", "minimal", "low", "medium", "high"],
        default=DEFAULT_THINKING_LEVEL,
        help=(
            "Gemini 3.x thinking level for Q&A generation. Use medium for the full run, "
            "minimal for cheap smoke tests, or none to omit the setting."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=(
            "Optional Gemini 2.5-style thinking budget. If set, this overrides "
            "--thinking-level."
        ),
    )
    args = parser.parse_args()

    try:
        generator = DakotaQAGenerator(
            args.extracted_dir,
            model_name=args.model,
            max_output_tokens=args.max_output_tokens,
            thinking_level=args.thinking_level,
            thinking_budget=args.thinking_budget,
        )
        logger.info("Starting Dakota synthetic QA generation...")
        generator.generate_training_set(
            args.output_file,
            pairs_per_language=args.pairs_per_language,
            context_size=args.context_size,
        )
        logger.info("Training set generation completed successfully")
    except Exception as e:
        logger.error(f"Error during training set generation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

