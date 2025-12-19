from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tinker import EncodedTextChunk, ModelInput, SamplingParams, ServiceClient
from tinker.types import ImageChunk
from tinker_cookbook.tokenizer_utils import get_tokenizer

from dakota_extraction.core.extraction_prompt import build_extraction_prompt
from dakota_extraction.schemas.dictionary_schema import DictionaryEntry, expand_pos, validate_entry

from .image_utils import PreparedImage, prepare_image_for_tinker, prepare_pil_image_for_tinker


_PAGE_RE = re.compile(r".*?_(\d{4})\.(jpg|jpeg|png)$", re.IGNORECASE)


def _page_number_from_filename(path: Path) -> Optional[int]:
    match = _PAGE_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _maybe_extract_json(text: str) -> str:
    stripped = text.strip()

    if "```json" in stripped:
        start = stripped.find("```json") + 7
        end = stripped.find("```", start)
        if end != -1:
            return stripped[start:end].strip()

    if "```" in stripped:
        start = stripped.find("```") + 3
        end = stripped.find("```", start)
        if end != -1:
            return stripped[start:end].strip()

    # Fallback: try to slice between outermost braces.
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return stripped[brace_start : brace_end + 1].strip()

    return stripped


def _default_tokenizer_model(tinker_model: str) -> str:
    if "/" in tinker_model:
        return tinker_model
    return f"Qwen/{tinker_model}"


@dataclass(frozen=True)
class TinkerVisionConfig:
    model: str
    tokenizer_model: str
    resize_long_edge: Optional[int]
    patch_size: int
    temperature: float
    split_columns: bool
    split_margin_frac: float


class Qwen3VLTinkerPageProcessor:
    """
    Extract Dakota dictionary entries using Thinking Machines Tinker + Qwen3-VL.

    Output JSON matches the existing Claude-based pipeline:
      - top-level keys: page_metadata, entries, raw_response, metadata
      - entries are `DictionaryEntry.to_dict()` payloads
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = "data/extracted_qwen3vl_tinker",
        reasoning_dir: str = "data/reasoning_traces_qwen3vl_tinker",
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        tokenizer_model: Optional[str] = None,
        resize_long_edge: Optional[int] = None,
        patch_size: int = 28,
        temperature: float = 0.0,
        split_columns: bool = False,
        split_margin_frac: float = 0.02,
    ):
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError("TINKER_API_KEY must be set")

        self.config = TinkerVisionConfig(
            model=model,
            tokenizer_model=tokenizer_model or _default_tokenizer_model(model),
            resize_long_edge=resize_long_edge,
            patch_size=patch_size,
            temperature=temperature,
            split_columns=split_columns,
            split_margin_frac=split_margin_frac,
        )

        svc_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            svc_kwargs["base_url"] = base_url
        self.service_client = ServiceClient(**svc_kwargs)
        self.sampling_client = self.service_client.create_sampling_client(base_model=model)

        self.tokenizer = get_tokenizer(self.config.tokenizer_model)

        self.output_dir = Path(output_dir)
        self.reasoning_dir = Path(reasoning_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(
        self,
        image_path: Path,
        page_number: int,
        *,
        max_tokens: int = 16000,
        page_context: str = "",
    ) -> Dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*70}")
        print(f"Processing Page {page_number}: {image_path.name}")
        print(f"{'='*70}")
        print(f"Model: {self.config.model}")

        if self.config.split_columns:
            extraction, response_text, prepared = self._extract_two_column_page(
                image_path=image_path,
                page_number=page_number,
                max_tokens=max_tokens,
                page_context=page_context,
            )
        else:
            prompt = build_extraction_prompt(page_context)
            prepared = prepare_image_for_tinker(
                image_path,
                resize_long_edge=self.config.resize_long_edge,
                patch_size=self.config.patch_size,
            )
            response_text = self._call_model(prepared, prompt, max_tokens=max_tokens)
            extraction = self._parse_response(response_text, page_number, image_path.name, entry_offset=0)

        extraction["metadata"] = {
            "page_number": page_number,
            "image_path": str(image_path),
            "processed_at": datetime.now().isoformat(),
            "model": self.config.model,
            "tokenizer_model": self.config.tokenizer_model,
            "image_width": prepared.width,
            "image_height": prepared.height,
            "image_tokens_estimate": prepared.tokens,
        }

        self._validate_extraction(extraction)
        self._save_response(page_number, response_text)
        self._save_extraction(page_number, extraction)

        print(f"OK Extracted {len(extraction.get('entries', []))} entries")
        return extraction

    def batch_extract(
        self,
        *,
        image_dir: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_tokens: int = 16000,
        page_context: str = "",
        skip_existing: bool = True,
    ) -> list[Dict[str, Any]]:
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))

        numbered: list[tuple[int, Path]] = []
        for path in image_paths:
            pn = _page_number_from_filename(path)
            if pn is not None:
                numbered.append((pn, path))

        if numbered:
            numbered.sort(key=lambda x: x[0])
            if end_page is None:
                selected = [(pn, path) for pn, path in numbered if pn >= start_page]
            else:
                selected = [(pn, path) for pn, path in numbered if start_page <= pn <= end_page]
        else:
            if end_page:
                sliced = image_paths[start_page - 1 : end_page]
            else:
                sliced = image_paths[start_page - 1 :]
            selected = list(enumerate(sliced, start=start_page))

        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION (Tinker Qwen3-VL): {len(selected)} pages")
        print(f"{'='*70}\n")

        extractions: list[Dict[str, Any]] = []
        for page_number, image_path in selected:
            try:
                if skip_existing:
                    output_path = self.output_dir / f"page_{page_number:03d}.json"
                    if output_path.exists():
                        print(f"Skipping existing: page_{page_number:03d}.json")
                        continue
                extractions.append(
                    self.extract_page(
                        image_path=image_path,
                        page_number=page_number,
                        max_tokens=max_tokens,
                        page_context=page_context,
                    )
                )
            except Exception as exc:
                print(f"ERROR processing page {page_number}: {exc}")
                continue
        return extractions

    def _call_model(self, prepared: PreparedImage, prompt: str, *, max_tokens: int) -> str:
        image_chunk = self._build_image_chunk(prepared)
        text_tokens = self._encode_prompt(prompt)
        text_chunk = EncodedTextChunk(tokens=text_tokens)
        model_input = ModelInput(chunks=[image_chunk, text_chunk])

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=1.0,
        )
        resp = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
        ).result()
        seq = resp.sequences[0]
        return self.tokenizer.decode(seq.tokens, skip_special_tokens=True)

    def _extract_two_column_page(
        self,
        *,
        image_path: Path,
        page_number: int,
        max_tokens: int,
        page_context: str,
    ) -> tuple[Dict[str, Any], str, PreparedImage]:
        """
        Crop left/right halves and extract separately, then merge.

        This reduces truncation risk on dense two-column dictionary pages.
        """
        from PIL import Image

        with Image.open(image_path) as img:
            img.load()
            w, h = img.size
            split_x = w // 2
            margin = int(round(self.config.split_margin_frac * w))
            left_box = (0, 0, max(1, split_x - margin), h)
            right_box = (min(w - 1, split_x + margin), 0, w, h)
            left_img = img.crop(left_box)
            right_img = img.crop(right_box)

        left_prepared = prepare_pil_image_for_tinker(
            left_img,
            resize_long_edge=self.config.resize_long_edge,
            patch_size=self.config.patch_size,
        )
        right_prepared = prepare_pil_image_for_tinker(
            right_img,
            resize_long_edge=self.config.resize_long_edge,
            patch_size=self.config.patch_size,
        )

        left_ctx = (
            "This image is ONLY the LEFT column of a two-column dictionary page.\n"
            "Extract entries ONLY from this column. Set `column` to 1 for all entries.\n"
        )
        right_ctx = (
            "This image is ONLY the RIGHT column of a two-column dictionary page.\n"
            "Extract entries ONLY from this column. Set `column` to 2 for all entries.\n"
        )
        if page_context:
            left_ctx += f"\nPage context: {page_context}\n"
            right_ctx += f"\nPage context: {page_context}\n"

        left_prompt = build_extraction_prompt(left_ctx)
        right_prompt = build_extraction_prompt(right_ctx)

        left_text = self._call_model(left_prepared, left_prompt, max_tokens=max_tokens)
        left_parsed = self._parse_response(
            left_text,
            page_number,
            image_path.name,
            entry_offset=0,
            forced_column=1,
        )
        left_entries = list(left_parsed.get("entries", []))

        right_text = self._call_model(right_prepared, right_prompt, max_tokens=max_tokens)
        right_parsed = self._parse_response(
            right_text,
            page_number,
            image_path.name,
            entry_offset=len(left_entries),
            forced_column=2,
        )

        combined_entries = left_entries + list(right_parsed.get("entries", []))
        page_metadata = left_parsed.get("page_metadata") or right_parsed.get("page_metadata") or {"columns": 2}

        combined_text = f"<<LEFT_COLUMN>>\n{left_text}\n\n<<RIGHT_COLUMN>>\n{right_text}\n"
        combined = {
            "page_metadata": page_metadata,
            "entries": combined_entries,
            "raw_response": combined_text,
        }

        prepared_full = prepare_image_for_tinker(
            image_path,
            resize_long_edge=self.config.resize_long_edge,
            patch_size=self.config.patch_size,
        )
        return combined, combined_text, prepared_full

    def _build_image_chunk(self, prepared: PreparedImage) -> ImageChunk:
        annotations = getattr(ImageChunk, "__annotations__", {}) or {}
        if "expected_tokens" in annotations:
            # Newer Tinker SDKs treat expected_tokens as advisory and may reject mismatches.
            # We omit it so the backend can compute the tokenization itself.
            return ImageChunk(data=prepared.data, format=prepared.format, expected_tokens=None)
        # Back-compat for older SDKs.
        return ImageChunk(
            data=prepared.data,
            format=prepared.format,
            width=prepared.width,
            height=prepared.height,
            tokens=prepared.tokens,
        )

    def _encode_prompt(self, prompt: str) -> list[int]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return self.tokenizer.encode(rendered)
            except Exception:
                return self.tokenizer.encode(prompt)
        return self.tokenizer.encode(prompt)

    def _parse_response(
        self,
        response_text: str,
        page_number: int,
        source_image: str,
        *,
        entry_offset: int = 0,
        forced_column: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            json_text = _maybe_extract_json(response_text)
            raw_data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return {
                "page_metadata": {"error": f"JSON parsing failed: {exc}"},
                "entries": [],
                "raw_response": response_text,
            }

        entries = []
        for i, entry_data in enumerate(raw_data.get("entries", [])):
            entry_data = dict(entry_data)
            entry_id_num = i + 1 + entry_offset
            entry_data["entry_id"] = f"page_{page_number:03d}_entry_{entry_id_num:03d}"
            entry_data["page_number"] = page_number
            entry_data["source_image"] = source_image

            if forced_column is not None and entry_data.get("column") is None:
                entry_data["column"] = int(forced_column)

            if not entry_data.get("definition_primary") and entry_data.get("definition"):
                entry_data["definition_primary"] = entry_data.get("definition")
            if not entry_data.get("headword") and entry_data.get("dakota"):
                entry_data["headword"] = entry_data.get("dakota")
            if not entry_data.get("part_of_speech") and entry_data.get("pos"):
                entry_data["part_of_speech"] = entry_data.get("pos")

            if entry_data.get("part_of_speech"):
                entry_data["pos_full"] = expand_pos(entry_data["part_of_speech"])

            try:
                entry = DictionaryEntry(**entry_data)
                entries.append(entry.to_dict())
            except Exception as exc:
                print(f"  WARNING: Could not create entry {i+1}: {exc}")
                entries.append(entry_data)

        return {
            "page_metadata": raw_data.get("page_metadata", {}),
            "entries": entries,
            "raw_response": response_text,
        }

    def _validate_extraction(self, extraction: Dict[str, Any]) -> None:
        total = len(extraction.get("entries", []))
        valid_count = 0
        warnings: list[str] = []
        for entry_dict in extraction.get("entries", []):
            try:
                entry = DictionaryEntry(**entry_dict)
                is_valid, issues = validate_entry(entry)
                if is_valid:
                    valid_count += 1
                else:
                    warnings.extend([f"{entry.headword}: {issue}" for issue in issues])
            except Exception as exc:
                warnings.append(f"Entry validation error: {exc}")

        print(f"Validation: {valid_count}/{total} entries valid")
        for warning in warnings[:10]:
            print(f"  WARNING: {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    def _save_extraction(self, page_number: int, extraction: Dict[str, Any]) -> None:
        output_path = self.output_dir / f"page_{page_number:03d}.json"
        output_path.write_text(
            json.dumps(extraction, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved extraction to: {output_path}")

    def _save_response(self, page_number: int, response_text: str) -> None:
        response_path = self.reasoning_dir / f"page_{page_number:03d}_qwen3vl_response.txt"
        response_path.write_text(response_text, encoding="utf-8")
        print(f"Saved raw response to: {response_path}")
