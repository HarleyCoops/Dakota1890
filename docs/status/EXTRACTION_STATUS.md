# Dictionary Extraction Status

## Current Status

**Extraction Stopped**: No extraction run currently active

**Pages Extracted**: 239 JSON files, with 238 non-empty files covering pages 95-101, 103-143, and 145-334

**Progress**: 238 of 336 content scan pages have non-empty extraction JSON

## Target Goal

**Goal**: Complete vocabulary-page extraction before generating synthetic Q&A data
- **Vocabulary scope**: scans 95-430
- **Grammar scope**: scans 1-92 are intentionally handled by the grammar-rule extraction/RL pipeline, not by this vocab Q&A run
- **Known missing vocab extraction**: page 102 and pages 335-430
- **Known bad extraction**: page 144 exists but has zero entries and should be reprocessed
- **Non-content scans**: 431-441 are blank/back-cover/card/calibration material and should not be paid extraction targets
- **Source scan gap**: scan 437 is absent in the Internet Archive JP2 zip and local JP2/JPG folders; it is outside the content range

## Next Steps

When ready to continue extraction:

1. **Patch the one-page gap**:
   ```bash
   python scripts/extraction/extract_dakota_dictionary_v2.py --pages 102-102
   ```

2. **Reprocess the empty page**:
   ```bash
   python scripts/extraction/extract_dakota_dictionary_v2.py --pages 144-144
   ```

3. **Extract the remaining content range**:
   ```bash
   python scripts/extraction/extract_dakota_dictionary_v2.py --pages 335-430
   ```

## Notes

- Current extraction files are safe in `data/extracted/`
- Extraction can be resumed from any page number
- The extraction script skips valid, non-empty existing page JSON
- Enhanced generator will use Gemini 3.5 Flash by default and leverage rich dictionary metadata

## Files Ready

-  `generate_synthetic_dakota.py` - Updated to use Gemini 3.5 Flash
-  `ENHANCED_SYNTHETIC_QA_PLAN.md` - Plan for sophisticated Q&A generation
-  Vocabulary extraction missing pages audited before the May 2026 continuation run
