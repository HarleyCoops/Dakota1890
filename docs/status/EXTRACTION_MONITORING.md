# Dictionary Extraction Monitoring

## Extraction Not Running

**Status**: Stopped

**Last intended command**: `python scripts/extraction/extract_dakota_dictionary_v2.py --all-dictionary`

**Target**: Pages 95-430 (336 vocabulary scans)

## Progress Tracking

Last checked: May 2026 preflight audit
- Non-empty extraction JSON: 238 pages
- Extracted ranges: 95-101, 103-143, and 145-334
- Missing vocabulary extraction: 102 and 335-430
- Needs reprocess: 144 has zero entries
- Estimated remaining paid extraction: about 98 pages
- Grammar scans 1-92 are intentionally out of scope for this Q&A extraction run

## Next Check

No monitoring is scheduled until the extraction run is started.

## Monitoring Commands

To check manually:
```powershell
# Count extracted pages
Get-ChildItem data\extracted\page_*.json | Measure-Object | Select-Object Count

# See latest page
Get-ChildItem data\extracted\page_*.json | Sort-Object Name -Descending | Select-Object -First 1 Name

# Check if extraction is still running
Get-Process python | Where-Object {$_.CommandLine -like "*extract_dakota_dictionary*"}
```

## Notes

- Extraction uses the configured Anthropic model; current default is Claude Sonnet 4.6
- Each page takes ~2 minutes
- Files are saved incrementally (safe to check anytime)
- Process will continue even if terminal is closed (if started as background job)

