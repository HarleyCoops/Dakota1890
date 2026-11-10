# Dakota1890 Adaption recipe / specification notes

Use this file as recipe notes for an Adaptive Data `instruction_dataset`.
Source gold is repaired Riggs grammar tasks (`grammar_tasks_complete_v2.jsonl` from PR 110).
Do not call paid APIs from this recipe.

## Orthography and gold

- 1890 Riggs Dakota orthography (Dakota-English Dictionary grammar; preserve ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú and other attested marks exactly).
- For Dakota-production tasks, gold must be Dakota.
- Every required_affix must appear in gold.
- Do not replace attested Dakota with an English glossary.
- Do not invent stems.

Dakota-production task types: `reverse_translation` (English → Dakota), `affix_insertion`, and `morphology` when the completion is a Dakota surface form.
Comprehension tasks (`word_translation`, Dakota → English) have English gold by design; do not rewrite those completions into unattested Dakota.

## Columns

- `prompt`: the user / task prompt
- `completion`: gold
- Extra columns Adaption can ignore or we can rejoin later: `task_type`, `difficulty`, `required_affixes` (JSON string), `source_id` (`info.rule_id`; v2 has no native source_id)

## Constraints

- Preserve attested Dakota surface forms from the 1890 grammar exactly.
- Nonempty `required_affixes` are supervision: each affix must occur in the gold completion.
- Empty `required_affixes` is allowed and still scores 0.0 on the live grant-clean rubric.
- Frozen held-out v1 (`grammar_tasks_heldout.jsonl`) is the eval set; do not overwrite it and do not treat it as train gold.
