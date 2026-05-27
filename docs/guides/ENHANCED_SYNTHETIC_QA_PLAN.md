# Enhanced Synthetic Q&A Generation for Dakota Dictionary

## Overview

The Dakota dictionary extraction produces much richer linguistic data than simple word-definition pairs. This document outlines how to leverage this richness for sophisticated bidirectional Q&A generation.

## Rich Dictionary Structure

Each extracted entry contains:

```json
{
  "headword": "a-na'-hdo-ka",
  "definition_primary": "to wear a hole in, as in a moccasin, on something",
  "definition_secondary": "additional meanings if present",
  "part_of_speech": "v. a.",
  "pos_full": "verb active",
  "derived_from": "nahdoka",
  "root_meaning": "meaning of root word",
  "inflected_forms": ["anawahdoka", "uŋkćakapį"],
  "grammatical_notes": "detailed grammatical explanations",
  "usage_notes": "contextual usage information",
  "example_phrases": ["example sentences"],
  "see_also": ["related_words"],
  "compare_with": ["similar_words"]
}
```

## Enhanced Q&A Types

### 1. Basic Translation (Existing)
- **English → Dakota**: "What is the Dakota word for 'to hide'?"
- **Dakota → English**: "What does a-na'-hma mean?"

### 2. Part of Speech Classification
- "What part of speech is a-na'-hdo-ka?"
- "Is a-na'-hma a verb or noun?"
- "How do you conjugate the verb a-na'-hdo-ka?"

### 3. Inflected Forms
- "What are the inflected forms of a-na'-hdo-ka?"
- "How do you say 'to wear a hole' in past tense?"
- "What's the plural form of a-na'-hma?"

### 4. Etymology & Word Formation
- "What is a-na'-hdo-ka derived from?"
- "What root word does a-na'-hdo-ka come from?"
- "How is a-na'-hdo-ka formed from nahdoka?"

### 5. Multiple Meanings (Polysemy)
- "What are all the meanings of 'a' in Dakota?"
- "How many different meanings does 'a' have?"
- "In what context would you use 'a' to mean 'the armpit'?"

### 6. Grammatical Usage
- "How is 'a' used as an inseparable prefix?"
- "What grammatical function does 'a' serve in 'awašteḳa'?"
- "When would you use 'a' to express incredulity?"

### 7. Comparative & Cross-References
- "What's the difference between a-na'-hma and woanahbe?"
- "How does a-na'-hdo-ka compare to nahdoka?"
- "What words are related to a-na'-hma?"

### 8. Contextual Usage
- "When would you use a-na'-hdo-ka in a sentence?"
- "Give an example sentence using a-na'-hma."
- "What is the proper context for using 'a' as an interjection?"

### 9. Cultural & Semantic Depth
- "What cultural concept does a-na'-hdo-ka express?"
- "What does 'wearing a hole in a moccasin' signify in Dakota culture?"
- "How does Dakota express the concept of 'hiding'?"

### 10. Complex Morphological Questions
- "How does prefixing 'a' change the meaning of nahdoka?"
- "What happens when you add 'a' to a verb in Dakota?"
- "Explain the morphological structure of a-na'-hdo-ka."

## Implementation Strategy

### Phase 1: Enhance Entry Context
- Include `part_of_speech`, `inflected_forms`, `derived_from`, `grammatical_notes` in prompts
- Group entries by related words (using `see_also` and `compare_with`)
- Identify polysemous words (same headword, different POS/meanings)

### Phase 2: Diversified Question Generation
- Generate 10+ question types per entry batch
- Weight question types based on entry richness:
  - Basic entries: Translation questions
  - Rich entries: Etymology, inflections, grammatical usage
  - Cross-referenced entries: Comparative questions

### Phase 3: Context-Aware Answers
- Include grammatical notes in answers
- Reference inflected forms when relevant
- Explain word formation patterns
- Connect to related words

### Phase 4: Bidirectional Sophistication
- **English → Dakota**: Not just "word for X" but "How do you express X concept grammatically?"
- **Dakota → English**: Not just "what does Y mean" but "How is Y used? What are its forms?"

## Benefits

1. **Grammar Integration**: Connects dictionary to grammar rules (RL training)
2. **Morphological Understanding**: Teaches word formation patterns
3. **Semantic Depth**: Explores meanings beyond simple translation
4. **Cultural Context**: Preserves cultural concepts embedded in language
5. **Linguistic Accuracy**: Maintains precision with special characters and grammatical detail

## Example Enhanced Q&A Pairs

```json
{
  "question": "What is the Dakota verb for 'to hide' and how is it conjugated?",
  "answer": "The Dakota verb for 'to hide' is a-na'-hma (v. a.). Its inflected forms include anawahbe, anayahbe, and anaunhmanpi. This verb comes from the root word and can be used with various prefixes to express different aspects of hiding.",
  "question_type": "translation_with_grammar",
  "source_language": "english",
  "entry_ids": ["page_095_entry_002"]
}
```

```json
{
  "question": "How is the prefix 'a' used grammatically in Dakota, and what does it change when added to verbs?",
  "answer": "The prefix 'a' in Dakota serves multiple grammatical functions: 1) As an inseparable preposition/prefix meaning 'on' or 'upon' when prefixed to verbs and adjectives; 2) It can give a causative meaning to verbs; 3) It forms collective plurals with some motion verbs; 4) Prefixed to nouns, it creates adverbs; 5) It can make nouns from verbs. For example, 'a-na'-hdo-ka' (to wear a hole) is derived from 'nahdoka' with the 'a' prefix.",
  "question_type": "grammatical_analysis",
  "source_language": "dakota",
  "entry_ids": ["page_095_entry_001"]
}
```

```json
{
  "question": "What are all the different meanings of 'a' in Dakota?",
  "answer": "The word 'a' in Dakota has multiple meanings depending on context: 1) As the first letter of the alphabet; 2) As an interjection (inj.) expressing incredulity or contempt (usually followed by ka or kada); 3) As a noun meaning 'the armpit'; 4) As an imperative verb meaning 'only; hark, listen' (with forms: a, a wo, a yo; plural: a po, am, a miyo).",
  "question_type": "polysemy_exploration",
  "source_language": "dakota",
  "entry_ids": ["page_095_entry_001", "page_095_entry_002", "page_095_entry_003", "page_095_entry_004"]
}
```

## Next Steps

1. Complete dictionary extraction (pages 95-430) - IN PROGRESS
2. Analyze extracted data structure:
   - Count entries by part of speech
   - Identify polysemous words
   - Map cross-references
   - Extract morphological patterns
3. Create enhanced `generate_synthetic_dakota_enhanced.py`:
   - Rich context building
   - Multi-type question generation
   - Grammar-aware answers
   - Cross-reference integration
4. Generate sophisticated Q&A pairs
5. Validate quality and linguistic accuracy

