"""Offline tests for lexgym: orthography, scoring, hack probes, split leakage."""
import json
from pathlib import Path

import pytest

from lexgym.orthography import canon, is_suspect, parse_headword, plain
from lexgym.scoring import extract_answer, score

TASKS = Path("data/lexgym_tasks")


def _task(kind, answer, accepted=None, **info):
    return {"task": kind, "answer": answer, "accepted": accepted or [answer], "info": info}


def test_canon_maps_variant_glyphs():
    assert canon("Šuηka") == canon("śuŋka") == "śuŋka"
    assert plain("a-hi'-yo-ka-sin") == "ahiyokasin"
    assert is_suspect("kahтaŋ") and not is_suspect("ćaŋ-ka'-ḣu")


def test_parse_headword():
    p = parse_headword("a-hi'-yo-ka-sin")
    assert p["syllables"] == ["a", "hi", "yo", "ka", "sin"] and p["stress"] == 1
    assert parse_headword("a-to-ka-tu") is None  # no accent mark: not trusted


def test_only_final_answer_is_scored():
    t = _task("en_to_dak", "śuŋka")
    assert score(t, "ANSWER: śuŋka")[0] == 1.0
    assert score(t, "The word śuŋka means dog.\nANSWER: mato")[0] < 0.3  # gold inside reasoning
    assert extract_answer("<think>śuŋka</think>ANSWER: mato")[0] == "mato"


def test_hack_probes_earn_little():
    t = _task("en_to_dak", "śuŋka")
    assert score(t, "")[0] == 0.0
    assert score(t, "ANSWER: śuŋka mato wakaŋ tipi ćaŋ mini peta maka wi haŋ")[0] < 0.5  # word stuffing
    s = _task("syllabify", "a-hi'-yo-ka-sin")
    assert score(s, "ANSWER: ahiyokasin")[0] == 0.0 or score(s, "ANSWER: ahiyokasin")[0] < 0.2  # prompt echo
    assert score(s, "ANSWER: a-hi'-yo-ka-sin")[0] == 1.0
    assert score(s, "ANSWER: a-hi'-yo-ka-zin")[0] == 0.0  # changed letters
    assert 0 < score(s, "ANSWER: a-hi-yo'-ka-sin")[0] < 1.0  # right cuts, wrong stress


def test_loose_nasal_match_is_partial_not_exact():
    r, led = score(_task("en_to_dak", "amdakedaŋ"), "ANSWER: amdakedan")
    assert led["exact"] == 0.0 and led["loose_match"] == 1.0 and r == pytest.approx(0.7)


def test_set_tasks():
    t = _task("inflect", "mduhepa, uŋyuhepapi", ["mduhepa", "uŋyuhepapi"])
    assert score(t, "ANSWER: mduhepa, uŋyuhepapi")[0] == 1.0
    assert 0 < score(t, "ANSWER: mduhepa")[0] < 1.0


@pytest.mark.skipif(not (TASKS / "rl_train.jsonl").exists(), reason="run python -m lexgym.tasks first")
def test_generated_data_gold_and_leakage():
    load = lambda n: [json.loads(l) for l in (TASKS / f"{n}.jsonl").open(encoding="utf-8")]
    train, held = load("rl_train"), load("eval_heldout_rule")
    for r in train[::7] + held[::3]:
        assert score(r, "ANSWER: " + r["answer"])[0] == pytest.approx(1.0), r["id"]
    fam = lambda rows: {r["info"]["family"] for r in rows if "family" in r["info"]}
    assert not fam(train) & fam(held)
    held_words = {r["prompt"].rsplit(": ", 1)[-1] for r in held if r["task"] == "syllabify"}
    sft = load("sft_train")
    assert not any(m["messages"][1]["content"].rsplit(": ", 1)[-1] in held_words for m in sft if m["task"] == "syllabify")
    assert all(r["info"].get("page") and r["info"].get("source_image") for r in train[::11])
