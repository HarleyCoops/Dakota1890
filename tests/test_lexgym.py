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


def test_unterminated_thinking_block_has_no_answer():
    t = _task("en_to_dak", "śuŋka")
    assert score(t, "<think>\nśuŋka")[0] == 0.0
    assert score(t, "<think>\nANSWER: śuŋka")[0] == 0.0
    assert score(t, "<think>x</think>\nANSWER: śuŋka")[0] == 1.0


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


def test_gated_arm_drops_char_dice_from_reward():
    """Arms a1/a2/c1: a near miss earns nothing; the ledger keeps char_dice.

    Baseline behaviour (no "arm" key, or arm="baseline") is unchanged: the
    character-overlap term still pays for a plausible-looking wrong word.
    """
    t = _task("en_to_dak", "śuŋka")
    t_gated = dict(t, arm="gated")
    r_base, led_base = score(t, "ANSWER: śuŋkata")
    r_gated, led_gated = score(t_gated, "ANSWER: śuŋkata")
    assert led_base["char_dice"] > 0.0 and led_gated["char_dice"] > 0.0  # diagnostic stays
    assert r_base > 0.0           # baseline still pays for character overlap
    assert r_gated == 0.0         # gated arm pays nothing for a near miss
    assert score(t_gated, "ANSWER: śuŋka")[0] == pytest.approx(1.0)
    # explicit baseline spelling behaves like the default
    assert score(dict(t, arm="baseline"), "ANSWER: śuŋkata")[0] == pytest.approx(r_base)
    # the n / ŋ partial channel survives the gate
    t2_gated = dict(_task("en_to_dak", "amdakedaŋ"), arm="gated")
    r, led = score(t2_gated, "ANSWER: amdakedan")
    assert r == pytest.approx(0.7) and led["exact"] == 0.0 and led["loose_match"] == 1.0


def test_gated_arm_is_gated_at_the_letters():
    """The letters_ok gate still zeroes a syllabify answer with a changed letter."""
    s_gated = dict(_task("syllabify", "a-hi'-yo-ka-sin"), arm="gated")
    assert score(s_gated, "ANSWER: a-hi'-yo-ka-zin")[0] == 0.0
    assert score(s_gated, "ANSWER: a-hi'-yo-ka-sin")[0] == pytest.approx(1.0)


def test_scorer_version_marks_the_arm():
    from lexgym.scoring import SCORER_VERSIONS, scorer_version
    assert scorer_version("baseline") != scorer_version("gated")
    assert set(SCORER_VERSIONS) >= {"baseline", "gated"}
    assert scorer_version("baseline") == "2026-11-10.2"


def test_group_advantages_center_and_floor():
    """Arm a2: std-scaled advantages stay centred and cannot blow up on noise."""
    from lexgym.advantages import group_advantages
    advs = group_advantages([0.0, 0.0, 0.0, 1.0])
    assert sum(advs) == pytest.approx(0.0, abs=1e-9)
    assert advs[3] > 0.0 > advs[0]
    # a near-constant group is floored at sigma_min instead of exploding
    tiny = group_advantages([0.0, 0.01], sigma_min=0.25)
    assert max(abs(x) for x in tiny) == pytest.approx(0.01 / 0.25)
    # a fully constant group has zero signal, as in the baseline loop
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


@pytest.mark.skipif(not (TASKS / "rl_train.jsonl").exists(), reason="run python -m lexgym.tasks first")
def test_generated_data_gold_and_leakage():
    def load(n):
        return [json.loads(line) for line in (TASKS / f"{n}.jsonl").open(encoding="utf-8")]
    train, held = load("rl_train"), load("eval_heldout_rule")
    for r in train[::7] + held[::3]:
        assert score(r, "ANSWER: " + r["answer"])[0] == pytest.approx(1.0), r["id"]
    def fam(rows):
        return {r["info"]["family"] for r in rows if "family" in r["info"]}
    assert not fam(train) & fam(held)
    held_words = {r["prompt"].rsplit(": ", 1)[-1] for r in held if r["task"] == "syllabify"}
    sft = load("sft_train")
    assert not any(m["messages"][1]["content"].rsplit(": ", 1)[-1] in held_words for m in sft if m["task"] == "syllabify")
    assert all(r["info"].get("page") and r["info"].get("source_image") for r in train[::11])
    # two-witness rows accept only forms that are also headwords in the Dakota half
    headwords = set(json.loads((TASKS / "lookup_index.json").read_text(encoding="utf-8")))
    from lexgym.orthography import plain as _plain
    e2d = [r for r in train if r["task"] == "en_to_dak"]
    assert e2d and all(r["info"]["gold_tier"] == "two_witnesses" for r in e2d)
    assert all(_plain(a) in headwords for r in e2d[::5] for a in r["accepted"])


def test_hub_package_scores_with_the_same_code():
    """The Prime Intellect package ships copies of the scorer and orthography."""
    pkg = Path("environments/riggs1890_lexgym/riggs1890_lexgym")
    for name in ("scoring.py", "orthography.py"):
        assert (pkg / name).read_bytes() == (Path("lexgym") / name).read_bytes(), name
