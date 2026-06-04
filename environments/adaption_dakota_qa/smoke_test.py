import asyncio

from adaption_dakota_qa import load_environment
from adaption_dakota_qa.environment import AdaptionDakotaRubric


def main():
    env = load_environment(max_examples=25, eval_examples=5, seed=1890)
    assert len(env.dataset) > 0
    assert env.eval_dataset is not None and len(env.eval_dataset) > 0
    row = env.dataset[0]
    assert row["question"]
    assert row["answer"]
    assert "dakota_terms" in row["info"]

    rubric = AdaptionDakotaRubric()
    answer = row["answer"]
    good = [{"role": "assistant", "content": answer}]
    bad = [{"role": "assistant", "content": "I do not know."}]

    async def score_all():
        good_scores = []
        bad_scores = []
        for func in rubric.funcs:
            good_scores.append(await func(completion=good, answer=answer, info=row["info"]))
            bad_scores.append(await func(completion=bad, answer=answer, info=row["info"]))
        return good_scores, bad_scores

    good_scores, bad_scores = asyncio.run(score_all())
    assert sum(good_scores) > sum(bad_scores), (good_scores, bad_scores)
    print({
        "train_examples": len(env.dataset),
        "eval_examples": len(env.eval_dataset),
        "sample_id": row["id"],
        "sample_terms": row["info"]["dakota_terms"][:5],
        "good_component_scores": good_scores,
        "bad_component_scores": bad_scores,
    })


if __name__ == "__main__":
    main()
