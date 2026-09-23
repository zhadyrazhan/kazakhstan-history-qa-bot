"""Evaluate the QA bot against the hand-verified golden set.

Scores two things the LLM-judge in the notebook does not separate cleanly:

  fact recall  - for answerable questions, did the answer contain the expected
                 facts (matched via aliases, so paraphrase is not penalised)?
  refusal      - for questions NOT answerable from pages 3-23, did the model
                 correctly decline instead of inventing an answer?

Both matter: a model that answers everything scores well on recall and badly on
refusal, and a model that refuses everything does the reverse.

Usage:
    uvicorn server:app --port 8000          # in webapp/, first
    python evals/run_eval.py
    python evals/run_eval.py --url http://127.0.0.1:8000 --out results.json
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

GOLDEN_SET = Path(__file__).parent.parent / "data" / "golden_set.json"

REFUSAL_MARKERS = [
    "не указан",
    "не указано",
    "нет информации",
    "не сказано",
    "отсутствует",
    "не содержится",
    "не упоминается",
    "не могу",
    "нет данных",
]


def ask(url: str, question: str, timeout: int = 120) -> str:
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["answer"]
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("detail", str(e))
        raise RuntimeError(f"server returned {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"cannot reach {url} - is the webapp running? ({e.reason})"
        ) from e


def normalize(text: str) -> str:
    # Models emit typographic variants that break naive substring matching:
    # «Дивани лугат ат‑тюрк» with U+2011 non-breaking hyphen would otherwise
    # miss the expected "ат-тюрк". Also folds ё/е and collapses whitespace.
    for dash in "‐‑‒–—−":
        text = text.replace(dash, "-")
    return " ".join(text.lower().replace("ё", "е").split())


def fact_found(answer: str, fact: str, aliases: dict[str, list[str]]) -> bool:
    candidates = [fact] + aliases.get(fact, [])
    normalized = normalize(answer)
    return any(normalize(c) in normalized for c in candidates)


def is_refusal(answer: str) -> bool:
    normalized = normalize(answer)
    return any(normalize(m) in normalized for m in REFUSAL_MARKERS)


def evaluate(items: list[dict], url: str) -> list[dict]:
    results = []
    for item in items:
        try:
            answer = ask(url, item["question"])
            error = None
        except RuntimeError as e:
            answer, error = "", str(e)

        aliases = item.get("acceptable_aliases", {})
        expected = item.get("expected_facts", [])

        if error:
            passed, detail = False, f"ERROR: {error}"
        elif item.get("expected_refusal"):
            passed = is_refusal(answer)
            detail = "correctly refused" if passed else "answered instead of refusing"
        else:
            hits = [f for f in expected if fact_found(answer, f, aliases)]
            passed = len(hits) == len(expected)
            detail = f"{len(hits)}/{len(expected)} facts: {hits or 'none'}"

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "passed": passed,
                "detail": detail,
                "answerable": item["answerable_from_source"],
                "from_brief": item.get("from_project_brief", False),
            }
        )
    return results


def report(results: list[dict]) -> None:
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}")
        print(f"      Q: {r['question']}")
        print(f"      A: {r['answer'][:160] or '(no answer)'}")
        print(f"      -> {r['detail']}\n")

    def rate(subset: list[dict]) -> str:
        if not subset:
            return "n/a"
        n = sum(r["passed"] for r in subset)
        return f"{n}/{len(subset)} ({n / len(subset):.0%})"

    answerable = [r for r in results if r["answerable"]]
    refusals = [r for r in results if not r["answerable"]]
    brief = [r for r in results if r["from_brief"]]

    print("=" * 52)
    print(f"Fact recall (answerable):   {rate(answerable)}")
    print(f"Correct refusals:           {rate(refusals)}")
    print(f"Project-brief questions:    {rate(brief)}")
    print(f"Overall:                    {rate(results)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="webapp base URL")
    parser.add_argument("--out", help="write full results to this JSON file")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    results = evaluate(golden["items"], args.url)
    report(results)

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
