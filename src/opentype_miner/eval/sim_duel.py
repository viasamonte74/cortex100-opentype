"""Local paired duel simulator → scoring.verdict (optimize this, not train loss)."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from opentype_challenge import harness, scoring, tracks
from opentype_challenge.generator import Gold
from opentype_challenge.scoring import CaseScore, Paired

DEFAULT_WEIGHTS = {t: p.weight for t, p in tracks.DEFAULT_PLAN.items()}

ChatGen = Callable[[list[dict[str, Any]], int], str]


def score_read_row(
    gold_json: Mapping[str, list[Any]],
    answers: Mapping[str, Any] | None,
    reads: Mapping[str, Any] | None,
) -> CaseScore:
    gold = {qid: Gold.from_json(v) for qid, v in gold_json.items()}
    return scoring.score_case(gold, answers, reads)


def score_harness_row(track: str, body: dict[str, Any], outputs: list[str] | None) -> CaseScore:
    if outputs is None:
        return scoring.harness_score(1.0)
    _, loss = harness.replay(tracks.ENVS[track], body, outputs)
    if loss is None:
        # depict without judge: treat as forfeit in local sim unless judge plugged in
        return scoring.harness_score(1.0)
    return scoring.harness_score(loss)


async def run_harness_transcript(track: str, body: dict[str, Any], generate: ChatGen) -> list[str]:
    async def _gen(messages: list[dict[str, Any]], seed: int) -> str:
        return generate(messages, seed)

    return await harness.run_episode(tracks.ENVS[track], body, _gen)


def soft_answer_from_gold(gold: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perfect calibrated answer (oracle reader) for headroom tables."""
    kind, options, probs = gold
    reads = {"label_mass": 0.99, "argmax_is_label": True}
    if kind == "noul":
        return {"noul": float(probs[0])}, reads
    if kind == "score":
        legend = {str(i): name for i, name in enumerate(options)}
        return {
            "score": int(max(range(len(probs)), key=probs.__getitem__)),
            "legend": legend,
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
            "confidence": float(max(probs)),
        }, reads
    return {
        "choice": options[max(range(len(probs)), key=probs.__getitem__)],
        "probabilities": {o: float(p) for o, p in zip(options, probs, strict=True)},
        "confidence": float(max(probs)),
    }, reads


def blur_probs(probs: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        return probs
    logits = [math.log(max(p, 1e-12)) / temperature for p in probs]
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    z = sum(ex)
    return [e / z for e in ex]


def side_from_skill(skill: str, gold_json: Mapping[str, list[Any]]) -> tuple[dict, dict]:
    """Synthetic side for smoke sims: 'oracle' | 'blur:T' | 'uniform' | 'forfeit'."""
    answers: dict[str, Any] = {}
    reads: dict[str, Any] = {}
    for qid, gold in gold_json.items():
        kind, options, probs = gold
        if skill == "forfeit":
            answers[qid] = {}
            reads[qid] = {"label_mass": 0.1, "argmax_is_label": False}
            continue
        if skill == "uniform":
            probs = [1.0 / len(options)] * len(options)
        elif skill.startswith("blur:"):
            probs = blur_probs(list(map(float, probs)), float(skill.split(":", 1)[1]))
        g2 = [kind, options, probs]
        answers[qid], reads[qid] = soft_answer_from_gold(g2)
    return answers, reads


def sim_jsonl(
    cases_path: Path,
    *,
    champion_skill: str = "blur:1.3",
    challenger_skill: str = "blur:1.05",
    retired: set[int] | None = None,
    weights: Mapping[str, float] | None = None,
    max_cases: int | None = None,
    scores_out: Path | None = None,
) -> dict[str, Any]:
    """Simulate a duel from a mixed JSONL of public cases (reads + harness oracles)."""
    retired = retired or set()
    weights = dict(weights or DEFAULT_WEIGHTS)
    pairs: list[Paired] = []
    score_rows: list[dict[str, Any]] = []
    with cases_path.open() as handle:
        for index, line in enumerate(handle):
            if max_cases is not None and index >= max_cases:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            track = row["track"]
            level = int(row.get("level", 0))
            if track in tracks.ENVS:
                # Champion / challenger use oracle with occasional drop for skill knobs
                oracle = row.get("oracle")
                champ_out = oracle if champion_skill != "forfeit" else None
                chall_out = oracle if challenger_skill != "forfeit" else None
                if champion_skill.startswith("blur:") and oracle:
                    # drop last turn with small probability encoded by blur
                    if float(champion_skill.split(":")[1]) > 1.2 and len(oracle) > 1:
                        champ_out = oracle[:-1]
                champ = score_harness_row(track, row["request"], champ_out)
                chall = score_harness_row(track, row["request"], chall_out)
            else:
                a0, r0 = side_from_skill(champion_skill, row["gold"])
                a1, r1 = side_from_skill(challenger_skill, row["gold"])
                champ = score_read_row(row["gold"], a0, r0)
                chall = score_read_row(row["gold"], a1, r1)
            pairs.append(Paired(index, level, champ, chall, track=track))
            score_rows.append(
                {
                    "index": index,
                    "track": track,
                    "level": level,
                    "champion_loss": champ.loss,
                    "challenger_loss": chall.loss,
                }
            )
    if scores_out:
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        with scores_out.open("w") as out:
            for r in score_rows:
                out.write(json.dumps(r) + "\n")
    # Restrict weights to tracks present
    present = {p.track for p in pairs}
    w = {t: weights[t] for t in present if t in weights and weights[t] > 0}
    total = sum(w.values()) or 1.0
    w = {t: v / total for t, v in w.items()}
    verdict = scoring.verdict(pairs, retired, stopped=False, weights=w)
    # Headroom table
    by_track: dict[str, dict[str, float]] = {}
    for t in sorted(present):
        rows = [p for p in pairs if p.track == t]
        by_track[t] = {
            "pairs": len(rows),
            "champion_loss": sum(p.champion.loss for p in rows),
            "challenger_loss": sum(p.challenger.loss for p in rows),
            "rel_cut": 1.0
            - (
                sum(p.challenger.loss for p in rows)
                / max(sum(p.champion.loss for p in rows), 1e-9)
            ),
        }
    verdict["headroom"] = by_track
    return verdict


def headroom_table(verdict: Mapping[str, Any]) -> str:
    lines = [
        f"crown={verdict.get('crown')} g={verdict.get('g'):.4f} "
        f"g_lcb={verdict.get('g_lcb'):.4f} g_min={verdict.get('g_min'):.4f}",
        f"halves={verdict.get('g_lcb_halves')}",
        "track\tpairs\tchamp_loss\tchall_loss\trel_cut\tg\tregressed\tgained",
    ]
    tracks_v = verdict.get("tracks") or {}
    head = verdict.get("headroom") or {}
    for t, h in head.items():
        meta = tracks_v.get(t, {})
        lines.append(
            f"{t}\t{h['pairs']}\t{h['champion_loss']:.3f}\t{h['challenger_loss']:.3f}\t"
            f"{h['rel_cut']:.3f}\t{meta.get('g', float('nan')):.4f}\t"
            f"{meta.get('regressed')}\t{meta.get('gained')}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Local OpenType duel simulator")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--champion-skill", default="blur:1.3")
    p.add_argument("--challenger-skill", default="blur:1.05")
    p.add_argument("--max-cases", type=int)
    p.add_argument("--scores-out", type=Path)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args(argv)
    verdict = sim_jsonl(
        args.cases,
        champion_skill=args.champion_skill,
        challenger_skill=args.challenger_skill,
        max_cases=args.max_cases,
        scores_out=args.scores_out,
    )
    print(headroom_table(verdict))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # verdict may contain non-JSON types; coerce
        args.json_out.write_text(json.dumps(verdict, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
