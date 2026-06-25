#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
BAYES_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = BAYES_DIR.parent
INVOCATION_CWD = Path.cwd().resolve()
if str(BAYES_DIR) not in sys.path:
    sys.path.insert(0, str(BAYES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_bayes_seeded_local_online as local
from multilingual_nn.languages import ALL_LANGUAGES, LanguageSpec
from multilingual_nn.phrase_translation import GoogleTranslatePhraseTranslator


DEFAULT_INPUT_CSV = local.DEFAULT_INPUT_CSV
DEFAULT_TARGET_MODEL = "gpt-5.4-nano"
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
DEFAULT_LANGUAGE_CODES = ["my", "km", "ug", "ky", "ne"]
DEFAULT_OUTPUT_DIR = BAYES_DIR / "runs" / f"zero_history_suboptimal_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_BUCKET_MANIFEST = BAYES_DIR / "runs" / "phrase_bucket_pmax_experiment_live" / "bucket_manifest.csv"
ALGORITHM_LABEL_SG_CE_V1 = "zero_history_signal_gated_ce_v1"
ALGORITHM_LABEL_LEGACY = "zero_history_multi_basin_suboptimal_search_v2"


@dataclass(frozen=True)
class FrontierSeed:
    assignment: tuple[int, ...]
    assignment_id: int
    score: float
    normalized_score: float
    round_index: int
    observed_index: int


@dataclass(frozen=True)
class BasinState:
    basin_id: int
    representative: FrontierSeed
    members: tuple[FrontierSeed, ...]
    best_score: float
    mean_score: float
    richness_score: float
    exploration_bonus: float
    allocation_weight: float
    stagnation_rounds: int


@dataclass(frozen=True)
class Proposal:
    assignment: tuple[int, ...]
    assignment_id: int
    selected_languages: tuple[str, ...]
    proposal_source: str
    basin_id: int | None
    secondary_basin_id: int | None
    basin_richness: float | None
    basin_allocation_weight: float | None
    search_score: float
    exploit_score: float
    uncertainty_score: float
    diversity_score: float
    ce_probability: float
    signal_regime: str
    elite_count: int
    seed_assignment_id: int | None
    seed_score: float | None
    mutation_radius: int | None
    round_index: int


@dataclass(frozen=True)
class ObservedPoint:
    assignment: tuple[int, ...]
    assignment_id: int
    judge_score: int | None
    normalized_reward: float
    round_index: int
    observed_index: int
    proposal_source: str
    basin_id: int | None
    secondary_basin_id: int | None
    basin_richness: float | None
    search_score: float
    exploit_score: float
    uncertainty_score: float
    diversity_score: float
    ce_probability: float
    signal_regime: str
    elite_count: int
    seed_assignment_id: int | None
    seed_score: float | None
    mutation_radius: int | None
    result: local.OnlineResult


@dataclass(frozen=True)
class QuerySearchStats:
    row_index: int
    example_id: str
    phrase_count: int
    total_combo_count: int
    rounds_run: int
    evaluated_count: int
    success_count_at_threshold: int
    best_judge_score: int | None
    first_success_candidate_index: int | None
    archive_seed_peak_size: int
    basin_peak_count: int
    peak_elite_count: int
    final_signal_regime: str
    pool_peak_size: int
    stopped_reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-history online search over multilingual phrase assignments. Each query starts "
            "from scratch, bootstraps with diverse samples, then progressively refines "
            "suboptimal regions using only same-query observed judge scores."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-behavior", choices=["source", "attack"], default="source")
    parser.add_argument(
        "--search-strategy",
        choices=["sg_ce_v1", "legacy_multi_basin"],
        default="sg_ce_v1",
        help="Promoted query-local search policy. legacy_multi_basin is retained only for comparison/debugging.",
    )
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGE_CODES)
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--final-m", type=int, default=20, help="Maximum online evaluations per query.")
    parser.add_argument(
        "--bucket-manifest",
        default=None,
        help="Optional metadata-only manifest used to filter input queries before collection.",
    )
    parser.add_argument(
        "--pmax",
        type=int,
        default=None,
        help="Keep only manifest rows with phrase_count <= pmax. Requires --bucket-manifest.",
    )
    parser.add_argument("--bootstrap-size", type=int, default=6, help="First-round evaluations before refinement.")
    parser.add_argument("--round-batch-size", type=int, default=4, help="Evaluations per refinement round.")
    parser.add_argument("--target-success-count", type=int, default=1, help="Stop a query after this many score hits.")
    parser.add_argument("--success-score", type=int, default=10)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=10)
    parser.add_argument("--frontier-topk", type=int, default=6, help="Maximum archive seeds retained per query-level basin archive view.")
    parser.add_argument("--frontier-quantile", type=float, default=0.7)
    parser.add_argument("--frontier-min-score", type=float, default=3.0)
    parser.add_argument("--suboptimal-margin", type=float, default=2.0)
    parser.add_argument("--max-basins", type=int, default=6, help="Maximum suboptimal basins retained per query.")
    parser.add_argument(
        "--basin-radius-frac",
        type=float,
        default=0.35,
        help="Assignments within this normalized Hamming radius are grouped into the same basin.",
    )
    parser.add_argument("--mutation-radius-min", type=int, default=1)
    parser.add_argument("--mutation-radius-max", type=int, default=4)
    parser.add_argument("--local-frac", type=float, default=0.45)
    parser.add_argument("--cross-basin-frac", type=float, default=0.15)
    parser.add_argument("--random-frac", type=float, default=0.20)
    parser.add_argument("--slot-prior-mean", type=float, default=0.35)
    parser.add_argument("--slot-prior-strength", type=float, default=1.0)
    parser.add_argument("--stagnation-widen-after", type=int, default=2)
    parser.add_argument("--ce-exploration-floor", type=float, default=0.12)
    parser.add_argument("--ce-no-signal-elite-frac", type=float, default=0.50)
    parser.add_argument("--ce-weak-signal-elite-frac", type=float, default=0.30)
    parser.add_argument("--ce-strong-signal-elite-frac", type=float, default=0.20)
    parser.add_argument("--ce-no-signal-alpha", type=float, default=0.15)
    parser.add_argument("--ce-weak-signal-alpha", type=float, default=0.35)
    parser.add_argument("--ce-strong-signal-alpha", type=float, default=0.60)
    parser.add_argument("--ce-conditioned-frac", type=float, default=0.25)
    parser.add_argument("--score-workers", type=int, default=2)
    parser.add_argument("--segment-backend", choices=["auto", "spacy", "regex"], default="auto")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--phrase-max-tokens", type=int, default=6)
    parser.add_argument("--translate-timeout", type=float, default=15.0)
    parser.add_argument("--translate-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--error-retry-sleep-seconds",
        type=float,
        default=15.0,
        help="Unexpected runtime errors block and retry after this sleep interval.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Disable blocking retry loops and raise unexpected exceptions immediately.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-text", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def get_language_specs(codes: list[str]) -> list[LanguageSpec]:
    by_code = {language.translate_code: language for language in ALL_LANGUAGES}
    specs: list[LanguageSpec] = []
    for code in codes:
        key = str(code).strip().lower()
        if key not in by_code:
            raise ValueError(
                f"Unsupported language code: {code}. Supported codes: {', '.join(sorted(by_code.keys()))}"
            )
        specs.append(by_code[key])
    return specs


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise ValueError(f"Hamming distance requires equal lengths, got {len(left)} and {len(right)}")
    return int(sum(a != b for a, b in zip(left, right)))


def normalized_hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left and not right:
        return 0.0
    return float(hamming_distance(left, right)) / float(max(1, len(left)))


def seed_from_observed(item: ObservedPoint) -> FrontierSeed:
    return FrontierSeed(
        assignment=item.assignment,
        assignment_id=int(item.assignment_id),
        score=float(item.judge_score if item.judge_score is not None else -1.0),
        normalized_score=float(item.normalized_reward),
        round_index=int(item.round_index),
        observed_index=int(item.observed_index),
    )


def weighted_sample_without_replacement(
    population: list[int],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    items = list(population)
    item_weights = [max(float(weight), 1e-6) for weight in weights]
    selected: list[int] = []
    limit = min(len(items), max(0, int(k)))
    while items and len(selected) < limit:
        index = rng.choices(range(len(items)), weights=item_weights, k=1)[0]
        selected.append(items.pop(index))
        item_weights.pop(index)
    return selected


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def ensure_output_dir(path: Path, allow_overwrite: bool) -> None:
    tracked_files = [
        path / "candidates.csv",
        path / "detail.csv",
        path / "summary.json",
        path / "search_trace.csv",
        path / "search_stats.csv",
    ]
    if path.exists() and not allow_overwrite:
        if any(file.exists() for file in tracked_files):
            raise FileExistsError(
                f"Refusing to overwrite existing artifacts in {path}. "
                "Choose a new --output-dir or pass --allow-overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def resolve_cli_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (INVOCATION_CWD / path).resolve()
    else:
        path = path.resolve()
    return path


def normalize_example_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def parse_manifest_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "y"}


def run_with_blocking_retry(
    operation: Callable[[], Any],
    *,
    context: str,
    fail_fast: bool,
    retry_sleep_seconds: float,
) -> Any:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if fail_fast:
                raise
            attempt += 1
            sleep_seconds = max(1.0, float(retry_sleep_seconds))
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(
                f"[{timestamp}] blocking retry in {context} attempt={attempt} "
                f"sleep={sleep_seconds:.1f}s error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            time.sleep(sleep_seconds)


def filter_eval_rows_with_manifest(eval_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.bucket_manifest is None and args.pmax is None:
        args._input_query_count = len(eval_rows)
        args._filtered_query_count = len(eval_rows)
        return eval_rows
    if args.bucket_manifest is None:
        raise ValueError("--pmax requires --bucket-manifest.")
    manifest_path = resolve_cli_path(args.bucket_manifest)
    manifest_df = pd.read_csv(manifest_path)
    if "example_id" not in manifest_df.columns:
        raise ValueError(f"bucket manifest missing example_id column: {manifest_path}")
    if "phrase_count" not in manifest_df.columns:
        raise ValueError(f"bucket manifest missing phrase_count column: {manifest_path}")
    manifest = manifest_df.copy()
    if "eligible" in manifest.columns:
        manifest = manifest[manifest["eligible"].map(parse_manifest_bool)].copy()
    if args.pmax is not None:
        manifest = manifest[pd.to_numeric(manifest["phrase_count"], errors="coerce") <= int(args.pmax)].copy()
    keep_ids = {normalize_example_id(value) for value in manifest["example_id"].tolist() if normalize_example_id(value)}
    filtered = [
        row
        for row in eval_rows
        if normalize_example_id(row.get("example_id") or row.get("id") or "") in keep_ids
    ]
    args._input_query_count = len(eval_rows)
    args._filtered_query_count = len(filtered)
    args._bucket_manifest_path = str(manifest_path)
    return filtered


def posterior_mean(count: float, reward_sum: float, prior_mean: float, prior_strength: float) -> float:
    return float((reward_sum + (prior_mean * prior_strength)) / (count + prior_strength))


def uncertainty_bonus(total_evals: int, count: float, prior_strength: float) -> float:
    log_term = math.log(max(2.0, float(total_evals) + 2.0))
    current = math.sqrt(log_term / max(1e-6, count + prior_strength))
    maximum = math.sqrt(log_term / max(1e-6, prior_strength))
    return clamp01(current / max(maximum, 1e-6))


def slot_scores_for_assignment(
    assignment: tuple[int, ...],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
) -> tuple[float, float]:
    means: list[float] = []
    bonuses: list[float] = []
    for position, language_index in enumerate(assignment):
        count = float(slot_counts[position, language_index])
        reward_sum = float(slot_reward_sums[position, language_index])
        means.append(posterior_mean(count, reward_sum, prior_mean, prior_strength))
        bonuses.append(uncertainty_bonus(total_evals, count, prior_strength))
    if not means:
        return 0.0, 0.0
    return float(np.mean(means)), float(np.mean(bonuses))


def assignment_diversity_score(assignment: tuple[int, ...], evaluated: set[tuple[int, ...]]) -> float:
    if not assignment:
        return 0.0
    if not evaluated:
        return 1.0
    min_distance = min(hamming_distance(assignment, previous) for previous in evaluated)
    return float(min_distance) / float(len(assignment))


def infer_signal_regime(best_score: float) -> str:
    value = float(best_score)
    if value <= 1.0:
        return "no_signal"
    if value >= 8.0:
        return "strong_signal"
    return "weak_signal"


def regime_elite_fraction(signal_regime: str, args: argparse.Namespace) -> float:
    if signal_regime == "strong_signal":
        return float(args.ce_strong_signal_elite_frac)
    if signal_regime == "weak_signal":
        return float(args.ce_weak_signal_elite_frac)
    return float(args.ce_no_signal_elite_frac)


def regime_alpha(signal_regime: str, args: argparse.Namespace) -> float:
    if signal_regime == "strong_signal":
        return float(args.ce_strong_signal_alpha)
    if signal_regime == "weak_signal":
        return float(args.ce_weak_signal_alpha)
    return float(args.ce_no_signal_alpha)


def mixed_with_uniform(weights: np.ndarray, floor: float) -> np.ndarray:
    if weights.ndim != 1 or weights.size <= 0:
        return weights
    floor_value = clamp01(float(floor))
    normalized = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(normalized))
    if total <= 0.0:
        normalized = np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)
    else:
        normalized = normalized / total
    normalized = ((1.0 - floor_value) * normalized) + (floor_value / float(weights.size))
    normalized /= max(float(np.sum(normalized)), 1e-9)
    return normalized.astype(np.float32)


def build_uniform_ce_distribution(phrase_count: int, base: int) -> np.ndarray:
    if phrase_count <= 0 or base <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.full((phrase_count, base), 1.0 / float(base), dtype=np.float32)


def build_balanced_bootstrap_assignments(
    phrase_count: int,
    base: int,
    bootstrap_size: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    if phrase_count <= 0 or base <= 0 or bootstrap_size <= 0:
        return []
    assignments: list[tuple[int, ...]] = []
    cyclic_count = min(int(bootstrap_size), max(1, base))
    for shift in range(cyclic_count):
        assignments.append(tuple((position + shift) % base for position in range(phrase_count)))
    while len(assignments) < int(bootstrap_size):
        best_candidate = None
        best_value = None
        attempts = max(64, base * max(1, phrase_count) * 8)
        for _ in range(attempts):
            candidate = random_assignment(phrase_count, base, rng)
            if candidate in assignments:
                continue
            if not assignments:
                value = float(phrase_count)
            else:
                value = min(hamming_distance(candidate, existing) for existing in assignments)
            if best_value is None or value > best_value:
                best_value = value
                best_candidate = candidate
        if best_candidate is None:
            fallback = random_assignment(phrase_count, base, rng)
            if fallback not in assignments:
                assignments.append(fallback)
                continue
            break
        assignments.append(best_candidate)
    return assignments[: int(bootstrap_size)]


def elite_subset(
    observed: list[ObservedPoint],
    signal_regime: str,
    args: argparse.Namespace,
) -> list[ObservedPoint]:
    if not observed:
        return []
    ordered = sorted(
        observed,
        key=lambda item: (
            -(float(item.judge_score) if item.judge_score is not None else -1.0),
            -float(item.search_score),
            int(item.assignment_id),
        ),
    )
    fraction = regime_elite_fraction(signal_regime, args)
    elite_count = max(1, int(math.ceil(len(ordered) * fraction)))
    if len(ordered) >= 2:
        elite_count = max(2, elite_count)
    elite_count = min(len(ordered), elite_count)
    return ordered[:elite_count]


def fit_ce_distribution(
    previous: np.ndarray,
    observed: list[ObservedPoint],
    signal_regime: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[ObservedPoint]]:
    if previous.size == 0:
        return previous, []
    elite = elite_subset(observed, signal_regime, args)
    if not elite:
        return previous, []
    phrase_count, base = previous.shape
    empirical = np.zeros((phrase_count, base), dtype=np.float64)
    total_weight = 0.0
    for item in elite:
        weight = max(1.0, float(item.judge_score if item.judge_score is not None else 0.0))
        total_weight += weight
        for position, language_index in enumerate(item.assignment):
            empirical[position, language_index] += weight
    if total_weight <= 0.0:
        empirical = np.full(previous.shape, 1.0 / float(base), dtype=np.float64)
    else:
        empirical /= total_weight
    alpha = clamp01(regime_alpha(signal_regime, args))
    updated = ((1.0 - alpha) * previous.astype(np.float64)) + (alpha * empirical)
    floor = clamp01(float(args.ce_exploration_floor))
    for position in range(phrase_count):
        updated[position] = mixed_with_uniform(updated[position], floor)
    return updated.astype(np.float32), elite


def assignment_ce_probability(assignment: tuple[int, ...], ce_distribution: np.ndarray) -> float:
    if not assignment or ce_distribution.size == 0:
        return 0.0
    probability = 1.0
    for position, language_index in enumerate(assignment):
        probability *= max(1e-8, float(ce_distribution[position, language_index]))
    return float(probability)


def sample_assignment_from_ce(ce_distribution: np.ndarray, rng: random.Random) -> tuple[int, ...]:
    if ce_distribution.size == 0:
        return ()
    assignment: list[int] = []
    for position in range(int(ce_distribution.shape[0])):
        weights = [float(value) for value in ce_distribution[position]]
        assignment.append(int(rng.choices(range(len(weights)), weights=weights, k=1)[0]))
    return tuple(assignment)


def conditioned_ce_assignment(
    elite_seed: ObservedPoint,
    ce_distribution: np.ndarray,
    signal_regime: str,
    rng: random.Random,
) -> tuple[tuple[int, ...], int]:
    assignment = list(elite_seed.assignment)
    phrase_count = len(assignment)
    if phrase_count <= 0:
        return tuple(assignment), 0
    if signal_regime == "strong_signal":
        max_radius = min(3, phrase_count)
    else:
        max_radius = min(4, phrase_count)
    radius = max(1, min(max_radius, 1 + (phrase_count // 4)))
    positions = rng.sample(list(range(phrase_count)), k=radius)
    for position in positions:
        weights = [float(value) for value in ce_distribution[position]]
        assignment[position] = int(rng.choices(range(len(weights)), weights=weights, k=1)[0])
    return tuple(assignment), radius


def compute_search_score(
    assignment: tuple[int, ...],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    evaluated: set[tuple[int, ...]],
    current_best_score: float,
    success_score: int,
    prior_mean: float,
    prior_strength: float,
    seed_score: float | None,
    mutation_radius: int | None,
) -> tuple[float, float, float, float]:
    exploit_score, uncertainty_score = slot_scores_for_assignment(
        assignment=assignment,
        slot_counts=slot_counts,
        slot_reward_sums=slot_reward_sums,
        total_evals=total_evals,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
    )
    diversity_score = assignment_diversity_score(assignment, evaluated)
    seed_bonus = 0.0 if seed_score is None else clamp01(float(seed_score) / max(1.0, float(success_score)))
    radius_bonus = 0.5
    if mutation_radius is not None and assignment:
        radius_bonus = 1.0 - min(1.0, float(mutation_radius) / float(len(assignment)))
    progress = clamp01(current_best_score / max(1.0, float(success_score)))
    weight_exploit = 0.45 + (0.25 * progress)
    weight_uncertainty = 0.35 - (0.15 * progress)
    weight_seed = 0.10 + (0.10 * progress)
    weight_diversity = 0.10
    weight_radius = 0.05 + (0.10 * progress)
    total_weight = weight_exploit + weight_uncertainty + weight_seed + weight_diversity + weight_radius
    search_score = (
        (weight_exploit * exploit_score)
        + (weight_uncertainty * uncertainty_score)
        + (weight_seed * seed_bonus)
        + (weight_diversity * diversity_score)
        + (weight_radius * radius_bonus)
    ) / max(total_weight, 1e-9)
    return clamp01(search_score), exploit_score, uncertainty_score, diversity_score


def language_sampling_weights(
    position: int,
    current_language: int | None,
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    current_best_score: float,
    success_score: int,
) -> list[float]:
    base = int(slot_counts.shape[1])
    progress = clamp01(current_best_score / max(1.0, float(success_score)))
    exploit_weight = 0.55 + (0.20 * progress)
    explore_weight = 0.45 - (0.20 * progress)
    temperature = max(0.35, 1.0 - (0.45 * progress))
    weights: list[float] = []
    for language_index in range(base):
        count = float(slot_counts[position, language_index])
        reward_sum = float(slot_reward_sums[position, language_index])
        mean = posterior_mean(count, reward_sum, prior_mean, prior_strength)
        bonus = uncertainty_bonus(total_evals, count, prior_strength)
        score = (exploit_weight * mean) + (explore_weight * bonus)
        weight = math.exp(score / max(temperature, 1e-6))
        if current_language is not None and language_index == int(current_language) and base > 1:
            weight *= 0.08
        weights.append(max(weight, 1e-6))
    return weights


def random_assignment(phrase_count: int, base: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(base) for _ in range(phrase_count))


def guided_assignment(
    phrase_count: int,
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    current_best_score: float,
    success_score: int,
    rng: random.Random,
) -> tuple[int, ...]:
    assignment: list[int] = []
    for position in range(phrase_count):
        weights = language_sampling_weights(
            position=position,
            current_language=None,
            slot_counts=slot_counts,
            slot_reward_sums=slot_reward_sums,
            total_evals=total_evals,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            current_best_score=current_best_score,
            success_score=success_score,
        )
        assignment.append(int(rng.choices(range(len(weights)), weights=weights, k=1)[0]))
    return tuple(assignment)


def choose_frontier_seed(frontier: list[FrontierSeed], rng: random.Random) -> FrontierSeed:
    weights = [0.1 + (seed.normalized_score**2) for seed in frontier]
    return rng.choices(frontier, weights=weights, k=1)[0]


def choose_basin_member(basin: BasinState, rng: random.Random) -> FrontierSeed:
    if len(basin.members) == 1:
        return basin.members[0]
    weights = []
    for member in basin.members:
        distance = normalized_hamming_distance(member.assignment, basin.representative.assignment)
        weights.append(0.2 + (member.normalized_score**2) + (0.15 * (1.0 - distance)))
    return rng.choices(list(basin.members), weights=weights, k=1)[0]


def sample_basin_pair(basins: list[BasinState], rng: random.Random) -> tuple[BasinState, BasinState]:
    if len(basins) == 1:
        return basins[0], basins[0]
    weights = [0.1 + basin.allocation_weight + basin.richness_score for basin in basins]
    first = rng.choices(basins, weights=weights, k=1)[0]
    others = [basin for basin in basins if basin.basin_id != first.basin_id]
    if not others:
        return first, first
    second_weights = [0.1 + basin.allocation_weight + basin.richness_score for basin in others]
    second = rng.choices(others, weights=second_weights, k=1)[0]
    return first, second


def select_mutation_positions(
    assignment: tuple[int, ...],
    radius: int,
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    current_best_score: float,
    success_score: int,
    rng: random.Random,
) -> list[int]:
    weights: list[float] = []
    base = int(slot_counts.shape[1])
    for position, current_language in enumerate(assignment):
        current_count = float(slot_counts[position, current_language])
        current_sum = float(slot_reward_sums[position, current_language])
        current_mean = posterior_mean(current_count, current_sum, prior_mean, prior_strength)
        current_bonus = uncertainty_bonus(total_evals, current_count, prior_strength)
        alt_scores: list[float] = []
        for language_index in range(base):
            if language_index == current_language:
                continue
            alt_count = float(slot_counts[position, language_index])
            alt_sum = float(slot_reward_sums[position, language_index])
            alt_mean = posterior_mean(alt_count, alt_sum, prior_mean, prior_strength)
            alt_bonus = uncertainty_bonus(total_evals, alt_count, prior_strength)
            alt_scores.append(0.7 * alt_mean + 0.3 * alt_bonus)
        best_alt = max(alt_scores) if alt_scores else current_mean
        weight = 0.1 + current_bonus + max(0.0, best_alt - current_mean)
        weights.append(weight)
    return weighted_sample_without_replacement(list(range(len(assignment))), weights, radius, rng)


def choose_mutation_radius(
    seed_score: float,
    phrase_count: int,
    current_best_score: float,
    success_score: int,
    mutation_radius_min: int,
    mutation_radius_max: int,
    stagnation_rounds: int,
    stagnation_widen_after: int,
    rng: random.Random,
) -> int:
    if phrase_count <= 0:
        return 0
    min_radius = max(1, min(int(mutation_radius_min), phrase_count))
    max_radius = max(min_radius, min(int(mutation_radius_max), phrase_count))
    progress = max(
        clamp01(float(seed_score) / max(1.0, float(success_score))),
        clamp01(float(current_best_score) / max(1.0, float(success_score))),
    )
    adaptive_max = int(round(min_radius + ((max_radius - min_radius) * (1.0 - progress))))
    if stagnation_rounds >= int(stagnation_widen_after):
        adaptive_max = min(max_radius, adaptive_max + 1)
    adaptive_max = max(min_radius, adaptive_max)
    radii = list(range(min_radius, adaptive_max + 1))
    weights = [1.0 / float(radius) for radius in radii]
    return int(rng.choices(radii, weights=weights, k=1)[0])


def mutate_frontier_seed(
    seed: FrontierSeed,
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    current_best_score: float,
    success_score: int,
    mutation_radius_min: int,
    mutation_radius_max: int,
    stagnation_rounds: int,
    stagnation_widen_after: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], int]:
    radius = choose_mutation_radius(
        seed_score=float(seed.score),
        phrase_count=len(seed.assignment),
        current_best_score=current_best_score,
        success_score=success_score,
        mutation_radius_min=mutation_radius_min,
        mutation_radius_max=mutation_radius_max,
        stagnation_rounds=stagnation_rounds,
        stagnation_widen_after=stagnation_widen_after,
        rng=rng,
    )
    positions = select_mutation_positions(
        assignment=seed.assignment,
        radius=radius,
        slot_counts=slot_counts,
        slot_reward_sums=slot_reward_sums,
        total_evals=total_evals,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        current_best_score=current_best_score,
        success_score=success_score,
        rng=rng,
    )
    mutated = list(seed.assignment)
    for position in positions:
        weights = language_sampling_weights(
            position=position,
            current_language=mutated[position],
            slot_counts=slot_counts,
            slot_reward_sums=slot_reward_sums,
            total_evals=total_evals,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            current_best_score=current_best_score,
            success_score=success_score,
        )
        mutated[position] = int(rng.choices(range(len(weights)), weights=weights, k=1)[0])
    return tuple(mutated), radius


def recombine_frontier_seeds(
    frontier: list[FrontierSeed],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    current_best_score: float,
    success_score: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], FrontierSeed]:
    if len(frontier) == 1:
        return frontier[0].assignment, frontier[0]
    first_seed = choose_frontier_seed(frontier, rng)
    remaining = [seed for seed in frontier if seed.assignment_id != first_seed.assignment_id]
    second_seed = choose_frontier_seed(remaining or frontier, rng)
    primary_seed = first_seed if first_seed.score >= second_seed.score else second_seed
    assignment: list[int] = []
    for position in range(len(primary_seed.assignment)):
        first_language = first_seed.assignment[position]
        second_language = second_seed.assignment[position]
        if first_language == second_language:
            assignment.append(first_language)
            continue
        first_count = float(slot_counts[position, first_language])
        first_sum = float(slot_reward_sums[position, first_language])
        second_count = float(slot_counts[position, second_language])
        second_sum = float(slot_reward_sums[position, second_language])
        first_score = posterior_mean(first_count, first_sum, prior_mean, prior_strength) + (0.25 * uncertainty_bonus(total_evals, first_count, prior_strength))
        second_score = posterior_mean(second_count, second_sum, prior_mean, prior_strength) + (0.25 * uncertainty_bonus(total_evals, second_count, prior_strength))
        if first_score == second_score:
            assignment.append(int(rng.choice([first_language, second_language])))
        else:
            better_language = first_language if first_score > second_score else second_language
            other_language = second_language if better_language == first_language else first_language
            assignment.append(int(rng.choices([better_language, other_language], weights=[0.7, 0.3], k=1)[0]))
    return tuple(assignment), primary_seed


def recombine_two_seeds(
    first_seed: FrontierSeed,
    second_seed: FrontierSeed,
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    prior_mean: float,
    prior_strength: float,
    rng: random.Random,
) -> tuple[tuple[int, ...], FrontierSeed, FrontierSeed]:
    primary_seed = first_seed if first_seed.score >= second_seed.score else second_seed
    assignment: list[int] = []
    for position in range(len(primary_seed.assignment)):
        first_language = first_seed.assignment[position]
        second_language = second_seed.assignment[position]
        if first_language == second_language:
            assignment.append(first_language)
            continue
        first_count = float(slot_counts[position, first_language])
        first_sum = float(slot_reward_sums[position, first_language])
        second_count = float(slot_counts[position, second_language])
        second_sum = float(slot_reward_sums[position, second_language])
        first_score = posterior_mean(first_count, first_sum, prior_mean, prior_strength) + (0.25 * uncertainty_bonus(total_evals, first_count, prior_strength))
        second_score = posterior_mean(second_count, second_sum, prior_mean, prior_strength) + (0.25 * uncertainty_bonus(total_evals, second_count, prior_strength))
        if first_score == second_score:
            assignment.append(int(rng.choice([first_language, second_language])))
        else:
            better_language = first_language if first_score > second_score else second_language
            other_language = second_language if better_language == first_language else first_language
            assignment.append(int(rng.choices([better_language, other_language], weights=[0.7, 0.3], k=1)[0]))
    return tuple(assignment), primary_seed, second_seed


def make_proposal(
    assignment: tuple[int, ...],
    languages: list[str],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    evaluated: set[tuple[int, ...]],
    current_best_score: float,
    success_score: int,
    prior_mean: float,
    prior_strength: float,
    proposal_source: str,
    round_index: int,
    basin_id: int | None,
    secondary_basin_id: int | None,
    basin_richness: float | None,
    basin_allocation_weight: float | None,
    seed_assignment_id: int | None,
    seed_score: float | None,
    mutation_radius: int | None,
    signal_regime: str = "legacy",
    elite_count: int = 0,
    ce_distribution: np.ndarray | None = None,
) -> Proposal:
    search_score, exploit_score, uncertainty_score, diversity_score = compute_search_score(
        assignment=assignment,
        slot_counts=slot_counts,
        slot_reward_sums=slot_reward_sums,
        total_evals=total_evals,
        evaluated=evaluated,
        current_best_score=current_best_score,
        success_score=success_score,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        seed_score=seed_score,
        mutation_radius=mutation_radius,
    )
    ce_probability = 0.0 if ce_distribution is None else assignment_ce_probability(assignment, ce_distribution)
    if ce_distribution is not None and assignment:
        ce_bonus = clamp01(ce_probability * float(len(assignment)))
        search_score += 0.10 * ce_bonus
    if basin_richness is not None:
        search_score += 0.12 * float(basin_richness)
    if basin_allocation_weight is not None:
        search_score += 0.08 * float(basin_allocation_weight)
    search_score = clamp01(search_score)
    assignment_id = int(local.encode_assignment_id(list(assignment), len(languages)))
    return Proposal(
        assignment=assignment,
        assignment_id=assignment_id,
        selected_languages=tuple(languages[index] for index in assignment),
        proposal_source=proposal_source,
        basin_id=basin_id,
        secondary_basin_id=secondary_basin_id,
        basin_richness=basin_richness,
        basin_allocation_weight=basin_allocation_weight,
        search_score=search_score,
        exploit_score=exploit_score,
        uncertainty_score=uncertainty_score,
        diversity_score=diversity_score,
        ce_probability=ce_probability,
        signal_regime=str(signal_regime),
        elite_count=int(elite_count),
        seed_assignment_id=seed_assignment_id,
        seed_score=seed_score,
        mutation_radius=mutation_radius,
        round_index=round_index,
    )


def frontier_threshold(observed: list[ObservedPoint], args: argparse.Namespace) -> float | None:
    if not observed:
        return None
    scores = np.asarray([float(item.judge_score if item.judge_score is not None else -1) for item in observed], dtype=np.float32)
    best = float(np.max(scores))
    quantile_value = float(np.quantile(scores, clamp01(float(args.frontier_quantile))))
    near_best = best - float(args.suboptimal_margin)
    return max(float(args.frontier_min_score), quantile_value, near_best)


def basin_radius_limit(phrase_count: int, args: argparse.Namespace) -> int:
    if phrase_count <= 0:
        return 0
    return max(1, min(phrase_count, int(math.ceil(float(args.basin_radius_frac) * phrase_count))))


def compute_basin_richness(members: list[FrontierSeed], phrase_count: int, base: int) -> float:
    if not members or phrase_count <= 0 or base <= 0:
        return 0.0
    if len(members) == 1:
        return 0.0
    counts = np.zeros((phrase_count, base), dtype=np.float32)
    for member in members:
        for position, language_index in enumerate(member.assignment):
            counts[position, language_index] += 1.0
    entropy_values: list[float] = []
    denom = math.log(max(2, base))
    for position in range(phrase_count):
        total = float(np.sum(counts[position]))
        if total <= 0.0:
            entropy_values.append(0.0)
            continue
        probabilities = counts[position] / total
        entropy = 0.0
        for probability in probabilities:
            if probability > 0.0:
                entropy -= float(probability) * math.log(float(probability))
        entropy_values.append(clamp01(entropy / max(denom, 1e-6)))
    pairwise_diversities: list[float] = []
    for left_index in range(len(members)):
        for right_index in range(left_index + 1, len(members)):
            pairwise_diversities.append(
                normalized_hamming_distance(members[left_index].assignment, members[right_index].assignment)
            )
    mean_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
    mean_pairwise = float(np.mean(pairwise_diversities)) if pairwise_diversities else 0.0
    return clamp01((0.65 * mean_entropy) + (0.35 * mean_pairwise))


def choose_basin_representative(members: list[FrontierSeed]) -> FrontierSeed:
    if len(members) == 1:
        return members[0]
    best_member = members[0]
    best_value = None
    for candidate in members:
        centrality = float(
            np.mean(
                [
                    1.0 - normalized_hamming_distance(candidate.assignment, other.assignment)
                    for other in members
                ]
            )
        )
        value = (0.75 * float(candidate.normalized_score)) + (0.25 * centrality)
        if best_value is None or value > best_value:
            best_value = value
            best_member = candidate
    return best_member


def build_basin_archive(
    observed: list[ObservedPoint],
    phrase_count: int,
    base: int,
    round_index: int,
    args: argparse.Namespace,
) -> tuple[list[BasinState], float | None]:
    if not observed:
        return [], None
    threshold = frontier_threshold(observed, args)
    ordered = sorted(
        observed,
        key=lambda item: (
            -(float(item.judge_score) if item.judge_score is not None else -1.0),
            -float(item.search_score),
            int(item.assignment_id),
        ),
    )
    eligible_points = [item for item in ordered if threshold is not None and (item.judge_score or -1) >= threshold]
    if not eligible_points:
        eligible_points = ordered[: max(1, min(len(ordered), int(args.max_basins)))]
    eligible_seeds = [seed_from_observed(item) for item in eligible_points]
    radius_limit = basin_radius_limit(phrase_count, args)

    raw_basins: list[list[FrontierSeed]] = []
    for seed in eligible_seeds:
        best_index = None
        best_distance = None
        for basin_index, members in enumerate(raw_basins):
            distance = min(hamming_distance(seed.assignment, member.assignment) for member in members)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = basin_index
        if best_index is not None and best_distance is not None and best_distance <= radius_limit:
            raw_basins[best_index].append(seed)
            continue
        if len(raw_basins) < int(args.max_basins):
            raw_basins.append([seed])
            continue
        if best_index is not None:
            raw_basins[best_index].append(seed)

    basin_states: list[BasinState] = []
    for members in raw_basins:
        unique_members = list({member.assignment_id: member for member in members}.values())
        unique_members.sort(key=lambda item: (-float(item.score), -float(item.normalized_score), int(item.assignment_id)))
        unique_members = unique_members[: max(1, int(args.frontier_topk))]
        representative = choose_basin_representative(unique_members)
        best_score = float(max(member.score for member in unique_members))
        mean_score = float(np.mean([member.score for member in unique_members])) if unique_members else -1.0
        richness_score = compute_basin_richness(unique_members, phrase_count, base)
        exploration_bonus = clamp01(1.0 / math.sqrt(len(unique_members) + 1.0))
        last_improvement_round = max(member.round_index for member in unique_members if float(member.score) == best_score)
        stagnation_rounds = max(0, int(round_index) - int(last_improvement_round))
        stagnation_penalty = clamp01(stagnation_rounds / max(1.0, float(args.stagnation_widen_after) + 2.0))
        best_norm = clamp01(best_score / max(1.0, float(args.success_score)))
        allocation_weight = max(
            1e-6,
            (0.45 * best_norm) + (0.35 * richness_score) + (0.20 * exploration_bonus) - (0.10 * stagnation_penalty),
        )
        basin_states.append(
            BasinState(
                basin_id=0,
                representative=representative,
                members=tuple(unique_members),
                best_score=best_score,
                mean_score=mean_score,
                richness_score=richness_score,
                exploration_bonus=exploration_bonus,
                allocation_weight=allocation_weight,
                stagnation_rounds=stagnation_rounds,
            )
        )

    basin_states.sort(
        key=lambda item: (
            -float(item.allocation_weight),
            -float(item.best_score),
            -float(item.richness_score),
            int(item.representative.assignment_id),
        )
    )
    ranked_basins: list[BasinState] = []
    for basin_index, basin in enumerate(basin_states, start=1):
        ranked_basins.append(
            BasinState(
                basin_id=int(basin_index),
                representative=basin.representative,
                members=basin.members,
                best_score=basin.best_score,
                mean_score=basin.mean_score,
                richness_score=basin.richness_score,
                exploration_bonus=basin.exploration_bonus,
                allocation_weight=basin.allocation_weight,
                stagnation_rounds=basin.stagnation_rounds,
            )
        )
    return ranked_basins, threshold


def add_proposal_if_new(
    pool: dict[tuple[int, ...], Proposal],
    proposal: Proposal,
    already_evaluated: set[tuple[int, ...]],
) -> None:
    if proposal.assignment in already_evaluated:
        return
    existing = pool.get(proposal.assignment)
    if existing is None or proposal.search_score > existing.search_score:
        pool[proposal.assignment] = proposal


def allocate_basin_targets(total: int, basins: list[BasinState]) -> dict[int, int]:
    if total <= 0 or not basins:
        return {}
    ordered = list(basins)
    counts = {basin.basin_id: 0 for basin in ordered}
    seeded = min(total, len(ordered))
    for basin in ordered[:seeded]:
        counts[basin.basin_id] += 1
    remaining = total - seeded
    if remaining <= 0:
        return counts
    weights = np.asarray([max(1e-6, float(basin.allocation_weight)) for basin in ordered], dtype=np.float64)
    weights /= np.sum(weights)
    raw = weights * remaining
    floors = np.floor(raw).astype(int)
    for basin, floor_value in zip(ordered, floors):
        counts[basin.basin_id] += int(floor_value)
    leftover = remaining - int(np.sum(floors))
    if leftover > 0:
        remainders = [float(value - math.floor(value)) for value in raw]
        ranked = sorted(range(len(ordered)), key=lambda index: (-remainders[index], -ordered[index].allocation_weight, ordered[index].basin_id))
        for index in ranked[:leftover]:
            counts[ordered[index].basin_id] += 1
    return counts


def build_candidate_pool(
    phrase_count: int,
    languages: list[str],
    basins: list[BasinState],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    evaluated: set[tuple[int, ...]],
    current_best_score: float,
    round_index: int,
    args: argparse.Namespace,
    batch_size: int,
    stagnation_rounds: int,
    rng: random.Random,
) -> list[Proposal]:
    base = len(languages)
    if phrase_count <= 0 or base <= 0 or batch_size <= 0:
        return []
    target_pool_size = max(batch_size, int(batch_size) * max(1, int(args.candidate_pool_multiplier)))
    local_target = int(round(target_pool_size * float(args.local_frac))) if basins else 0
    cross_target = int(round(target_pool_size * float(args.cross_basin_frac))) if len(basins) >= 2 else 0
    random_target = int(round(target_pool_size * float(args.random_frac)))
    guided_target = max(0, target_pool_size - local_target - cross_target - random_target)
    basin_recombine_target = 0
    if basins and local_target > 0:
        basin_recombine_target = max(0, local_target // 3)
        local_target -= basin_recombine_target

    pool: dict[tuple[int, ...], Proposal] = {}

    def top_up_random(target_count: int) -> None:
        attempts = 0
        limit = max(50, target_count * 40)
        while len([item for item in pool.values() if item.proposal_source == "random"]) < target_count and attempts < limit:
            attempts += 1
            assignment = random_assignment(phrase_count, base, rng)
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="random",
                round_index=round_index,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=None,
                seed_score=None,
                mutation_radius=None,
            )
            add_proposal_if_new(pool, proposal, evaluated)

    def top_up_guided(target_count: int) -> None:
        attempts = 0
        limit = max(50, target_count * 40)
        while len([item for item in pool.values() if item.proposal_source == "guided"]) < target_count and attempts < limit:
            attempts += 1
            assignment = guided_assignment(
                phrase_count=phrase_count,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                rng=rng,
            )
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="guided",
                round_index=round_index,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=None,
                seed_score=None,
                mutation_radius=None,
            )
            add_proposal_if_new(pool, proposal, evaluated)

    def top_up_local_by_basin(target_count: int) -> None:
        if target_count <= 0 or not basins:
            return
        targets = allocate_basin_targets(target_count, basins)
        basin_by_id = {basin.basin_id: basin for basin in basins}
        for basin_id, basin_target in targets.items():
            basin = basin_by_id[basin_id]
            attempts = 0
            produced = 0
            limit = max(50, basin_target * 50)
            while produced < basin_target and attempts < limit:
                attempts += 1
                seed = choose_basin_member(basin, rng)
                assignment, radius = mutate_frontier_seed(
                    seed=seed,
                    slot_counts=slot_counts,
                    slot_reward_sums=slot_reward_sums,
                    total_evals=total_evals,
                    prior_mean=float(args.slot_prior_mean),
                    prior_strength=float(args.slot_prior_strength),
                    current_best_score=current_best_score,
                    success_score=int(args.success_score),
                    mutation_radius_min=int(args.mutation_radius_min),
                    mutation_radius_max=int(args.mutation_radius_max),
                    stagnation_rounds=max(stagnation_rounds, basin.stagnation_rounds),
                    stagnation_widen_after=int(args.stagnation_widen_after),
                    rng=rng,
                )
                proposal = make_proposal(
                    assignment=assignment,
                    languages=languages,
                    slot_counts=slot_counts,
                    slot_reward_sums=slot_reward_sums,
                    total_evals=total_evals,
                    evaluated=evaluated,
                    current_best_score=current_best_score,
                    success_score=int(args.success_score),
                    prior_mean=float(args.slot_prior_mean),
                    prior_strength=float(args.slot_prior_strength),
                    proposal_source="basin_local",
                    round_index=round_index,
                    basin_id=int(basin.basin_id),
                    secondary_basin_id=None,
                    basin_richness=float(basin.richness_score),
                    basin_allocation_weight=float(basin.allocation_weight),
                    seed_assignment_id=int(seed.assignment_id),
                    seed_score=float(seed.score),
                    mutation_radius=int(radius),
                )
                before_size = len(pool)
                add_proposal_if_new(pool, proposal, evaluated)
                if len(pool) > before_size:
                    produced += 1

    def top_up_basin_recombine(target_count: int) -> None:
        if target_count <= 0 or not basins:
            return
        candidates = [basin for basin in basins if len(basin.members) >= 2]
        if not candidates:
            return
        targets = allocate_basin_targets(target_count, candidates)
        basin_by_id = {basin.basin_id: basin for basin in candidates}
        for basin_id, basin_target in targets.items():
            basin = basin_by_id[basin_id]
            attempts = 0
            produced = 0
            limit = max(50, basin_target * 50)
            while produced < basin_target and attempts < limit:
                attempts += 1
                first_seed = choose_basin_member(basin, rng)
                second_seed = choose_basin_member(basin, rng)
                if first_seed.assignment_id == second_seed.assignment_id and len(basin.members) > 1:
                    continue
                assignment, primary_seed, secondary_seed = recombine_two_seeds(
                    first_seed=first_seed,
                    second_seed=second_seed,
                    slot_counts=slot_counts,
                    slot_reward_sums=slot_reward_sums,
                    total_evals=total_evals,
                    prior_mean=float(args.slot_prior_mean),
                    prior_strength=float(args.slot_prior_strength),
                    rng=rng,
                )
                radius = min(
                    hamming_distance(assignment, primary_seed.assignment),
                    hamming_distance(assignment, secondary_seed.assignment),
                )
                proposal = make_proposal(
                    assignment=assignment,
                    languages=languages,
                    slot_counts=slot_counts,
                    slot_reward_sums=slot_reward_sums,
                    total_evals=total_evals,
                    evaluated=evaluated,
                    current_best_score=current_best_score,
                    success_score=int(args.success_score),
                    prior_mean=float(args.slot_prior_mean),
                    prior_strength=float(args.slot_prior_strength),
                    proposal_source="basin_recombine",
                    round_index=round_index,
                    basin_id=int(basin.basin_id),
                    secondary_basin_id=None,
                    basin_richness=float(basin.richness_score),
                    basin_allocation_weight=float(basin.allocation_weight),
                    seed_assignment_id=int(primary_seed.assignment_id),
                    seed_score=float(primary_seed.score),
                    mutation_radius=int(radius),
                )
                before_size = len(pool)
                add_proposal_if_new(pool, proposal, evaluated)
                if len(pool) > before_size:
                    produced += 1

    def top_up_cross_basin(target_count: int) -> None:
        if target_count <= 0 or len(basins) < 2:
            return
        attempts = 0
        produced = 0
        limit = max(50, target_count * 60)
        while produced < target_count and attempts < limit:
            attempts += 1
            first_basin, second_basin = sample_basin_pair(basins, rng)
            if first_basin.basin_id == second_basin.basin_id:
                continue
            first_seed = choose_basin_member(first_basin, rng)
            second_seed = choose_basin_member(second_basin, rng)
            assignment, primary_seed, secondary_seed = recombine_two_seeds(
                first_seed=first_seed,
                second_seed=second_seed,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                rng=rng,
            )
            radius = min(
                hamming_distance(assignment, primary_seed.assignment),
                hamming_distance(assignment, secondary_seed.assignment),
            )
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="cross_basin",
                round_index=round_index,
                basin_id=int(first_basin.basin_id),
                secondary_basin_id=int(second_basin.basin_id),
                basin_richness=float((first_basin.richness_score + second_basin.richness_score) / 2.0),
                basin_allocation_weight=float((first_basin.allocation_weight + second_basin.allocation_weight) / 2.0),
                seed_assignment_id=int(primary_seed.assignment_id),
                seed_score=float(primary_seed.score),
                mutation_radius=int(radius),
            )
            before_size = len(pool)
            add_proposal_if_new(pool, proposal, evaluated)
            if len(pool) > before_size:
                produced += 1

    top_up_local_by_basin(local_target)
    top_up_basin_recombine(basin_recombine_target)
    top_up_cross_basin(cross_target)
    top_up_guided(guided_target)
    top_up_random(random_target)

    if len(pool) < target_pool_size:
        top_up_guided(target_pool_size - len(pool))
    if len(pool) < target_pool_size:
        top_up_random(target_pool_size - len(pool))

    return list(pool.values())


def build_sg_ce_candidate_pool(
    phrase_count: int,
    languages: list[str],
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    total_evals: int,
    evaluated: set[tuple[int, ...]],
    current_best_score: float,
    round_index: int,
    args: argparse.Namespace,
    batch_size: int,
    ce_distribution: np.ndarray,
    elite_observed: list[ObservedPoint],
    signal_regime: str,
    rng: random.Random,
) -> list[Proposal]:
    base = len(languages)
    if phrase_count <= 0 or base <= 0 or batch_size <= 0:
        return []
    target_pool_size = max(batch_size, int(batch_size) * max(1, int(args.candidate_pool_multiplier)))
    elite_count = len(elite_observed)
    pool: dict[tuple[int, ...], Proposal] = {}

    if signal_regime == "strong_signal":
        conditioned_target = int(round(target_pool_size * float(args.ce_conditioned_frac)))
        random_target = max(1, int(round(target_pool_size * 0.10)))
        guided_target = max(1, int(round(target_pool_size * 0.20)))
    elif signal_regime == "weak_signal":
        conditioned_target = 0
        random_target = max(1, int(round(target_pool_size * 0.20)))
        guided_target = max(1, int(round(target_pool_size * 0.25)))
    else:
        conditioned_target = 0
        random_target = max(1, int(round(target_pool_size * 0.35)))
        guided_target = max(1, int(round(target_pool_size * 0.35)))
    ce_target = max(0, target_pool_size - conditioned_target - random_target - guided_target)

    def add_ce_sample(target_count: int) -> None:
        attempts = 0
        limit = max(50, target_count * 50)
        while len([item for item in pool.values() if item.proposal_source == "ce_sample"]) < target_count and attempts < limit:
            attempts += 1
            assignment = sample_assignment_from_ce(ce_distribution, rng)
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="ce_sample",
                round_index=round_index,
                signal_regime=signal_regime,
                elite_count=elite_count,
                ce_distribution=ce_distribution,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=None,
                seed_score=None,
                mutation_radius=None,
            )
            add_proposal_if_new(pool, proposal, evaluated)

    def add_conditioned(target_count: int) -> None:
        if target_count <= 0 or not elite_observed:
            return
        attempts = 0
        limit = max(50, target_count * 60)
        while len([item for item in pool.values() if item.proposal_source == "ce_conditioned"]) < target_count and attempts < limit:
            attempts += 1
            seed = rng.choice(elite_observed)
            assignment, radius = conditioned_ce_assignment(seed, ce_distribution, signal_regime, rng)
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="ce_conditioned",
                round_index=round_index,
                signal_regime=signal_regime,
                elite_count=elite_count,
                ce_distribution=ce_distribution,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=int(seed.assignment_id),
                seed_score=float(seed.judge_score if seed.judge_score is not None else 0.0),
                mutation_radius=int(radius),
            )
            add_proposal_if_new(pool, proposal, evaluated)

    def add_guided(target_count: int) -> None:
        attempts = 0
        limit = max(50, target_count * 40)
        while len([item for item in pool.values() if item.proposal_source == "guided"]) < target_count and attempts < limit:
            attempts += 1
            assignment = guided_assignment(
                phrase_count=phrase_count,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                rng=rng,
            )
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="guided",
                round_index=round_index,
                signal_regime=signal_regime,
                elite_count=elite_count,
                ce_distribution=ce_distribution,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=None,
                seed_score=None,
                mutation_radius=None,
            )
            add_proposal_if_new(pool, proposal, evaluated)

    def add_random(target_count: int) -> None:
        attempts = 0
        limit = max(50, target_count * 40)
        while len([item for item in pool.values() if item.proposal_source == "random"]) < target_count and attempts < limit:
            attempts += 1
            assignment = random_assignment(phrase_count, base, rng)
            proposal = make_proposal(
                assignment=assignment,
                languages=languages,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=total_evals,
                evaluated=evaluated,
                current_best_score=current_best_score,
                success_score=int(args.success_score),
                prior_mean=float(args.slot_prior_mean),
                prior_strength=float(args.slot_prior_strength),
                proposal_source="random",
                round_index=round_index,
                signal_regime=signal_regime,
                elite_count=elite_count,
                ce_distribution=ce_distribution,
                basin_id=None,
                secondary_basin_id=None,
                basin_richness=None,
                basin_allocation_weight=None,
                seed_assignment_id=None,
                seed_score=None,
                mutation_radius=None,
            )
            add_proposal_if_new(pool, proposal, evaluated)

    add_ce_sample(ce_target)
    add_conditioned(conditioned_target)
    add_guided(guided_target)
    add_random(random_target)
    if len(pool) < target_pool_size:
        add_ce_sample(target_pool_size - len(pool))
    if len(pool) < target_pool_size:
        add_guided(target_pool_size - len(pool))
    if len(pool) < target_pool_size:
        add_random(target_pool_size - len(pool))
    return list(pool.values())


def select_batch_from_pool(pool: list[Proposal], batch_size: int) -> list[Proposal]:
    if not pool or batch_size <= 0:
        return []
    ranked = sorted(
        pool,
        key=lambda item: (
            -float(item.search_score),
            -float(item.exploit_score),
            -float(item.uncertainty_score),
            int(item.assignment_id),
        ),
    )
    if all(item.basin_id is None and item.secondary_basin_id is None for item in ranked):
        selected: list[Proposal] = []
        remaining = list(ranked)
        while remaining and len(selected) < batch_size:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = 0
            best_value = None
            for index, candidate in enumerate(remaining):
                diversity = min(
                    hamming_distance(candidate.assignment, previous.assignment) / float(len(candidate.assignment))
                    for previous in selected
                )
                value = (
                    (0.78 * float(candidate.search_score))
                    + (0.12 * diversity)
                    + (0.10 * clamp01(float(candidate.ce_probability) * max(1, len(candidate.assignment))))
                )
                if best_value is None or value > best_value:
                    best_value = value
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected
    selected: list[Proposal] = []
    remaining = list(ranked)
    represented_basins: set[int] = set()

    basin_first_choices: dict[int, Proposal] = {}
    for proposal in ranked:
        if proposal.basin_id is None or proposal.basin_id in basin_first_choices:
            continue
        basin_first_choices[proposal.basin_id] = proposal
    for proposal in basin_first_choices.values():
        if len(selected) >= batch_size:
            break
        selected.append(proposal)
        remaining = [item for item in remaining if item.assignment_id != proposal.assignment_id]
        if proposal.basin_id is not None:
            represented_basins.add(int(proposal.basin_id))

    while remaining and len(selected) < batch_size:
        if not selected:
            proposal = remaining.pop(0)
            selected.append(proposal)
            if proposal.basin_id is not None:
                represented_basins.add(int(proposal.basin_id))
            continue
        best_index = 0
        best_value = None
        for index, candidate in enumerate(remaining):
            diversity = min(
                hamming_distance(candidate.assignment, previous.assignment) / float(len(candidate.assignment))
                for previous in selected
            )
            basin_bonus = 0.0
            if candidate.basin_id is not None and int(candidate.basin_id) not in represented_basins:
                basin_bonus += 0.08
            if candidate.secondary_basin_id is not None and int(candidate.secondary_basin_id) not in represented_basins:
                basin_bonus += 0.05
            value = (0.80 * float(candidate.search_score)) + (0.15 * diversity) + basin_bonus
            if best_value is None or value > best_value:
                best_value = value
                best_index = index
        proposal = remaining.pop(best_index)
        selected.append(proposal)
        if proposal.basin_id is not None:
            represented_basins.add(int(proposal.basin_id))
        if proposal.secondary_basin_id is not None:
            represented_basins.add(int(proposal.secondary_basin_id))
    return selected


def update_slot_stats(
    slot_counts: np.ndarray,
    slot_reward_sums: np.ndarray,
    assignment: tuple[int, ...],
    normalized_reward: float,
) -> None:
    for position, language_index in enumerate(assignment):
        slot_counts[position, language_index] += 1.0
        slot_reward_sums[position, language_index] += float(normalized_reward)


def build_trace_row(
    row_index: int,
    example_id: str,
    basin_count: int,
    archive_seed_count: int,
    frontier_floor: float | None,
    pool_size: int,
    success_score: int,
    proposal: Proposal,
    observed_index: int,
    result: local.OnlineResult,
) -> dict[str, Any]:
    return {
        "row_index": int(row_index),
        "example_id": str(example_id),
        "round_index": int(proposal.round_index),
        "observed_index": int(observed_index),
        "candidate_index": result.candidate_index,
        "proposal_source": proposal.proposal_source,
        "assignment_id": int(proposal.assignment_id),
        "selected_languages": ",".join(proposal.selected_languages),
        "basin_id": proposal.basin_id,
        "secondary_basin_id": proposal.secondary_basin_id,
        "basin_richness": proposal.basin_richness,
        "basin_allocation_weight": proposal.basin_allocation_weight,
        "search_score": float(proposal.search_score),
        "exploit_score": float(proposal.exploit_score),
        "uncertainty_score": float(proposal.uncertainty_score),
        "diversity_score": float(proposal.diversity_score),
        "ce_probability": float(proposal.ce_probability),
        "signal_regime": str(proposal.signal_regime),
        "elite_count": int(proposal.elite_count),
        "seed_assignment_id": proposal.seed_assignment_id,
        "seed_score": proposal.seed_score,
        "mutation_radius": proposal.mutation_radius,
        "basin_count": int(basin_count),
        "archive_seed_count": int(archive_seed_count),
        "frontier_floor": frontier_floor,
        "pool_size": int(pool_size),
        "target_status": result.target_status,
        "judge_score": result.judge_score,
        "judge_status": result.judge_status,
        "success_hit": int((result.judge_score or 0) >= int(success_score)),
    }


def evaluate_candidate_batch(
    row_index: int,
    example_id: str,
    query: str,
    phrases: list[str],
    total_combo_count: int,
    proposals: list[Proposal],
    basin_count: int,
    archive_seed_count: int,
    frontier_floor: float | None,
    pool_size: int,
    translator: GoogleTranslatePhraseTranslator,
    language_specs: list[Any],
    args: argparse.Namespace,
    global_candidate_index_start: int,
    current_success_hits: int,
) -> tuple[list[ObservedPoint], list[local.LocalCandidate], list[local.OnlineResult], list[dict[str, Any]]]:
    if not proposals:
        return [], [], [], []
    local_candidates: list[local.LocalCandidate] = []
    for offset, proposal in enumerate(proposals):
        local_candidates.append(
            local.LocalCandidate(
                candidate_index=int(global_candidate_index_start + offset),
                assignment_id=int(proposal.assignment_id),
                selected_languages=tuple(str(code) for code in proposal.selected_languages),
                confidence=float(proposal.search_score),
                min_radius_to_seed=None if proposal.mutation_radius is None else int(proposal.mutation_radius),
                source_seed_assignment_id=None if proposal.seed_assignment_id is None else int(proposal.seed_assignment_id),
                source_seed_confidence=None
                if proposal.seed_score is None
                else clamp01(float(proposal.seed_score) / max(1.0, float(args.success_score))),
            )
        )
    candidate_set = local.CandidateSet(
        row_index=int(row_index),
        example_id=str(example_id),
        query=str(query),
        phrases=list(phrases),
        phrase_count=len(phrases),
        total_combo_count=int(total_combo_count),
        requested_m=int(args.final_m),
        selected_count=len(local_candidates),
        seed_count=int(archive_seed_count),
        local_pool_count=int(pool_size),
        best_seed_probability=None if frontier_floor is None else clamp01(float(frontier_floor) / max(1.0, float(args.success_score))),
        best_candidate_probability=float(max(candidate.confidence for candidate in local_candidates)),
        candidates=local_candidates,
    )
    remaining_successes = max(0, int(args.target_success_count) - int(current_success_hits))
    batch_results = local.score_candidate_set(
        candidate_set=candidate_set,
        translator=translator,
        language_specs=language_specs,
        target_model=args.target_model,
        judge_model=args.judge_model,
        judge_behavior=args.judge_behavior,
        success_score=int(args.success_score),
        score_all_candidates=remaining_successes > 1,
        dry_run=bool(args.dry_run),
        block_on_error=not bool(args.fail_fast),
        error_retry_sleep_seconds=float(args.error_retry_sleep_seconds),
        error_context=(
            "zero_history_batch_score "
            f"row_index={row_index} example_id={example_id} "
            f"round_start_candidate={global_candidate_index_start}"
        ),
    )
    proposal_by_assignment_id = {int(proposal.assignment_id): proposal for proposal in proposals}
    observed_points: list[ObservedPoint] = []
    evaluated_candidates: list[local.LocalCandidate] = []
    trace_rows: list[dict[str, Any]] = []
    for observed_offset, result in enumerate(batch_results, start=0):
        if result.assignment_id is None:
            continue
        proposal = proposal_by_assignment_id.get(int(result.assignment_id))
        if proposal is None:
            continue
        normalized_reward = 0.0
        if result.judge_score is not None:
            normalized_reward = clamp01(float(result.judge_score) / max(1.0, float(args.success_score)))
        observed_index = int(global_candidate_index_start + observed_offset)
        observed_points.append(
            ObservedPoint(
                assignment=proposal.assignment,
                assignment_id=int(proposal.assignment_id),
                judge_score=result.judge_score,
                normalized_reward=normalized_reward,
                round_index=int(proposal.round_index),
                observed_index=observed_index,
                proposal_source=proposal.proposal_source,
                basin_id=proposal.basin_id,
                secondary_basin_id=proposal.secondary_basin_id,
                basin_richness=proposal.basin_richness,
                search_score=float(proposal.search_score),
                exploit_score=float(proposal.exploit_score),
                uncertainty_score=float(proposal.uncertainty_score),
                diversity_score=float(proposal.diversity_score),
                ce_probability=float(proposal.ce_probability),
                signal_regime=str(proposal.signal_regime),
                elite_count=int(proposal.elite_count),
                seed_assignment_id=proposal.seed_assignment_id,
                seed_score=proposal.seed_score,
                mutation_radius=proposal.mutation_radius,
                result=result,
            )
        )
        evaluated_candidates.append(local_candidates[observed_offset])
        trace_rows.append(
            build_trace_row(
                row_index=row_index,
                example_id=example_id,
                basin_count=basin_count,
                archive_seed_count=archive_seed_count,
                frontier_floor=frontier_floor,
                pool_size=pool_size,
                success_score=int(args.success_score),
                proposal=proposal,
                observed_index=observed_index,
                result=result,
            )
        )
    return observed_points, evaluated_candidates, batch_results, trace_rows


def run_query_search(
    row_index: int,
    row: dict[str, str],
    segmenter: Any,
    translator: GoogleTranslatePhraseTranslator,
    language_specs: list[Any],
    args: argparse.Namespace,
) -> tuple[local.CandidateSet, list[local.OnlineResult], list[dict[str, Any]], QuerySearchStats]:
    query = str(row.get("source_query") or row.get("query") or "").strip()
    example_id = str(row.get("example_id") or row.get("id") or row_index)
    phrases = segmenter.segment(query)
    phrase_count = len(phrases)
    base = len(args.languages)
    total_combo_count = base**phrase_count if phrase_count > 0 else 0
    if phrase_count <= 0 or total_combo_count <= 0:
        candidate_set = local.CandidateSet(
            row_index=int(row_index),
            example_id=example_id,
            query=query,
            phrases=list(phrases),
            phrase_count=phrase_count,
            total_combo_count=total_combo_count,
            requested_m=int(args.final_m),
            selected_count=0,
            seed_count=0,
            local_pool_count=0,
            best_seed_probability=None,
            best_candidate_probability=None,
            candidates=[],
        )
        results = local.score_candidate_set(
            candidate_set=candidate_set,
            translator=translator,
            language_specs=language_specs,
            target_model=args.target_model,
            judge_model=args.judge_model,
            judge_behavior=args.judge_behavior,
            success_score=int(args.success_score),
            score_all_candidates=False,
            dry_run=bool(args.dry_run),
        )
        return (
            candidate_set,
            results,
            [],
            QuerySearchStats(
                row_index=int(row_index),
                example_id=example_id,
                phrase_count=phrase_count,
                total_combo_count=total_combo_count,
                rounds_run=0,
                evaluated_count=0,
                success_count_at_threshold=0,
                best_judge_score=None,
                first_success_candidate_index=None,
                archive_seed_peak_size=0,
                basin_peak_count=0,
                peak_elite_count=0,
                final_signal_regime="empty_query",
                pool_peak_size=0,
                stopped_reason="empty_query",
            ),
        )

    rng = random.Random(int(args.seed) + (int(row_index) * 1_000_003))
    slot_counts = np.zeros((phrase_count, base), dtype=np.float32)
    slot_reward_sums = np.zeros((phrase_count, base), dtype=np.float32)
    ce_distribution = build_uniform_ce_distribution(phrase_count, base)
    observed_points: list[ObservedPoint] = []
    evaluated_results: list[local.OnlineResult] = []
    evaluated_candidates: list[local.LocalCandidate] = []
    trace_rows: list[dict[str, Any]] = []
    evaluated_assignments: set[tuple[int, ...]] = set()
    success_hits = 0
    current_best_score = 0.0
    candidate_index_cursor = 1
    rounds_run = 0
    stagnation_rounds = 0
    archive_seed_peak_size = 0
    basin_peak_count = 0
    peak_elite_count = 0
    pool_peak_size = 0
    first_success_candidate_index: int | None = None
    final_signal_regime = "no_signal"
    stopped_reason = "budget_exhausted"

    while len(evaluated_results) < int(args.final_m):
        if success_hits >= int(args.target_success_count):
            stopped_reason = "target_success_count_reached"
            break
        rounds_run += 1
        round_batch_size = int(args.bootstrap_size) if not observed_points else int(args.round_batch_size)
        remaining_budget = int(args.final_m) - len(evaluated_results)
        batch_size = min(round_batch_size, remaining_budget)
        basins: list[BasinState] = []
        frontier_floor = None
        archive_seed_count = 0
        if str(args.search_strategy) == "legacy_multi_basin":
            basins, frontier_floor = build_basin_archive(
                observed=observed_points,
                phrase_count=phrase_count,
                base=base,
                round_index=rounds_run,
                args=args,
            )
            basin_peak_count = max(basin_peak_count, len(basins))
            archive_seed_count = sum(len(basin.members) for basin in basins)
            archive_seed_peak_size = max(archive_seed_peak_size, archive_seed_count)
            pool = build_candidate_pool(
                phrase_count=phrase_count,
                languages=list(args.languages),
                basins=basins,
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                total_evals=len(observed_points),
                evaluated=evaluated_assignments,
                current_best_score=current_best_score,
                round_index=rounds_run,
                args=args,
                batch_size=batch_size,
                stagnation_rounds=stagnation_rounds,
                rng=rng,
            )
        else:
            if not observed_points:
                bootstrap_assignments = build_balanced_bootstrap_assignments(
                    phrase_count=phrase_count,
                    base=base,
                    bootstrap_size=batch_size,
                    rng=rng,
                )
                pool = [
                    make_proposal(
                        assignment=assignment,
                        languages=list(args.languages),
                        slot_counts=slot_counts,
                        slot_reward_sums=slot_reward_sums,
                        total_evals=len(observed_points),
                        evaluated=evaluated_assignments,
                        current_best_score=current_best_score,
                        success_score=int(args.success_score),
                        prior_mean=float(args.slot_prior_mean),
                        prior_strength=float(args.slot_prior_strength),
                        proposal_source="bootstrap_balanced",
                        round_index=rounds_run,
                        signal_regime="bootstrap",
                        elite_count=0,
                        ce_distribution=ce_distribution,
                        basin_id=None,
                        secondary_basin_id=None,
                        basin_richness=None,
                        basin_allocation_weight=None,
                        seed_assignment_id=None,
                        seed_score=None,
                        mutation_radius=None,
                    )
                    for assignment in bootstrap_assignments
                ]
            else:
                final_signal_regime = infer_signal_regime(current_best_score)
                ce_distribution, elite_observed = fit_ce_distribution(
                    previous=ce_distribution,
                    observed=observed_points,
                    signal_regime=final_signal_regime,
                    args=args,
                )
                peak_elite_count = max(peak_elite_count, len(elite_observed))
                archive_seed_count = len(elite_observed)
                archive_seed_peak_size = max(archive_seed_peak_size, archive_seed_count)
                frontier_floor = (
                    float(min(item.judge_score for item in elite_observed if item.judge_score is not None))
                    if elite_observed and any(item.judge_score is not None for item in elite_observed)
                    else None
                )
                pool = build_sg_ce_candidate_pool(
                    phrase_count=phrase_count,
                    languages=list(args.languages),
                    slot_counts=slot_counts,
                    slot_reward_sums=slot_reward_sums,
                    total_evals=len(observed_points),
                    evaluated=evaluated_assignments,
                    current_best_score=current_best_score,
                    round_index=rounds_run,
                    args=args,
                    batch_size=batch_size,
                    ce_distribution=ce_distribution,
                    elite_observed=elite_observed,
                    signal_regime=final_signal_regime,
                    rng=rng,
                )
        pool_peak_size = max(pool_peak_size, len(pool))
        selected_batch = select_batch_from_pool(pool, batch_size)
        if not selected_batch:
            stopped_reason = "no_new_candidates"
            break
        new_observed, new_candidates, new_results, new_trace_rows = evaluate_candidate_batch(
            row_index=row_index,
            example_id=example_id,
            query=query,
            phrases=list(phrases),
            total_combo_count=total_combo_count,
            proposals=selected_batch,
            basin_count=len(basins),
            archive_seed_count=archive_seed_count,
            frontier_floor=frontier_floor,
            pool_size=len(pool),
            translator=translator,
            language_specs=language_specs,
            args=args,
            global_candidate_index_start=candidate_index_cursor,
            current_success_hits=success_hits,
        )
        if not new_results:
            stopped_reason = "batch_without_results"
            break
        improved = False
        for observed_point in new_observed:
            evaluated_assignments.add(observed_point.assignment)
            update_slot_stats(
                slot_counts=slot_counts,
                slot_reward_sums=slot_reward_sums,
                assignment=observed_point.assignment,
                normalized_reward=float(observed_point.normalized_reward),
            )
            observed_points.append(observed_point)
            if observed_point.judge_score is not None and observed_point.judge_score > current_best_score:
                current_best_score = float(observed_point.judge_score)
                improved = True
            if observed_point.judge_score is not None and observed_point.judge_score >= int(args.success_score):
                success_hits += 1
                if first_success_candidate_index is None:
                    first_success_candidate_index = int(observed_point.result.candidate_index or 0)
        if improved:
            stagnation_rounds = 0
        else:
            stagnation_rounds += 1
        evaluated_results.extend(new_results)
        evaluated_candidates.extend(new_candidates)
        trace_rows.extend(new_trace_rows)
        candidate_index_cursor += len(new_candidates)

    if str(args.search_strategy) != "legacy_multi_basin":
        final_signal_regime = "bootstrap" if not observed_points else infer_signal_regime(current_best_score)

    if len(evaluated_results) >= int(args.final_m) and success_hits < int(args.target_success_count):
        stopped_reason = "budget_exhausted"

    best_candidate_probability = None
    if evaluated_candidates:
        best_candidate_probability = float(max(candidate.confidence for candidate in evaluated_candidates))
    best_seed_probability = None
    if observed_points:
        best_seed_probability = clamp01(current_best_score / max(1.0, float(args.success_score)))
    candidate_set = local.CandidateSet(
        row_index=int(row_index),
        example_id=example_id,
        query=query,
        phrases=list(phrases),
        phrase_count=phrase_count,
        total_combo_count=total_combo_count,
        requested_m=int(args.final_m),
        selected_count=len(evaluated_candidates),
        seed_count=int(archive_seed_peak_size),
        local_pool_count=int(pool_peak_size),
        best_seed_probability=best_seed_probability,
        best_candidate_probability=best_candidate_probability,
        candidates=evaluated_candidates,
    )
    best_judge_score = None
    scored_values = [item.judge_score for item in observed_points if item.judge_score is not None]
    if scored_values:
        best_judge_score = int(max(scored_values))
    stats = QuerySearchStats(
        row_index=int(row_index),
        example_id=example_id,
        phrase_count=phrase_count,
        total_combo_count=total_combo_count,
        rounds_run=rounds_run,
        evaluated_count=len(evaluated_results),
        success_count_at_threshold=success_hits,
        best_judge_score=best_judge_score,
        first_success_candidate_index=first_success_candidate_index,
        archive_seed_peak_size=archive_seed_peak_size,
        basin_peak_count=basin_peak_count,
        peak_elite_count=peak_elite_count,
        final_signal_regime=final_signal_regime,
        pool_peak_size=pool_peak_size,
        stopped_reason=stopped_reason,
    )
    return candidate_set, evaluated_results, trace_rows, stats


def render_progress(completed_queries: int, total_queries: int, results: list[local.OnlineResult], latest: local.OnlineResult | None) -> None:
    width = 28
    ratio = 0.0 if total_queries <= 0 else completed_queries / total_queries
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    per_query: dict[int, list[local.OnlineResult]] = {}
    for item in results:
        per_query.setdefault(item.row_index, []).append(item)
    success10 = sum(1 for items in per_query.values() if any((item.judge_score or 0) >= 10 for item in items))
    evaluated = sum(1 for item in results if item.candidate_index is not None)
    latest_score = "-" if latest is None or latest.judge_score is None else str(latest.judge_score)
    latest_search = "-" if latest is None or latest.confidence is None else f"{latest.confidence:.3f}"
    line = (
        f"\r[{bar}] {completed_queries}/{total_queries} "
        f"evals={evaluated} success10={success10} "
        f"last_score={latest_score} last_search={latest_search}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def search_stats_to_rows(stats_items: list[QuerySearchStats]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": int(item.row_index),
            "example_id": str(item.example_id),
            "phrase_count": int(item.phrase_count),
            "total_combo_count": int(item.total_combo_count),
            "rounds_run": int(item.rounds_run),
            "evaluated_count": int(item.evaluated_count),
            "success_count_at_threshold": int(item.success_count_at_threshold),
            "best_judge_score": item.best_judge_score,
            "first_success_candidate_index": item.first_success_candidate_index,
            "archive_seed_peak_size": int(item.archive_seed_peak_size),
            "basin_peak_count": int(item.basin_peak_count),
            "peak_elite_count": int(item.peak_elite_count),
            "final_signal_regime": str(item.final_signal_regime),
            "pool_peak_size": int(item.pool_peak_size),
            "stopped_reason": str(item.stopped_reason),
        }
        for item in sorted(stats_items, key=lambda value: value.row_index)
    ]


def save_trace(path: Path, trace_rows: list[dict[str, Any]]) -> None:
    if not trace_rows:
        pd.DataFrame(
            columns=[
                "row_index",
                "example_id",
                "round_index",
                "observed_index",
                "candidate_index",
                "proposal_source",
                "assignment_id",
                "selected_languages",
                "basin_id",
                "secondary_basin_id",
                "basin_richness",
                "basin_allocation_weight",
                "search_score",
                "exploit_score",
                "uncertainty_score",
                "diversity_score",
                "ce_probability",
                "signal_regime",
                "elite_count",
                "seed_assignment_id",
                "seed_score",
                "mutation_radius",
                "basin_count",
                "archive_seed_count",
                "frontier_floor",
                "pool_size",
                "target_status",
                "judge_score",
                "judge_status",
                "success_hit",
            ]
        ).to_csv(path, index=False)
        return
    pd.DataFrame(sorted(trace_rows, key=lambda row: (row["row_index"], row["observed_index"]))).to_csv(path, index=False)


def build_summary(
    args: argparse.Namespace,
    candidate_sets: list[local.CandidateSet],
    results: list[local.OnlineResult],
    query_stats: list[QuerySearchStats],
    trace_rows: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    per_query_results: dict[int, list[local.OnlineResult]] = {}
    for item in results:
        per_query_results.setdefault(item.row_index, []).append(item)
    evaluated_results = [item for item in results if item.candidate_index is not None]
    score_counts = Counter(int(item.judge_score) for item in evaluated_results if item.judge_score is not None)
    success_query_count = sum(
        1 for items in per_query_results.values() if any((item.judge_score or 0) >= int(args.success_score) for item in items)
    )
    sample_success_count = sum(1 for item in evaluated_results if (item.judge_score or 0) >= int(args.success_score))
    proposal_counts = Counter(str(row.get("proposal_source") or "") for row in trace_rows)
    proposal_success_counts = Counter(
        str(row.get("proposal_source") or "") for row in trace_rows if int(row.get("success_hit") or 0) > 0
    )
    query_count = len(candidate_sets)
    mean_first_success_index = float(
        np.mean([item.first_success_candidate_index for item in query_stats if item.first_success_candidate_index is not None])
    ) if any(item.first_success_candidate_index is not None for item in query_stats) else None
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "algorithm": ALGORITHM_LABEL_LEGACY if str(args.search_strategy) == "legacy_multi_basin" else ALGORITHM_LABEL_SG_CE_V1,
        "search_strategy": str(args.search_strategy),
        "input": str(resolve_cli_path(args.input)),
        "bucket_manifest": str(getattr(args, "_bucket_manifest_path", "")) or None,
        "pmax": int(args.pmax) if args.pmax is not None else None,
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "judge_behavior": args.judge_behavior,
        "languages": list(args.languages),
        "source_language": args.source_language,
        "final_m": int(args.final_m),
        "bootstrap_size": int(args.bootstrap_size),
        "round_batch_size": int(args.round_batch_size),
        "target_success_count": int(args.target_success_count),
        "success_score": int(args.success_score),
        "candidate_pool_multiplier": int(args.candidate_pool_multiplier),
        "frontier_topk": int(args.frontier_topk),
        "frontier_quantile": float(args.frontier_quantile),
        "frontier_min_score": float(args.frontier_min_score),
        "suboptimal_margin": float(args.suboptimal_margin),
        "max_basins": int(args.max_basins),
        "basin_radius_frac": float(args.basin_radius_frac),
        "mutation_radius_min": int(args.mutation_radius_min),
        "mutation_radius_max": int(args.mutation_radius_max),
        "local_frac": float(args.local_frac),
        "cross_basin_frac": float(args.cross_basin_frac),
        "random_frac": float(args.random_frac),
        "guided_frac": float(max(0.0, 1.0 - float(args.local_frac) - float(args.cross_basin_frac) - float(args.random_frac))),
        "slot_prior_mean": float(args.slot_prior_mean),
        "slot_prior_strength": float(args.slot_prior_strength),
        "stagnation_widen_after": int(args.stagnation_widen_after),
        "ce_exploration_floor": float(args.ce_exploration_floor),
        "ce_no_signal_elite_frac": float(args.ce_no_signal_elite_frac),
        "ce_weak_signal_elite_frac": float(args.ce_weak_signal_elite_frac),
        "ce_strong_signal_elite_frac": float(args.ce_strong_signal_elite_frac),
        "ce_no_signal_alpha": float(args.ce_no_signal_alpha),
        "ce_weak_signal_alpha": float(args.ce_weak_signal_alpha),
        "ce_strong_signal_alpha": float(args.ce_strong_signal_alpha),
        "ce_conditioned_frac": float(args.ce_conditioned_frac),
        "score_workers": int(args.score_workers),
        "error_retry_sleep_seconds": float(args.error_retry_sleep_seconds),
        "fail_fast": bool(args.fail_fast),
        "dry_run": bool(args.dry_run),
        "save_text": bool(args.save_text),
        "query_count_loaded": int(getattr(args, "_input_query_count", query_count)),
        "query_count_after_filter": int(getattr(args, "_filtered_query_count", query_count)),
        "query_count_total": int(query_count),
        "candidate_query_count": int(sum(1 for item in candidate_sets if item.selected_count > 0)),
        "evaluated_candidate_count": int(len(evaluated_results)),
        "sample_success_count_at_target": int(sample_success_count),
        "query_success_count_at_target": int(success_query_count),
        "query_success_rate_at_target": (success_query_count / query_count) if query_count else 0.0,
        "sample_precision_at_target": (sample_success_count / len(evaluated_results)) if evaluated_results else 0.0,
        "mean_evaluated_per_query": float(np.mean([item.evaluated_count for item in query_stats])) if query_stats else 0.0,
        "mean_rounds_per_query": float(np.mean([item.rounds_run for item in query_stats])) if query_stats else 0.0,
        "mean_archive_seed_peak_size": float(np.mean([item.archive_seed_peak_size for item in query_stats])) if query_stats else 0.0,
        "mean_basin_peak_count": float(np.mean([item.basin_peak_count for item in query_stats])) if query_stats else 0.0,
        "mean_peak_elite_count": float(np.mean([item.peak_elite_count for item in query_stats])) if query_stats else 0.0,
        "mean_pool_peak_size": float(np.mean([item.pool_peak_size for item in query_stats])) if query_stats else 0.0,
        "mean_first_success_candidate_index": mean_first_success_index,
        "score_counts": dict(sorted(score_counts.items())),
        "proposal_source_counts": dict(sorted(proposal_counts.items())),
        "proposal_source_success_counts": dict(sorted(proposal_success_counts.items())),
        "final_signal_regime_counts": dict(sorted(Counter(item.final_signal_regime for item in query_stats).items())),
        "stop_reasons": dict(sorted(Counter(item.stopped_reason for item in query_stats).items())),
    }


def persist_artifacts(
    *,
    args: argparse.Namespace,
    candidates_path: Path,
    detail_path: Path,
    trace_path: Path,
    search_stats_path: Path,
    summary_path: Path,
    candidate_sets: list[local.CandidateSet],
    query_stats: list[QuerySearchStats],
    results: list[local.OnlineResult],
    trace_rows: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    pd.DataFrame(local.candidate_set_to_rows(candidate_sets, save_text=bool(args.save_text))).to_csv(
        candidates_path,
        index=False,
    )
    local.save_detail(detail_path, results, save_text=bool(args.save_text))
    save_trace(trace_path, trace_rows)
    pd.DataFrame(search_stats_to_rows(query_stats)).to_csv(search_stats_path, index=False)
    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = build_summary(
        args=args,
        candidate_sets=candidate_sets,
        results=results,
        query_stats=query_stats,
        trace_rows=trace_rows,
        started_at=started_at,
        finished_at=finished_at,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if int(args.final_m) <= 0:
        raise ValueError("--final-m must be positive.")
    if args.pmax is not None and int(args.pmax) <= 0:
        raise ValueError("--pmax must be positive.")
    if int(args.bootstrap_size) <= 0 or int(args.round_batch_size) <= 0:
        raise ValueError("batch sizes must be positive.")
    if int(args.target_success_count) <= 0:
        raise ValueError("--target-success-count must be positive.")
    if int(args.success_score) <= 0:
        raise ValueError("--success-score must be positive.")
    if int(args.candidate_pool_multiplier) <= 0:
        raise ValueError("--candidate-pool-multiplier must be positive.")
    if int(args.frontier_topk) <= 0:
        raise ValueError("--frontier-topk must be positive.")
    if int(args.max_basins) <= 0:
        raise ValueError("--max-basins must be positive.")
    if not 0.0 <= float(args.frontier_quantile) <= 1.0:
        raise ValueError("--frontier-quantile must be in [0,1].")
    if not 0.0 < float(args.basin_radius_frac) <= 1.0:
        raise ValueError("--basin-radius-frac must be in (0,1].")
    if int(args.mutation_radius_min) <= 0 or int(args.mutation_radius_max) <= 0:
        raise ValueError("mutation radii must be positive.")
    if int(args.mutation_radius_min) > int(args.mutation_radius_max):
        raise ValueError("--mutation-radius-min must be <= --mutation-radius-max.")
    if not 0.0 <= float(args.local_frac) <= 1.0:
        raise ValueError("--local-frac must be in [0,1].")
    if not 0.0 <= float(args.cross_basin_frac) <= 1.0:
        raise ValueError("--cross-basin-frac must be in [0,1].")
    if not 0.0 <= float(args.random_frac) <= 1.0:
        raise ValueError("--random-frac must be in [0,1].")
    if float(args.local_frac) + float(args.cross_basin_frac) + float(args.random_frac) > 1.0:
        raise ValueError("--local-frac + --cross-basin-frac + --random-frac must be <= 1.")
    if not 0.0 <= float(args.ce_exploration_floor) <= 1.0:
        raise ValueError("--ce-exploration-floor must be in [0,1].")
    for name in [
        "ce_no_signal_elite_frac",
        "ce_weak_signal_elite_frac",
        "ce_strong_signal_elite_frac",
        "ce_no_signal_alpha",
        "ce_weak_signal_alpha",
        "ce_strong_signal_alpha",
        "ce_conditioned_frac",
    ]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1].")
    if float(args.slot_prior_strength) <= 0.0:
        raise ValueError("--slot-prior-strength must be positive.")
    if float(args.error_retry_sleep_seconds) <= 0.0:
        raise ValueError("--error-retry-sleep-seconds must be positive.")
    if args.bucket_manifest is None and (args.pmax is not None):
        raise ValueError("--pmax requires --bucket-manifest.")

    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir = (
        resolve_cli_path(args.output_dir)
        if args.output_dir
        else default_output_dir()
    )
    ensure_output_dir(output_dir, allow_overwrite=bool(args.allow_overwrite))
    candidates_path = output_dir / "candidates.csv"
    detail_path = output_dir / "detail.csv"
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "search_trace.csv"
    search_stats_path = output_dir / "search_stats.csv"

    eval_rows = run_with_blocking_retry(
        lambda: filter_eval_rows_with_manifest(
            local.load_csv_rows(resolve_cli_path(args.input), max_samples=args.max_samples),
            args,
        ),
        context="input_load_and_filter",
        fail_fast=bool(args.fail_fast),
        retry_sleep_seconds=float(args.error_retry_sleep_seconds),
    )
    if not eval_rows:
        raise RuntimeError("No queries remain after applying manifest/P_max filtering.")
    language_specs = get_language_specs(list(args.languages))
    segmenter = local.build_segmenter(args.segment_backend, args.spacy_model, int(args.phrase_max_tokens))
    translator = GoogleTranslatePhraseTranslator(
        source_language=str(args.source_language),
        timeout=float(args.translate_timeout),
        max_retries=int(args.translate_retries),
        language_specs=language_specs,
    )

    candidate_sets_by_index: list[local.CandidateSet | None] = [None] * len(eval_rows)
    query_stats_by_index: list[QuerySearchStats | None] = [None] * len(eval_rows)
    results: list[local.OnlineResult] = []
    trace_rows: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    total_queries = len(eval_rows)
    print(
        "Running zero-history online search: "
        f"strategy={args.search_strategy} "
        f"queries={total_queries} final_m={args.final_m} bootstrap={args.bootstrap_size} "
        f"round_batch={args.round_batch_size} max_basins={args.max_basins} "
        f"pmax={args.pmax if args.pmax is not None else '-'} "
        f"block_on_error={int(not bool(args.fail_fast))} output_dir={output_dir}",
        flush=True,
    )
    render_progress(0, total_queries, results, None)

    def run_query_search_with_retry(row_index: int, row: dict[str, str]):
        example_id = normalize_example_id(row.get("example_id") or row.get("id") or row_index)
        return run_with_blocking_retry(
            lambda: run_query_search(
                row_index,
                row,
                segmenter,
                translator,
                language_specs,
                args,
            ),
            context=f"query_search row_index={row_index} example_id={example_id}",
            fail_fast=bool(args.fail_fast),
            retry_sleep_seconds=float(args.error_retry_sleep_seconds),
        )

    with ThreadPoolExecutor(max_workers=max(1, int(args.score_workers))) as executor:
        future_to_index = {
            executor.submit(
                run_query_search_with_retry,
                row_index,
                row,
            ): row_index
            for row_index, row in enumerate(eval_rows)
        }
        for future in as_completed(future_to_index):
            row_index = future_to_index[future]
            candidate_set, query_results, query_trace_rows, query_stats = future.result()
            with results_lock:
                candidate_sets_by_index[row_index] = candidate_set
                query_stats_by_index[row_index] = query_stats
                results.extend(query_results)
                trace_rows.extend(query_trace_rows)
                ready_candidate_sets = [item for item in candidate_sets_by_index if item is not None]
                ready_query_stats = [item for item in query_stats_by_index if item is not None]
                run_with_blocking_retry(
                    lambda: persist_artifacts(
                        args=args,
                        candidates_path=candidates_path,
                        detail_path=detail_path,
                        trace_path=trace_path,
                        search_stats_path=search_stats_path,
                        summary_path=summary_path,
                        candidate_sets=ready_candidate_sets,
                        query_stats=ready_query_stats,
                        results=results,
                        trace_rows=trace_rows,
                        started_at=started_at,
                    ),
                    context=f"artifact_checkpoint row_index={row_index}",
                    fail_fast=bool(args.fail_fast),
                    retry_sleep_seconds=float(args.error_retry_sleep_seconds),
                )
                latest = query_results[-1] if query_results else None
                render_progress(len(ready_candidate_sets), total_queries, results, latest)

    sys.stdout.write("\n")
    final_candidate_sets = [item for item in candidate_sets_by_index if item is not None]
    final_query_stats = [item for item in query_stats_by_index if item is not None]
    summary = run_with_blocking_retry(
        lambda: persist_artifacts(
            args=args,
            candidates_path=candidates_path,
            detail_path=detail_path,
            trace_path=trace_path,
            search_stats_path=search_stats_path,
            summary_path=summary_path,
            candidate_sets=final_candidate_sets,
            query_stats=final_query_stats,
            results=results,
            trace_rows=trace_rows,
            started_at=started_at,
        ),
        context="artifact_final_write",
        fail_fast=bool(args.fail_fast),
        retry_sleep_seconds=float(args.error_retry_sleep_seconds),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved candidates: {candidates_path}", flush=True)
    print(f"Saved detail: {detail_path}", flush=True)
    print(f"Saved search trace: {trace_path}", flush=True)
    print(f"Saved search stats: {search_stats_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
