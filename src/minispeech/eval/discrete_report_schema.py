"""Versioned, fail-closed schema helpers for formal D1c discrete-LM evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_CANDIDATE_STEP_NAME_RE = re.compile(r"candidate_step_(\d+)\.pt$", re.IGNORECASE)

REPORT_SCHEMA = "minispeech.discrete_eval_report"
REPORT_SCHEMA_VERSION = "2.0.0"
DECODE_BUNDLE_SCHEMA = "minispeech.discrete_decode_bundle"
DECODE_BUNDLE_SCHEMA_VERSION = "2.0.0"
CONDITIONS = (
    "model",
    "copy_synth",
    "wrong_utt",
    "random_codes",
    "shuffle_text",
    "mask_text",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON shared by all v2 hashes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any, *, exclude: str | Sequence[str] = ()) -> str:
    """Hash a JSON value after excluding top-level self-hash fields."""
    omitted = {exclude} if isinstance(exclude, str) else set(exclude)
    body = {k: v for k, v in value.items() if k not in omitted} if isinstance(value, Mapping) else value
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_utt_ids_sha256(utt_ids: Sequence[str]) -> str:
    return sha256_json([str(uid) for uid in utt_ids])


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([dict(row) for row in rows])


def _fail(message: str) -> None:
    raise ValueError(f"formal D1c report refused: {message}")


def _require_mapping(obj: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        _fail(f"{label} must be an object")
    return obj


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} missing")
    return value


def validate_bound_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_utt_ids: Sequence[str],
    *,
    condition: str = "condition",
    require_artifacts: bool = True,
) -> None:
    """Prove exact ordered binding; no missing/extra/duplicate/reordered rows."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        _fail(f"{condition}.rows must be an array")
    expected = [str(uid) for uid in expected_utt_ids]
    actual: list[str] = []
    for ordinal, row in enumerate(rows):
        r = _require_mapping(row, f"{condition}.rows[{ordinal}]")
        if r.get("ordinal") != ordinal:
            _fail(f"{condition}.rows[{ordinal}].ordinal must equal {ordinal}")
        uid = _require_nonempty_str(r.get("utt_id"), f"{condition}.rows[{ordinal}].utt_id")
        actual.append(uid)
        if require_artifacts:
            for key in ("normalized_reference_sha256", "generated_codes_sha256"):
                _require_nonempty_str(r.get(key), f"{condition}.rows[{ordinal}].{key}")
            wav = _require_mapping(r.get("wav"), f"{condition}.rows[{ordinal}].wav")
            _require_nonempty_str(wav.get("path"), f"{condition}.rows[{ordinal}].wav.path")
            _require_nonempty_str(wav.get("file_sha256"), f"{condition}.rows[{ordinal}].wav.file_sha256")
            _require_nonempty_str(wav.get("pcm_sha256"), f"{condition}.rows[{ordinal}].wav.pcm_sha256")
            _require_mapping(r.get("generation"), f"{condition}.rows[{ordinal}].generation")
            if r.get("cer") is None:
                _fail(f"{condition}.rows[{ordinal}].cer missing")
    if len(actual) != len(set(actual)):
        _fail(f"{condition}.rows contain duplicate utt_id")
    if actual != expected:
        _fail(f"{condition}.rows utt_ids differ from expected ordered utt_ids")


def _validate_condition(name: str, payload: Mapping[str, Any], expected_utt_ids: Sequence[str]) -> None:
    _require_nonempty_str(payload.get("subset_key"), f"conditions.{name}.subset_key")
    _require_nonempty_str(payload.get("subset_path"), f"conditions.{name}.subset_path")
    _require_nonempty_str(payload.get("subset_file_sha256"), f"conditions.{name}.subset_file_sha256")
    if payload.get("expected_count") != len(expected_utt_ids):
        _fail(f"conditions.{name}.expected_count mismatch")
    expected_hash = ordered_utt_ids_sha256(expected_utt_ids)
    if payload.get("ordered_utt_ids_sha256") != expected_hash:
        _fail(f"conditions.{name}.ordered_utt_ids_sha256 mismatch")
    if payload.get("evaluated_count") != len(expected_utt_ids):
        _fail(f"conditions.{name}.evaluated_count mismatch")
    rows = payload.get("rows")
    validate_bound_rows(rows, expected_utt_ids, condition=f"conditions.{name}")
    if payload.get("rows_sha256") != rows_sha256(rows):
        _fail(f"conditions.{name}.rows_sha256 mismatch")


def validate_formal_report(
    report: Mapping[str, Any],
    *,
    expected_utt_ids: Sequence[str] | None = None,
    require_formal: bool = True,
    allowed_candidate_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate the complete evaluator-produced v2 report, fail closed."""
    data = dict(_require_mapping(report, "report"))
    if data.get("schema") != REPORT_SCHEMA or data.get("schema_version") != REPORT_SCHEMA_VERSION:
        _fail("wrong report schema/version")
    if require_formal and data.get("formal_eligible") is not True:
        _fail("formal_eligible must be true")
    for key in ("checkpoint", "provenance", "infer", "teacher_forced_metrics", "conditions"):
        _require_mapping(data.get(key), key)
    checkpoint = _require_mapping(data["checkpoint"], "checkpoint")
    for key in ("path", "sha256", "step", "schema", "schema_version"):
        if checkpoint.get(key) in (None, ""):
            _fail(f"checkpoint.{key} missing")
    step = checkpoint.get("step")
    # bool is a subclass of int — reject True/False explicitly.
    if isinstance(step, bool) or not isinstance(step, int) or int(step) <= 0:
        _fail("checkpoint.step must be a positive integer (not 0/missing)")
    allowed_steps = None
    if allowed_candidate_steps is not None:
        allowed_steps = [int(s) for s in allowed_candidate_steps]
    if allowed_steps is not None and int(step) not in allowed_steps:
        _fail(f"checkpoint.step={step} not in allowed candidate_steps={allowed_steps}")
    ckpt_path = Path(str(checkpoint["path"]))
    name_match = _CANDIDATE_STEP_NAME_RE.search(ckpt_path.name)
    if name_match is not None and int(name_match.group(1)) != int(step):
        _fail(
            f"checkpoint.path step={int(name_match.group(1))} != checkpoint.step={int(step)}"
        )
    # Trainer-stamped run identity: required so filename alone cannot invent a step.
    hashes = checkpoint.get("hashes")
    if not isinstance(hashes, Mapping):
        _fail("checkpoint.hashes missing")
    for key in (
        "eval_contract",
        "recipe",
        "recipe_source_manifest",
        "run_manifest",
    ):
        value = hashes.get(key)
        if value is None or not str(value).strip() or str(value) == ("0" * 64):
            _fail(f"checkpoint.hashes.{key} missing")
    # When the checkpoint file is reachable, claimed sha256 must match on-disk bytes.
    if ckpt_path.is_file():
        live = file_sha256(ckpt_path)
        claimed = str(checkpoint["sha256"])
        if live != claimed:
            _fail(
                f"checkpoint.sha256 does not match file at checkpoint.path "
                f"(claimed={claimed} live={live})"
            )
    tf = _require_mapping(data["teacher_forced_metrics"], "teacher_forced_metrics")
    if tf.get("teacher_forced_ce_source") != "eval_computed":
        _fail("teacher_forced_ce_source must be eval_computed")
    total_ce = tf.get("total_ce")
    if not isinstance(total_ce, (int, float)) or not math.isfinite(float(total_ce)):
        _fail("teacher_forced_metrics.total_ce must be finite")

    binding = _require_mapping(data.get("row_binding"), "row_binding")
    report_ids = binding.get("ordered_utt_ids")
    if not isinstance(report_ids, list) or not report_ids:
        _fail("row_binding.ordered_utt_ids missing")
    report_ids = [str(uid) for uid in report_ids]
    if len(report_ids) != len(set(report_ids)):
        _fail("row_binding.ordered_utt_ids contains duplicates")
    if expected_utt_ids is not None and report_ids != [str(uid) for uid in expected_utt_ids]:
        _fail("report ordered utt_ids differ from expected ordered utt_ids")
    if binding.get("ordered_utt_ids_sha256") != ordered_utt_ids_sha256(report_ids):
        _fail("row_binding.ordered_utt_ids_sha256 mismatch")

    conditions = _require_mapping(data["conditions"], "conditions")
    missing = set(CONDITIONS) - set(conditions)
    extra = set(conditions) - set(CONDITIONS)
    if missing or extra:
        _fail(f"condition set mismatch missing={sorted(missing)} extra={sorted(extra)}")
    for name in CONDITIONS:
        _validate_condition(name, _require_mapping(conditions[name], f"conditions.{name}"), report_ids)

    if data.get("report_sha256") != sha256_json(data, exclude="report_sha256"):
        _fail("report_sha256 mismatch")
    return data


def build_formal_report(**fields: Any) -> dict[str, Any]:
    """Stamp a v2 report after validating its exact row binding."""
    report = copy.deepcopy(fields)
    report["schema"] = REPORT_SCHEMA
    report["schema_version"] = REPORT_SCHEMA_VERSION
    conditions = report.get("conditions") or {}
    for condition in CONDITIONS:
        row_block = conditions.get(condition)
        if isinstance(row_block, dict) and isinstance(row_block.get("rows"), list):
            row_block["rows_sha256"] = rows_sha256(row_block["rows"])
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_json(report)
    validate_formal_report(report)
    return report


def validate_decode_bundle(bundle: Mapping[str, Any], *, expected_utt_ids: Sequence[str] | None = None) -> dict[str, Any]:
    data = dict(_require_mapping(bundle, "decode bundle"))
    if data.get("schema") != DECODE_BUNDLE_SCHEMA or data.get("schema_version") != DECODE_BUNDLE_SCHEMA_VERSION:
        _fail("wrong decode bundle schema/version")
    if data.get("formal_eligible") is not True:
        _fail("decode bundle formal_eligible must be true")
    if data.get("bundle_sha256") != sha256_json(data, exclude="bundle_sha256"):
        _fail("decode bundle hash mismatch")
    if expected_utt_ids is not None:
        ids = data.get("ordered_utt_ids")
        if ids != [str(x) for x in expected_utt_ids]:
            _fail("decode bundle ordered utt_ids mismatch")
    return data


def build_decode_bundle(**fields: Any) -> dict[str, Any]:
    bundle = copy.deepcopy(fields)
    bundle["schema"] = DECODE_BUNDLE_SCHEMA
    bundle["schema_version"] = DECODE_BUNDLE_SCHEMA_VERSION
    bundle.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = sha256_json(bundle)
    validate_decode_bundle(bundle)
    return bundle
