"""
ablation_analysis.py
作者：Accilia
创建时间：2026-04-07
用途说明：
分析不同消融实验的结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
ABLATION_ROOT = BASE_DIR / "qa_eval_ablation"
FULL_ROOT = BASE_DIR / "qa_eval_output" / "glm-4-plus"
OUTPUT_DIR = BASE_DIR / "analysis_tools" / "data" / "ablation"

GROUP_MAPPING = {
    "A1": "no_rewrite",
    "A2": "no_cls",
    "A3": "no_cls_rewrite",
    "A4": "no_sparse",
    "A5": "no_rerank",
    "A6": "no_reverse_verification",
}

JSON_PATTERNS = (
    "qa_eval_results_*.json",
    "qa_intermediate_results_*.json",
)

SEMANTIC_PATH_CANDIDATES = {
    "query": ("query",),
    "metrics": ("metrics",),
    "timing": ("timing",),
    "classification": (
        "classification_result",
        "metrics.classification_accuracy",
        "metrics.domain_correct",
        "metrics.label_correct",
    ),
    "domain": (
        "metrics.ground_truth_domain",
        "metrics.predicted_domain",
        "classification_result.domain_name_zh",
        "classification_result.domain_id",
        "retrieval_results.evidence[].domain",
    ),
    "question_type": (
        "question_type",
        "type",
        "metrics.question_type",
        "metrics.query_type",
        "ground_truth.question_type",
        "raw.question_type",
    ),
}

NORMALIZED_FIELD_CANDIDATES = {
    "group_id": (),
    "query": ("query",),
    "domain": (
        "metrics.ground_truth_domain",
        "metrics.predicted_domain",
        "classification_result.domain_name_zh",
        "classification_result.domain_id",
        "retrieval_results.evidence[].domain",
    ),
    "question_type": (
        "question_type",
        "type",
        "metrics.question_type",
        "metrics.query_type",
        "ground_truth.question_type",
        "raw.question_type",
    ),
    "metrics": ("metrics",),
    "timing": ("timing",),
    "classification": ("classification_result",),
}

AGGREGATION_METRICS = {
    "BLEU": "bleu_score",
    "ROUGE-1": "rouge_1_f1",
    "ROUGE-L": "rouge_l_f1",
    "BERTScore": "bert_score_f1",
    "Answer Length": "answer_length",
    "SP_1": "source_precision_sp1",
    "SP_2": "source_precision_sp2",
    "SQC": "source_query_coverage",
    "RP": "response_precision",
    "RQC": "response_query_coverage",
    "RSD": "response_self_distinctness",
    "Groundedness": "groundedness",
}


@dataclass
class AblationAnalysisContext:
    detection_payload: Optional[Dict[str, Any]] = None
    detection_path: Optional[Path] = None
    group_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mapping_payload: Optional[Dict[str, Any]] = None
    mapping_path: Optional[Path] = None
    preview_payload: Optional[Dict[str, Any]] = None
    preview_path: Optional[Path] = None
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )
    overall_summary_payload: Optional[Dict[str, Any]] = None
    overall_summary_json_path: Optional[Path] = None
    overall_summary_excel_path: Optional[Path] = None
    comparison_payload: Optional[Dict[str, Any]] = None
    comparison_json_path: Optional[Path] = None
    comparison_excel_path: Optional[Path] = None
    classification_scope_payload: Optional[Dict[str, Any]] = None
    classification_scope_path: Optional[Path] = None
    classification_analysis_payload: Optional[Dict[str, Any]] = None
    classification_analysis_json_path: Optional[Path] = None
    classification_analysis_excel_path: Optional[Path] = None
    classification_analysis_scope_path: Optional[Path] = None
    a23_domain_thelma_payload: Optional[Dict[str, Any]] = None
    a23_domain_thelma_json_path: Optional[Path] = None
    a23_domain_thelma_excel_path: Optional[Path] = None
    a23_domain_thelma_scope_path: Optional[Path] = None


def sorted_existing_files(directory: Path, pattern: str) -> List[Path]:
    return sorted(
        (path for path in directory.glob(pattern) if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def choose_preferred_json(
    directory: Path,
) -> Tuple[Optional[Path], Optional[str], Dict[str, List[str]]]:
    candidates: Dict[str, List[Path]] = {
        pattern: sorted_existing_files(directory, pattern) for pattern in JSON_PATTERNS
    }

    for pattern in JSON_PATTERNS:
        if candidates[pattern]:
            selected = candidates[pattern][0]
            return (
                selected,
                pattern,
                {
                    key: [str(path) for path in paths]
                    for key, paths in candidates.items()
                },
            )

    return (
        None,
        None,
        {key: [str(path) for path in paths] for key, paths in candidates.items()},
    )


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def walk_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        if prefix:
            yield prefix, "dict"
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_paths(value, next_prefix)
        return

    if isinstance(obj, list):
        list_prefix = f"{prefix}[]" if prefix else "[]"
        yield list_prefix, "list"
        for item in obj:
            yield from walk_paths(item, list_prefix)
        return

    value_type = type(obj).__name__
    if prefix:
        yield prefix, value_type


def collect_field_inventory(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    path_types: Dict[str, Set[str]] = defaultdict(set)

    for record in records:
        for path, value_type in walk_paths(record):
            path_types[path].add(value_type)

    available_paths = sorted(path_types.keys())
    top_level_fields = sorted(
        {path.split(".", 1)[0].replace("[]", "") for path in available_paths}
    )
    section_fields = {
        "metrics": [],
        "timing": [],
        "classification_result": [],
        "retrieval_results": [],
    }

    for section in section_fields:
        prefix = f"{section}."
        nested = sorted(
            path[len(prefix) :] for path in available_paths if path.startswith(prefix)
        )
        section_fields[section] = nested

    return {
        "top_level_fields": top_level_fields,
        "all_paths": available_paths,
        "path_types": {
            path: sorted(types) for path, types in sorted(path_types.items())
        },
        "section_fields": section_fields,
    }


def normalize_candidate_path(path: str) -> str:
    return path.replace("[]", "")


def detect_semantic_fields(available_paths: Iterable[str]) -> Dict[str, Any]:
    available_set = set(available_paths)
    normalized_map = {normalize_candidate_path(path): path for path in available_paths}

    detection: Dict[str, Any] = {}
    for semantic_name, candidates in SEMANTIC_PATH_CANDIDATES.items():
        matched = []
        for candidate in candidates:
            if candidate in available_set:
                matched.append(candidate)
                continue
            normalized_candidate = normalize_candidate_path(candidate)
            if normalized_candidate in normalized_map:
                matched.append(normalized_map[normalized_candidate])

        detection[semantic_name] = {
            "available": bool(matched),
            "matched_paths": matched,
            "candidate_paths": list(candidates),
        }
    return detection


def build_group_report(
    group_id: str, folder_name: str, root_dir: Path
) -> Dict[str, Any]:
    group_dir = root_dir / folder_name
    report: Dict[str, Any] = {
        "group_id": group_id,
        "folder_name": folder_name,
        "source_dir": str(group_dir),
        "exists": group_dir.is_dir(),
        "selected_json": None,
        "selected_json_rule": None,
        "candidate_files": {},
        "record_count": 0,
        "available_fields": {},
        "field_detection": {},
        "status": "missing",
    }

    if not group_dir.is_dir():
        return report

    selected_json, selected_rule, candidate_files = choose_preferred_json(group_dir)
    report["candidate_files"] = candidate_files

    if selected_json is None:
        report["status"] = "missing_json"
        return report

    report["selected_json"] = str(selected_json)
    report["selected_json_rule"] = selected_rule

    records = load_json_records(selected_json)
    report["record_count"] = len(records)
    if not records:
        report["status"] = "empty_json"
        return report

    inventory = collect_field_inventory(records)
    report["available_fields"] = inventory
    report["field_detection"] = detect_semantic_fields(inventory["all_paths"])
    report["status"] = "ready"
    return report


def latest_matching_output(pattern: str) -> Optional[Path]:
    matches = sorted(
        OUTPUT_DIR.glob(pattern),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return matches[0] if matches else None


def write_json_output(payload: Dict[str, Any], filename: str) -> Path:
    output_path = OUTPUT_DIR / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def build_detection_payload() -> Dict[str, Any]:
    group_reports = {
        group_id: build_group_report(group_id, folder_name, ABLATION_ROOT)
        for group_id, folder_name in GROUP_MAPPING.items()
    }
    output_groups = {}
    for group_id, report in group_reports.items():
        output_groups[group_id] = {
            "group_id": report["group_id"],
            "folder_name": report["folder_name"],
            "source_dir": report["source_dir"],
            "exists": report["exists"],
            "selected_json": report["selected_json"],
            "selected_json_rule": report["selected_json_rule"],
            "candidate_files": report["candidate_files"],
            "available_fields": report["available_fields"],
            "field_detection": report["field_detection"],
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "phase": "field_detection_only",
        "notes": [
            "This output only covers group discovery, JSON selection, and field detection.",
            "No metric aggregation or final ablation statistics are computed in this phase.",
            "JSON is preferred over XLSX; XLSX is not used in this phase.",
        ],
        "group_mapping": GROUP_MAPPING,
        "groups": output_groups,
    }


def get_value_by_path(record: Dict[str, Any], path: str) -> Any:
    current: Any = record
    for segment in path.split("."):
        if current is None:
            return None

        is_list = segment.endswith("[]")
        key = segment[:-2] if is_list else segment

        if not isinstance(current, dict) or key not in current:
            return None

        current = current.get(key)

        if is_list:
            if not isinstance(current, list):
                return None
            return current

    return current


def simplify_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): simplify_value(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [simplify_value(item) for item in value]
        compact = [item for item in items if item not in (None, "", [], {})]
        if not compact:
            return None
        if all(not isinstance(item, (dict, list)) for item in compact):
            uniq = []
            for item in compact:
                if item not in uniq:
                    uniq.append(item)
            if len(uniq) == 1:
                return uniq[0]
            return uniq
        return compact
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def choose_existing_path(
    available_paths: Iterable[str], candidates: Iterable[str]
) -> Optional[str]:
    available = set(available_paths)
    normalized = {normalize_candidate_path(path): path for path in available_paths}
    for candidate in candidates:
        if candidate in available:
            return candidate
        normalized_candidate = normalize_candidate_path(candidate)
        if normalized_candidate in normalized:
            return normalized[normalized_candidate]
    return None


def build_group_field_mapping(group_report: Dict[str, Any]) -> Dict[str, Any]:
    available_fields = group_report.get("available_fields", {})
    available_paths = available_fields.get("all_paths", [])
    mapping: Dict[str, Any] = {}

    for field_name, candidates in NORMALIZED_FIELD_CANDIDATES.items():
        if field_name == "group_id":
            mapping[field_name] = {
                "available": True,
                "source_path": None,
                "notes": "Filled from group context.",
            }
            continue

        selected_path = choose_existing_path(available_paths, candidates)
        mapping[field_name] = {
            "available": selected_path is not None,
            "source_path": selected_path,
            "candidate_paths": list(candidates),
        }

    return mapping


def normalize_record(
    group_id: str,
    record: Dict[str, Any],
    field_mapping: Dict[str, Any],
    top_level_fields: List[str],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "group_id": group_id,
        "query": None,
        "domain": None,
        "question_type": None,
        "metrics": None,
        "timing": None,
        "classification": None,
        "raw_available_fields": {
            "record_top_level_fields": sorted(record.keys()),
            "group_top_level_fields": top_level_fields,
            "selected_source_paths": {
                key: value.get("source_path")
                for key, value in field_mapping.items()
                if key != "group_id"
            },
        },
    }

    for field_name in (
        "query",
        "domain",
        "question_type",
        "metrics",
        "timing",
        "classification",
    ):
        source_path = field_mapping.get(field_name, {}).get("source_path")
        if source_path is None:
            normalized[field_name] = None
            continue

        raw_value = get_value_by_path(record, source_path)
        normalized[field_name] = simplify_value(raw_value)

    return normalized


def build_normalized_preview(
    group_reports: Dict[str, Dict[str, Any]],
    sample_size: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    mapping_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "normalization_preparation_only",
        "normalized_schema": {
            "group_id": "str",
            "query": "str | null",
            "domain": "str | list | null",
            "question_type": "str | list | null",
            "metrics": "dict | null",
            "timing": "dict | null",
            "classification": "dict | null",
            "raw_available_fields": "dict",
        },
        "groups": {},
    }

    preview_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "normalization_preparation_only",
        "notes": [
            "This preview only contains normalized records for schema validation.",
            "No aggregation, baseline comparison, or specialized analysis is computed here.",
            "Full baseline only uses qa_eval_output/glm-4-plus.",
        ],
        "groups": {},
    }
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]] = {}

    for group_id, group_report in group_reports.items():
        field_mapping = build_group_field_mapping(group_report)
        top_level_fields = group_report.get("available_fields", {}).get(
            "top_level_fields", []
        )

        records: List[Dict[str, Any]] = []
        selected_json = group_report.get("selected_json")
        if selected_json:
            records = load_json_records(Path(selected_json))

        all_normalized_records = [
            normalize_record(group_id, record, field_mapping, top_level_fields)
            for record in records
        ]
        normalized_records = all_normalized_records[:sample_size]

        normalized_records_by_group[group_id] = all_normalized_records

        mapping_payload["groups"][group_id] = {
            "group_id": group_id,
            "folder_name": group_report.get("folder_name"),
            "source_dir": group_report.get("source_dir"),
            "selected_json": selected_json,
            "available_fields_summary": {
                "top_level_fields": top_level_fields,
                "metrics_fields": group_report.get("available_fields", {})
                .get("section_fields", {})
                .get("metrics", []),
                "timing_fields": group_report.get("available_fields", {})
                .get("section_fields", {})
                .get("timing", []),
                "classification_fields": group_report.get("available_fields", {})
                .get("section_fields", {})
                .get("classification_result", []),
            },
            "normalized_field_mapping": field_mapping,
        }

        preview_payload["groups"][group_id] = {
            "group_id": group_id,
            "selected_json": selected_json,
            "samples": normalized_records,
        }

    return mapping_payload, preview_payload, normalized_records_by_group


def compute_numeric_metric_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "value": None,
            "n_valid": 0,
            "status": "skipped",
        }

    return {
        "value": round(sum(values) / len(values), 6),
        "n_valid": len(values),
        "status": "available",
    }


def compute_time_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "mean": {"value": None, "n_valid": 0, "status": "skipped"},
            "median": {"value": None, "n_valid": 0, "status": "skipped"},
            "p95": {"value": None, "n_valid": 0, "status": "skipped"},
        }

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    median = (
        sorted_values[mid]
        if n % 2 == 1
        else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    )
    p95_index = max(0, min(n - 1, int((n - 1) * 0.95)))
    p95 = sorted_values[p95_index]

    return {
        "mean": {
            "value": round(sum(sorted_values) / n, 6),
            "n_valid": n,
            "status": "available",
        },
        "median": {
            "value": round(median, 6),
            "n_valid": n,
            "status": "available",
        },
        "p95": {
            "value": round(p95, 6),
            "n_valid": n,
            "status": "available",
        },
    }


def build_overall_summary(
    group_reports: Dict[str, Dict[str, Any]],
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    summary_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "overall_group_aggregation_only",
        "notes": [
            "This output only contains within-group overall aggregation.",
            "No Full-vs-ablation comparison table is generated in this phase.",
            "Full uses qa_eval_output/glm-4-plus only.",
        ],
        "groups": {},
    }

    excel_rows: List[Dict[str, Any]] = []

    for group_id, records in normalized_records_by_group.items():
        group_report = group_reports[group_id]
        group_summary: Dict[str, Any] = {
            "group_id": group_id,
            "folder_name": group_report.get("folder_name"),
            "selected_json": group_report.get("selected_json"),
            "metrics": {},
            "response_time": {},
        }

        metrics_values: Dict[str, List[float]] = {
            name: [] for name in AGGREGATION_METRICS
        }
        timing_values: List[float] = []

        for record in records:
            metrics = (
                record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            )
            timing = (
                record.get("timing") if isinstance(record.get("timing"), dict) else {}
            )

            for metric_name, metric_key in AGGREGATION_METRICS.items():
                numeric_value = safe_float(metrics.get(metric_key))
                if numeric_value is not None:
                    metrics_values[metric_name].append(numeric_value)

            total_time = safe_float(timing.get("total_time"))
            if total_time is not None:
                timing_values.append(total_time)

        for metric_name, values in metrics_values.items():
            metric_summary = compute_numeric_metric_summary(values)
            group_summary["metrics"][metric_name] = metric_summary["value"]
            excel_rows.append(
                {
                    "group_id": group_id,
                    "folder_name": group_report.get("folder_name"),
                    "category": "metric",
                    "metric": metric_name,
                    "value": metric_summary["value"],
                }
            )

        time_summary = compute_time_summary(timing_values)
        group_summary["response_time"] = {
            stat_name: stat_summary["value"]
            for stat_name, stat_summary in time_summary.items()
        }
        for stat_name, stat_summary in time_summary.items():
            excel_rows.append(
                {
                    "group_id": group_id,
                    "folder_name": group_report.get("folder_name"),
                    "category": "response_time",
                    "metric": stat_name,
                    "value": stat_summary["value"],
                }
            )

        summary_payload["groups"][group_id] = group_summary

    return summary_payload, pd.DataFrame(excel_rows)


def compute_delta_pct(
    value: Optional[float], full_value: Optional[float]
) -> Tuple[Optional[float], Optional[str]]:
    if value is None or full_value is None:
        return None, "full_or_group_value_missing"
    if full_value == 0:
        return None, "full_value_zero"
    return round((value - full_value) / full_value, 6), None


def build_comparison_summary(
    overall_summary: Dict[str, Any]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    groups = overall_summary.get("groups", {})
    full_group = groups.get("Full", {})

    comparison_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "full_comparison_only",
        "notes": [
            "This output compares A1-A6 against Full within the available group-level aggregates.",
            "Full uses qa_eval_output/glm-4-plus only.",
            "Missing metrics are skipped instead of raising errors.",
        ],
        "full_group_id": "Full",
        "rows": [],
    }

    excel_rows: List[Dict[str, Any]] = []

    full_metrics = full_group.get("metrics", {})
    full_time = full_group.get("response_time", {})

    metric_specs: List[Tuple[str, str, Dict[str, Any]]] = []
    for metric_name, metric_value in full_metrics.items():
        metric_specs.append(("metric", metric_name, metric_value))
    for stat_name, stat_value in full_time.items():
        metric_specs.append(("response_time", stat_name, stat_value))

    for group_id, group_payload in groups.items():
        if group_id == "Full":
            continue

        group_metrics = group_payload.get("metrics", {})
        group_time = group_payload.get("response_time", {})

        for category, metric_name, full_value in metric_specs:
            group_value = (
                group_metrics.get(metric_name)
                if category == "metric"
                else group_time.get(metric_name)
            )
            delta = None
            delta_pct = None
            delta_pct_reason = None

            if group_value is not None and full_value is not None:
                delta = round(group_value - full_value, 6)
                delta_pct, delta_pct_reason = compute_delta_pct(group_value, full_value)
            elif full_value is None:
                delta_pct_reason = "full_or_group_value_missing"
            elif group_value is None:
                delta_pct_reason = "full_or_group_value_missing"

            row = {
                "group_id": group_id,
                "category": category,
                "metric": metric_name,
                "value": group_value,
                "full_value": full_value,
                "delta": delta,
                "delta_pct": delta_pct,
                "delta_pct_reason": delta_pct_reason,
            }
            comparison_payload["rows"].append(row)
            excel_rows.append(row)

    return comparison_payload, pd.DataFrame(excel_rows)


def build_classification_scope_check(
    group_reports: Dict[str, Dict[str, Any]],
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    target_groups = ("A1", "A2", "A3")
    results: Dict[str, Any] = {}

    for group_id in target_groups:
        records = normalized_records_by_group.get(group_id, [])
        group_report = group_reports.get(group_id, {})

        classification_records = [
            record.get("classification")
            for record in records
            if isinstance(record.get("classification"), dict)
            and record.get("classification")
        ]
        metrics_records = [
            record.get("metrics")
            for record in records
            if isinstance(record.get("metrics"), dict) and record.get("metrics")
        ]

        methods = sorted(
            {
                cls.get("method")
                for cls in classification_records
                if cls.get("method") is not None
            }
        )
        domain_ids = {
            cls.get("domain_id")
            for cls in classification_records
            if cls.get("domain_id") not in (None, "")
        }
        primary_label_ids = {
            cls.get("primary_label_id")
            for cls in classification_records
            if cls.get("primary_label_id") not in (None, "")
        }

        metric_keys = ("classification_accuracy", "domain_correct", "label_correct")
        metric_value_sets: Dict[str, List[float]] = {}
        nonzero_metric_count = 0
        for key in metric_keys:
            values = []
            for metrics in metrics_records:
                numeric_value = safe_float(metrics.get(key))
                if numeric_value is not None:
                    values.append(numeric_value)
            unique_values = sorted(set(values))
            metric_value_sets[key] = unique_values
            nonzero_metric_count += sum(1 for value in values if value != 0.0)

        has_meaningful_label_diversity = (
            len(domain_ids) > 1 or len(primary_label_ids) > 1
        )
        has_metric_variation = any(
            len(values) > 1 for values in metric_value_sets.values()
        )
        has_nonzero_classification_metrics = nonzero_metric_count > 0

        include_in_analysis = bool(
            classification_records
            and metrics_records
            and (
                has_meaningful_label_diversity
                or has_metric_variation
                or has_nonzero_classification_metrics
            )
        )

        if include_in_analysis:
            conclusion = "include"
            reason = (
                "Classification-related fields are present and show meaningful variation in "
                "predicted labels and/or classification metrics."
            )
        else:
            conclusion = "exclude"
            reason = (
                "Classification-related fields exist, but current outputs are degenerate for "
                "supplemental analysis: labels do not vary meaningfully and classification metrics "
                "are constant or zero."
            )

        results[group_id] = {
            "group_id": group_id,
            "folder_name": group_report.get("folder_name"),
            "selected_json": group_report.get("selected_json"),
            "include_in_classification_analysis": include_in_analysis,
            "conclusion": conclusion,
            "reason": reason,
            "evidence": {
                "classification_fields_detected": group_report.get(
                    "field_detection", {}
                ).get("classification", {}),
                "classification_methods": methods,
                "domain_ids": sorted(domain_ids),
                "primary_label_ids": sorted(primary_label_ids),
                "classification_metric_unique_values": metric_value_sets,
            },
        }

    included_groups = [
        gid
        for gid, item in results.items()
        if item["include_in_classification_analysis"]
    ]
    excluded_groups = [
        gid
        for gid, item in results.items()
        if not item["include_in_classification_analysis"]
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "phase": "classification_scope_check_only",
        "summary": {
            "include_groups": included_groups,
            "exclude_groups": excluded_groups,
            "judgement_basis": [
                "classification_result field availability",
                "classification metric field availability",
                "variation of classification methods",
                "variation of domain_id and primary_label_id",
                "variation and non-zero presence of classification metrics",
            ],
        },
        "groups": results,
    }


def build_classification_analysis(
    group_reports: Dict[str, Dict[str, Any]],
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]],
    scope_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    metric_mapping = {
        "Domain Correctness": "domain_correct",
        "Label Correctness": "label_correct",
        "Classification Accuracy": "classification_accuracy",
        "Rank Score": "rank_score",
        "Weighted Score": "weighted_score",
    }

    analysis_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "a1_classification_analysis_only",
        "groups": {},
    }

    scope_output: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "a1_classification_analysis_scope",
        "included_groups": ["Full", "A1"],
        "excluded_groups": ["A2", "A3"],
        "field_mapping": {
            "Domain Correctness": {
                "source": "metrics.domain_correct",
                "mode": "direct",
            },
            "Label Correctness": {
                "source": "metrics.label_correct",
                "mode": "direct",
            },
            "Classification Accuracy": {
                "source": "metrics.classification_accuracy",
                "mode": "direct",
                "fallback_formula": "Domain Correctness × Label Correctness",
            },
            "Rank Score": {
                "source": "metrics.rank_score",
                "mode": "direct",
            },
            "Weighted Score": {
                "source": "metrics.weighted_score",
                "mode": "direct",
                "fallback_formula": "0.7 × Classification Accuracy + 0.3 × Rank Score",
            },
        },
        "groups": {},
    }

    excel_rows: List[Dict[str, Any]] = []

    for group_id in ("Full", "A1"):
        records = normalized_records_by_group.get(group_id, [])
        group_report = group_reports.get(group_id, {})
        group_entry: Dict[str, Any] = {
            "group_id": group_id,
            "metrics": {},
        }

        for display_name, metric_key in metric_mapping.items():
            values: List[float] = []
            for record in records:
                metrics = (
                    record.get("metrics")
                    if isinstance(record.get("metrics"), dict)
                    else {}
                )
                numeric_value = safe_float(metrics.get(metric_key))
                if numeric_value is not None:
                    values.append(numeric_value)

            metric_summary = compute_numeric_metric_summary(values)
            group_entry["metrics"][display_name] = metric_summary["value"]
            field_source = f"metrics.{metric_key}"
            excel_rows.append(
                {
                    "group_id": group_id,
                    "metric": display_name,
                    "value": metric_summary["value"],
                }
            )
            scope_output["groups"].setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "included": True,
                    "selected_json": group_report.get("selected_json"),
                    "used_fields": {},
                },
            )
            scope_output["groups"][group_id]["used_fields"][display_name] = {
                "source": field_source,
                "mode": "direct",
                "available": metric_summary["status"] == "available",
            }

        analysis_payload["groups"][group_id] = group_entry

    scope_groups = (scope_payload or {}).get("groups", {})
    for group_id in ("A2", "A3"):
        scope_info = scope_groups.get(group_id, {})
        exclusion_reason = (
            scope_info.get("reason") or "classification analysis unavailable"
        )
        scope_output["groups"][group_id] = {
            "group_id": group_id,
            "included": False,
            "selected_json": group_reports.get(group_id, {}).get("selected_json"),
            "exclusion_reason": exclusion_reason,
            "scope_evidence": scope_info.get("evidence", {}),
        }

    return analysis_payload, scope_output, pd.DataFrame(excel_rows)


def build_a23_domain_thelma_analysis(
    normalized_records_by_group: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    domain_field = "metrics.ground_truth_domain"
    metric_mapping = {
        "SP_1": "source_precision_sp1",
        "SP_2": "source_precision_sp2",
        "SQC": "source_query_coverage",
        "RP": "response_precision",
        "RQC": "response_query_coverage",
        "RSD": "response_self_distinctness",
        "Groundedness": "groundedness",
    }
    target_groups = ("Full", "A2", "A3")

    domain_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        gid: defaultdict(list) for gid in target_groups
    }
    for group_id in target_groups:
        for record in normalized_records_by_group.get(group_id, []):
            domain = record.get("domain")
            if isinstance(domain, list):
                domain = domain[0] if domain else None
            if isinstance(domain, str):
                domain = domain.strip()
            if domain:
                domain_records[group_id][domain].append(record)

    all_domains = sorted(
        set().union(*(set(domain_records[gid].keys()) for gid in target_groups))
    )
    domain_available = bool(all_domains)

    scope_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "a23_domain_thelma_scope",
        "domain_field": {
            "source": domain_field,
            "available": domain_available,
            "reason": None
            if domain_available
            else "No usable domain field values were found in normalized records.",
        },
        "metric_field_mapping": {
            metric_name: f"metrics.{metric_key}"
            for metric_name, metric_key in metric_mapping.items()
        },
        "unavailable_parts": [],
    }

    if not domain_available:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "base_dir": str(BASE_DIR),
            "ablation_root": str(ABLATION_ROOT),
            "full_root": str(FULL_ROOT),
            "phase": "a23_domain_thelma_only",
            "rows": [],
        }
        return (
            payload,
            scope_payload,
            pd.DataFrame(
                columns=[
                    "domain",
                    "metric",
                    "full_value",
                    "a2_value",
                    "a2_delta",
                    "a2_delta_pct",
                    "a3_value",
                    "a3_delta",
                    "a3_delta_pct",
                ]
            ),
        )

    rows: List[Dict[str, Any]] = []
    excel_rows: List[Dict[str, Any]] = []

    for domain in all_domains:
        for metric_name, metric_key in metric_mapping.items():
            values_by_group: Dict[str, Optional[float]] = {}
            for group_id in target_groups:
                values = []
                for record in domain_records[group_id].get(domain, []):
                    metrics = (
                        record.get("metrics")
                        if isinstance(record.get("metrics"), dict)
                        else {}
                    )
                    numeric_value = safe_float(metrics.get(metric_key))
                    if numeric_value is not None:
                        values.append(numeric_value)
                summary = compute_numeric_metric_summary(values)
                values_by_group[group_id] = summary["value"]

            full_value = values_by_group.get("Full")
            a2_value = values_by_group.get("A2")
            a3_value = values_by_group.get("A3")

            a2_delta = (
                round(a2_value - full_value, 6)
                if a2_value is not None and full_value is not None
                else None
            )
            a2_delta_pct, _ = compute_delta_pct(a2_value, full_value)
            a3_delta = (
                round(a3_value - full_value, 6)
                if a3_value is not None and full_value is not None
                else None
            )
            a3_delta_pct, _ = compute_delta_pct(a3_value, full_value)

            row = {
                "domain": domain,
                "metric": metric_name,
                "full_value": full_value,
                "a2_value": a2_value,
                "a2_delta": a2_delta,
                "a2_delta_pct": a2_delta_pct,
                "a3_value": a3_value,
                "a3_delta": a3_delta,
                "a3_delta_pct": a3_delta_pct,
            }
            rows.append(row)
            excel_rows.append(row)

            if full_value is None:
                scope_payload["unavailable_parts"].append(
                    {
                        "domain": domain,
                        "metric": metric_name,
                        "reason": "full_value_unavailable",
                    }
                )
            if a2_value is None:
                scope_payload["unavailable_parts"].append(
                    {
                        "domain": domain,
                        "metric": metric_name,
                        "reason": "a2_value_unavailable",
                    }
                )
            if a3_value is None:
                scope_payload["unavailable_parts"].append(
                    {
                        "domain": domain,
                        "metric": metric_name,
                        "reason": "a3_value_unavailable",
                    }
                )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "ablation_root": str(ABLATION_ROOT),
        "full_root": str(FULL_ROOT),
        "phase": "a23_domain_thelma_only",
        "rows": rows,
    }
    return payload, scope_payload, pd.DataFrame(excel_rows)


def run_field_detection() -> AblationAnalysisContext:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = AblationAnalysisContext()
    ablation_group_reports = {
        group_id: build_group_report(group_id, folder_name, ABLATION_ROOT)
        for group_id, folder_name in GROUP_MAPPING.items()
    }
    detection_payload = build_detection_payload()
    detection_path = write_json_output(detection_payload, "field_detection.json")
    context.detection_payload = detection_payload
    context.detection_path = detection_path

    full_group_report = build_group_report(
        "Full", "glm-4-plus", BASE_DIR / "qa_eval_output"
    )
    context.group_reports = {
        **ablation_group_reports,
        "Full": full_group_report,
    }

    print(f"Field detection written to: {detection_path}")
    return context


def run_normalization(context: AblationAnalysisContext) -> AblationAnalysisContext:
    (
        mapping_payload,
        preview_payload,
        normalized_records_by_group,
    ) = build_normalized_preview(context.group_reports)
    mapping_payload["field_detection_source"] = (
        str(context.detection_path) if context.detection_path else None
    )
    preview_payload["field_detection_source"] = (
        str(context.detection_path) if context.detection_path else None
    )

    mapping_path = write_json_output(mapping_payload, "normalization_mapping.json")
    preview_path = write_json_output(preview_payload, "normalized_preview.json")

    context.mapping_payload = mapping_payload
    context.mapping_path = mapping_path
    context.preview_payload = preview_payload
    context.preview_path = preview_path
    context.normalized_records_by_group = normalized_records_by_group

    print(f"Normalization mapping written to: {mapping_path}")
    print(f"Normalized preview written to: {preview_path}")
    return context


def run_overall_summary(context: AblationAnalysisContext) -> AblationAnalysisContext:
    summary_payload, summary_df = build_overall_summary(
        context.group_reports, context.normalized_records_by_group
    )
    summary_payload["field_detection_source"] = (
        str(context.detection_path) if context.detection_path else None
    )
    summary_payload["normalization_mapping_source"] = (
        str(context.mapping_path) if context.mapping_path else None
    )
    summary_payload["normalized_preview_source"] = (
        str(context.preview_path) if context.preview_path else None
    )

    summary_json_path = write_json_output(
        summary_payload, "ablation_summary_overall.json"
    )
    summary_excel_path = OUTPUT_DIR / "ablation_summary_overall.xlsx"
    summary_df.to_excel(summary_excel_path, index=False)

    context.overall_summary_payload = summary_payload
    context.overall_summary_json_path = summary_json_path
    context.overall_summary_excel_path = summary_excel_path

    print(f"Overall summary written to: {summary_json_path}")
    print(f"Overall summary Excel written to: {summary_excel_path}")
    return context


def run_comparison(context: AblationAnalysisContext) -> AblationAnalysisContext:
    comparison_payload, comparison_df = build_comparison_summary(
        context.overall_summary_payload or {}
    )
    comparison_payload["field_detection_source"] = (
        str(context.detection_path) if context.detection_path else None
    )
    comparison_payload["normalization_mapping_source"] = (
        str(context.mapping_path) if context.mapping_path else None
    )
    comparison_payload["normalized_preview_source"] = (
        str(context.preview_path) if context.preview_path else None
    )
    comparison_payload["overall_summary_source"] = (
        str(context.overall_summary_json_path)
        if context.overall_summary_json_path
        else None
    )

    comparison_json_path = write_json_output(
        comparison_payload, "ablation_comparison.json"
    )
    comparison_excel_path = OUTPUT_DIR / "ablation_comparison.xlsx"
    comparison_df.to_excel(comparison_excel_path, index=False)

    context.comparison_payload = comparison_payload
    context.comparison_json_path = comparison_json_path
    context.comparison_excel_path = comparison_excel_path

    print(f"Comparison summary written to: {comparison_json_path}")
    print(f"Comparison Excel written to: {comparison_excel_path}")
    return context


def run_classification_scope_check(
    context: AblationAnalysisContext,
) -> AblationAnalysisContext:
    payload = build_classification_scope_check(
        context.group_reports, context.normalized_records_by_group
    )
    output_path = write_json_output(payload, "classification_scope_check.json")
    context.classification_scope_payload = payload
    context.classification_scope_path = output_path

    print(f"Classification scope check written to: {output_path}")
    return context


def run_classification_analysis(
    context: AblationAnalysisContext,
) -> AblationAnalysisContext:
    analysis_payload, scope_output, analysis_df = build_classification_analysis(
        context.group_reports,
        context.normalized_records_by_group,
        context.classification_scope_payload or {},
    )

    analysis_payload["classification_scope_source"] = (
        str(context.classification_scope_path)
        if context.classification_scope_path
        else None
    )
    analysis_json_path = write_json_output(
        analysis_payload, "ablation_a1_classification.json"
    )
    analysis_excel_path = OUTPUT_DIR / "ablation_a1_classification.xlsx"
    analysis_df.to_excel(analysis_excel_path, index=False)

    scope_output["classification_scope_check_source"] = (
        str(context.classification_scope_path)
        if context.classification_scope_path
        else None
    )
    scope_json_path = write_json_output(
        scope_output, "ablation_a1_classification_scope.json"
    )

    context.classification_analysis_payload = analysis_payload
    context.classification_analysis_json_path = analysis_json_path
    context.classification_analysis_excel_path = analysis_excel_path
    context.classification_analysis_scope_path = scope_json_path

    print(f"A1 classification analysis written to: {analysis_json_path}")
    print(f"A1 classification Excel written to: {analysis_excel_path}")
    print(f"A1 classification scope written to: {scope_json_path}")
    return context


def run_a23_domain_thelma_analysis(
    context: AblationAnalysisContext,
) -> AblationAnalysisContext:
    payload, scope_payload, df = build_a23_domain_thelma_analysis(
        context.normalized_records_by_group
    )
    json_path = write_json_output(payload, "ablation_a23_by_domain_thelma.json")
    excel_path = OUTPUT_DIR / "ablation_a23_by_domain_thelma.xlsx"
    df.to_excel(excel_path, index=False)
    scope_path = write_json_output(
        scope_payload, "ablation_a23_domain_thelma_scope.json"
    )

    context.a23_domain_thelma_payload = payload
    context.a23_domain_thelma_json_path = json_path
    context.a23_domain_thelma_excel_path = excel_path
    context.a23_domain_thelma_scope_path = scope_path

    print(f"A2/A3 domain THELMA analysis written to: {json_path}")
    print(f"A2/A3 domain THELMA Excel written to: {excel_path}")
    print(f"A2/A3 domain THELMA scope written to: {scope_path}")
    return context


def main() -> None:
    context = run_field_detection()
    context = run_normalization(context)
    context = run_overall_summary(context)
    context = run_comparison(context)
    context = run_classification_scope_check(context)
    context = run_classification_analysis(context)
    context = run_a23_domain_thelma_analysis(context)


if __name__ == "__main__":
    main()
