#!/usr/bin/env python
"""
Complaint clustering and Gemma4 31B labeling pipeline.

This script is designed for Python 3.10+ and the Korean civil complaint
dataset layout used in this workspace.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Set

DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_DATA_GLOB_TRAIN = "**/1.Training/라벨링데이터/TL1.zip"
DEFAULT_DATA_GLOB_VALIDATION = "**/2.Validation/라벨링데이터/VL1.zip"
LABEL_FIELDNAMES = [
    "cluster_id", "representative_id", "importance", "urgency", "basis_tags", "reason",
    "confidence", "needs_review", "model", "label_source", "raw_response",
]
SYSTEM_INSTRUCTION = "You classify Korean civil complaints. Return only valid JSON."
BATCH_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}

ENTITY_KEYWORD_EXCLUDE_LABELS = {"LOC", "PER"}
RISK_CATEGORIES = {"안전건설", "보건소", "환경미화", "건축허가", "교통"}
RISK_KEYWORDS = {
    "사망", "부상", "중상", "의식불명", "생명", "인명", "위험", "위해", "사고위험",
    "다칠", "다침", "추락", "낙상", "붕괴", "붕괴위험", "균열", "싱크홀", "화재", "폭발",
    "가스", "누출", "감전", "침수", "홍수", "산사태", "낙석", "지진", "대피",
    "감염", "전염", "방역", "집단감염", "식중독", "오염", "폐수", "석면", "유독", "악취", "위생",
    "어린이", "유치원", "초등학교", "어린이보호구역", "스쿨존", "노인", "장애인", "환자", "임산부",
    "불법", "위반", "무허가", "단속", "불법주정차", "불법 주정차", "불법투기", "불법 투기",
    "주민들", "시민들", "다수", "집단", "아파트", "마을", "상습", "반복", "여러",
    "테러", "폭발물", "국가안보", "군사", "보안시설",
}


def _require_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "This pipeline requires Python 3.10+. "
            f"Current interpreter is {sys.version.split()[0]}."
        )


def _import_orjson():
    try:
        import orjson  # type: ignore

        return orjson
    except ImportError:
        return None


def _read_json_bytes(raw: bytes):
    orjson = _import_orjson()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _progress(iterable, total=None, desc: str = "", unit: str = "it"):
    try:
        from tqdm import tqdm  # type: ignore

        kwargs = {"total": total, "desc": desc, "unit": unit, "dynamic_ncols": True, "ascii": True}
        if total is not None:
            kwargs["bar_format"] = "{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]"
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _progress_bar(total: int, desc: str = "", unit: str = "it"):
    if total <= 0:
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            ascii=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) [{elapsed}<{remaining}, {rate_fmt}]",
        )
    except ImportError:
        return None


def _progress_write(message: str) -> None:
    try:
        from tqdm import tqdm  # type: ignore

        tqdm.write(message)
    except ImportError:
        print(message)


def _resolve_first(pattern: str, label: str) -> Path:
    matches = sorted(Path(".").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {label} zip with pattern: {pattern}")
    return matches[0]


def _normalize_space(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_keyword(value) -> str:
    text = _normalize_space(value).lower()
    text = re.sub(r"#@[^#]+#", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "", text)
    return text.strip()


def _normalize_text_for_cluster(value) -> str:
    text = _normalize_space(value).lower()
    text = re.sub(r"#@[^#]+#", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _unique_ordered(values):
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _bucket_key(category: str, subcategory: str, predication: str, content_keywords: str) -> str:
    keyword_part = content_keywords if content_keywords else "__no_keyword__"
    raw = "|".join([
        _normalize_keyword(category) or "__no_category__",
        _normalize_keyword(subcategory) or "__no_subcategory__",
        _normalize_keyword(predication) or "__no_predication__",
        keyword_part,
    ])
    return raw


def _extract_documents(payload):
    if isinstance(payload, dict):
        docs = payload.get("documents", [])
        return docs if isinstance(docs, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _iter_label_zip(zip_path: Path, split: str, max_docs: Optional[int] = None):
    emitted = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        entries = sorted([entry for entry in archive.infolist() if entry.filename.lower().endswith(".json")], key=lambda e: e.filename)
        for entry in entries:
            with archive.open(entry, "r") as handle:
                payload = _read_json_bytes(handle.read())
            for doc in _extract_documents(payload):
                if max_docs is not None and emitted >= max_docs:
                    return
                yield _flatten_doc(doc, split, entry.filename)
                emitted += 1


def _flatten_doc(doc: dict, split: str, source_entry: str) -> dict:
    labeling = doc.get("labeling") or {}
    intent = labeling.get("intent") or {}
    keyword_items = labeling.get("keyword") or []
    entity_items = labeling.get("entities") or []

    excluded_forms = {
        _normalize_keyword(entity.get("form"))
        for entity in entity_items
        if _normalize_space(entity.get("label")) in ENTITY_KEYWORD_EXCLUDE_LABELS
    }

    keywords = []
    content_keywords = []
    for keyword in keyword_items:
        form = _normalize_space(keyword.get("form") if isinstance(keyword, dict) else keyword)
        normalized = _normalize_keyword(form)
        if not form:
            continue
        keywords.append(form)
        if normalized and normalized not in excluded_forms:
            content_keywords.append(normalized)

    keywords = _unique_ordered(keywords)
    content_keywords = sorted(set(content_keywords))

    category = _normalize_space(intent.get("category"))
    subcategory = _normalize_space(intent.get("subcategory"))
    predication = _normalize_space(intent.get("predication"))
    content_keyword_text = "|".join(content_keywords)

    return {
        "id": _normalize_space(doc.get("id")),
        "split": split,
        "source_entry": source_entry,
        "publish_date": _normalize_space(doc.get("publish_date")),
        "Q_refined": _normalize_space(doc.get("Q_refined")),
        "category": category,
        "subcategory": subcategory,
        "predication": predication,
        "department": _normalize_space(labeling.get("department")),
        "keywords": "|".join(keywords),
        "content_keywords": content_keyword_text,
        "related_law": _normalize_space(labeling.get("related_law")),
        "bucket_key": _bucket_key(category, subcategory, predication, content_keyword_text),
        "cluster_text_norm": _normalize_text_for_cluster(doc.get("Q_refined")),
    }


def command_flatten(args: argparse.Namespace) -> None:
    _require_python()
    train_zip = Path(args.train_label_zip) if args.train_label_zip else _resolve_first(DEFAULT_DATA_GLOB_TRAIN, "training label")
    validation_zip = Path(args.validation_label_zip) if args.validation_label_zip else _resolve_first(DEFAULT_DATA_GLOB_VALIDATION, "validation label")
    output = Path(args.output)
    _ensure_parent(output)

    fieldnames = [
        "id", "split", "source_entry", "publish_date", "Q_refined", "category", "subcategory",
        "predication", "department", "keywords", "content_keywords", "related_law", "bucket_key",
        "cluster_text_norm",
    ]

    counts = Counter()
    missing = Counter()
    category_counts = Counter()

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, zip_path in [("train", train_zip), ("validation", validation_zip)]:
            rows = _iter_label_zip(zip_path, split, args.max_docs)
            for row in _progress(rows, total=args.max_docs, desc=f"flatten:{split}", unit="docs"):
                writer.writerow(row)
                counts[split] += 1
                category_counts[row["category"] or "__missing__"] += 1
                for key in ["id", "Q_refined", "category", "subcategory", "predication", "keywords"]:
                    if not row[key]:
                        missing[key] += 1

    report = {
        "output": str(output),
        "train_label_zip": str(train_zip),
        "validation_label_zip": str(validation_zip),
        "counts": dict(counts),
        "total": sum(counts.values()),
        "missing": dict(missing),
        "top_categories": category_counts.most_common(30),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _lazy_pandas():
    import pandas as pd  # type: ignore

    return pd


def _lazy_cluster_libs():
    import numpy as np  # type: ignore
    from scipy import sparse  # type: ignore
    from scipy.sparse.csgraph import connected_components  # type: ignore
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    return np, sparse, connected_components, MiniBatchKMeans, TfidfVectorizer, NearestNeighbors


def _duplicate_edge_matrix(norm_texts, sparse_module):
    rows = []
    cols = []
    by_text = defaultdict(list)
    for idx, text in enumerate(norm_texts):
        if text:
            by_text[text].append(idx)
    for indices in by_text.values():
        if len(indices) < 2:
            continue
        root = indices[0]
        for idx in indices[1:]:
            rows.extend([root, idx])
            cols.extend([idx, root])
    if not rows:
        return sparse_module.csr_matrix((len(norm_texts), len(norm_texts)), dtype=bool)
    data = [True] * len(rows)
    return sparse_module.csr_matrix((data, (rows, cols)), shape=(len(norm_texts), len(norm_texts)), dtype=bool)


def _components_for_matrix(x_matrix, norm_texts, threshold: float, n_neighbors: int, sparse_module, connected_components_fn, NearestNeighbors):
    n_rows = x_matrix.shape[0]
    if n_rows == 1:
        return [0]

    neighbors = min(max(2, n_neighbors), n_rows)
    nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute", n_jobs=1)
    nn.fit(x_matrix)
    graph = nn.kneighbors_graph(x_matrix, mode="distance")
    graph.data = 1.0 - graph.data
    graph = graph.tocsr()
    graph.data = graph.data >= threshold
    graph.eliminate_zeros()
    graph = graph.astype(bool)
    graph = graph.maximum(graph.T)
    graph = graph.maximum(_duplicate_edge_matrix(norm_texts, sparse_module))
    _, labels = connected_components_fn(graph, directed=False, return_labels=True)
    return labels.tolist()


def _fallback_exact_clusters(norm_texts):
    seen = {}
    labels = []
    for text in norm_texts:
        key = text or f"__empty_{len(labels)}"
        if key not in seen:
            seen[key] = len(seen)
        labels.append(seen[key])
    return labels


def _representative_positions(np_module, x_matrix, local_positions):
    reps = {}
    for label, positions in local_positions.items():
        if len(positions) == 1:
            reps[label] = positions[0]
            continue
        x_cluster = x_matrix[positions]
        centroid = x_cluster.mean(axis=0)
        scores = np_module.asarray(x_cluster @ centroid.T).ravel()
        reps[label] = positions[int(scores.argmax())]
    return reps


def _process_subbucket(df, sub_indices, x_matrix, threshold, n_neighbors, libs):
    np_module, sparse_module, connected_components_fn, _, _, NearestNeighbors = libs
    norm_texts = df.loc[sub_indices, "cluster_text_norm"].fillna("").tolist()
    try:
        labels = _components_for_matrix(
            x_matrix,
            norm_texts,
            threshold,
            n_neighbors,
            sparse_module,
            connected_components_fn,
            NearestNeighbors,
        )
    except Exception:
        labels = _fallback_exact_clusters(norm_texts)

    by_component = defaultdict(list)
    for pos, label in enumerate(labels):
        by_component[label].append(pos)

    rep_local_positions = _representative_positions(np_module, x_matrix, by_component)
    components = []
    sub_indices_list = list(sub_indices)
    for label, positions in by_component.items():
        global_indices = [sub_indices_list[pos] for pos in positions]
        rep_idx = sub_indices_list[rep_local_positions[label]]
        components.append((global_indices, rep_idx))
    return components


def _risk_score_for_row(row) -> int:
    text = " ".join([
        str(row.get("representative_text", "")),
        str(row.get("keywords", "")),
        str(row.get("content_keywords", "")),
    ])
    score = 0
    if row.get("category") in RISK_CATEGORIES:
        score += 2
    for keyword in RISK_KEYWORDS:
        if keyword and keyword in text:
            score += 1
    return min(score, 10)


RULE_KEYWORD_TAGS = {
    "LIFE_SAFETY": {
        "사망", "부상", "중상", "의식불명", "생명", "인명", "위험", "위해", "사고위험",
        "다칠", "다침", "추락", "낙상", "붕괴", "붕괴위험", "균열", "싱크홀", "화재", "폭발",
        "가스", "누출", "감전", "침수", "홍수", "산사태", "낙석", "지진", "대피",
    },
    "PUBLIC_HEALTH": {
        "감염", "전염", "감염병", "방역", "집단감염", "식중독", "오염", "폐수", "석면",
        "유독", "악취", "위생", "보건", "소독", "해충", "급식",
    },
    "VULNERABLE_GROUP": {
        "어린이", "유치원", "초등학교", "어린이보호구역", "스쿨존", "노인", "장애인",
        "환자", "임산부", "청소년", "학생", "아이", "아이들", "아동", "복지시설", "요양원",
    },
    "TRAFFIC_SAFETY": {
        "신호등", "횡단보도", "어린이보호구역", "스쿨존", "보행자", "보도", "인도", "차도",
        "도로", "교차로", "과속방지턱", "표지판", "불법주정차", "불법 주정차", "주정차",
        "차량", "버스정류장", "자전거도로", "통행",
    },
    "PUBLIC_INFRA": {
        "가로등", "보안등", "맨홀", "교량", "다리", "난간", "계단", "놀이터", "놀이기구",
        "시설물", "도로파손", "포트홀", "파손", "고장", "균열", "싱크홀", "하수구", "배수로",
    },
    "LEGAL_VIOLATION": {
        "불법", "위반", "무허가", "단속", "과태료", "불법건축물", "불법주정차",
        "불법 주정차", "불법투기", "불법 투기", "무단", "점유", "방치", "영업신고",
    },
    "MULTIPLE_RESIDENTS": {
        "주민들", "시민들", "여러", "다수", "집단", "아파트", "마을", "상습",
        "반복", "계속", "매일", "수차례", "피해자", "학생들",
    },
    "SECURITY_RISK": {"테러", "폭발물", "국가안보", "군사", "보안시설", "간첩", "협박"},
    "FACILITY_MAINTENANCE": {
        "보수", "수리", "정비", "고장", "파손", "교체", "설치", "철거", "청소", "개선",
        "시설물", "도로", "가로등", "보안등", "놀이터", "표지판", "하수구", "배수로",
    },
    "SERVICE_DELAY": {"지연", "처리", "답변", "신청", "접수", "허가", "승인", "발급", "심사", "약속", "기한"},
    "THANKS": {"감사", "고맙", "수고하셨"},
    "STANDARD_INQUIRY": {
        "문의", "질문", "궁금", "알려", "정보", "방법", "절차", "담당자", "부서",
        "번호", "가능한가", "어떻게", "어디", "언제",
    },
}
URGENT_KEYWORDS = {
    "지금", "즉시", "긴급", "당장", "바로", "빠른", "빨리", "조속", "시급", "오늘",
    "현재", "방금", "계속", "대피", "확산", "새고", "누출", "화재", "폭발", "붕괴",
    "감전", "침수", "사고", "다쳤", "다침", "사망", "부상", "식중독", "집단감염",
}
CRITICAL_DANGER_KEYWORDS = {
    "사망", "부상", "중상", "의식불명", "화재", "폭발", "가스", "누출", "감전",
    "붕괴", "붕괴위험", "침수", "홍수", "싱크홀", "산사태", "낙석", "지진", "대피",
}
REQUEST_ACTION_KEYWORDS = {
    "요청", "신고", "고발", "개선", "조치", "처리", "단속", "설치", "보수", "수리",
    "정비", "점검", "철거", "확인", "해결", "민원", "불편", "피해", "위험",
}
RULE_V3_LABEL_FIELDS = [
    "importance", "urgency", "priority_type", "cap_reason", "evidence_tags",
    "risk_evidence", "counter_evidence", "confidence", "needs_review", "trainable",
    "validation_flags", "reason", "model", "label_source",
]
RULE_V3_LIST_FIELDS = {"evidence_tags", "risk_evidence", "counter_evidence", "validation_flags"}
RULE_V3_INQUIRY_KEYWORDS = {
    "문의", "질문", "질의", "궁금", "알려", "방법", "절차", "담당부서", "담당자",
    "전화번호", "연락처", "어디", "어떻게", "언제", "가능한가", "가능한지", "확인",
    "발급", "서류", "제출", "증명서", "신청 방법", "납부 방법", "가상계좌", "영수증",
}
RULE_V3_PAYMENT_KEYWORDS = {
    "과태료", "납부", "세금", "자동차세", "재산세", "지방세", "주민세", "취득세",
    "가상계좌", "영수증", "체납", "고지서", "납세", "환급",
}
RULE_V3_ENFORCEMENT_KEYWORDS = {
    "과태료 부과", "부과", "단속", "신고", "고발", "견인", "행정처분", "불법",
    "위반", "불법주차", "불법 주차", "불법주정차", "불법 주정차",
}
RULE_V3_ACTION_KEYWORDS = {
    "요청", "조치", "제거", "견인", "점검", "단속", "부과", "처리", "수리",
    "보수", "설치", "철거", "정비", "개선", "해결", "바랍니다", "부탁드립니다",
}
RULE_V3_ENVIRONMENT_KEYWORDS = {
    "잡초", "잡풀", "풀", "제초", "풀베기", "가로수", "나무", "수목", "벌목",
    "가지치기", "화단", "녹지", "수풀", "덩굴",
}
RULE_V3_VISIBILITY_KEYWORDS = {"시야", "가려", "가림", "보이지", "안보", "사각", "침범"}
RULE_V3_TRAFFIC_CONTEXT_KEYWORDS = {
    "도로", "교차로", "횡단보도", "통학로", "어린이보호구역", "스쿨존", "인도",
    "보도", "차도", "차량", "버스정류장", "신호등", "커브길", "좌회전", "우회전",
}
RULE_V3_ACCIDENT_RISK_KEYWORDS = {
    "사고 위험", "사고위험", "위험", "위험합니다", "다칠", "다침", "부상",
    "사고가", "사고 발생", "추돌", "충돌",
}
RULE_V3_HYDRANT_KEYWORDS = {"소화전"}
RULE_V3_FIRE_SUPPRESSION_KEYWORDS = {"화재진압", "소방차", "소방", "불이 났", "화재 시", "불났"}
RULE_V3_FOOD_POISONING_KEYWORDS = {"식중독", "구토", "설사", "복통", "급식", "뷔페", "음식점"}
RULE_V3_HEALTH_SPREAD_KEYWORDS = {
    "집단감염", "감염 확산", "전염", "감염병", "방역", "격리", "확진", "식중독",
    "구토", "설사", "응급실", "급식", "여러 사람", "여러 명", "다수",
}
RULE_V3_SANITATION_KEYWORDS = {"위생", "오염", "폐수", "석면", "유독", "악취", "해충", "소독", "음식점"}
RULE_V3_DISASTER_KEYWORDS = {
    "화재", "가스", "누출", "감전", "붕괴", "붕괴위험", "침수", "홍수", "싱크홀",
    "산사태", "낙석", "지진", "폭발", "대피", "전선", "맨홀 열림", "맨홀열림",
}
RULE_V3_INJURY_KEYWORDS = {"사망", "부상", "중상", "의식불명", "응급", "응급실", "다쳤", "다침"}
RULE_V3_SECURITY_KEYWORDS = {"테러", "폭발물", "국가안보", "군사", "보안시설", "협박"}
RULE_V3_ACCESSIBILITY_KEYWORDS = {
    "장애인 주차구역", "장애인주차구역", "장애인 주차", "장애인구역", "장애인",
}
RULE_V3_VULNERABLE_KEYWORDS = {
    "어린이", "유치원", "초등학교", "어린이보호구역", "스쿨존", "노인", "장애인",
    "환자", "임산부", "요양원",
}
RULE_V3_MULTIPLE_KEYWORDS = {"주민들", "시민들", "여러", "다수", "집단", "아파트", "마을", "상습", "반복", "계속"}
RULE_V3_STRONG_EVIDENCE_TAGS = {
    "FIRE_SUPPRESSION_BLOCKED", "FOOD_POISONING_CLUSTER", "PUBLIC_HEALTH_SPREAD",
    "ACTIVE_FIRE_GAS_ELECTRIC", "COLLAPSE_FLOOD_DISASTER", "ACTIVE_INJURY",
    "SECURITY_RISK", "CRITICAL_INFRA_FAILURE",
}


def _rule_text(row: dict) -> str:
    return " ".join([
        _normalize_space(row.get("representative_text")),
        _normalize_space(row.get("keywords")),
        _normalize_space(row.get("content_keywords")),
        _normalize_space(row.get("category")),
        _normalize_space(row.get("subcategory")),
        _normalize_space(row.get("predication")),
    ])


def _rule_body_text(row: dict) -> str:
    return " ".join([
        _normalize_space(row.get("representative_text")),
        _normalize_space(row.get("keywords")),
        _normalize_space(row.get("content_keywords")),
    ])


def _matched_rule_tags(text: str) -> list[str]:
    tags = []
    for tag, keywords in RULE_KEYWORD_TAGS.items():
        if any(keyword and keyword in text for keyword in keywords):
            tags.append(tag)
    return tags


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def _basis_tags(tags: list[str], preferred: list[str], fallback: str) -> list[str]:
    selected = [tag for tag in preferred if tag in tags]
    return selected or [fallback]


def _cluster_size_num(row: dict) -> int:
    try:
        return max(1, int(float(row.get("cluster_size", 1) or 1)))
    except (TypeError, ValueError):
        return 1


def _is_true_thanks(text: str, predication: str, tags: list[str]) -> bool:
    risk_tags = {
        "LIFE_SAFETY", "PUBLIC_HEALTH", "VULNERABLE_GROUP", "TRAFFIC_SAFETY",
        "PUBLIC_INFRA", "LEGAL_VIOLATION", "MULTIPLE_RESIDENTS", "SECURITY_RISK",
    }
    has_risk = bool(set(tags) & risk_tags)
    if "감사" in predication and not has_risk:
        return True
    compact = re.sub(r"\s+", "", text)
    has_thanks_word = any(keyword in text for keyword in RULE_KEYWORD_TAGS["THANKS"])
    has_action = _contains_any(text, REQUEST_ACTION_KEYWORDS - {"처리", "민원"})
    return has_thanks_word and len(compact) <= 80 and not has_risk and not has_action


def _is_simple_inquiry(text: str, predication: str, tags: list[str]) -> bool:
    if not ("문의" in predication or "질의" in predication or "STANDARD_INQUIRY" in tags):
        return False
    risk_tags = {
        "LIFE_SAFETY", "PUBLIC_HEALTH", "VULNERABLE_GROUP", "TRAFFIC_SAFETY",
        "PUBLIC_INFRA", "LEGAL_VIOLATION", "MULTIPLE_RESIDENTS", "SECURITY_RISK",
    }
    has_risk = bool(set(tags) & risk_tags)
    has_action = _contains_any(text, REQUEST_ACTION_KEYWORDS - {"처리", "민원"})
    return not has_risk and not has_action


def build_rule_label(row: dict) -> dict:
    text = _rule_text(row)
    body_text = _rule_body_text(row)
    tags = _matched_rule_tags(text)
    body_tags = _matched_rule_tags(body_text)
    predication = _normalize_space(row.get("predication"))
    category = _normalize_space(row.get("category"))
    cluster_size = _cluster_size_num(row)

    tag_set = set(tags)
    has_life = "LIFE_SAFETY" in tag_set
    has_health = "PUBLIC_HEALTH" in tag_set
    has_vulnerable = "VULNERABLE_GROUP" in tag_set
    has_traffic = "TRAFFIC_SAFETY" in tag_set
    has_infra = "PUBLIC_INFRA" in tag_set
    has_legal = "LEGAL_VIOLATION" in tag_set
    has_many = "MULTIPLE_RESIDENTS" in tag_set or cluster_size >= 10
    has_security = "SECURITY_RISK" in tag_set
    has_facility = "FACILITY_MAINTENANCE" in tag_set
    has_service = "SERVICE_DELAY" in tag_set
    has_immediate = _contains_any(text, URGENT_KEYWORDS)
    has_critical_danger = _contains_any(text, CRITICAL_DANGER_KEYWORDS)

    if _is_true_thanks(body_text, predication, body_tags):
        importance = 1
        urgency = 1
        basis_tags = ["THANKS"]
        confidence = 0.9
    elif _is_simple_inquiry(body_text, predication, body_tags):
        importance = 1
        urgency = 1
        basis_tags = ["STANDARD_INQUIRY"]
        confidence = 0.86
    elif has_security:
        importance = 5
        urgency = 5 if has_immediate or has_critical_danger else 4
        basis_tags = _basis_tags(tags, ["SECURITY_RISK", "LIFE_SAFETY"], "SECURITY_RISK")
        confidence = 0.9
    elif has_health and (has_immediate or has_many or has_vulnerable or _contains_any(text, {"집단감염", "식중독", "급식"})):
        importance = 5
        urgency = 5 if has_immediate else 4
        basis_tags = _basis_tags(tags, ["PUBLIC_HEALTH", "VULNERABLE_GROUP", "MULTIPLE_RESIDENTS"], "PUBLIC_HEALTH")
        confidence = 0.88
    elif has_life and has_critical_danger:
        importance = 5
        urgency = 5 if has_immediate else 4
        basis_tags = _basis_tags(tags, ["LIFE_SAFETY", "PUBLIC_INFRA", "MULTIPLE_RESIDENTS"], "LIFE_SAFETY")
        confidence = 0.88
    elif has_life and has_vulnerable and (has_traffic or has_infra or has_immediate):
        importance = 5
        urgency = 5 if has_immediate else 4
        basis_tags = _basis_tags(tags, ["LIFE_SAFETY", "VULNERABLE_GROUP", "TRAFFIC_SAFETY", "PUBLIC_INFRA"], "LIFE_SAFETY")
        confidence = 0.84
    elif (has_traffic or has_infra) and (has_life or has_vulnerable or has_legal or has_many):
        importance = 4
        urgency = 4 if has_immediate or has_life or has_vulnerable else 3
        basis_tags = _basis_tags(
            tags,
            ["VULNERABLE_GROUP", "TRAFFIC_SAFETY", "PUBLIC_INFRA", "LEGAL_VIOLATION", "LIFE_SAFETY"],
            "PUBLIC_SAFETY",
        )
        confidence = 0.82
    elif has_health or (has_legal and (has_many or category in RISK_CATEGORIES)) or (has_many and category in RISK_CATEGORIES):
        importance = 4
        urgency = 4 if has_immediate else 3
        basis_tags = _basis_tags(tags, ["PUBLIC_HEALTH", "LEGAL_VIOLATION", "MULTIPLE_RESIDENTS"], "PUBLIC_RISK")
        confidence = 0.8
    elif has_legal:
        importance = 3
        urgency = 3
        basis_tags = _basis_tags(tags, ["LEGAL_VIOLATION"], "LEGAL_VIOLATION")
        confidence = 0.76
    elif has_many or has_facility or has_service:
        importance = 3
        urgency = 3 if has_immediate else 2
        basis_tags = _basis_tags(
            tags,
            ["MULTIPLE_RESIDENTS", "FACILITY_MAINTENANCE", "SERVICE_DELAY", "TRAFFIC_SAFETY", "PUBLIC_INFRA"],
            "REPEATED_INCONVENIENCE",
        )
        confidence = 0.74
    else:
        importance = 2
        urgency = 2
        basis_tags = tags or ["ROUTINE_REQUEST"]
        confidence = 0.68

    reason = (
        f"민원처리법의 통상 처리 기준, 재난안전·감염병·공공안전 기준을 운영 점수화한 룰입니다. "
        f"{', '.join(basis_tags)} 근거로 중요도 {importance}, 긴급도 {urgency}로 분류했습니다."
    )
    return {
        "importance": importance,
        "urgency": urgency,
        "basis_tags": basis_tags,
        "reason": reason,
        "confidence": confidence,
        "needs_review": confidence < 0.7,
    }


def _rule_v3_primary_text(row: dict) -> str:
    return _normalize_space(
        row.get("representative_text")
        or row.get("Q_refined")
        or row.get("text")
        or ""
    )


def _rule_v3_body_text(row: dict) -> str:
    return " ".join([
        _rule_v3_primary_text(row),
        _normalize_space(row.get("keywords")),
        _normalize_space(row.get("content_keywords")),
    ]).strip()


def _rule_v3_meta_text(row: dict) -> str:
    return " ".join([
        _normalize_space(row.get("category")),
        _normalize_space(row.get("subcategory")),
        _normalize_space(row.get("predication")),
        _normalize_space(row.get("department")),
    ]).strip()


def _rule_v3_compact(text: str) -> str:
    return _normalize_keyword(text)


def _rule_v3_has_any(text: str, compact: str, keywords: set[str]) -> bool:
    for keyword in keywords:
        if not keyword:
            continue
        normalized = _normalize_keyword(keyword)
        if keyword in text or (len(normalized) >= 3 and normalized in compact):
            return True
    return False


def _rule_v3_matches(text: str, compact: str, keywords: set[str]) -> list[str]:
    matches = []
    for keyword in sorted(keywords):
        if not keyword:
            continue
        normalized = _normalize_keyword(keyword)
        if keyword in text or (len(normalized) >= 3 and normalized in compact):
            matches.append(keyword)
    return matches


def _rule_v3_append_matches(target: list[str], text: str, compact: str, keywords: set[str]) -> None:
    target.extend(keyword for keyword in _rule_v3_matches(text, compact, keywords) if keyword not in target)


def _rule_v3_has_symptom_signal(text: str) -> bool:
    symptom_text = text.replace("건설사", "")
    return any(keyword in symptom_text for keyword in {"구토", "설사", "복통"})


def _rule_v3_has_food_poisoning_signal(text: str, compact: str) -> bool:
    core_keywords = RULE_V3_FOOD_POISONING_KEYWORDS - {"구토", "설사", "복통"}
    return _rule_v3_has_any(text, compact, core_keywords) or _rule_v3_has_symptom_signal(text)


def _rule_v3_has_health_spread_signal(text: str, compact: str) -> bool:
    core_keywords = RULE_V3_HEALTH_SPREAD_KEYWORDS - {"구토", "설사", "여러 사람", "여러 명", "다수"}
    return _rule_v3_has_any(text, compact, core_keywords) or _rule_v3_has_symptom_signal(text)


def _rule_v3_tag_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value if str(part)]
    return [part for part in str(value or "").split("|") if part]


def validate_rule_v3_label(label: dict) -> list[str]:
    flags = []
    importance = _label_int(label.get("importance"))
    urgency = _label_int(label.get("urgency"))
    priority_type = _normalize_space(label.get("priority_type"))
    tags = set(_rule_v3_tag_list(label.get("evidence_tags")))
    counter = set(_rule_v3_tag_list(label.get("counter_evidence")))

    if priority_type == "SIMPLE_INQUIRY" and (importance >= 3 or urgency >= 3):
        flags.append("SIMPLE_INQUIRY_HIGH_SCORE")
    if importance == 5 and not tags.intersection(RULE_V3_STRONG_EVIDENCE_TAGS):
        flags.append("CRITICAL_WITHOUT_STRONG_EVIDENCE")
    if priority_type == "FACILITY_ENVIRONMENT" and importance >= 5:
        flags.append("ENVIRONMENT_CRITICAL_REVIEW")
    if "PAYMENT_ADMIN_SIGNAL" in counter and importance >= 4:
        flags.append("PAYMENT_ADMIN_HIGH_SCORE_REVIEW")
    if importance >= 4 and not tags:
        flags.append("HIGH_SCORE_WITHOUT_EVIDENCE")
    if urgency > importance and priority_type not in {"SIMPLE_INQUIRY", "ROUTINE_ADMIN"}:
        flags.append("URGENCY_EXCEEDS_IMPORTANCE_REVIEW")
    return flags


def _finalize_rule_v3_label(label: dict) -> dict:
    validation_flags = validate_rule_v3_label(label)
    confidence = _label_float(label.get("confidence"))
    needs_review = bool(validation_flags) or confidence < 0.75
    label["validation_flags"] = validation_flags
    label["needs_review"] = needs_review
    label["trainable"] = not needs_review
    label["model"] = "rubric_rules_v3"
    label["label_source"] = "row_rule_v3"
    return label


def _rule_v3_label(
    importance: int,
    urgency: int,
    priority_type: str,
    cap_reason: str,
    evidence_tags: list[str],
    risk_evidence: list[str],
    counter_evidence: list[str],
    confidence: float,
    reason: str,
) -> dict:
    return _finalize_rule_v3_label({
        "importance": importance,
        "urgency": urgency,
        "priority_type": priority_type,
        "cap_reason": cap_reason,
        "evidence_tags": _unique_ordered(evidence_tags),
        "risk_evidence": _unique_ordered(risk_evidence),
        "counter_evidence": _unique_ordered(counter_evidence),
        "confidence": round(confidence, 3),
        "reason": reason,
    })


def build_rule_v3_label(row: dict) -> dict:
    body_text = _rule_v3_body_text(row)
    meta_text = _rule_v3_meta_text(row)
    all_text = " ".join([body_text, meta_text]).strip()
    body_compact = _rule_v3_compact(body_text)
    all_compact = _rule_v3_compact(all_text)
    predication = _normalize_space(row.get("predication"))
    cluster_size = _cluster_size_num(row)

    risk_evidence = []
    counter_evidence = []

    has_inquiry_signal = (
        "문의" in predication
        or "질의" in predication
        or _rule_v3_has_any(all_text, all_compact, RULE_V3_INQUIRY_KEYWORDS)
    )
    has_payment = _rule_v3_has_any(body_text, body_compact, RULE_V3_PAYMENT_KEYWORDS)
    has_enforcement = _rule_v3_has_any(body_text, body_compact, RULE_V3_ENFORCEMENT_KEYWORDS)
    has_action = _rule_v3_has_any(body_text, body_compact, RULE_V3_ACTION_KEYWORDS)
    has_environment = _rule_v3_has_any(body_text, body_compact, RULE_V3_ENVIRONMENT_KEYWORDS)
    has_visibility = _rule_v3_has_any(body_text, body_compact, RULE_V3_VISIBILITY_KEYWORDS)
    has_traffic_context = _rule_v3_has_any(body_text, body_compact, RULE_V3_TRAFFIC_CONTEXT_KEYWORDS)
    has_accident_risk = _rule_v3_has_any(body_text, body_compact, RULE_V3_ACCIDENT_RISK_KEYWORDS)
    has_hydrant = _rule_v3_has_any(body_text, body_compact, RULE_V3_HYDRANT_KEYWORDS)
    has_fire_suppression = _rule_v3_has_any(body_text, body_compact, RULE_V3_FIRE_SUPPRESSION_KEYWORDS)
    has_food_poisoning = _rule_v3_has_food_poisoning_signal(body_text, body_compact)
    has_health_spread = _rule_v3_has_health_spread_signal(body_text, body_compact)
    has_sanitation = _rule_v3_has_any(body_text, body_compact, RULE_V3_SANITATION_KEYWORDS)
    has_disaster = _rule_v3_has_any(body_text, body_compact, RULE_V3_DISASTER_KEYWORDS)
    has_injury = _rule_v3_has_any(body_text, body_compact, RULE_V3_INJURY_KEYWORDS)
    has_security = _rule_v3_has_any(body_text, body_compact, RULE_V3_SECURITY_KEYWORDS)
    has_accessibility = _rule_v3_has_any(body_text, body_compact, RULE_V3_ACCESSIBILITY_KEYWORDS)
    has_vulnerable = _rule_v3_has_any(body_text, body_compact, RULE_V3_VULNERABLE_KEYWORDS)
    has_multiple = cluster_size >= 10 or _rule_v3_has_any(body_text, body_compact, RULE_V3_MULTIPLE_KEYWORDS)
    has_child_zone = _rule_v3_has_any(body_text, body_compact, {"어린이보호구역", "스쿨존", "통학로"})
    has_crosswalk_or_busstop = _rule_v3_has_any(body_text, body_compact, {"횡단보도", "버스정류장"})
    has_illegal_parking = _rule_v3_has_any(
        body_text,
        body_compact,
        {"불법주차", "불법 주차", "불법주정차", "불법 주정차", "주정차"},
    )

    food_cluster = has_food_poisoning and (
        has_multiple
        or _rule_v3_has_any(body_text, body_compact, {"여러 사람", "여러 명", "집단", "구토", "설사", "응급실", "급식"})
    )
    health_spread = has_health_spread and (
        has_multiple
        or _rule_v3_has_any(body_text, body_compact, {"집단감염", "감염 확산", "전염", "응급실", "급식"})
    )
    fire_suppression_blocked = has_hydrant and (has_fire_suppression or _rule_v3_has_any(body_text, body_compact, {"화재", "불이 났", "불났"}))
    preventive_safety = has_disaster and _rule_v3_has_any(
        all_text,
        all_compact,
        {"발생시", "발생 시", "예방", "대비", "행동요령", "피난", "설계", "검토", "대피를 위해"},
    ) and not has_injury and not _rule_v3_has_any(body_text, body_compact, {"지금", "현재", "방금", "새고", "누출", "불이 났", "불났"})
    direct_disaster = has_disaster and not preventive_safety and (
        has_accident_risk
        or has_injury
        or _rule_v3_has_any(body_text, body_compact, {"즉시", "긴급", "새고", "누출"})
    )
    simple_payment_inquiry = has_payment and has_inquiry_signal and not has_enforcement
    simple_inquiry = has_inquiry_signal and not has_action and not has_enforcement
    information_only_risk_topic = simple_inquiry and _rule_v3_has_any(
        all_text,
        all_compact,
        {"행동요령", "담당", "담당부서", "어디", "알고 싶", "알려", "방법", "절차", "문의"},
    )

    if simple_payment_inquiry:
        _rule_v3_append_matches(counter_evidence, body_text, body_compact, RULE_V3_PAYMENT_KEYWORDS)
        return _rule_v3_label(
            1,
            1,
            "SIMPLE_INQUIRY",
            "SIMPLE_INQUIRY_CAP",
            ["STANDARD_INQUIRY"],
            [],
            ["PAYMENT_ADMIN_SIGNAL", "NO_ACTION_REQUEST"],
            0.94,
            "납부·세금·과태료 절차를 묻는 단순 정보 요청이라 중요도와 긴급도를 낮게 둡니다.",
        )

    if fire_suppression_blocked:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_HYDRANT_KEYWORDS | RULE_V3_FIRE_SUPPRESSION_KEYWORDS)
        return _rule_v3_label(
            5,
            5,
            "LIFE_SAFETY",
            "DIRECT_LIFE_SAFETY",
            ["FIRE_SUPPRESSION_BLOCKED", "ILLEGAL_PARKING"],
            risk_evidence,
            [],
            0.93,
            "소화전·화재진압 방해는 실제 화재 대응 지연으로 이어질 수 있어 최우선 처리 대상으로 둡니다.",
        )

    if food_cluster:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_FOOD_POISONING_KEYWORDS | RULE_V3_HEALTH_SPREAD_KEYWORDS)
        return _rule_v3_label(
            5,
            5,
            "PUBLIC_HEALTH",
            "PUBLIC_HEALTH_SPREAD",
            ["FOOD_POISONING_CLUSTER", "PUBLIC_HEALTH_SPREAD"],
            risk_evidence,
            [],
            0.92,
            "여러 사람의 식중독·구토·설사·응급실 등 확산성 보건 위험 신호가 있어 긴급 점검 대상으로 둡니다.",
        )

    if health_spread:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_HEALTH_SPREAD_KEYWORDS)
        return _rule_v3_label(
            5,
            5,
            "PUBLIC_HEALTH",
            "PUBLIC_HEALTH_SPREAD",
            ["PUBLIC_HEALTH_SPREAD"],
            risk_evidence,
            [],
            0.9,
            "감염 확산이나 집단 보건 위험 신호가 있어 지연 시 피해가 커질 수 있습니다.",
        )

    if has_security:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_SECURITY_KEYWORDS)
        return _rule_v3_label(
            5,
            5,
            "SECURITY_RISK",
            "SECURITY_CRITICAL",
            ["SECURITY_RISK"],
            risk_evidence,
            [],
            0.9,
            "테러·폭발물·보안시설 등 공공안전 위협 신호가 있어 최상위 우선순위로 둡니다.",
        )

    if information_only_risk_topic:
        return _rule_v3_label(
            1,
            1,
            "SIMPLE_INQUIRY",
            "SIMPLE_INQUIRY_CAP",
            ["STANDARD_INQUIRY"],
            [],
            ["INFORMATION_ONLY_RISK_TOPIC", "NO_ACTION_REQUEST"],
            0.88,
            "안전·재난 주제를 포함하더라도 행동요령·담당부서·절차를 묻는 정보성 문의라 낮게 둡니다.",
        )

    if preventive_safety:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_DISASTER_KEYWORDS)
        return _rule_v3_label(
            4,
            3,
            "SAFETY_PREVENTION",
            "PREVENTIVE_SAFETY_CAP",
            ["PUBLIC_SAFETY_PREVENTION"],
            risk_evidence,
            ["NO_ACTIVE_EMERGENCY_SIGNAL"],
            0.82,
            "화재·대피·피난 같은 안전 주제이지만 현재 진행 중인 사고가 아니라 예방·설계 검토 요청으로 봅니다.",
        )

    if direct_disaster:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_DISASTER_KEYWORDS | RULE_V3_INJURY_KEYWORDS)
        tags = ["COLLAPSE_FLOOD_DISASTER" if has_disaster else "ACTIVE_INJURY"]
        if has_injury:
            tags.append("ACTIVE_INJURY")
        return _rule_v3_label(
            5,
            5 if _rule_v3_has_any(body_text, body_compact, {"즉시", "긴급", "지금", "대피", "누출", "화재"}) else 4,
            "LIFE_SAFETY",
            "DIRECT_LIFE_SAFETY",
            tags,
            risk_evidence,
            [],
            0.88,
            "화재·가스·감전·붕괴·침수·부상 등 직접 안전 위험이 명확해 높은 우선순위로 둡니다.",
        )

    if simple_inquiry:
        return _rule_v3_label(
            1,
            1,
            "SIMPLE_INQUIRY",
            "SIMPLE_INQUIRY_CAP",
            ["STANDARD_INQUIRY"],
            [],
            ["NO_ACTION_REQUEST"],
            0.9,
            "질문·절차·정보 확인 중심이며 직접 조치나 위험 근거가 없어 낮은 우선순위로 둡니다.",
        )

    if has_accessibility and has_illegal_parking:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_ACCESSIBILITY_KEYWORDS | {"불법주차", "불법 주차", "불법주정차", "불법 주정차"})
        return _rule_v3_label(
            4,
            3,
            "ACCESSIBILITY_ENFORCEMENT",
            "ACCESSIBILITY_PUBLIC_INTEREST",
            ["VULNERABLE_ACCESS", "ILLEGAL_PARKING"],
            risk_evidence,
            [],
            0.86,
            "장애인 주차구역 침해는 취약계층 이용권과 공익성이 있어 일반 단속 요청보다 높게 둡니다.",
        )

    if has_visibility and has_traffic_context and has_accident_risk:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_VISIBILITY_KEYWORDS | RULE_V3_TRAFFIC_CONTEXT_KEYWORDS | RULE_V3_ACCIDENT_RISK_KEYWORDS)
        return _rule_v3_label(
            4,
            4,
            "TRAFFIC_SAFETY",
            "EXPLICIT_SAFETY_RISK",
            ["VISIBILITY_BLOCKED", "ACCIDENT_RISK"],
            risk_evidence,
            [],
            0.87,
            "시야 가림과 사고 위험이 같이 나타나 단순 환경정비가 아니라 교통안전 민원으로 봅니다.",
        )

    if has_illegal_parking and (has_child_zone or has_crosswalk_or_busstop) and (has_vulnerable or has_accident_risk or has_child_zone):
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_TRAFFIC_CONTEXT_KEYWORDS | RULE_V3_VULNERABLE_KEYWORDS)
        return _rule_v3_label(
            4,
            4 if has_accident_risk or has_child_zone else 3,
            "TRAFFIC_SAFETY",
            "PUBLIC_SAFETY_ENFORCEMENT",
            ["TRAFFIC_SAFETY", "ILLEGAL_PARKING"],
            risk_evidence,
            [],
            0.84,
            "횡단보도·버스정류장·어린이보호구역 등의 불법주정차는 공공 교통안전 위험으로 둡니다.",
        )

    if has_environment:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_ENVIRONMENT_KEYWORDS)
        if _rule_v3_has_any(body_text, body_compact, {"보행불편", "통행불편", "해충", "모기", "악취"}):
            return _rule_v3_label(
                3,
                2,
                "FACILITY_ENVIRONMENT",
                "LIMITED_PUBLIC_INCONVENIENCE",
                ["ENVIRONMENT_MAINTENANCE"],
                risk_evidence,
                ["NO_DIRECT_LIFE_SAFETY_SIGNAL"],
                0.8,
                "제초·수목 정비 요청이지만 직접 사고·생명 위험 근거는 없어 중간 이하로 제한합니다.",
            )
        return _rule_v3_label(
            2,
            2,
            "FACILITY_ENVIRONMENT",
            "NO_EXPLICIT_SAFETY_RISK",
            ["WEEDS"],
            risk_evidence,
            ["NO_EXPLICIT_SAFETY_RISK"],
            0.82,
            "도로변 풀·잡초 정비는 명시적 사고 위험이 없으면 통상 환경정비 요청으로 처리합니다.",
        )

    if has_enforcement:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_ENFORCEMENT_KEYWORDS)
        return _rule_v3_label(
            3,
            3,
            "ENFORCEMENT",
            "ENFORCEMENT_CAP",
            ["LEGAL_ENFORCEMENT"],
            risk_evidence,
            ["NO_DIRECT_LIFE_SAFETY_SIGNAL"],
            0.83,
            "불법행위 확인·과태료·단속 요청이지만 직접 생명·재난 위험 근거가 없어 일반 조치 수준으로 제한합니다.",
        )

    if has_sanitation:
        _rule_v3_append_matches(risk_evidence, body_text, body_compact, RULE_V3_SANITATION_KEYWORDS)
        return _rule_v3_label(
            3,
            3,
            "PUBLIC_HEALTH_INSPECTION",
            "SANITATION_INSPECTION",
            ["SANITATION_RISK"],
            risk_evidence,
            ["NO_SPREAD_OR_EMERGENCY_SIGNAL"],
            0.79,
            "위생·오염 점검 필요성은 있으나 집단 증상이나 응급 신호가 없어 일반 점검 우선순위로 둡니다.",
        )

    if has_multiple or has_action:
        return _rule_v3_label(
            3,
            2,
            "ROUTINE_ACTION",
            "NO_EXPLICIT_HIGH_RISK",
            ["ROUTINE_ACTION_REQUEST"],
            [],
            ["NO_EXPLICIT_HIGH_RISK"],
            0.76,
            "구체적 조치 요청은 있으나 안전·보건·다수 피해의 강한 근거가 없어 일반 처리 대상으로 둡니다.",
        )

    return _rule_v3_label(
        2,
        2,
        "ROUTINE_ADMIN",
        "LOW_SIGNAL_DEFAULT",
        ["ROUTINE_REQUEST"],
        [],
        ["LOW_RISK_SIGNAL"],
        0.72,
        "명확한 위험 근거가 부족한 일반 민원이라 보수적으로 낮은 기본 점수를 부여하고 검토 대상으로 둡니다.",
    )


def _label_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _label_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _split_basis_tags(value) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value if str(part)]
    return [part for part in str(value or "").split("|") if part]


def merge_priority_label(cluster_label: dict, row_label: dict) -> dict:
    cluster_priority = (_label_int(cluster_label.get("importance")), _label_int(cluster_label.get("urgency")))
    row_priority = (_label_int(row_label.get("importance")), _label_int(row_label.get("urgency")))
    if row_priority <= cluster_priority:
        return cluster_label

    base_source = str(cluster_label.get("label_source") or "cluster_label")
    merged = dict(row_label)
    merged["basis_tags"] = _split_basis_tags(row_label.get("basis_tags"))
    merged["model"] = "rubric_rules_v2_row_escalation"
    merged["label_source"] = f"row_rule_escalated_from_{base_source}"
    merged["reason"] = (
        "행 본문의 위험 신호가 군집 대표 라벨보다 강해 상향 보정했습니다. "
        f"{row_label.get('reason', '')}"
    )
    return merged


def _rule_row_from_propagated_record(record: dict) -> dict:
    return {
        "representative_text": _normalize_space(record.get("Q_refined") or record.get("text")),
        "category": _normalize_space(record.get("category")),
        "subcategory": _normalize_space(record.get("subcategory")),
        "predication": _normalize_space(record.get("predication")),
        "department": _normalize_space(record.get("department")),
        "keywords": _normalize_space(record.get("keywords")),
        "content_keywords": _normalize_space(record.get("content_keywords")),
        "cluster_size": _normalize_space(record.get("cluster_size") or "1"),
    }


def _apply_row_rule_escalation(propagated):
    needed_cols = [
        "Q_refined", "category", "subcategory", "predication", "department", "keywords",
        "content_keywords", "cluster_size", "importance", "urgency", "basis_tags", "reason",
        "confidence", "needs_review", "model", "label_source",
    ]
    available_cols = [col for col in needed_cols if col in propagated.columns]
    updates = {col: [] for col in ["importance", "urgency", "basis_tags", "reason", "confidence", "needs_review", "model", "label_source"]}
    records = propagated[available_cols].to_dict("records")
    for record in _progress(records, total=len(records), desc="export:row-rules", unit="row"):
        cluster_label = {
            "importance": _label_int(record.get("importance")),
            "urgency": _label_int(record.get("urgency")),
            "basis_tags": _split_basis_tags(record.get("basis_tags")),
            "reason": _normalize_space(record.get("reason")),
            "confidence": _label_float(record.get("confidence")),
            "needs_review": _bool_to_text(record.get("needs_review")) == "true",
            "model": _normalize_space(record.get("model")),
            "label_source": _normalize_space(record.get("label_source")),
        }
        row_label = build_rule_label(_rule_row_from_propagated_record(record))
        merged = merge_priority_label(cluster_label, row_label)
        updates["importance"].append(str(_label_int(merged.get("importance"))))
        updates["urgency"].append(str(_label_int(merged.get("urgency"))))
        updates["basis_tags"].append("|".join(_split_basis_tags(merged.get("basis_tags"))))
        updates["reason"].append(_normalize_space(merged.get("reason")))
        updates["confidence"].append(str(_label_float(merged.get("confidence"))))
        updates["needs_review"].append(_bool_to_text(merged.get("needs_review")))
        updates["model"].append(_normalize_space(merged.get("model")))
        updates["label_source"].append(_normalize_space(merged.get("label_source")))

    for col, values in updates.items():
        propagated[col] = values
    return propagated


def _content_keyword_count(value: str) -> int:
    return len([part for part in str(value or "").split("|") if part.strip()])


def _should_auto_link_keyword_bucket(df, group_indices, args: argparse.Namespace) -> bool:
    if args.disable_keyword_bucket_auto_link:
        return False
    if len(group_indices) > args.keyword_bucket_auto_link_size:
        return False
    first = df.loc[group_indices[0]]
    if _content_keyword_count(first.get("content_keywords", "")) < args.min_auto_link_keywords:
        return False
    required = ["category", "subcategory", "predication"]
    return all(_normalize_space(first.get(column, "")) for column in required)


def _choose_length_central_representative(df, indices):
    lengths = [len(str(df.loc[idx, "Q_refined"])) for idx in indices]
    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2]
    best_pos = min(range(len(indices)), key=lambda pos: (abs(lengths[pos] - median), lengths[pos]))
    return indices[best_pos]


def command_cluster(args: argparse.Namespace) -> None:
    _require_python()
    pd = _lazy_pandas()
    libs = _lazy_cluster_libs()
    np_module, _, _, MiniBatchKMeans, TfidfVectorizer, _ = libs

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    _ensure_parent(output_path)
    _ensure_parent(summary_path)

    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str).fillna("")
    if "cluster_text_norm" not in df.columns:
        df["cluster_text_norm"] = df["Q_refined"].map(_normalize_text_for_cluster)
    if "bucket_key" not in df.columns:
        df["bucket_key"] = df.apply(
            lambda row: _bucket_key(row.get("category", ""), row.get("subcategory", ""), row.get("predication", ""), row.get("content_keywords", "")),
            axis=1,
        )

    df["cluster_id"] = ""
    df["cluster_size"] = 0
    df["is_cluster_representative"] = False
    summaries = []
    cluster_counter = 0

    grouped = df.groupby("bucket_key", sort=False).groups
    total_groups = len(grouped)
    print(f"Clustering {len(df):,} complaints across {total_groups:,} buckets")

    group_iter = _progress(grouped.items(), total=total_groups, desc="cluster:buckets", unit="bucket")
    for group_no, (bucket_key, index_obj) in enumerate(group_iter, start=1):
        group_indices = list(index_obj)
        texts = df.loc[group_indices, "Q_refined"].fillna("").tolist()
        norm_texts = df.loc[group_indices, "cluster_text_norm"].fillna("").tolist()
        if hasattr(group_iter, "set_postfix"):
            group_iter.set_postfix(rows=len(group_indices), clusters=cluster_counter)

        if len(group_indices) == 1:
            components = [(group_indices, group_indices[0])]
        elif _should_auto_link_keyword_bucket(df, group_indices, args):
            components = [(group_indices, _choose_length_central_representative(df, group_indices))]
        else:
            try:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(args.ngram_min, args.ngram_max),
                    min_df=1,
                    max_features=args.max_features,
                    dtype=np_module.float32,
                )
                x_group = vectorizer.fit_transform(texts)
                if len(group_indices) > args.large_bucket_size:
                    k = max(2, int(math.ceil(len(group_indices) / args.large_bucket_size)))
                    k = min(k, len(group_indices))
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=args.random_state,
                        batch_size=args.kmeans_batch_size,
                        n_init=3,
                    )
                    sub_labels = kmeans.fit_predict(x_group)
                    components = []
                    for sub_label in sorted(set(sub_labels)):
                        local_positions = [pos for pos, value in enumerate(sub_labels) if value == sub_label]
                        sub_indices = [group_indices[pos] for pos in local_positions]
                        components.extend(
                            _process_subbucket(
                                df,
                                sub_indices,
                                x_group[local_positions],
                                args.similarity_threshold,
                                args.n_neighbors,
                                libs,
                            )
                        )
                else:
                    components = _process_subbucket(
                        df,
                        group_indices,
                        x_group,
                        args.similarity_threshold,
                        args.n_neighbors,
                        libs,
                    )
            except Exception as exc:
                _progress_write(f"[cluster] fallback exact grouping for bucket {bucket_key[:80]}: {exc}")
                labels = _fallback_exact_clusters(norm_texts)
                by_label = defaultdict(list)
                for idx, label in zip(group_indices, labels):
                    by_label[label].append(idx)
                components = [(indices, indices[0]) for indices in by_label.values()]

        for indices, rep_idx in components:
            cluster_counter += 1
            cluster_id = f"C{cluster_counter:08d}"
            df.loc[indices, "cluster_id"] = cluster_id
            df.loc[indices, "cluster_size"] = len(indices)
            df.loc[rep_idx, "is_cluster_representative"] = True
            rep = df.loc[rep_idx].to_dict()
            summary_row = {
                "cluster_id": cluster_id,
                "cluster_size": len(indices),
                "representative_id": rep.get("id", ""),
                "representative_text": rep.get("Q_refined", ""),
                "category": rep.get("category", ""),
                "subcategory": rep.get("subcategory", ""),
                "predication": rep.get("predication", ""),
                "department": rep.get("department", ""),
                "keywords": rep.get("keywords", ""),
                "content_keywords": rep.get("content_keywords", ""),
                "related_law": rep.get("related_law", ""),
                "bucket_key": bucket_key,
            }
            summary_row["risk_score"] = _risk_score_for_row(summary_row)
            summaries.append(summary_row)

    print("[cluster] writing clustered complaints CSV")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries)
    summary_df.sort_values(["risk_score", "cluster_size"], ascending=[False, False], inplace=True)
    print("[cluster] writing cluster summary CSV")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path} and {summary_path}")


def command_sample(args: argparse.Namespace) -> None:
    _require_python()
    pd = _lazy_pandas()
    cluster_summary_path = Path(args.cluster_summary)
    output_path = Path(args.output)
    _ensure_parent(output_path)

    df = pd.read_csv(cluster_summary_path, encoding="utf-8-sig", dtype=str).fillna("")
    if df.empty:
        raise RuntimeError("cluster_summary is empty")

    df["cluster_size_num"] = pd.to_numeric(df["cluster_size"], errors="coerce").fillna(1).clip(lower=1)
    if "risk_score" not in df.columns:
        df["risk_score"] = df.apply(_risk_score_for_row, axis=1)
    df["risk_score_num"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    df["priority"] = df["risk_score_num"] * 1000 + df["cluster_size_num"].map(lambda value: math.log1p(float(value)))
    df["sample_key"] = df["category"].astype(str) + "|" + df["subcategory"].astype(str)

    target_size = min(args.target_size, len(df))
    selected_indices = set()

    group_weights = df.groupby("sample_key")["cluster_size_num"].sum().map(lambda value: math.sqrt(float(value)))
    weight_sum = float(group_weights.sum()) or 1.0
    allocations = {
        key: max(1, int(round(target_size * float(weight) / weight_sum)))
        for key, weight in group_weights.items()
    }

    for key, allocation in _progress(allocations.items(), total=len(allocations), desc="sample:strata", unit="group"):
        if len(selected_indices) >= target_size:
            continue
        group = df[df["sample_key"] == key].sort_values(["risk_score_num", "cluster_size_num", "priority"], ascending=False)
        for idx in group.head(allocation).index:
            if len(selected_indices) >= target_size:
                break
            selected_indices.add(idx)

    if len(selected_indices) < target_size:
        remaining = df.drop(index=list(selected_indices), errors="ignore").sort_values(
            ["risk_score_num", "cluster_size_num", "priority"],
            ascending=False,
        )
        for idx in _progress(list(remaining.index), total=len(remaining), desc="sample:fill", unit="cluster"):
            selected_indices.add(idx)
            if len(selected_indices) >= target_size:
                break

    selected = df.loc[sorted(selected_indices)].copy()
    selected.sort_values(["risk_score_num", "cluster_size_num"], ascending=False, inplace=True)
    selected["label_status"] = "pending"
    selected.drop(columns=["cluster_size_num", "risk_score_num", "priority", "sample_key"], errors="ignore", inplace=True)
    print("[sample] writing label targets CSV")
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(selected):,} label targets to {output_path}")


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        return


def _read_rubric_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_label_prompt(row: dict, rubric_text: str) -> str:
    return f"""
다음은 행정 민원 중요도/긴급도 라벨링 작업입니다.
아래 기준을 반드시 따르고, 최종 답변은 JSON 객체 하나만 출력하세요.

[라벨링 기준]
{rubric_text}

[민원 정보]
- 대표 민원 ID: {row.get("representative_id", "")}
- 대분류: {row.get("category", "")}
- 세부분류: {row.get("subcategory", "")}
- 민원 의도: {row.get("predication", "")}
- 담당부서: {row.get("department", "")}
- 키워드: {row.get("keywords", "")}
- 같은 내용 클러스터 크기: {row.get("cluster_size", "")}
- 민원 본문: {row.get("representative_text", "")}

[출력 JSON 스키마]
{{
  "importance": 1,
  "urgency": 1,
  "basis_tags": ["STANDARD_INQUIRY"],
  "reason": "판단 근거를 한국어 한 문장으로 작성",
  "confidence": 0.0,
  "needs_review": false
}}

규칙:
- importance와 urgency는 1부터 5까지의 정수입니다.
- confidence는 0.0부터 1.0까지의 숫자입니다.
- basis_tags는 영문 대문자 태그 배열입니다.
- 근거가 애매하면 confidence를 낮추고 needs_review를 true로 설정하세요.
- JSON 바깥의 설명, 마크다운, 코드블록을 절대 쓰지 마세요.
""".strip()


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(cleaned)):
        char = cleaned[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : pos + 1])
    raise ValueError("Unclosed JSON object in model response")


def _validate_label(payload: dict) -> dict:
    result = dict(payload)
    for key in ["importance", "urgency"]:
        value = int(result.get(key))
        if value < 1 or value > 5:
            raise ValueError(f"{key} out of range: {value}")
        result[key] = value

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    result["confidence"] = confidence

    basis_tags = result.get("basis_tags", [])
    if isinstance(basis_tags, str):
        basis_tags = [basis_tags]
    if not isinstance(basis_tags, list):
        basis_tags = []
    result["basis_tags"] = [str(tag).strip().upper() for tag in basis_tags if str(tag).strip()]

    reason = _normalize_space(result.get("reason"))
    result["reason"] = reason

    needs_review = bool(result.get("needs_review", False))
    if confidence < 0.65:
        needs_review = True
    if result["importance"] >= 4 and len(reason) < 8:
        needs_review = True
    result["needs_review"] = needs_review
    return result


def _existing_cluster_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    existing = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("cluster_id"):
                existing.add(row["cluster_id"])
    return existing


def _batch_key_for_row(row: dict) -> str:
    cluster_id = _normalize_space(row.get("cluster_id"))
    if cluster_id:
        return cluster_id
    fallback = "|".join([
        _normalize_space(row.get("representative_id")),
        _normalize_space(row.get("representative_text")),
    ])
    return f"missing-cluster-{_stable_hash(fallback)}"


def _row_needs_label(row: dict, include_completed: bool) -> bool:
    if include_completed:
        return True
    status = _normalize_space(row.get("label_status")).lower()
    return status in {"", "pending", "failed", "needs_review"}


def build_batch_generate_content_request(row: dict, rubric_text: str) -> dict:
    prompt = _build_label_prompt(row, rubric_text)
    return {
        "key": _batch_key_for_row(row),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "config": {
                "response_mime_type": "application/json",
                "system_instruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION}],
                },
            },
        },
    }


def _count_batch_target_rows(targets_path: Path, limit: Optional[int], include_completed: bool) -> int:
    count = 0
    with targets_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if limit is not None and count >= limit:
                break
            if not _row_needs_label(row, include_completed):
                continue
            count += 1
    return count


def _count_nonblank_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_batch_requests_jsonl(
    targets_path: Path,
    output_path: Path,
    rubric_text: str,
    limit: Optional[int] = None,
    include_completed: bool = False,
    map_output_path: Optional[Path] = None,
    show_progress: bool = False,
) -> int:
    _ensure_parent(output_path)
    if map_output_path is not None:
        _ensure_parent(map_output_path)

    count = 0
    progress = _progress_bar(_count_batch_target_rows(targets_path, limit, include_completed), desc="batch:prepare", unit="req") if show_progress else None
    with targets_path.open("r", encoding="utf-8-sig", newline="") as input_handle, output_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        reader = csv.DictReader(input_handle)
        map_handle = None
        map_writer = None
        try:
            if map_output_path is not None:
                map_handle = map_output_path.open("w", encoding="utf-8-sig", newline="")
                map_writer = csv.DictWriter(map_handle, fieldnames=["key", "cluster_id", "representative_id"])
                map_writer.writeheader()

            for row in reader:
                if limit is not None and count >= limit:
                    break
                if not _row_needs_label(row, include_completed):
                    continue
                record = build_batch_generate_content_request(row, rubric_text)
                out_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if map_writer is not None:
                    map_writer.writerow({
                        "key": record["key"],
                        "cluster_id": _normalize_space(row.get("cluster_id")),
                        "representative_id": _normalize_space(row.get("representative_id")),
                    })
                count += 1
                if progress is not None:
                    progress.update(1)
        finally:
            if map_handle is not None:
                map_handle.close()
            if progress is not None:
                progress.close()
    return count


def _load_batch_request_map(path: Optional[Path]) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["key"]: {
                "cluster_id": row.get("cluster_id", ""),
                "representative_id": row.get("representative_id", ""),
            }
            for row in csv.DictReader(handle)
            if row.get("key")
        }


def _record_key(record: dict) -> str:
    key = _normalize_space(record.get("key"))
    if key:
        return key
    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict):
        return _normalize_space(metadata.get("key") or metadata.get("cluster_id"))
    return ""


def _record_metadata(record: dict, request_lookup: Optional[dict[str, dict[str, str]]] = None) -> dict[str, str]:
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    key = _record_key(record)
    mapped = request_lookup.get(key, {}) if request_lookup else {}
    cluster_id = _normalize_space(metadata.get("cluster_id") or mapped.get("cluster_id") or key)
    representative_id = _normalize_space(metadata.get("representative_id") or mapped.get("representative_id"))
    return {
        "key": key,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
    }


def _record_error(record: dict) -> Optional[dict]:
    if isinstance(record.get("error"), dict):
        return record["error"]
    if isinstance(record.get("status"), dict):
        return record["status"]
    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("error"), dict):
        return output["error"]
    return None


def _record_response(record: dict) -> Optional[dict]:
    if isinstance(record.get("response"), dict):
        return record["response"]
    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("response"), dict):
        return output["response"]
    return None


def _response_text(response: dict) -> str:
    text = _normalize_space(response.get("text"))
    if text:
        return text
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("No candidates found in batch response")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    texts = [_normalize_space(part.get("text")) for part in parts if isinstance(part, dict)]
    raw_text = "\n".join([part for part in texts if part])
    if not raw_text:
        raise ValueError("No text part found in batch response")
    return raw_text


def batch_result_record_to_label_row(
    record: dict,
    model: str,
    request_lookup: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    metadata = _record_metadata(record, request_lookup)
    error = _record_error(record)
    if error is not None:
        return None, {
            "key": metadata["key"],
            "cluster_id": metadata["cluster_id"],
            "representative_id": metadata["representative_id"],
            "error": _normalize_space(error.get("message") or error.get("status") or error),
            "error_code": error.get("code", ""),
            "model": model,
        }

    try:
        response = _record_response(record)
        if response is None:
            raise ValueError("No response object found in batch result")
        raw_text = _response_text(response)
        label = _validate_label(_extract_json_object(raw_text))
        return {
            "cluster_id": metadata["cluster_id"],
            "representative_id": metadata["representative_id"],
            "importance": label["importance"],
            "urgency": label["urgency"],
            "basis_tags": "|".join(label["basis_tags"]),
            "reason": label["reason"],
            "confidence": label["confidence"],
            "needs_review": label["needs_review"],
            "model": model,
            "label_source": "gemma_batch_cluster_representative",
            "raw_response": raw_text,
        }, None
    except Exception as exc:
        return None, {
            "key": metadata["key"],
            "cluster_id": metadata["cluster_id"],
            "representative_id": metadata["representative_id"],
            "error": str(exc),
            "error_code": "",
            "model": model,
            "raw_record": record,
        }


def collect_batch_results_jsonl(
    results_path: Path,
    output_path: Path,
    failed_path: Path,
    model: str,
    resume: bool = False,
    request_map_path: Optional[Path] = None,
    show_progress: bool = False,
) -> dict[str, int]:
    _ensure_parent(output_path)
    _ensure_parent(failed_path)
    request_lookup = _load_batch_request_map(request_map_path)
    existing = _existing_cluster_ids(output_path) if resume else set()
    write_header = not output_path.exists() or not resume
    mode = "a" if resume else "w"
    stats = {"processed": 0, "failed": 0, "skipped": 0}
    progress = _progress_bar(_count_nonblank_lines(results_path), desc="batch:collect", unit="row") if show_progress else None

    with results_path.open("r", encoding="utf-8") as input_handle, output_path.open(mode, encoding="utf-8-sig", newline="") as out_handle, failed_path.open("a", encoding="utf-8") as fail_handle:
        writer = csv.DictWriter(out_handle, fieldnames=LABEL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for line_no, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail_handle.write(json.dumps({"line_no": line_no, "error": str(exc), "model": model}, ensure_ascii=False) + "\n")
                stats["failed"] += 1
                if progress is not None:
                    progress.update(1)
                continue

            label_row, failure = batch_result_record_to_label_row(record, model, request_lookup)
            if label_row is not None:
                if resume and label_row["cluster_id"] in existing:
                    stats["skipped"] += 1
                    if progress is not None:
                        progress.update(1)
                    continue
                writer.writerow(label_row)
                stats["processed"] += 1
            elif failure is not None:
                fail_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                stats["failed"] += 1
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return stats


def _batch_state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", "")
    if hasattr(state, "name"):
        return state.name
    return str(state)


def _batch_dest_file_name(batch_job: Any) -> str:
    dest = getattr(batch_job, "dest", None) or getattr(batch_job, "output", None)
    if dest is None:
        return ""
    return _normalize_space(getattr(dest, "file_name", "") or getattr(dest, "responses_file", "") or getattr(dest, "responsesFile", ""))


def _int_attr(value: Any, name: str) -> int:
    try:
        return int(getattr(value, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def batch_status_summary(batch_job: Any) -> dict:
    summary = {
        "job_name": getattr(batch_job, "name", ""),
        "state": _batch_state_name(batch_job),
        "dest_file_name": _batch_dest_file_name(batch_job),
    }
    stats = getattr(batch_job, "completion_stats", None)
    if stats is None:
        summary["progress_percent"] = None
        return summary

    successful = _int_attr(stats, "successful_count")
    failed = _int_attr(stats, "failed_count")
    incomplete = _int_attr(stats, "incomplete_count")
    done = successful + failed
    total = done + incomplete
    summary["completion_stats"] = {
        "successful": successful,
        "failed": failed,
        "incomplete": incomplete,
        "done": done,
        "total": total,
    }
    summary["progress_percent"] = round((done / total) * 100, 2) if total else None
    return summary


def _format_batch_status_line(summary: dict) -> str:
    stats = summary.get("completion_stats") or {}
    percent = summary.get("progress_percent")
    if percent is None:
        progress = "progress=unknown"
    else:
        progress = f"progress={percent:.2f}% done={stats.get('done', 0)}/{stats.get('total', 0)}"
    return f"{time.strftime('%Y-%m-%d %H:%M:%S')} state={summary.get('state', '')} {progress}"


def ensure_batch_model_supported(model: str) -> None:
    if "gemma" not in model.lower():
        return
    raise RuntimeError(
        "Gemma models are free on the normal generateContent API, but they do not support "
        "Gemini Batch API in this project. Use: python scripts\\pipeline.py label "
        "--targets outputs\\gemma_label_targets.csv --resume"
    )


def command_label(args: argparse.Namespace) -> None:
    _require_python()
    _load_dotenv_if_present()

    model = args.model or os.getenv("GEMMA_MODEL", DEFAULT_MODEL)
    if model != DEFAULT_MODEL:
        raise RuntimeError(
            f"This pipeline is fixed to {DEFAULT_MODEL}. "
            f"Refusing to use configured model: {model}"
        )
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in the environment or .env file.")

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

    rubric_text = _read_rubric_text(Path(args.rubric))
    targets_path = Path(args.targets)
    output_path = Path(args.output)
    failed_path = Path(args.failed_output)
    _ensure_parent(output_path)
    _ensure_parent(failed_path)

    client = genai.Client()
    existing = _existing_cluster_ids(output_path) if args.resume else set()

    with targets_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    write_header = not output_path.exists() or not args.resume
    mode = "a" if args.resume else "w"

    processed = 0
    skipped = 0
    failed = 0
    with output_path.open(mode, encoding="utf-8-sig", newline="") as out_handle, failed_path.open("a", encoding="utf-8") as fail_handle:
        writer = csv.DictWriter(out_handle, fieldnames=LABEL_FIELDNAMES)
        if write_header:
            writer.writeheader()

        row_iter = _progress(rows, total=len(rows), desc="label:gemma", unit="req")
        for row in row_iter:
            cluster_id = row.get("cluster_id", "")
            if args.resume and cluster_id in existing:
                skipped += 1
                if hasattr(row_iter, "set_postfix"):
                    row_iter.set_postfix(done=processed, skipped=skipped, failed=failed)
                continue
            prompt = _build_label_prompt(row, rubric_text)
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            system_instruction="You classify Korean civil complaints. Return only valid JSON.",
                        ),
                    )
                    raw_text = response.text or ""
                    label = _validate_label(_extract_json_object(raw_text))
                    writer.writerow({
                        "cluster_id": cluster_id,
                        "representative_id": row.get("representative_id", ""),
                        "importance": label["importance"],
                        "urgency": label["urgency"],
                        "basis_tags": "|".join(label["basis_tags"]),
                        "reason": label["reason"],
                        "confidence": label["confidence"],
                        "needs_review": label["needs_review"],
                        "model": model,
                        "label_source": "gemma4_31b_cluster_representative",
                        "raw_response": raw_text,
                    })
                    processed += 1
                    if hasattr(row_iter, "set_postfix"):
                        row_iter.set_postfix(done=processed, skipped=skipped, failed=failed)
                    if args.sleep_seconds:
                        time.sleep(args.sleep_seconds)
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep_seconds * attempt)
                    else:
                        fail_handle.write(json.dumps({
                            "cluster_id": cluster_id,
                            "representative_id": row.get("representative_id", ""),
                            "error": last_error,
                            "model": model,
                        }, ensure_ascii=False) + "\n")
                        fail_handle.flush()
                        failed += 1
                        if hasattr(row_iter, "set_postfix"):
                            row_iter.set_postfix(done=processed, skipped=skipped, failed=failed)

    print(f"Wrote {processed:,} labels to {output_path} (skipped={skipped:,}, failed={failed:,})")


def command_label_rules(args: argparse.Namespace) -> None:
    _require_python()
    targets_path = Path(args.targets)
    output_path = Path(args.output)
    _ensure_parent(output_path)
    existing = _existing_cluster_ids(output_path) if args.resume else set()

    with targets_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    write_header = not output_path.exists() or not args.resume
    mode = "a" if args.resume else "w"
    processed = 0
    skipped = 0
    with output_path.open(mode, encoding="utf-8-sig", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=LABEL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in _progress(rows, total=len(rows), desc="label:rules", unit="row"):
            cluster_id = _normalize_space(row.get("cluster_id"))
            if args.resume and cluster_id in existing:
                skipped += 1
                continue
            label = build_rule_label(row)
            raw_response = json.dumps(label, ensure_ascii=False)
            writer.writerow({
                "cluster_id": cluster_id,
                "representative_id": _normalize_space(row.get("representative_id")),
                "importance": label["importance"],
                "urgency": label["urgency"],
                "basis_tags": "|".join(label["basis_tags"]),
                "reason": label["reason"],
                "confidence": label["confidence"],
                "needs_review": label["needs_review"],
                "model": "rubric_rules_v2",
                "label_source": "rubric_rules_cluster_representative",
                "raw_response": raw_response,
            })
            processed += 1
    print(f"Wrote {processed:,} rule labels to {output_path} (skipped={skipped:,})")


def _rule_v3_output_fieldnames(input_fieldnames: list[str] | None) -> list[str]:
    fieldnames = list(input_fieldnames or [])
    for field in RULE_V3_LABEL_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def _serialize_rule_v3_label(label: dict) -> dict:
    serialized = {}
    for field in RULE_V3_LABEL_FIELDS:
        value = label.get(field, "")
        if field in RULE_V3_LIST_FIELDS:
            serialized[field] = "|".join(_rule_v3_tag_list(value))
        elif isinstance(value, bool):
            serialized[field] = _bool_to_text(value)
        else:
            serialized[field] = _normalize_space(value)
    return serialized


def _count_csv_data_rows(path: Path, limit: int | None = None) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        total = max(0, sum(1 for _ in handle) - 1)
    return min(total, limit) if limit is not None else total


def command_rule_v3(args: argparse.Namespace) -> None:
    _require_python()
    input_path = Path(args.input)
    all_output = Path(args.all_output)
    trainable_output = Path(args.trainable_output)
    needs_review_output = Path(args.needs_review_output)
    review_sample_output = Path(args.review_sample_output)
    report_output = Path(args.report_output)

    for path in [all_output, trainable_output, needs_review_output, review_sample_output, report_output]:
        _ensure_parent(path)

    total = _count_csv_data_rows(input_path, args.limit)
    rng = random.Random(args.random_state)
    processed = 0
    trainable_count = 0
    needs_review_count = 0
    importance_counts = Counter()
    urgency_counts = Counter()
    priority_type_counts = Counter()
    validation_flag_counts = Counter()
    review_seen = Counter()
    review_samples: dict[str, list[dict]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8-sig", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = _rule_v3_output_fieldnames(reader.fieldnames)
        with all_output.open("w", encoding="utf-8-sig", newline="") as all_handle, \
            trainable_output.open("w", encoding="utf-8-sig", newline="") as train_handle, \
            needs_review_output.open("w", encoding="utf-8-sig", newline="") as needs_handle:
            all_writer = csv.DictWriter(all_handle, fieldnames=fieldnames)
            train_writer = csv.DictWriter(train_handle, fieldnames=fieldnames)
            needs_writer = csv.DictWriter(needs_handle, fieldnames=fieldnames)
            all_writer.writeheader()
            train_writer.writeheader()
            needs_writer.writeheader()

            row_iter = _progress(reader, total=total, desc="rule-v3", unit="row")
            for row in row_iter:
                if args.limit is not None and processed >= args.limit:
                    break
                label = build_rule_v3_label(row)
                serialized_label = _serialize_rule_v3_label(label)
                out_row = dict(row)
                out_row.update(serialized_label)

                all_writer.writerow(out_row)
                if label["trainable"]:
                    train_writer.writerow(out_row)
                    trainable_count += 1
                else:
                    needs_writer.writerow(out_row)
                    needs_review_count += 1

                importance_key = str(label["importance"])
                importance_counts[importance_key] += 1
                urgency_counts[str(label["urgency"])] += 1
                priority_type_counts[label["priority_type"]] += 1
                validation_flag_counts.update(label["validation_flags"])

                review_seen[importance_key] += 1
                bucket = review_samples[importance_key]
                if len(bucket) < args.review_per_importance:
                    bucket.append(dict(out_row))
                else:
                    replace_at = rng.randint(1, review_seen[importance_key])
                    if replace_at <= args.review_per_importance:
                        bucket[replace_at - 1] = dict(out_row)

                processed += 1

    with review_sample_output.open("w", encoding="utf-8-sig", newline="") as sample_handle:
        sample_writer = csv.DictWriter(sample_handle, fieldnames=fieldnames)
        sample_writer.writeheader()
        for importance in sorted(review_samples, key=lambda value: int(value)):
            sample_writer.writerows(review_samples[importance])

    report = {
        "input": str(input_path),
        "outputs": {
            "all": str(all_output),
            "trainable": str(trainable_output),
            "needs_review": str(needs_review_output),
            "review_sample": str(review_sample_output),
        },
        "processed": processed,
        "trainable": trainable_count,
        "needs_review": needs_review_count,
        "importance_counts": dict(sorted(importance_counts.items(), key=lambda item: int(item[0]))),
        "urgency_counts": dict(sorted(urgency_counts.items(), key=lambda item: int(item[0]))),
        "priority_type_counts": dict(priority_type_counts.most_common()),
        "validation_flag_counts": dict(validation_flag_counts.most_common()),
        "review_sample_rows": sum(len(rows) for rows in review_samples.values()),
        "model": "rubric_rules_v3",
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_label_batch_prepare(args: argparse.Namespace) -> None:
    _require_python()
    rubric_text = _read_rubric_text(Path(args.rubric))
    count = write_batch_requests_jsonl(
        targets_path=Path(args.targets),
        output_path=Path(args.output),
        rubric_text=rubric_text,
        limit=args.limit,
        include_completed=args.include_completed,
        map_output_path=Path(args.map_output) if args.map_output else None,
        show_progress=True,
    )
    print(f"Wrote {count:,} batch requests to {args.output}")
    if args.map_output:
        print(f"Wrote request map to {args.map_output}")


def command_label_batch_create(args: argparse.Namespace) -> None:
    _require_python()
    _load_dotenv_if_present()

    model = args.model or os.getenv("GEMMA_MODEL", DEFAULT_MODEL)
    ensure_batch_model_supported(model)
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in the environment or .env file.")

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

    requests_output = Path(args.requests_output)
    map_output = Path(args.map_output)
    rubric_text = _read_rubric_text(Path(args.rubric))
    count = write_batch_requests_jsonl(
        targets_path=Path(args.targets),
        output_path=requests_output,
        rubric_text=rubric_text,
        limit=args.limit,
        include_completed=args.include_completed,
        map_output_path=map_output,
        show_progress=True,
    )
    if count == 0:
        raise RuntimeError("No batch requests were written. Check targets or --include-completed.")

    client = genai.Client()
    uploaded_file = client.files.upload(
        file=str(requests_output),
        config=types.UploadFileConfig(display_name=requests_output.stem, mime_type="jsonl"),
    )
    display_name = args.display_name or f"complaint-importance-{int(time.time())}"
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )

    job_info = {
        "job_name": getattr(batch_job, "name", ""),
        "state": _batch_state_name(batch_job),
        "model": model,
        "display_name": display_name,
        "uploaded_file": getattr(uploaded_file, "name", ""),
        "request_count": count,
        "requests_output": str(requests_output),
        "request_map": str(map_output),
    }
    job_output = Path(args.job_output)
    _ensure_parent(job_output)
    job_output.write_text(json.dumps(job_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(job_info, ensure_ascii=False, indent=2))


def command_label_batch_status(args: argparse.Namespace) -> None:
    _require_python()
    _load_dotenv_if_present()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in the environment or .env file.")

    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

    client = genai.Client()
    while True:
        batch_job = client.batches.get(name=args.job_name)
        status = batch_status_summary(batch_job)
        state = status["state"]
        dest_file_name = status["dest_file_name"]
        if args.watch:
            print(_format_batch_status_line(status), flush=True)
        if not args.watch or state in BATCH_COMPLETED_STATES:
            break
        time.sleep(args.poll_seconds)

    if args.download_output:
        if state not in BATCH_COMPLETED_STATES:
            raise RuntimeError(f"Batch job is not complete yet: {state}")
        if not dest_file_name:
            raise RuntimeError("Batch job has no downloadable result file.")
        output_path = Path(args.download_output)
        _ensure_parent(output_path)
        content = client.files.download(file=dest_file_name)
        output_path.write_bytes(content)
        status["download_output"] = str(output_path)

    if not args.watch:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status, ensure_ascii=False, indent=2))


def command_label_batch_collect(args: argparse.Namespace) -> None:
    _require_python()
    stats = collect_batch_results_jsonl(
        results_path=Path(args.results),
        output_path=Path(args.output),
        failed_path=Path(args.failed_output),
        model=args.model or DEFAULT_MODEL,
        resume=args.resume,
        request_map_path=Path(args.request_map) if args.request_map else None,
        show_progress=True,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _bool_to_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return "true" if text in {"true", "1", "yes"} else "false"


def command_export(args: argparse.Namespace) -> None:
    _require_python()
    pd = _lazy_pandas()

    print("[export] loading targets")
    targets = pd.read_csv(args.targets, encoding="utf-8-sig", dtype=str).fillna("")
    print("[export] loading labels")
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", dtype=str).fillna("")
    print("[export] loading clustered complaints")
    clustered = pd.read_csv(args.clustered, encoding="utf-8-sig", dtype=str).fillna("")

    label_cols = ["cluster_id", "importance", "urgency", "basis_tags", "reason", "confidence", "needs_review", "model", "label_source"]
    labels = labels[label_cols].drop_duplicates("cluster_id", keep="last")

    print("[export] building representative CSV")
    representatives = targets.merge(labels, on="cluster_id", how="inner")
    representatives["text"] = representatives["representative_text"]
    representatives["needs_review"] = representatives["needs_review"].map(_bool_to_text)
    rep_cols = [
        "text", "importance", "urgency", "category", "subcategory", "predication", "department",
        "keywords", "cluster_id", "cluster_size", "confidence", "needs_review", "basis_tags",
        "reason", "model", "label_source",
    ]

    rep_output = Path(args.representatives_output)
    propagated_output = Path(args.propagated_output)
    _ensure_parent(rep_output)
    _ensure_parent(propagated_output)
    representatives[rep_cols].to_csv(rep_output, index=False, encoding="utf-8-sig")

    print("[export] building cluster-propagated CSV")
    propagated = clustered.merge(labels, on="cluster_id", how="inner")
    propagated["text"] = propagated["Q_refined"]
    propagated["label_source"] = "cluster_propagated_from_" + propagated["label_source"].astype(str)
    propagated = _apply_row_rule_escalation(propagated)
    propagated["needs_review"] = propagated["needs_review"].map(_bool_to_text)
    prop_cols = [
        "text", "importance", "urgency", "category", "subcategory", "predication", "department",
        "keywords", "cluster_id", "cluster_size", "confidence", "needs_review", "basis_tags",
        "reason", "model", "label_source",
    ]
    propagated[prop_cols].to_csv(propagated_output, index=False, encoding="utf-8-sig")
    print(f"Wrote {rep_output} ({len(representatives):,} rows)")
    print(f"Wrote {propagated_output} ({len(propagated):,} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Civil complaint clustering and Gemma4 31B labeling pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    flatten = subparsers.add_parser("flatten", help="Extract labeled JSON zip files to a flat CSV")
    flatten.add_argument("--train-label-zip", default=None)
    flatten.add_argument("--validation-label-zip", default=None)
    flatten.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "complaints_flat.csv"))
    flatten.add_argument("--max-docs", type=int, default=None, help="Optional per-split document limit for smoke tests")
    flatten.set_defaults(func=command_flatten)

    cluster = subparsers.add_parser("cluster", help="Cluster similar complaints inside category/keyword buckets")
    cluster.add_argument("--input", default=str(DEFAULT_OUTPUT_DIR / "complaints_flat.csv"))
    cluster.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "complaints_clustered.csv"))
    cluster.add_argument("--summary-output", default=str(DEFAULT_OUTPUT_DIR / "cluster_summary.csv"))
    cluster.add_argument("--similarity-threshold", type=float, default=0.82)
    cluster.add_argument("--n-neighbors", type=int, default=20)
    cluster.add_argument("--large-bucket-size", type=int, default=20000)
    cluster.add_argument("--kmeans-batch-size", type=int, default=4096)
    cluster.add_argument("--max-features", type=int, default=50000)
    cluster.add_argument("--ngram-min", type=int, default=3)
    cluster.add_argument("--ngram-max", type=int, default=5)
    cluster.add_argument("--random-state", type=int, default=42)
    cluster.add_argument("--keyword-bucket-auto-link-size", type=int, default=50)
    cluster.add_argument("--min-auto-link-keywords", type=int, default=2)
    cluster.add_argument("--disable-keyword-bucket-auto-link", action="store_true")
    cluster.set_defaults(func=command_cluster)

    sample = subparsers.add_parser("sample", help="Select representative clusters for Gemma labeling")
    sample.add_argument("--cluster-summary", default=str(DEFAULT_OUTPUT_DIR / "cluster_summary.csv"))
    sample.add_argument("--target-size", type=int, default=15000)
    sample.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    sample.set_defaults(func=command_sample)

    label = subparsers.add_parser("label", help="Label representative clusters with Gemma4 31B")
    label.add_argument("--targets", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    label.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_labels.csv"))
    label.add_argument("--failed-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_failed.jsonl"))
    label.add_argument("--rubric", default=str(Path("config") / "importance_rubric.yaml"))
    label.add_argument("--model", default=None)
    label.add_argument("--limit", type=int, default=None)
    label.add_argument("--resume", action="store_true")
    label.add_argument("--retries", type=int, default=3)
    label.add_argument("--sleep-seconds", type=float, default=0.0)
    label.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    label.set_defaults(func=command_label)

    label_rules = subparsers.add_parser("label-rules", help="Label representative clusters with local rubric rules, no API calls")
    label_rules.add_argument("--targets", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    label_rules.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_labels.csv"))
    label_rules.add_argument("--limit", type=int, default=None)
    label_rules.add_argument("--resume", action="store_true")
    label_rules.set_defaults(func=command_label_rules)

    rule_v3 = subparsers.add_parser("rule-v3", help="Label every clustered complaint with conservative rubric v3 rules")
    rule_v3.add_argument("--input", default=str(DEFAULT_OUTPUT_DIR / "complaints_clustered.csv"))
    rule_v3.add_argument("--all-output", default=str(DEFAULT_OUTPUT_DIR / "rule_v3_all.csv"))
    rule_v3.add_argument("--trainable-output", default=str(DEFAULT_OUTPUT_DIR / "rule_v3_trainable.csv"))
    rule_v3.add_argument("--needs-review-output", default=str(DEFAULT_OUTPUT_DIR / "rule_v3_needs_review.csv"))
    rule_v3.add_argument("--review-sample-output", default=str(DEFAULT_OUTPUT_DIR / "rule_v3_review_2500.csv"))
    rule_v3.add_argument("--report-output", default=str(DEFAULT_OUTPUT_DIR / "rule_v3_report.json"))
    rule_v3.add_argument("--review-per-importance", type=int, default=500)
    rule_v3.add_argument("--random-state", type=int, default=42)
    rule_v3.add_argument("--limit", type=int, default=None)
    rule_v3.set_defaults(func=command_rule_v3)

    batch_prepare = subparsers.add_parser("label-batch-prepare", help="Prepare Gemini Batch API JSONL requests without calling the API")
    batch_prepare.add_argument("--targets", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    batch_prepare.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_requests.jsonl"))
    batch_prepare.add_argument("--map-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_request_map.csv"))
    batch_prepare.add_argument("--rubric", default=str(Path("config") / "importance_rubric.yaml"))
    batch_prepare.add_argument("--limit", type=int, default=None)
    batch_prepare.add_argument("--include-completed", action="store_true")
    batch_prepare.set_defaults(func=command_label_batch_prepare)

    batch_create = subparsers.add_parser("label-batch-create", help="Prepare, upload, and create a Gemini Batch API labeling job")
    batch_create.add_argument("--targets", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    batch_create.add_argument("--requests-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_requests.jsonl"))
    batch_create.add_argument("--map-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_request_map.csv"))
    batch_create.add_argument("--job-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_job.json"))
    batch_create.add_argument("--rubric", default=str(Path("config") / "importance_rubric.yaml"))
    batch_create.add_argument("--model", default=None)
    batch_create.add_argument("--limit", type=int, default=None)
    batch_create.add_argument("--display-name", default=None)
    batch_create.add_argument("--include-completed", action="store_true")
    batch_create.set_defaults(func=command_label_batch_create)

    batch_status = subparsers.add_parser("label-batch-status", help="Check a Gemini Batch API job and optionally download results")
    batch_status.add_argument("--job-name", required=True)
    batch_status.add_argument("--download-output", default=None)
    batch_status.add_argument("--watch", action="store_true", help="Poll until the batch job reaches a terminal state")
    batch_status.add_argument("--poll-seconds", type=float, default=60.0, help="Seconds between --watch status checks")
    batch_status.set_defaults(func=command_label_batch_status)

    batch_collect = subparsers.add_parser("label-batch-collect", help="Collect downloaded Gemini Batch API JSONL results into label CSV")
    batch_collect.add_argument("--results", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_results.jsonl"))
    batch_collect.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_labels.csv"))
    batch_collect.add_argument("--failed-output", default=str(DEFAULT_OUTPUT_DIR / "gemma_failed.jsonl"))
    batch_collect.add_argument("--request-map", default=str(DEFAULT_OUTPUT_DIR / "gemma_batch_request_map.csv"))
    batch_collect.add_argument("--model", default=DEFAULT_MODEL)
    batch_collect.add_argument("--resume", action="store_true")
    batch_collect.set_defaults(func=command_label_batch_collect)

    export = subparsers.add_parser("export", help="Export Gemma labels to Orange3-ready CSV files")
    export.add_argument("--targets", default=str(DEFAULT_OUTPUT_DIR / "gemma_label_targets.csv"))
    export.add_argument("--labels", default=str(DEFAULT_OUTPUT_DIR / "gemma_labels.csv"))
    export.add_argument("--clustered", default=str(DEFAULT_OUTPUT_DIR / "complaints_clustered.csv"))
    export.add_argument("--representatives-output", default=str(DEFAULT_OUTPUT_DIR / "orange_representatives.csv"))
    export.add_argument("--propagated-output", default=str(DEFAULT_OUTPUT_DIR / "orange_cluster_propagated.csv"))
    export.set_defaults(func=command_export)

    return parser


def main(argv=None) -> int:
    if sys.version_info < (3, 10):
        print(
            "ERROR: This pipeline requires Python 3.10+. "
            f"Current interpreter is {sys.version.split()[0]}. "
            r"Use .\.venv\Scripts\activate.bat in CMD or run .\.venv\Scripts\python.exe directly.",
            file=sys.stderr,
        )
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
