import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pipeline  # noqa: E402


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _target_row(**overrides):
    row = {
        "cluster_id": "C00000001",
        "representative_id": "B1-0",
        "representative_text": "놀이터 시설물이 파손되어 어린이가 다칠 위험이 있습니다.",
        "category": "안전건설",
        "subcategory": "시설물",
        "predication": "요청/개선",
        "department": "구청",
        "keywords": "놀이터|파손|어린이",
        "cluster_size": "7",
    }
    row.update(overrides)
    return row


def test_build_batch_request_uses_generate_content_json_contract():
    record = pipeline.build_batch_generate_content_request(
        _target_row(),
        rubric_text="importance: test",
    )

    assert record["key"] == "C00000001"
    request = record["request"]
    assert request["contents"][0]["role"] == "user"
    prompt = request["contents"][0]["parts"][0]["text"]
    assert "놀이터 시설물이 파손" in prompt
    assert "importance: test" in prompt
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["system_instruction"]["parts"][0]["text"]


def test_write_batch_requests_jsonl_writes_one_request_per_pending_target(tmp_path):
    targets = tmp_path / "targets.csv"
    out = tmp_path / "requests.jsonl"
    rows = [
        _target_row(cluster_id="C1", label_status="pending"),
        _target_row(cluster_id="C2", label_status="done"),
        _target_row(cluster_id="C3", label_status=""),
    ]
    with targets.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    count = pipeline.write_batch_requests_jsonl(
        targets_path=targets,
        output_path=out,
        rubric_text="rubric",
        limit=None,
        include_completed=False,
    )

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert [line["key"] for line in lines] == ["C1", "C3"]
    assert all("request" in line for line in lines)


def test_batch_result_record_to_label_row_parses_successful_response():
    record = {
        "metadata": {"cluster_id": "C1", "representative_id": "B1"},
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "importance": 5,
                                        "urgency": 4,
                                        "basis_tags": ["life_safety"],
                                        "reason": "어린이 안전 위험이 명확합니다.",
                                        "confidence": 0.91,
                                        "needs_review": False,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ]
        },
    }

    label, failure = pipeline.batch_result_record_to_label_row(record, model="gemma-4-31b-it")

    assert failure is None
    assert label["cluster_id"] == "C1"
    assert label["importance"] == 5
    assert label["urgency"] == 4
    assert label["basis_tags"] == "LIFE_SAFETY"
    assert label["label_source"] == "gemma_batch_cluster_representative"


def test_batch_result_record_to_label_row_returns_failure_for_api_error():
    record = {
        "metadata": {"cluster_id": "C1", "representative_id": "B1"},
        "error": {"code": 429, "message": "quota exceeded"},
    }

    label, failure = pipeline.batch_result_record_to_label_row(record, model="gemma-4-31b-it")

    assert label is None
    assert failure["cluster_id"] == "C1"
    assert failure["error"] == "quota exceeded"
    assert failure["error_code"] == 429


def test_collect_batch_results_writes_labels_and_failures(tmp_path):
    results = tmp_path / "results.jsonl"
    labels = tmp_path / "labels.csv"
    failures = tmp_path / "failed.jsonl"
    success = {
        "metadata": {"cluster_id": "C1", "representative_id": "B1"},
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "importance": 3,
                                        "urgency": 2,
                                        "basis_tags": ["facility_maintenance"],
                                        "reason": "시설 보수 요청입니다.",
                                        "confidence": 0.8,
                                        "needs_review": False,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ]
        },
    }
    failure = {
        "metadata": {"cluster_id": "C2", "representative_id": "B2"},
        "error": {"message": "bad request"},
    }
    results.write_text(
        json.dumps(success, ensure_ascii=False)
        + "\n"
        + json.dumps(failure, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    stats = pipeline.collect_batch_results_jsonl(
        results_path=results,
        output_path=labels,
        failed_path=failures,
        model="gemma-4-31b-it",
        resume=False,
    )

    assert stats == {"processed": 1, "failed": 1, "skipped": 0}
    with labels.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["cluster_id"] == "C1"
    failed_rows = [json.loads(line) for line in failures.read_text(encoding="utf-8").splitlines()]
    assert failed_rows[0]["cluster_id"] == "C2"


def test_batch_status_summary_calculates_percent_when_completion_stats_exist():
    batch_job = _Obj(
        name="batches/test",
        state=_Obj(name="JOB_STATE_RUNNING"),
        dest=_Obj(file_name="files/result"),
        completion_stats=_Obj(
            successful_count=70,
            failed_count=5,
            incomplete_count=25,
        ),
    )

    summary = pipeline.batch_status_summary(batch_job)

    assert summary["job_name"] == "batches/test"
    assert summary["state"] == "JOB_STATE_RUNNING"
    assert summary["dest_file_name"] == "files/result"
    assert summary["completion_stats"] == {
        "successful": 70,
        "failed": 5,
        "incomplete": 25,
        "done": 75,
        "total": 100,
    }
    assert summary["progress_percent"] == 75.0


def test_ensure_batch_model_supported_rejects_gemma_model():
    try:
        pipeline.ensure_batch_model_supported("gemma-4-31b-it")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected Gemma batch model rejection")

    assert "Gemma" in message
    assert "batch" in message.lower()
    assert "pipeline.py label" in message


def test_ensure_batch_model_supported_accepts_gemini_model():
    pipeline.ensure_batch_model_supported("gemini-2.5-flash")


def test_build_rule_label_marks_life_safety_complaint_high_importance():
    row = _target_row(
        representative_text="놀이터 시설물이 파손되어 어린이가 다칠 위험이 있습니다. 빠른 조치 바랍니다.",
        category="안전건설",
        predication="요청/개선",
        keywords="놀이터|파손|어린이|위험",
        cluster_size="7",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] == 5
    assert label["urgency"] >= 4
    assert "LIFE_SAFETY" in label["basis_tags"]
    assert label["confidence"] >= 0.8


def test_build_rule_label_marks_simple_inquiry_low_importance():
    row = _target_row(
        representative_text="주차장 이용 시간이 궁금합니다.",
        category="교통",
        predication="문의(질의)",
        keywords="주차장|이용시간",
        cluster_size="1",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] == 1
    assert label["urgency"] == 1
    assert label["basis_tags"] == ["STANDARD_INQUIRY"]


def test_build_rule_label_does_not_treat_polite_request_as_thanks():
    row = _target_row(
        representative_text="신호등 설치 약속은 언제 지키시나요? 시청과 농협 사이에 신호등 설치를 약속했는데 아무런 변화가 없습니다. 빠른 조치 부탁드립니다. 감사하겠습니다.",
        category="교통",
        subcategory="기타",
        predication="요청/개선",
        keywords="시청|농협|신호등설치",
        cluster_size="4",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] >= 3
    assert label["urgency"] >= 2
    assert "THANKS" not in label["basis_tags"]


def test_build_rule_label_marks_earthquake_or_collapse_report_critical():
    row = _target_row(
        representative_text="방금 지진으로 건물 벽에 큰 균열이 생겼고 붕괴 위험이 있습니다. 주민 대피와 긴급 점검이 필요합니다.",
        category="안전건설",
        subcategory="시설물",
        predication="신고/고발",
        keywords="지진|균열|붕괴위험|주민대피",
        cluster_size="12",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] == 5
    assert label["urgency"] == 5
    assert "LIFE_SAFETY" in label["basis_tags"]


def test_build_rule_label_marks_child_zone_illegal_parking_as_public_safety():
    row = _target_row(
        representative_text="어린이보호구역 횡단보도 앞 불법주정차 때문에 아이들이 차 사이로 나와 사고 위험이 큽니다. 단속 요청합니다.",
        category="교통",
        subcategory="불법 주정차",
        predication="신고/고발",
        keywords="어린이보호구역|횡단보도|불법주정차|사고위험",
        cluster_size="6",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] >= 4
    assert label["urgency"] >= 4
    assert {"VULNERABLE_GROUP", "TRAFFIC_SAFETY", "LEGAL_VIOLATION"} & set(label["basis_tags"])


def test_build_rule_label_keeps_true_thanks_low_importance():
    row = _target_row(
        representative_text="친절하게 민원을 처리해 주셔서 감사합니다. 수고하셨습니다.",
        category="기타",
        subcategory="기타",
        predication="감사",
        keywords="감사|친절",
        cluster_size="1",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] == 1
    assert label["urgency"] == 1
    assert label["basis_tags"] == ["THANKS"]


def test_build_rule_label_marks_public_health_spread_high_priority():
    row = _target_row(
        representative_text="학교 급식 후 여러 학생이 식중독 증상을 보이고 있습니다. 집단감염 우려가 있어 즉시 방역과 조사가 필요합니다.",
        category="보건소",
        subcategory="위생",
        predication="신고/고발",
        keywords="학교|급식|식중독|집단감염|방역",
        cluster_size="20",
    )

    label = pipeline.build_rule_label(row)

    assert label["importance"] == 5
    assert label["urgency"] >= 4
    assert "PUBLIC_HEALTH" in label["basis_tags"]


def test_merge_priority_label_escalates_when_row_text_has_stronger_risk():
    cluster_label = {
        "importance": 3,
        "urgency": 2,
        "basis_tags": ["MULTIPLE_RESIDENTS"],
        "reason": "cluster label",
        "confidence": 0.74,
        "needs_review": False,
        "model": "rubric_rules_v2",
        "label_source": "cluster_propagated_from_rubric_rules_cluster_representative",
    }
    row_label = {
        "importance": 5,
        "urgency": 5,
        "basis_tags": ["PUBLIC_HEALTH"],
        "reason": "row label",
        "confidence": 0.88,
        "needs_review": False,
    }

    merged = pipeline.merge_priority_label(cluster_label, row_label)

    assert merged["importance"] == 5
    assert merged["urgency"] == 5
    assert merged["basis_tags"] == ["PUBLIC_HEALTH"]
    assert merged["model"] == "rubric_rules_v2_row_escalation"
    assert merged["label_source"].startswith("row_rule_escalated_from_")


def test_merge_priority_label_keeps_cluster_label_when_row_signal_is_weaker():
    cluster_label = {
        "importance": 4,
        "urgency": 4,
        "basis_tags": ["TRAFFIC_SAFETY"],
        "reason": "cluster label",
        "confidence": 0.82,
        "needs_review": False,
        "model": "rubric_rules_v2",
        "label_source": "cluster_propagated_from_rubric_rules_cluster_representative",
    }
    row_label = {
        "importance": 1,
        "urgency": 1,
        "basis_tags": ["STANDARD_INQUIRY"],
        "reason": "row label",
        "confidence": 0.86,
        "needs_review": False,
    }

    merged = pipeline.merge_priority_label(cluster_label, row_label)

    assert merged == cluster_label


def test_rule_v3_marks_payment_inquiry_low_and_trainable():
    row = _target_row(
        representative_text="자동차세 과태료 납부 가상계좌가 궁금합니다. 납부 방법을 알려주세요.",
        category="세무",
        subcategory="지방세",
        predication="문의(질의)",
        keywords="자동차세|과태료|납부|가상계좌",
        cluster_size="3",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 1
    assert label["urgency"] == 1
    assert label["priority_type"] == "SIMPLE_INQUIRY"
    assert label["cap_reason"] == "SIMPLE_INQUIRY_CAP"
    assert label["trainable"] is True


def test_rule_v3_caps_ordinary_fine_request_at_normal_action():
    row = _target_row(
        representative_text="인도에 불법 주차한 차량 확인 후 과태료 부과 부탁드립니다.",
        category="자동차",
        subcategory="불법주정차",
        predication="요청/개선",
        keywords="인도|불법주차|과태료",
        cluster_size="2",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 3
    assert label["urgency"] == 3
    assert label["priority_type"] == "ENFORCEMENT"
    assert label["cap_reason"] == "ENFORCEMENT_CAP"


def test_rule_v3_keeps_simple_roadside_weeds_below_high_priority():
    row = _target_row(
        representative_text="도로 옆 잡초와 풀이 많이 자라서 제초 작업을 요청합니다.",
        category="산림",
        subcategory="기타",
        predication="요청/개선",
        keywords="도로|잡초|풀|제초",
        cluster_size="1",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] <= 3
    assert label["urgency"] <= 2
    assert label["priority_type"] == "FACILITY_ENVIRONMENT"
    assert label["cap_reason"] == "NO_EXPLICIT_SAFETY_RISK"


def test_rule_v3_marks_visibility_blocked_accident_risk_high_not_critical():
    row = _target_row(
        representative_text="교차로 도로 옆 잡풀과 나무가 시야를 가려 사고 위험이 있습니다. 현장 확인 후 제거 바랍니다.",
        category="교통",
        subcategory="기타",
        predication="요청/개선",
        keywords="교차로|도로|잡풀|나무|시야|사고위험",
        cluster_size="3",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 4
    assert label["urgency"] == 4
    assert label["priority_type"] == "TRAFFIC_SAFETY"
    assert "VISIBILITY_BLOCKED" in label["evidence_tags"]


def test_rule_v3_marks_fire_hydrant_illegal_parking_critical():
    row = _target_row(
        representative_text="소화전 앞에 불법주정차된 차량 때문에 불이 났을 때 화재진압을 못할 것 같습니다. 견인 조치 바랍니다.",
        category="자동차",
        subcategory="불법주정차",
        predication="요청/개선",
        keywords="소화전|불법주정차|화재진압",
        cluster_size="20",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 5
    assert label["urgency"] == 5
    assert label["priority_type"] == "LIFE_SAFETY"
    assert "FIRE_SUPPRESSION_BLOCKED" in label["evidence_tags"]


def test_rule_v3_does_not_make_health_center_parking_critical():
    row = _target_row(
        representative_text="창원보건소 장애인 주차구역에 불법주차한 차량이 있습니다. 과태료 부과 바랍니다.",
        category="자동차",
        subcategory="불법주정차",
        predication="요청/개선",
        keywords="창원보건소|장애인 주차구역|불법주차|과태료",
        cluster_size="1",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 4
    assert label["urgency"] == 3
    assert label["priority_type"] == "ACCESSIBILITY_ENFORCEMENT"
    assert "PUBLIC_HEALTH" not in label["evidence_tags"]


def test_rule_v3_marks_food_poisoning_cluster_critical():
    row = _target_row(
        representative_text="예식장 뷔페 식사 후 여러 사람이 구토와 설사 등 식중독 증상으로 응급실에 갔습니다. 즉시 위생 점검 바랍니다.",
        category="위생",
        subcategory="기타",
        predication="신고/고발",
        keywords="뷔페|식중독|여러 사람|구토|설사|응급실",
        cluster_size="8",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 5
    assert label["urgency"] == 5
    assert label["priority_type"] == "PUBLIC_HEALTH"
    assert "FOOD_POISONING_CLUSTER" in label["evidence_tags"]


def test_rule_v3_keeps_disaster_preparedness_inquiry_low():
    row = _target_row(
        representative_text="지진대피 행동요령 관련하여 알고 싶습니다.",
        category="안전건설",
        subcategory="재난",
        predication="문의(질의)",
        keywords="지진대피|행동요령",
        cluster_size="1",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 1
    assert label["urgency"] == 1
    assert label["priority_type"] == "SIMPLE_INQUIRY"


def test_rule_v3_does_not_match_vomit_across_land_words():
    row = _target_row(
        representative_text="진해구 #@주소# 토지 관련해 민원 올립니다. 수목 제거 후 수목이식이 결정되었고 토지 보상이 먼저 해결되었으면 합니다.",
        category="산림",
        subcategory="수목",
        predication="건의/제기",
        keywords="토지|수목 제거|민원|이식|토지 보상",
        cluster_size="1",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] <= 3
    assert label["priority_type"] == "FACILITY_ENVIRONMENT"
    assert "FOOD_POISONING_CLUSTER" not in label["evidence_tags"]


def test_rule_v3_does_not_match_diarrhea_inside_construction_company():
    row = _target_row(
        representative_text="주차한 차에 시멘트 오물이 튀어 광택작업이 필요하지만 건설사 측에서 비용 지불을 거절하고 있습니다.",
        category="건설",
        subcategory="공사피해",
        predication="요청/개선",
        keywords="시멘트|오물|건설사|비용",
        cluster_size="3",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] <= 3
    assert label["priority_type"] != "PUBLIC_HEALTH"
    assert "FOOD_POISONING_CLUSTER" not in label["evidence_tags"]


def test_rule_v3_keeps_preventive_fire_escape_request_below_critical():
    row = _target_row(
        representative_text="화재 발생시 아파트 옥상 대피를 위해 피난에 방해가 되지 않도록 설계 검토를 요청드립니다.",
        category="안전건설",
        subcategory="건축",
        predication="요청/개선",
        keywords="화재 발생시|옥상 대피|피난|설계 검토",
        cluster_size="4",
    )

    label = pipeline.build_rule_v3_label(row)

    assert label["importance"] == 4
    assert label["urgency"] == 3
    assert label["priority_type"] == "SAFETY_PREVENTION"


def test_rule_v3_validation_flags_inconsistent_critical_label():
    flags = pipeline.validate_rule_v3_label(
        {
            "importance": 5,
            "priority_type": "FACILITY_ENVIRONMENT",
            "evidence_tags": ["WEEDS"],
            "counter_evidence": [],
        }
    )

    assert "CRITICAL_WITHOUT_STRONG_EVIDENCE" in flags
    assert "ENVIRONMENT_CRITICAL_REVIEW" in flags
