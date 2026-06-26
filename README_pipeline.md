# 민원 중요도 판별 AI 파이프라인

AI Hub 민원 데이터를 평탄화하고, 유사 민원을 군집화한 뒤, 군집 대표에 중요도/긴급도 라벨을 붙여 학습용 CSV를 만드는 프로젝트입니다.

현재 기본 방식은 **Gemma API가 아니라 로컬 룰 기반 라벨링 v3**입니다. Gemma 4 무료 API는 일반 `generateContent`만 지원하고 Batch API를 지원하지 않아 대량 라벨링에는 너무 느립니다. 대신 법률/절차 기준을 바탕으로 만든 운영 룰을 전체 데이터에 적용하고, 애매한 행은 `needs_review=true`로 분리합니다.

## 현재 산출물

```text
outputs/complaints_flat.csv
outputs/complaints_clustered.csv
outputs/cluster_summary.csv
outputs/rule_v3_all.csv
outputs/rule_v3_trainable.csv
outputs/rule_v3_needs_review.csv
outputs/rule_v3_review_2500.csv
outputs/rule_v3_report.json
```

핵심 학습 파일은 `outputs/rule_v3_trainable.csv`입니다. `outputs/rule_v3_all.csv`는 90만건 전체 라벨 파일이고, `outputs/rule_v3_needs_review.csv`는 룰만 믿고 학습시키기 애매한 제외 후보입니다.

## 기준 출처

법에 “민원 중요도 1~5” 같은 공식 우선처리 점수표는 없습니다. 그래서 아래 법률과 절차 기준에서 판단축을 가져와 운영 기준으로 점수화했습니다.

- 민원 처리에 관한 법률: https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=239293
- 민원 처리에 관한 법률 시행령: https://law.go.kr/LSW/lsInfoP.do?lsiSeq=181064
- 국민신문고 민원 처리기간 안내: https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnNtrcnContent.npaid
- 행정안전부 민원처리기준표 고시: https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000016&nttId=126603
- 재난 및 안전관리 기본법: https://www.law.go.kr/LSW/lsInfoP.do?ancYnChk=0&lsId=009640
- 감염병의 예방 및 관리에 관한 법률: https://www.law.go.kr/LSW/lsInfoP.do?ancYnChk=0&lsId=001792
- 안전신문고 안내: https://www.mois.go.kr/frt/sub/a06/b10/safetyReport/screen.do
- 국민보호와 공공안전을 위한 테러방지법: https://law.go.kr/LSW/lsInfoP.do?lsiSeq=181624

세부 기준과 검증 플래그는 `config/importance_rubric.yaml`에 정리되어 있습니다.

## 점수 의미

`importance`는 처리하지 않았을 때 피해 규모와 공익성을 봅니다.

- 5: 생명·신체·재산·재난·감염병 확산·국가안보 등 직접 중대 위험
- 4: 공공안전, 교통안전, 취약계층, 다수 주민 피해, 명확한 위법/단속 필요
- 3: 구체적 조치가 필요한 반복 불편, 시설 보수, 행정 지연, 일반 단속 요청
- 2: 개인 불편 중심의 일반 요청
- 1: 단순 문의, 절차/정보 요청, 감사, 일반 제안

`urgency`는 지연되면 피해가 얼마나 빨리 커지는지를 봅니다.

- 5: 지금/오늘 바로 위험이 진행 중
- 4: 며칠 내 피해 확대 가능성 큼
- 3: 일반보다 빠른 처리 필요
- 2: 통상 처리 가능
- 1: 시간 민감도 낮음

## 재생성 명령

이미 `complaints_flat.csv`, `complaints_clustered.csv`, `cluster_summary.csv`가 있으면 클러스터링은 다시 돌릴 필요 없습니다.

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py rule-v3
```

처음부터 다시 돌릴 때만 아래 순서로 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py flatten
.\.venv\Scripts\python.exe scripts\pipeline.py cluster
.\.venv\Scripts\python.exe scripts\pipeline.py rule-v3
```

## Colab에서 보기

큰 전체 파일 대신 아래 파일부터 확인하세요.

```text
outputs/rule_v3_review_2500.csv
outputs/rule_v3_needs_review.csv
outputs/rule_v3_trainable.csv
```

`rule_v3_review_2500.csv`는 중요도 1~5를 각 500개씩 뽑은 검토용 파일입니다.

## v3 품질 보정

v3 룰은 아래 오탐을 막도록 보수적으로 설계했습니다.

- 단순 문의는 안전·재난 단어가 있어도 행동요령/담당부서/방법 문의면 1/1로 둡니다.
- 도로 옆 풀·수목·제초는 사고위험/시야가림이 명시될 때만 4점 이상으로 올립니다.
- `보건소` 같은 기관명만으로는 보건위험으로 보지 않습니다.
- `진해구 토지`에서 `구토`, `건설사`에서 `설사`처럼 단어 경계가 붙어 생기는 오탐을 막습니다.
- 화재/대피 관련 예방·설계 검토 요청은 즉시위험 5점이 아니라 `SAFETY_PREVENTION` 4/3으로 분리합니다.

## 주의

이 라벨은 법적 판단이나 실제 우선처리 명령이 아니라 학습용 pseudo-label입니다. 모델 학습에는 우선 `trainable=true`인 행만 쓰고, `needs_review=true` 행은 검수 또는 룰 보정 후 추가하는 것이 좋습니다.
