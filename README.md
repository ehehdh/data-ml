# 민원 중요도 분류 AI

민원 내용을 입력하면 중요도 1~5단계를 예측하는 수행평가용 AI 프로젝트입니다.  
불필요한 단순 문의와 긴급 민원을 구분하여, 생명·안전·다수 피해와 관련된 민원을 먼저 확인할 수 있도록 돕는 것을 목표로 했습니다.

## 바로 보기

- 웹 데모: GitHub Pages 배포 후 `https://sionkk1.github.io/data/` 형식으로 접속
- Colab 노트북: `notebooks/complaint_priority_train_colab.ipynb`
- 학습 데이터: `data/processed/rule_v3_trainable.csv.gz`
- 검토 샘플: `data/sample/rule_v3_review_2500.csv`

## 저장소 구조

```text
config/       중요도·긴급도 라벨링 기준
data/         제출용 압축 학습 데이터와 검토 샘플
notebooks/    Colab 학습/평가/Gradio 테스트 노트북
scripts/      데이터 생성, 라벨링, 웹 모델 내보내기 코드
tests/        라벨링 기준과 파이프라인 검증 테스트
web/          GitHub Pages용 정적 웹 데모
outputs/      로컬 생성 산출물
```

## 연구 요약

민원 처리 과정에서는 단순 문의, 개인 불편, 시설 보수, 공공안전, 생명·재난 위험이 같은 접수 흐름 안에 섞일 수 있습니다. 이 프로젝트는 민원 텍스트를 분석해 중요도와 긴급도를 예측하고, 우선 확인이 필요한 민원을 빠르게 찾는 보조 모델을 만드는 데 초점을 두었습니다.

## 데이터와 라벨링

AI Hub 민원 데이터를 정리한 뒤 중복·유사 민원을 군집화하고, 법률 및 행정 절차 기준을 참고하여 중요도 1~5단계 라벨을 생성했습니다. 단순 키워드만으로 과하게 위험도를 올리는 문제를 줄이기 위해 반례 조건, 신뢰도, 검토 필요 플래그를 함께 두었습니다.

GitHub에는 용량 제한 때문에 900,000건짜리 대용량 CSV 원본 전체를 그대로 올리지 않고, Colab 학습에 바로 쓸 수 있는 압축 학습 데이터 `data/processed/rule_v3_trainable.csv.gz`와 중요도별 검토 샘플 `data/sample/rule_v3_review_2500.csv`를 포함했습니다.

중요도 기준은 다음과 같습니다.

- 1: 단순 문의, 정보 확인, 감사, 일반 제안
- 2: 개인 불편 중심의 일반 요청
- 3: 구체적인 조치가 필요한 일반 민원
- 4: 공공안전, 다수 피해, 취약계층, 위법 가능성 등 우선 검토 민원
- 5: 생명, 신체, 재난, 감염병, 안보 등 즉시 대응 가능성이 큰 민원

## 모델 설계

Colab 학습 모델은 민원 문장을 문자 n-gram 기반 벡터로 변환한 뒤 선형 분류 모델로 중요도 등급을 예측합니다. 웹 데모는 같은 계열의 경량 모델을 JSON으로 내보내 브라우저에서 바로 실행하고, 명백한 생명·안전 표현에는 안전 보정 규칙을 함께 적용하도록 구성했습니다.

## 실행 방법

로컬에서 데이터 생성과 라벨링을 다시 실행하려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py flatten
.\.venv\Scripts\python.exe scripts\pipeline.py cluster
.\.venv\Scripts\python.exe scripts\pipeline.py rule-v3
```

웹 데모용 모델을 다시 만들려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts\train_web_model.py
```

웹 데모를 로컬에서 확인하려면 다음 명령을 실행한 뒤 브라우저에서 표시된 주소를 엽니다.

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 -d web
```

## GitHub Pages 배포

1. 저장소를 GitHub에 올립니다.
2. GitHub 저장소에서 `Settings > Pages`로 이동합니다.
3. `Build and deployment`를 `Deploy from a branch`로 선택합니다.
4. Branch는 `main`, Folder는 `/web`으로 선택합니다.
5. 저장 후 표시되는 Pages 주소를 제출합니다.

## 한계와 제언

이 모델은 행정 처리 보조용 실험 모델입니다. 실제 우선 처리 여부는 담당자 검토와 기관 기준을 함께 적용해야 합니다. 향후에는 사람이 검수한 고품질 라벨을 추가하고, KoELECTRA나 KLUE-RoBERTa 같은 한국어 트랜스포머 모델과 비교하면 더 안정적인 성능을 확인할 수 있습니다.
