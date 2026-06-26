const MODEL_URL = "model/complaint_priority_model.json";

const labelBasis = {
  1: "단순 문의, 정보 확인, 감사 표현처럼 즉시 조치 필요성이 낮은 민원입니다.",
  2: "개인 불편이나 일반 요청 중심으로 통상 처리 절차에 따라 대응할 수 있는 민원입니다.",
  3: "시설 보수, 행정 조치, 반복 불편처럼 구체적인 확인과 처리가 필요한 일반 민원입니다.",
  4: "공공안전, 다수 주민 피해, 취약계층, 명확한 위법 가능성 등 우선 검토가 필요한 민원입니다.",
  5: "생명·신체 위험, 재난, 감염병, 안보 등 즉시 대응해야 할 가능성이 큰 민원입니다.",
};

let model = null;

function utf8Bytes(value) {
  return new TextEncoder().encode(value);
}

function murmurhash3(bytes, seed = 0) {
  let h1 = seed | 0;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;
  const roundedEnd = bytes.length & ~3;

  for (let i = 0; i < roundedEnd; i += 4) {
    let k1 = (bytes[i] & 0xff) | ((bytes[i + 1] & 0xff) << 8) | ((bytes[i + 2] & 0xff) << 16) | ((bytes[i + 3] & 0xff) << 24);
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);

    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = (Math.imul(h1, 5) + 0xe6546b64) | 0;
  }

  let k1 = 0;
  switch (bytes.length & 3) {
    case 3:
      k1 ^= (bytes[roundedEnd + 2] & 0xff) << 16;
    case 2:
      k1 ^= (bytes[roundedEnd + 1] & 0xff) << 8;
    case 1:
      k1 ^= bytes[roundedEnd] & 0xff;
      k1 = Math.imul(k1, c1);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, c2);
      h1 ^= k1;
  }

  h1 ^= bytes.length;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;
  return h1 | 0;
}

function charWbNgrams(text, minN, maxN) {
  const terms = [];
  const words = text.split(/\s+/).filter(Boolean);
  for (const rawWord of words) {
    const word = ` ${rawWord} `;
    for (let n = minN; n <= maxN; n += 1) {
      for (let i = 0; i <= word.length - n; i += 1) {
        terms.push(word.slice(i, i + n));
      }
    }
  }
  return terms;
}

function vectorize(text) {
  const counts = new Map();
  const [minN, maxN] = model.ngram_range;
  for (const term of charWbNgrams(text, minN, maxN)) {
    const hash = murmurhash3(utf8Bytes(term), 0);
    const index = ((hash % model.n_features) + model.n_features) % model.n_features;
    counts.set(index, (counts.get(index) || 0) + 1);
  }

  let norm = 0;
  const values = [];
  for (const [index, count] of counts.entries()) {
    const tf = model.sublinear_tf ? 1 + Math.log(count) : count;
    const value = tf * model.idf[index];
    values.push([index, value]);
    norm += value * value;
  }

  norm = Math.sqrt(norm) || 1;
  return values.map(([index, value]) => [index, value / norm]);
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function predict(text) {
  const features = vectorize(text);
  const scores = model.coef.map((row, classIndex) => {
    let score = model.intercept[classIndex];
    for (const [featureIndex, value] of features) {
      score += row[featureIndex] * value;
    }
    return score;
  });
  const probabilities = softmax(scores);
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }
  return {
    label: model.labels[bestIndex],
    probability: probabilities[bestIndex],
    probabilities,
  };
}

function hasAny(text, words) {
  return words.some((word) => text.includes(word));
}

function applyGuardrails(text, result) {
  const normalized = text.replace(/\s+/g, " ");
  const criticalTerms = [
    "사망", "생명", "의식불명", "심정지", "응급", "구조", "실종", "화재", "폭발", "가스누출",
    "감전", "붕괴", "침수", "산사태", "감염병", "집단감염", "테러", "흉기", "총기",
  ];
  const safetyTerms = [
    "위험", "사고", "파손", "고장", "균열", "붕괴", "추락", "급정거", "차도", "횡단보도",
    "신호등", "중앙분리대", "난간", "맨홀", "싱크홀", "전봇대", "전선", "누전", "감전",
    "침수", "누수", "불법주정차", "무단횡단", "공사장",
  ];
  const vulnerableTerms = ["학교", "학생", "어린이", "유치원", "초등", "노인", "장애인", "임산부"];
  const urgencyTerms = ["즉시", "긴급", "오늘", "바로", "조속", "빠른", "시급"];
  const publicImpactTerms = ["다수", "여러", "주민", "세대", "반복", "집단", "통행", "보행자"];
  const simpleQuestionTerms = ["궁금", "문의", "알려", "방법", "어디", "확인", "납부", "발급", "신청"];

  const hasCritical = hasAny(normalized, criticalTerms);
  const hasSafety = hasAny(normalized, safetyTerms);
  const hasVulnerable = hasAny(normalized, vulnerableTerms);
  const hasUrgency = hasAny(normalized, urgencyTerms);
  const hasPublicImpact = hasAny(normalized, publicImpactTerms);
  const isSimpleQuestion = hasAny(normalized, simpleQuestionTerms) && !hasSafety && !hasCritical;

  let label = result.label;
  let guardrailReason = "";

  if (hasCritical) {
    label = Math.max(label, 5);
    guardrailReason = "생명·재난·중대 사고 표현이 감지되어 즉시 대응 등급을 적용했습니다.";
  } else if (hasSafety && (hasVulnerable || hasUrgency || hasPublicImpact)) {
    label = Math.max(label, 4);
    guardrailReason = "공공안전 위험과 취약 대상·긴급성·다수 피해 표현이 함께 감지되어 우선 처리 등급을 적용했습니다.";
  } else if (hasSafety) {
    label = Math.max(label, 3);
    guardrailReason = "시설 위험 또는 사고 가능성 표현이 감지되어 일반 조치 이상으로 보정했습니다.";
  } else if (isSimpleQuestion) {
    label = Math.min(label, 2);
    guardrailReason = "위험 표현이 없는 절차·정보 문의로 판단되어 낮은 등급으로 보정했습니다.";
  }

  return {
    ...result,
    label,
    guardrailReason,
    adjusted: label !== result.label,
    modelLabel: result.label,
  };
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderPrediction(result) {
  const scoreEl = document.querySelector("#result-score");
  const titleEl = document.querySelector("#result-title");
  const badgeEl = document.querySelector("#priority-badge");
  const basisEl = document.querySelector("#basis-text");
  const probabilitiesEl = document.querySelector("#probabilities");

  scoreEl.textContent = result.label;
  titleEl.textContent = model.label_descriptions[String(result.label)] || "분류 결과";
  basisEl.textContent = result.guardrailReason
    ? `${labelBasis[result.label] || ""} ${result.guardrailReason}`
    : labelBasis[result.label] || "";
  badgeEl.textContent = result.label >= 4 ? "우선 처리 검토" : "일반 처리";
  badgeEl.style.background = result.label >= 4 ? "#fff1d6" : "#e8f5f4";
  badgeEl.style.color = result.label >= 4 ? "#8a4f08" : "#08736f";

  probabilitiesEl.innerHTML = "";
  model.labels.forEach((label, index) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    const bar = document.createElement("div");
    const fill = document.createElement("span");
    const pct = document.createElement("strong");

    name.textContent = `중요도 ${label}`;
    bar.className = "bar";
    fill.style.width = formatPercent(result.probabilities[index]);
    pct.textContent = formatPercent(result.probabilities[index]);

    bar.appendChild(fill);
    li.append(name, bar, pct);
    probabilitiesEl.appendChild(li);
  });
}

function runPrediction() {
  const text = document.querySelector("#complaint-text").value.trim();
  if (!model || !text) return;
  renderPrediction(applyGuardrails(text, predict(text)));
}

function renderMetrics() {
  const metrics = model.metrics || {};
  document.querySelector("#metric-accuracy").textContent = metrics.accuracy ? formatPercent(metrics.accuracy) : "-";
  document.querySelector("#metric-f1").textContent = metrics.macro_f1 ? formatPercent(metrics.macro_f1) : "-";
  document.querySelector("#metric-rows").textContent = metrics.test_rows ? metrics.test_rows.toLocaleString("ko-KR") : "-";
}

async function init() {
  const response = await fetch(MODEL_URL);
  model = await response.json();
  renderMetrics();
  document.querySelector("#predict-button").addEventListener("click", runPrediction);
  document.querySelector("#clear-button").addEventListener("click", () => {
    document.querySelector("#complaint-text").value = "";
  });
  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelector("#complaint-text").value = button.dataset.example;
      runPrediction();
    });
  });
  runPrediction();
}

init().catch((error) => {
  document.querySelector("#result-title").textContent = "모델 파일을 불러오지 못했습니다";
  document.querySelector("#basis-text").textContent = String(error);
});
