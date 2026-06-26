# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


OUT_DIR = Path("outputs/poster_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/malgunbd.ttf")

fm.fontManager.addfont(str(FONT_REGULAR))
fm.fontManager.addfont(str(FONT_BOLD))
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#173B6D"
BLUE = "#3667F6"
CYAN = "#26C6DA"
LIGHT = "#F7F9FC"
GRAY = "#667085"
DARK = "#111827"
ORANGE = "#FFB020"
GREEN = "#16A34A"
PURPLE = "#7C3AED"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def rounded(draw, box, radius=24, fill="white", outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(draw, text, font_obj, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            test = word if not line else line + " " + word
            if draw.textbbox((0, 0), test, font=font_obj)[2] <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def draw_wrapped(draw, xy, text, font_obj, fill, max_width, line_gap=8):
    x, y = xy
    for line in wrap_text(draw, text, font_obj, max_width):
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def gradient_header(img, left=(67, 56, 255), right=(74, 196, 236), height=150):
    width, _ = img.size
    px = img.load()
    for x in range(width):
        t = x / max(1, width - 1)
        color = tuple(int(left[i] * (1 - t) + right[i] * t) for i in range(3))
        for y in range(height):
            px[x, y] = color


def create_pipeline():
    img = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(img)
    gradient_header(img)
    draw.text((70, 42), "민원 중요도 분류 AI 구축 과정", font=font(46, True), fill="white")
    draw.text((72, 100), "원천 민원 데이터에서 학습 데이터와 웹 테스트 모델까지의 전체 흐름", font=font(24), fill="#EAF5FF")

    steps = [
        ("01", "데이터 수집", "AI Hub 민원 텍스트\n총 900,000건 확보", "#EAF5FF"),
        ("02", "전처리", "본문 정제, 공백 처리\n카테고리·키워드 정리", "#F0FDF4"),
        ("03", "룰 v3 라벨링", "중요도·긴급도 1~5 산정\n오탐 방지 규칙 적용", "#FFF7ED"),
        ("04", "학습 데이터 구성", "trainable 852,649건\nreview 47,351건 분리", "#F5F3FF"),
        ("05", "모델 학습", "문자 n-gram\nTF-IDF 선형 분류", "#ECFEFF"),
        ("06", "평가·시연", "혼동행렬·F1·Recall\nGradio 입력 테스트", "#FDF2F8"),
    ]

    start_x, y, card_w, card_h, gap = 70, 230, 255, 250, 30
    for i, (num, title, body, fill) in enumerate(steps):
        x = start_x + i * (card_w + gap)
        rounded(draw, (x, y, x + card_w, y + card_h), 26, fill=fill, outline="#D0D5DD", width=2)
        draw.ellipse((x + 22, y + 22, x + 78, y + 78), fill=BLUE if i % 2 == 0 else CYAN)
        draw.text((x + 38, y + 32), num, font=font(18, True), fill="white")
        draw.text((x + 28, y + 104), title, font=font(26, True), fill=NAVY)
        draw_wrapped(draw, (x + 28, y + 154), body, font(20), DARK, card_w - 56, line_gap=8)
        if i < len(steps) - 1:
            ax1 = x + card_w + 8
            ay = y + card_h // 2
            draw.line((ax1, ay, ax1 + 18, ay), fill=BLUE, width=5)
            draw.polygon([(ax1 + 18, ay - 10), (ax1 + 38, ay), (ax1 + 18, ay + 10)], fill=BLUE)

    rounded(draw, (90, 580, 1710, 805), 28, fill=LIGHT, outline="#D0D5DD")
    stats = [
        ("900,000건", "전체 라벨링 데이터"),
        ("852,649건", "학습 사용 데이터"),
        ("47,351건", "검토 제외 후보"),
        ("95.3%", "테스트 정확도"),
        ("90.7%", "Macro F1"),
    ]
    for i, (big, small) in enumerate(stats):
        x = 150 + i * 315
        draw.text((x, 630), big, font=font(42, True), fill=BLUE if i != 2 else ORANGE)
        draw.text((x, 690), small, font=font(22), fill=GRAY)
    draw.text(
        (90, 835),
        "핵심 원칙: 단순 문의는 낮게, 생명·보건·재난·안보 위험은 높게, 애매한 라벨은 학습에서 제외",
        font=font(22, True),
        fill=NAVY,
    )
    img.save(OUT_DIR / "infographic_pipeline.png")


def create_distribution():
    labels = ["1", "2", "3", "4", "5"]
    all_counts = np.array([162478, 66399, 599011, 64942, 7170])
    train_counts = np.array([162478, 19048, 599011, 64942, 7170])

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax.bar(x - width / 2, all_counts, width, label="전체 라벨", color="#A7C7FF")
    bars2 = ax.bar(x + width / 2, train_counts, width, label="학습 사용", color=BLUE)
    ax.set_title("중요도별 라벨 분포", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("중요도", fontsize=15)
    ax.set_ylabel("건수", fontsize=15)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=13)
    ax.bar_label(bars1, fmt=lambda v: f"{int(v):,}", padding=3, fontsize=10)
    ax.bar_label(bars2, fmt=lambda v: f"{int(v):,}", padding=3, fontsize=10)
    ax.text(
        0.01,
        -0.17,
        "해석: 중요도 3이 가장 많고 중요도 5는 적다. 학습 시 class_weight=balanced를 적용해 고우선 민원이 묻히지 않도록 했다.",
        transform=ax.transAxes,
        fontsize=12,
        color="#344054",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "infographic_label_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_confusion_matrix():
    labels = ["1", "2", "3", "4", "5"]
    cm = np.array(
        [
            [15504, 129, 577, 19, 19],
            [6, 1807, 92, 0, 0],
            [1864, 47, 57220, 589, 181],
            [5, 14, 321, 6143, 11],
            [19, 4, 83, 2, 609],
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 10), dpi=170)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("혼동행렬: 민원 중요도 분류 결과", fontsize=22, fontweight="bold", pad=22)
    ax.set_xlabel("예측 중요도", fontsize=15)
    ax.set_ylabel("실제 중요도", fontsize=15)
    ax.set_xticks(np.arange(5), labels)
    ax.set_yticks(np.arange(5), labels)
    threshold = cm.max() * 0.55
    for i in range(5):
        for j in range(5):
            color = "white" if cm[i, j] > threshold else NAVY
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    metrics_text = "Accuracy 95.3%   |   Macro F1 90.7%   |   중요도 5 Recall 84.9%   |   고우선(4·5) Recall 93.8%"
    fig.text(0.5, 0.03, metrics_text, ha="center", fontsize=13, color=NAVY, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "infographic_confusion_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_model_evaluation():
    img = Image.new("RGB", (1800, 950), "white")
    draw = ImageDraw.Draw(img)
    gradient_header(img, left=(23, 59, 109), right=(38, 198, 218))
    draw.text((70, 42), "모델 설계 및 평가 요약", font=font(46, True), fill="white")
    draw.text((72, 100), "룰 기반 pseudo-label을 활용한 1차 베이스라인 분류 모델", font=font(24), fill="#EAF5FF")

    cards = [
        ("입력 데이터", "민원 본문(Q_refined)\n학습 사용 852,649건\n검토 제외 47,351건", BLUE),
        ("벡터화", "HashingVectorizer\n문자 n-gram 2~5\n한국어 띄어쓰기·오타에 강함", CYAN),
        ("분류 모델", "SGDClassifier(log_loss)\n로지스틱 회귀 계열 선형 분류기\nclass_weight=balanced", PURPLE),
        ("웹 테스트", "Gradio UI 구성\n민원 직접 입력\n중요도별 확률 표시", GREEN),
    ]
    for i, (title, body, color) in enumerate(cards):
        x = 80 + i * 430
        y = 210
        rounded(draw, (x, y, x + 380, y + 260), 26, fill="white", outline="#D0D5DD")
        draw.rectangle((x, y, x + 380, y + 12), fill=color)
        draw.text((x + 28, y + 40), title, font=font(28, True), fill=NAVY)
        draw_wrapped(draw, (x + 28, y + 96), body, font(21), DARK, 320, line_gap=8)

    rounded(draw, (80, 530, 1720, 820), 28, fill=LIGHT, outline="#D0D5DD")
    draw.text((120, 570), "주요 평가 결과", font=font(32, True), fill=NAVY)
    metric_cards = [
        ("95.3%", "전체 정확도"),
        ("90.7%", "Macro F1"),
        ("84.9%", "중요도 5 Recall"),
        ("93.8%", "고우선(4·5) Recall"),
    ]
    for i, (big, label) in enumerate(metric_cards):
        x = 130 + i * 390
        rounded(draw, (x, 640, x + 310, 760), 20, fill="white", outline="#D0D5DD")
        draw.text((x + 34, 660), big, font=font(42, True), fill=BLUE if i < 2 else ORANGE)
        draw.text((x + 34, 720), label, font=font(21), fill=GRAY)
    draw.text(
        (120, 850),
        "해석: 전체 성능은 안정적이나, 실제 최우선 민원 누락을 줄이기 위해 중요도 5 Recall 개선과 사람 검수 데이터 추가가 필요하다.",
        font=font(24, True),
        fill=NAVY,
    )
    img.save(OUT_DIR / "infographic_model_evaluation.png")


def main():
    create_pipeline()
    create_distribution()
    create_confusion_matrix()
    create_model_evaluation()
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path.resolve())


if __name__ == "__main__":
    main()
