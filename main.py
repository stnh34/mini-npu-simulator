"""Mini NPU Simulator - MAC(Multiply-Accumulate) 연산 기반 패턴 판별기.

외부 라이브러리 없이 표준 라이브러리(json, time, re, os)만 사용한다.
"""

import json
import os
import re
import sys
import time

EPSILON = 1e-9          # 동점 판정 허용오차
REPEAT = 10             # 성능 측정 반복 횟수
LABEL_CROSS = "Cross"
LABEL_X = "X"
# 표준 라벨 목록. 새 라벨을 추가할 때는 여기와 _LABEL_MAP 두 곳만 수정한다(README "새 라벨 추가 절차").
STANDARD_LABELS = (LABEL_CROSS, LABEL_X)
UNDECIDED = "UNDECIDED"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
LINE = "#" + "-" * 40


# ----------------------------------------------------------------------
# 데이터 구조: n×n 2차원 행렬
# ----------------------------------------------------------------------
class Matrix:
    """n×n 패턴/필터 저장소 (리스트의 리스트)."""

    def __init__(self, n, rows=None):
        self.n = n
        if rows is None:
            self.data = [[0.0] * n for _ in range(n)]
        else:
            self.data = [[float(v) for v in row] for row in rows]

    def get(self, r, c):
        return self.data[r][c]

    def set(self, r, c, value):
        self.data[r][c] = float(value)

    def flatten(self):
        """보너스: 2차원 → 1차원(길이 N²) 변환."""
        flat = []
        for row in self.data:
            flat.extend(row)
        return flat

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트 검증 후 Matrix 생성. 정사각형이 아니면 ValueError."""
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("2차원 배열이 아니거나 비어 있음")
        n = len(rows)
        for i, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError("%d행이 배열이 아님" % (i + 1))
            if len(row) != n:
                raise ValueError("정사각형이 아님 (%d행 열 수 %d, 기대 %d)" % (i + 1, len(row), n))
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError("숫자가 아닌 값 포함: %r" % (v,))
        return cls(n, rows)


# ----------------------------------------------------------------------
# MAC 연산 (반복문 직접 구현)
# ----------------------------------------------------------------------
def mac(pattern, filt):
    """위치별 곱의 총합(유사도 점수). 크기가 다르면 ValueError.

    점수 해석: 패턴과 필터에서 '둘 다 값이 큰 칸'이 많을수록 곱이 커져 누적 점수가 커진다.
    따라서 MAC 점수가 높을수록 그 필터(모양)와 더 유사하다고 본다.
    예) 십자가 패턴 × 십자가 필터 = 5 (5칸 겹침), 십자가 패턴 × X 필터 = 1 (중심 1칸만 겹침)
    """
    if pattern.n != filt.n:
        raise ValueError("크기 불일치: 패턴 %d×%d, 필터 %d×%d" % (pattern.n, pattern.n, filt.n, filt.n))
    n = pattern.n
    total = 0.0
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * filt.get(r, c)
    return total


def mac_flat(pattern_flat, filter_flat):
    """보너스: 1차원 배열(길이 N²)에 대한 MAC 연산."""
    total = 0.0
    for i in range(len(pattern_flat)):
        total += pattern_flat[i] * filter_flat[i]
    return total


# ----------------------------------------------------------------------
# 판정 / 라벨 정규화
# ----------------------------------------------------------------------
def judge_scores(scores):
    """scores: {라벨: 점수}. 점수가 가장 높은(=가장 유사한) 라벨을 반환.

    1위와 2위의 차이가 EPSILON 미만이면 어느 쪽인지 결정할 수 없으므로 UNDECIDED.
    라벨 수에 관계없이 동작하므로 새 라벨을 추가해도 수정할 필요가 없다.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < EPSILON:
        return UNDECIDED
    return ranked[0][0]


def judge(score_a, score_b, label_a, label_b):
    """두 필터 비교(모드 1). 점수가 높은 쪽으로 판정, 동점이면 UNDECIDED."""
    return judge_scores({label_a: score_a, label_b: score_b})


# 입력 라벨(소문자) → 표준 라벨. expected 값과 filters 키 모두 이 표로 정규화한다.
_LABEL_MAP = {
    "+": LABEL_CROSS, "cross": LABEL_CROSS,
    "x": LABEL_X,
}


def normalize_label(raw):
    """'+', 'cross', 'Cross' → Cross / 'x', 'X' → X. 알 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return _LABEL_MAP.get(raw.strip().lower())


# ----------------------------------------------------------------------
# 패턴 생성기 (보너스)
# ----------------------------------------------------------------------
def make_cross(n):
    m = Matrix(n)
    mid = n // 2
    for i in range(n):
        m.set(mid, i, 1)
        m.set(i, mid, 1)
    return m


def make_x(n):
    m = Matrix(n)
    for i in range(n):
        m.set(i, i, 1)
        m.set(i, n - 1 - i, 1)
    return m


# ----------------------------------------------------------------------
# 성능 측정
# ----------------------------------------------------------------------
def measure_ms(func, *args, repeat=REPEAT):
    """연산 함수 호출 구간만 repeat회 측정 후 평균(ms)."""
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


def print_performance(entries, title="성능 분석"):
    """entries: [(n, pattern, filter), ...]"""
    print()
    print(LINE)
    print("# %s (평균/%d회)" % (title, REPEAT))
    print(LINE)
    print("%-8s %14s %14s %10s" % ("크기", "2D 평균(ms)", "1D 평균(ms)", "연산 횟수"))
    print("-" * 52)
    for n, pattern, filt in entries:
        t2d = measure_ms(mac, pattern, filt)
        pf, ff = pattern.flatten(), filt.flatten()
        t1d = measure_ms(mac_flat, pf, ff)
        size = "%d×%d" % (n, n)
        print("%-8s %14.4f %14.4f %10d" % (size, t2d, t1d, n * n))
    print("(2D: 기본 구현 / 1D: 보너스 최적화 - 동일 입력, 동일 반복 횟수)")


# ----------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ----------------------------------------------------------------------
def read_matrix(title, n=3):
    """n줄을 입력받아 Matrix 반환. 형식 오류 시 해당 행렬을 처음부터 재입력."""
    while True:
        print("%s (%d줄 입력, 공백 구분)" % (title, n))
        rows = []
        error = None
        for _ in range(n):
            line = input().lstrip("﻿")
            parts = line.split()
            if len(parts) != n:
                error = "입력 형식 오류: 각 줄에 %d개의 숫자를 공백으로 구분해 입력하세요." % n
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                error = "입력 형식 오류: 숫자만 입력하세요. (입력값: %s)" % line.strip()
                break
        if error is None:
            return Matrix(n, rows)
        print(error)
        print("→ %s을(를) 처음부터 다시 입력하세요.\n" % title)


def mode_user_input():
    print()
    print(LINE)
    print("# [1] 필터 입력")
    print(LINE)
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")
    print("\n✓ 필터 A, B 저장 완료")

    print()
    print(LINE)
    print("# [2] 패턴 입력")
    print(LINE)
    pattern = read_matrix("패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_ms(mac, pattern, filter_a)
    result = judge(score_a, score_b, "A", "B")

    print()
    print(LINE)
    print("# [3] MAC 결과")
    print(LINE)
    print("A 점수: %r" % score_a)
    print("B 점수: %r" % score_b)
    print("연산 시간(평균/%d회): %.4f ms" % (REPEAT, avg_ms))
    if result == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < %g)" % EPSILON)
    else:
        print("판정: %s" % result)

    print_performance([(3, pattern, filter_a)], "[4] 성능 분석")


# ----------------------------------------------------------------------
# 모드 2: data.json 분석
# ----------------------------------------------------------------------
PATTERN_KEY = re.compile(r"^size_(\d+)_(\d+)$")
FILTER_KEY = re.compile(r"^size_(\d+)$")


def load_filters(raw_filters):
    """{N: {표준 라벨: Matrix}} 반환. STANDARD_LABELS가 모두 있어야 로드 성공, 아니면 메시지 출력 후 제외."""
    filters = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 객체가 아닙니다.")
        return filters
    for key, group in raw_filters.items():
        m = FILTER_KEY.match(key)
        if not m:
            print("✗ %s: 키 형식 오류 (size_N 필요)" % key)
            continue
        n = int(m.group(1))
        if not isinstance(group, dict):
            print("✗ %s: 필터 그룹이 객체가 아님" % key)
            continue
        loaded = {}
        problems = []
        for label_key, rows in group.items():
            label = normalize_label(label_key)
            if label is None:
                problems.append("알 수 없는 라벨 '%s'" % label_key)
                continue
            try:
                mat = Matrix.from_rows(rows)
            except ValueError as e:
                problems.append("%s: %s" % (label, e))
                continue
            if mat.n != n:
                problems.append("%s: 크기 %d×%d (키는 %d×%d)" % (label, mat.n, mat.n, n, n))
                continue
            loaded[label] = mat
        missing = [l for l in STANDARD_LABELS if l not in loaded]
        if not missing:
            filters[n] = loaded
            print("✓ %-7s 필터 로드 완료 (%s)" % (key, ", ".join(STANDARD_LABELS)))
        else:
            print("✗ %-7s 필터 로드 실패 (누락: %s) %s" % (key, ", ".join(missing), "; ".join(problems)))
    return filters


def analyze_case(key, case, filters):
    """한 케이스 분석. (passed, reason) 반환. 예외 없이 FAIL 처리."""
    m = PATTERN_KEY.match(key)
    if not m:
        return False, "키 형식 오류 (size_N_idx 필요)"
    n = int(m.group(1))
    if not isinstance(case, dict) or "input" not in case or "expected" not in case:
        return False, "스키마 오류 (input/expected 누락)"
    expected = normalize_label(case["expected"])
    if expected is None:
        return False, "알 수 없는 expected 라벨: %r" % (case["expected"],)
    if n not in filters:
        return False, "size_%d 필터 없음" % n
    try:
        pattern = Matrix.from_rows(case["input"])
    except ValueError as e:
        return False, "패턴 형식 오류: %s" % e
    if pattern.n != n:
        return False, "크기 불일치: 키 size_%d, 패턴 %d×%d" % (n, pattern.n, pattern.n)

    # 표준 라벨마다 MAC 점수를 계산하고, 가장 높은(=가장 유사한) 라벨로 판정
    scores = {}
    for label in STANDARD_LABELS:
        scores[label] = mac(pattern, filters[n][label])
        print("%s 점수: %r" % (label, scores[label]))
    result = judge_scores(scores)
    if result == expected:
        print("판정: %s | expected: %s | PASS" % (result, expected))
        return True, ""
    if result == UNDECIDED:
        reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
        print("판정: %s | expected: %s | FAIL (동점 규칙)" % (result, expected))
    else:
        reason = "판정 %s ≠ expected %s" % (result, expected)
        print("판정: %s | expected: %s | FAIL" % (result, expected))
    return False, reason


def mode_json(path=DATA_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print("data.json 로드 실패: %s" % e)
        return

    print()
    print(LINE)
    print("# [1] 필터 로드")
    print(LINE)
    filters = load_filters(data.get("filters") if isinstance(data, dict) else None)

    print()
    print(LINE)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print(LINE)
    patterns = data.get("patterns", {}) if isinstance(data, dict) else {}
    if not isinstance(patterns, dict):
        patterns = {}
    total, passed, failures = 0, 0, []
    for key, case in patterns.items():
        total += 1
        print("--- %s ---" % key)
        try:
            ok, reason = analyze_case(key, case, filters)
        except Exception as e:  # 예상 못한 오류도 케이스 단위 FAIL
            ok, reason = False, "처리 중 오류: %s" % e
        if ok:
            passed += 1
        else:
            failures.append((key, reason))
            if not reason.startswith("동점") and not reason.startswith("판정"):
                print("FAIL: %s" % reason)

    entries = [(3, make_cross(3), make_cross(3))]
    for n in sorted(filters):
        if n == 3:
            continue
        entries.append((n, make_cross(n), filters[n][LABEL_CROSS]))
    print_performance(entries, "[3] 성능 분석")

    print()
    print(LINE)
    print("# [4] 결과 요약")
    print(LINE)
    print("총 테스트: %d개" % total)
    print("통과: %d개" % passed)
    print("실패: %d개" % len(failures))
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print("- %s: %s" % (key, reason))
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션 참고)')


# ----------------------------------------------------------------------
# 진입점
# ----------------------------------------------------------------------
def setup_console():
    """출력을 파일로 리다이렉트해도 cp949 콘솔에서 ✓ 등이 깨져 종료되지 않도록 UTF-8로 고정."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main():
    setup_console()
    print("=== Mini NPU Simulator ===")
    print()
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    while True:
        choice = input("선택: ").strip().lstrip("﻿")  # 파이프 입력 BOM 제거
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력하세요.")
    if choice == "1":
        mode_user_input()
    else:
        mode_json(sys.argv[1] if len(sys.argv) > 1 else DATA_FILE)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")
