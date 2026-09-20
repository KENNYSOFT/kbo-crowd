"""KBO 데이터 수집 공통 유틸리티.

표준 라이브러리만 쓴다. 매일 도는 수집이 의존성 설치 없이 끝나야
GitHub Actions 가 빠르고, 패키지 변화로 깨질 일도 없다.
분석 쪽(analysis/)만 pandas 계열을 쓴다.
"""

import csv
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Windows 콘솔 기본 코드페이지(cp949)로는 이 스크립트들이 찍는 한국어가 깨진다.
# Actions 는 UTF-8 이라 무관하지만 로컬 실행 결과도 읽을 수 있어야 한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

BASE = "https://www.koreabaseball.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"


def http(url, data=None, headers=None, retries=3, timeout=45):
    """GET 또는 POST 후 본문을 str 로 돌려준다. data 가 dict 면 form 인코딩한다."""
    body = None
    hdrs = {"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"}
    if data is not None:
        body = urllib.parse.urlencode(data, encoding="utf-8").encode()
        hdrs["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    if headers:
        hdrs.update(headers)

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.read().decode("utf-8", "replace")
        except (urllib.error.URLError, ssl.SSLError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError("요청 실패: %s (%s)" % (url, last))


def write_csv(path, header, rows):
    """CSV 를 원자적으로 쓴다.

    개행을 LF 로 고정하는 것이 중요하다. 로컬은 Windows, Actions 는 Linux 라서
    개행이 갈리면 내용이 같은데도 매일 전체 파일이 바뀐 것으로 잡힌다.
    행 정렬은 호출하는 쪽에서 이미 끝내 놓아야 한다.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)
    return len(rows)


def read_csv(path):
    """CSV 를 dict 리스트로 읽는다. 없으면 빈 리스트."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def data_path(name):
    return os.path.join(DATA_DIR, name)


def report(label, path, n):
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    print("%-10s %-22s %5d행" % (label, rel, n))
