# kbo-crowd

KBO 경기별 관중 기록을 매일 모아 CSV 로 쌓고, 좌석 제약에 잘린 수요를 되찾아 보는 개인 프로젝트.

## 이 데이터의 성질

KBO 가 공개하는 관중 수는 수요가 아니다. 표가 다 팔리면 기록이 구장 수용인원에서 멈추고, 그 위로 얼마나 더 원했는지는 사라진다. 2023년 이후 경기의 3할 남짓이 그렇게 상한에 눌려 있다. 그래서 평균 관중으로 인기를 재면 구장이 작은 팀이 체계적으로 저평가된다. 한화가 대표적이다. 평균 관중은 중하위권인데 매진율은 1위인데, 그것은 대전 구장이 17,000석이고 대구가 24,000석이기 때문이지 수요가 작아서가 아니다.

이 저장소의 분석이 검열 회귀(Tobit)로 시작하는 이유가 그것이다. 매진하지 않은 경기는 관측값이 곧 수요이고, 매진한 경기는 '수요가 최소한 좌석 수 이상'이라는 사실만 알려준다. 둘을 나눠 우도를 쓰면 잘린 부분을 메운 수요를 추정할 수 있다.

## 데이터

매일 전 시즌을 다시 받아 통째로 덮어쓴다. 요청이 시즌 수만큼이라 부담이 없고, KBO 가 과거 수치를 정정했을 때 그것도 따라온다. SQLite 대신 CSV 를 정본으로 두는 것도 같은 이유다. 연 720행이라 가볍고, 무엇보다 바뀐 줄이 diff 에 그대로 남는다.

| 파일 | 단위 | 내용 |
|---|---|---|
| `data/games.csv` | 경기 | 날짜, 요일, 홈, 원정, 구장, 관중 수 |
| `data/schedule.csv` | 편성 | 시작 시각, 스코어, 우천취소 여부 (취소된 경기도 남는다) |
| `data/standings.csv` | 경기 | 그 경기 **직전**의 순위, 승률, 게임차, 연승, 최근 10경기 |
| `data/weather.csv` | 시간 | 구장 인근 관측소의 기온, 강수, 습도, 풍속, 운량 |
| `data/stadiums.csv` | 구장 | 홈 구단, 돔 여부, 대응 관측소, 좌표 (손으로 관리) |
| `data/dataset.csv` | 경기 | 위를 모두 이어 붙이고 수용인원과 매진 여부를 붙인 모델 입력 |

출처는 KBO 기록실의 [경기 관중 현황](https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx), 일정 페이지의 내부 API, 그리고 [기상청 API 허브](https://apihub.kma.go.kr)다.

### 알아둘 것

경기 단위 관중 기록은 **2021년부터만** 나온다. 그 이전을 요청하면 빈 결과가 온다. 시즌과 팀 단위 총관중은 1982년까지 `History.aspx` 에 있지만 아직 수집하지 않는다.

수용인원은 원자료에 없어서 관측값에서 되찾는다. 표가 다 팔리면 발표 관중 수가 정확히 같은 숫자로 여러 번 찍히므로, (시즌, 구장) 별로 두 번 이상 나온 값 중 최댓값을 상한으로 본다. 시즌마다 따로 잡는 이유는 상한이 해마다 움직이기 때문이다. 대전은 2025년에 신구장으로 옮겼고, 사직과 창원은 좌석이 해마다 바뀌었다.

2021년과 2022년 상당 기간은 코로나 입장 제한이 걸려 있었다. 그때의 상한은 구장 크기가 아니라 방역 지침이라 수요를 재는 데 쓸 수 없다. `load_dataset` 이 기본으로 제외한다.

## 쓰는 법

수집 스크립트는 표준 라이브러리만 쓴다. 설치할 것이 없다.

```bash
python scripts/fetch_crowd.py
python scripts/fetch_schedule.py
python scripts/build_standings.py
python scripts/fetch_weather.py
python scripts/build_dataset.py
```

분석은 numpy, scipy, pandas 가 필요하다.

```bash
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
```

```bash
.venv\Scripts\python analysis/demand.py --holdout-from 2026-08-01
```

```bash
.venv\Scripts\python analysis/predict_today.py
```

```bash
.venv\Scripts\python analysis/test_model.py
```

`demand.py` 는 홈과 원정으로 나눈 티켓 파워와 잠재 수요 순위를 낸다. `predict_today.py` 는 그날 경기의 매진 확률과 예상 관중을 낸다. `rain_price.py` 는 비가 수요를 얼마나 깎는지 재는데 날씨 데이터가 있어야 돈다.

### 대시보드

`dashboard/index.html` 이 탐색용 화면이다. 데이터는 생성물이라 버전 관리하지 않으므로 먼저 만들어야 한다.

```bash
.venv\Scripts\python analysis/export_dashboard.py
```

그 다음 `dashboard` 디렉토리를 정적 서버로 열면 된다. 구장 정렬 기준은 화면에서 고를 수 있고, 무엇으로 정렬했는지 차트 위에 표시된다. 아래 수용인원 표도 같은 순서를 따라가므로 두 화면을 나란히 대조할 수 있다.

### 날씨

기상청 API 허브에서 인증키를 받아 `KMA_API_KEY` 환경변수로 넘긴다. 키가 없으면 날씨 수집만 건너뛰고 나머지는 그대로 돈다. GitHub Actions 에서는 저장소 시크릿에 같은 이름으로 넣는다.

## 자동화

`.github/workflows/daily.yml` 이 매일 KST 05:00 에 돌아 바뀐 데이터만 커밋한다. 시즌이 끝난 뒤에는 바뀔 것이 없으니 커밋도 생기지 않는다. 전 시즌을 다시 받고 싶으면 workflow_dispatch 에서 `full` 을 켠다.

## 모델을 손댈 때

`analysis/test_model.py` 를 먼저 돌린다. 참값을 아는 합성 데이터로 검열 회귀가 모수를 되찾는지, 기울기가 수치 미분과 맞는지, 검열이 없을 때 최소제곱과 같아지는지를 본다. 실제 관중 데이터로는 맞았는지 확인할 방법이 없다. 매진 경기의 진짜 수요는 아무도 모르기 때문이다.

검증은 시즌을 통째로 빼는 대신 날짜로 자른다. 시즌을 범주형으로 넣어서 학습에 없던 시즌은 예측할 수 없는데, KBO 관중이 해마다 크게 늘어 그 효과를 빼면 모델이 망가진다. 실제 쓰임새인 오늘 경기 예측이 '같은 시즌의 앞부분으로 학습해 뒷부분을 맞히는' 상황이라 그쪽이 맞기도 하다.
