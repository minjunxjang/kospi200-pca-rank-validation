# KOSPI 200 단기반전 × PCA 공통요인 제거 검증

과거 수익률에서 PCA 공통요인을 제거했을 때 KOSPI 200 종목의 횡단면 단기반전 신호가 미래 수익률을 더 잘 설명하는지 검증한 연구입니다. 일별 Rank IC와 Quintile IC를 계산하고, 같은 형성기간의 Raw 신호와 PCA 신호의 차이를 전체 기간 및 연도별로 비교합니다.

![PCA Rank Validation 3D 요약](results/figures/step1234%20%ED%86%B5%ED%95%A9%20%EC%82%AC%EC%A7%84.png)

## 연구 설계

| 항목 | 설정 |
| --- | --- |
| 신호 | 과거 `n`거래일 누적 로그수익률의 음수. Raw 또는 PCA 잔차 수익률 사용 |
| 형성기간 `n` | 5, 10, 20, 60거래일 |
| 미래기간 `k` | 5, 10, 20, 60거래일 |
| PCA 추정 창 `W` | 60, 120, 252거래일 |
| 제거 성분 | `PC1`, `PC1–2`, `PC1–3` |
| 비교 대상 | Raw 4개와 PCA 36개, 총 40개 case |
| Rank IC | 매 거래일 종목별 신호 순위와 미래수익률 순위의 Spearman 상관계수 |
| Quintile IC | 매 거래일 신호 분위수 번호(1–5)와 미래수익률 분위수 번호(1–5)의 Pearson 상관계수 |
| Delta IC | PCA case의 IC에서 동일한 `n`의 Raw case IC를 뺀 값 |

`full_summary.csv`는 전체 기간, `annual_summary.csv`는 연도별 결과입니다. 평균의 표준오차와 검정 통계량에는 `max(n, k) - 1` 시차의 HAC 추정이 쓰이고, 다중검정에는 Benjamini–Hochberg 보정을 적용합니다. Delta 지표는 PCA case 36개에만 정의됩니다. `annual_summary.csv`에는 분석 시작 전 준비 구간인 2008년 행도 포함되며, 관측치가 없는 결과는 빈 값입니다.

## 폴더 구성

```text
kospi200-pca-rank-validation/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── run_pca_rank_validation.py  # 원천 XLSX → 신호·IC·요약 CSV
│   ├── recompute_summaries.py      # 저장된 일별 IC → 요약 CSV 재계산
│   ├── build_visuals.py            # 3D 그림·지표별 영상 생성 함수
│   ├── run_analysis.py             # 저장소 경로를 지정하는 분석 실행 파일
│   └── run_visualization.py        # CSV에서 그림·영상 생성 실행 파일
├── data/
│   ├── raw/                      # 별도로 준비한 원천 XLSX 5개를 여기에 둠
│   └── processed/
│       ├── cases.csv
│       ├── full_summary.csv
│       └── annual_summary.csv
└── results/
    ├── figures/
    │   └── step1234 통합 사진.png
    └── videos/
        ├── step1 히트맵.mp4
        ├── step2 히트맵.mp4
        ├── step3 히트맵.mp4
        └── step4 히트맵.mp4
```

`data/processed/`와 `results/`에는 바로 열어볼 수 있는 결과물을 넣었습니다. `results/generated/`에는 새 그림·영상이, `results/recomputed/`에는 원천 XLSX로 다시 계산한 결과가 생성됩니다. 재실행해도 함께 제공된 결과 파일은 덮어쓰지 않습니다.

## 설치

Windows PowerShell에서 이 폴더로 이동한 뒤 실행합니다. Python 3.10 이상을 권장합니다.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux에서는 `python3 -m venv .venv`를 실행하고, 아래 명령의 `.\.venv\Scripts\python.exe`를 `.venv/bin/python`으로 바꾸면 됩니다. 한글 레이블을 표시할 수 있는 글꼴이 필요합니다. Windows에서는 스크립트가 `Malgun Gothic`을 우선 사용합니다.

## 포함된 CSV로 그림 다시 만들기

원천 XLSX가 없어도 가능합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_visualization.py
```

`results/generated/09_PCA_RANK_SPACE_OVERVIEW_CLEAR.png`가 4개 지표를 합친 그림이며, 같은 폴더에 지표별 PNG 4개도 생성됩니다. 4개 지표의 연도 선택 애니메이션까지 만들려면 다음 명령을 사용합니다. 영상 생성은 그림보다 오래 걸립니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_visualization.py --videos
```

영상 출력 파일은 `results/generated/10_1_RANK_IC_TIME_SELECTION.mp4`부터 `10_4_DELTA_QUINTILE_IC_TIME_SELECTION.mp4`까지입니다. 함께 제공된 `results/videos/step1–4 히트맵.mp4`는 발표용 최종 영상 파일입니다.

다른 CSV를 사용하려면 다음처럼 세 CSV가 들어 있는 폴더를 지정합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_visualization.py --data "C:\path\to\csv-folder" --output "C:\path\to\new-results"
```

## 원천 XLSX부터 전체 분석 다시 실행하기

아래 5개 파일을 `data/raw/`에 넣습니다. 원천 데이터는 이 프로젝트에 포함하지 않았습니다.

```text
00_COMMON252_BASE.xlsx
02_PCA_W60_CUMULATIVE.xlsx
04_PCA_W120_CUMULATIVE.xlsx
06_PCA_W252_CUMULATIVE.xlsx
FUTURE_LOG_RETURNS_5_10_20_60.xlsx
```

이름과 시트 구조가 위 분석 코드에서 기대하는 것과 같아야 합니다. 원천 파일에는 동일한 날짜·종목 순서의 데이터가 있어야 하며, 코드는 4,582거래일 × 407종목 형태를 기대합니다. 파일이 준비되면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_analysis.py
```

`results/recomputed/csv/`에 `cases.csv`, `full_summary.csv`, `annual_summary.csv`가 생깁니다. `results/recomputed/intermediate/`에는 일별 지표, `rank_cubes.npz`, 분위수 혼동표와 메타데이터가 저장됩니다. 다른 위치에 원천 파일이 있으면 `--source-xlsx-dir "C:\path\to\xlsx-folder"`를 지정합니다.

저장한 `rank_cubes.npz`에서 통계 요약만 다시 계산할 때는 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_analysis.py --summaries-only
```

새로 계산한 CSV로 그림을 그리려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_visualization.py --data results\recomputed\csv
```

