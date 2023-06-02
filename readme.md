# 2022 암 예후예측 데이터 구축 AI 경진대회
## 암 융합 데이터를 이용한 암 예후 예측

### 코드 구조

```
$/
├── /DATA
│   ├── /DATA/train
│   │		└── train.csv
│	├── /DATA/test	
│	│		└── test.csv
│   └── sample_submission.csv
└── /USER/baseline
    └── testing.py
```

- testing.py : 학습 및 EDA, TEST 중 어떤 작업을 할 것인지 Argument로 설정 가능, pretrain model은 사용하지 않았음.

### 필수 라이브러리
 - sklearn
 - pandas
 - pycox
 - pytorch
 - numpy

### 사용 모델
- model : CoxPH

### EDA

1. 'python testing.py --EDA'
	1. 코드내의 EDA코드를 실행시킴 

### 학습

1. 'python testing.py --TRAIN'
    1. 코드내에 설정한 파라미터로 학습 시작
2. '/USER/baseline/results/model' 내에 모델이 저장됨(.pt, .pickle)

### 추론
1. 'python testing.py --Predict'
	1. 코드내에 설정된 모델 이름에 해당하는 모을 불러오고, 해당 모델로 추론
2. '/USER/baseline/results/predict/' 내에 결과 csv 파일 생성
