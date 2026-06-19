# 작업지시서: 서버 최종 점수 모델을 V12 단일 출력으로 전환

## 1. 목표

현재 서버에 적용된 V6~V11 개별 모델 출력/분류 로직을 제거하고, 최종 점수와 폴더 분류는 **V12 Selected Ridge**만 사용하도록 변경하라.

중요:

```text
V12는 image-to-score 단독 모델이 아니라,
V6/V7/V8/V10-B/V11 Student/V11 Mission의 점수를 feature로 사용하는 meta ensemble이다.
```

따라서 V6~V11을 완전히 삭제하지 말고, 서버 외부 출력에서는 숨기고 내부 feature 생성용 helper로만 사용한다.

최종 서버 출력은 반드시 다음만 사용한다.

```text
final_model = V12 Selected Ridge
final_score = v12_score
final_pred = v12_pred
folder = final_pred_Star
```

---

## 2. 사용할 V12 artifact

서버 적용용 artifact:

```text
D:\Cos\taste_model\v12_results\v12_server_fullfit_selected_ridge.json
```

이 파일을 서버 시작 시 로드하라.

필수 정보:

```text
model_name: V12 Selected Ridge
run_key: v12_a_ridge|ridge|alpha0.1|direct

feature_columns:
- v6_score
- v7_score
- v8_score
- v10b_score
- v11_student_score
- v11_mission_score
- v10b_mine_prob
- v10b_gate
- score_spread

thresholds:
[2.052030349342921,
 2.253570380264428,
 3.4048756759645045,
 4.5925223831376485]

prediction_formula:
score = clip(intercept + sum(coef_i * ((x_i - scaler_mean_i) / scaler_scale_i)), 1, 5)
pred = 1 + sum(score >= thresholds_i)
```

---

## 3. 서버 구조 변경

### 3.1 기존 구조 제거 대상

외부 API, UI, 폴더 이동 로직에서 다음 모델을 최종 모델로 쓰는 코드를 제거하거나 비활성화한다.

```text
V6
V7
V8
V10-B
V11 Student
V11 Mission
```

이 모델들은 최종 결과로 노출하지 않는다.

금지:

```text
response에 v6_rating, v8_rating, v10b_rating, v11_rating 등을 최종 판단처럼 표시
폴더 이동 기준으로 V11 Mission 사용
V8 단독 점수로 폴더 분류
V10-B 단독 점수로 폴더 분류
```

허용:

```text
V12 feature 생성을 위해 내부적으로 V6~V11 점수를 계산
debug=true일 때만 내부 feature 표시
```

---

### 3.2 새 구조

새 이미지 입력 시 파이프라인은 다음과 같이 바꾼다.

```text
새 이미지
↓
기존 embedding/segmentation 생성
↓
V6 score 계산
V7 score 계산
V8 score 계산
V10-B score, pred, gate, mine_prob 계산
V11 Student score 계산
V11 Mission score 계산
↓
score_spread 계산
↓
V12 Selected Ridge 계산
↓
v12_score, v12_pred 산출
↓
v12_pred 기준으로 N_Star 폴더 이동
```

서버 응답 기본값:

```json
{
  "model": "V12 Selected Ridge",
  "run_key": "v12_a_ridge|ridge|alpha0.1|direct",
  "score": 4.37,
  "pred": 4,
  "folder": "4_Star"
}
```

debug 모드에서만 다음을 추가한다.

```json
{
  "debug_features": {
    "v6_score": 0,
    "v7_score": 0,
    "v8_score": 0,
    "v10b_score": 0,
    "v11_student_score": 0,
    "v11_mission_score": 0,
    "v10b_mine_prob": 0,
    "v10b_gate": 0,
    "score_spread": 0
  }
}
```

---

## 4. V12 Predictor 구현

`v12_server_fullfit_selected_ridge.json`을 로드해서 다음 클래스를 구현하라.

```python
class V12SelectedRidgePredictor:
    def __init__(self, artifact_path):
        ...
    
    def predict_from_features(self, features: dict) -> dict:
        ...
```

동작:

```python
x_scaled_i = (x_i - scaler_mean_i) / scaler_scale_i
score_raw = intercept + sum(coef_i * x_scaled_i)
score = min(5.0, max(1.0, score_raw))
pred = 1 + sum(score >= t for t in thresholds)
```

반환:

```python
{
    "model_name": "V12 Selected Ridge",
    "run_key": "v12_a_ridge|ridge|alpha0.1|direct",
    "score": score,
    "pred": pred,
    "features": features
}
```

주의:

```text
threshold는 반드시 artifact의 server full-fit threshold를 사용한다.
OOF fold threshold를 서버에 쓰지 않는다.
```

---

## 5. Feature 생성 로직

V12 입력 feature는 다음 9개다.

```text
v6_score
v7_score
v8_score
v10b_score
v11_student_score
v11_mission_score
v10b_mine_prob
v10b_gate
score_spread
```

`score_spread` 계산:

```python
base_scores = [
    v6_score,
    v7_score,
    v8_score,
    v10b_score,
    v11_student_score,
    v11_mission_score,
]

score_spread = max(base_scores) - min(base_scores)
```

만약 기존 prediction table이나 cache에 값이 있으면 재사용한다.

우선순위:

```text
1. DB에 기존 prediction이 있으면 로드
2. 없으면 해당 base model inference 실행
3. 결과를 DB에 저장
4. V12 feature로 사용
```

---

## 6. DB 저장 구조

V12 결과 저장 테이블을 만든다.

DB:

```text
D:\Cos\taste_model\aurarate.db
```

테이블 예시:

```sql
CREATE TABLE IF NOT EXISTS prediction_v12 (
    key TEXT PRIMARY KEY,
    filepath TEXT,

    model_name TEXT NOT NULL,
    run_key TEXT NOT NULL,

    score REAL NOT NULL,
    pred INTEGER NOT NULL,

    v6_score REAL,
    v7_score REAL,
    v8_score REAL,
    v10b_score REAL,
    v11_student_score REAL,
    v11_mission_score REAL,
    v10b_mine_prob REAL,
    v10b_gate REAL,
    score_spread REAL,

    artifact_path TEXT,
    artifact_json TEXT,

    status TEXT DEFAULT 'done',
    error TEXT,
    created_at TEXT,
    updated_at TEXT
);
```

폴더 이동 후 filepath가 바뀌면 `prediction_v12.filepath`도 갱신한다.

---

## 7. 폴더 이동 기준 변경

기존:

```text
V6/V8/V10/V11 중 하나의 pred 기준
```

새 기준:

```text
V12 pred 기준
```

폴더:

```text
D:\Cos\images\pending\1_Star
D:\Cos\images\pending\2_Star
D:\Cos\images\pending\3_Star
D:\Cos\images\pending\4_Star
D:\Cos\images\pending\5_Star
```

이동 로직:

```python
target_folder = f"D:\\Cos\\images\\pending\\{v12_pred}_Star"
```

파일 이동 후 DB 경로 업데이트.

---

## 8. API 변경

### 8.1 `/predict` 또는 기존 분석 API

응답에서 기본적으로 V12만 반환한다.

```json
{
  "status": "ok",
  "model": "V12 Selected Ridge",
  "run_key": "v12_a_ridge|ridge|alpha0.1|direct",
  "score": 4.12,
  "pred": 4,
  "folder": "4_Star"
}
```

### 8.2 debug 옵션

`debug=true`일 때만 base model feature를 반환한다.

```json
{
  "debug": {
    "v6_score": 3,
    "v7_score": 3,
    "v8_score": 4,
    "v10b_score": 4.2,
    "v11_student_score": 3,
    "v11_mission_score": 4,
    "v10b_mine_prob": 0.02,
    "v10b_gate": 0.63,
    "score_spread": 1.2
  }
}
```

---

## 9. 기존 모델 제거의 정확한 의미

삭제하면 안 되는 것:

```text
V6/V7/V8/V10-B/V11 inference 함수
prediction table
checkpoint
embedding 생성 코드
```

삭제하거나 비활성화할 것:

```text
V6/V7/V8/V10-B/V11을 최종 모델로 선택하는 옵션
UI에서 V6~V11 점수를 최종 점수처럼 보여주는 부분
V11 Mission을 폴더 이동 기준으로 쓰는 부분
서버 설정의 default_model = v11_mission
```

새 기본값:

```text
default_model = v12_selected_ridge
```

---

## 10. 검증 절차

### 10.1 Artifact 로드 테스트

```text
v12_server_fullfit_selected_ridge.json 로드 성공
feature_columns 9개 확인
thresholds 4개 확인
coef/scaler 누락 없음
```

### 10.2 기존 labeled sample 10개 테스트

DB에서 labeled image 10개를 뽑아 V12 prediction을 계산하고, `prediction_v12`에 저장되는지 확인한다.

### 10.3 기존 V12 holdout prediction 재현

가능하면 다음 파일과 일부 샘플을 비교한다.

```text
D:\Cos\taste_model\v12_results\v12_holdout_predictions.csv
```

오차 허용:

```text
score 오차 < 1e-6
pred 동일
```

### 10.4 새 이미지 end-to-end 테스트

새 이미지 1장에 대해:

```text
1. 이미지 입력
2. base feature 생성
3. V12 score/pred 생성
4. prediction_v12 저장
5. pred_Star 폴더로 이동
6. API 응답이 V12만 반환
```

---

## 11. 성공 조건

작업 성공 조건:

```text
서버 기본 모델이 V12 Selected Ridge로 바뀜
외부 응답에서 V6~V11 개별 점수가 기본 노출되지 않음
폴더 이동 기준이 V12 pred로 바뀜
prediction_v12 테이블에 결과 저장됨
debug=true일 때만 V6~V11 feature 확인 가능
기존 V11 Mission 기준 분류 코드가 더 이상 기본 경로에서 실행되지 않음
```

---

## 12. 주의사항

1. V12는 base model score를 feature로 쓰므로 V6~V11 내부 계산 코드까지 삭제하지 말 것.

2. V12 artifact의 threshold는 server full-fit threshold를 사용한다.

3. V8은 단독 severe_fn 이슈가 있으므로 V8 단독 분류를 절대 사용하지 말 것.

4. V11 Mission은 fallback으로 남겨도 되지만 default는 V12다.

5. base feature 중 하나라도 누락되면 V12를 계산하지 말고 error를 반환한다.

6. 새 이미지 처리 시 base model 결과는 cache하여 다음 호출에서 재사용한다.

---

## 13. 최종 서버 기본 설정

최종 설정값:

```text
DEFAULT_MODEL = "v12_selected_ridge"
V12_ARTIFACT = "D:\\Cos\\taste_model\\v12_results\\v12_server_fullfit_selected_ridge.json"
SAVE_TABLE = "prediction_v12"
DEBUG_BASE_MODELS = false
```

최종 한 줄 목표:

```text
서버는 외부적으로 V12 단일 모델처럼 동작해야 한다.
내부적으로만 V6~V11을 V12 feature 생성용 helper로 사용한다.
```
