# V12 실험계획서: Specialist Ensemble / Stacking

## 1. 실험 목적

V12의 목표는 새로운 단일 모델 하나를 무리하게 만드는 것이 아니라, 지금까지 만든 모델들의 강점을 조합해서 최종 점수를 더 안정적으로 끌어올리는 것이다.

현재 모델별 역할은 다음처럼 본다.

```text
V11 Mission = 안전 anchor / severe_fn=0 / high-score protection
V8          = QWK, Spearman, MAE, rescue가 강한 보조 모델
V10-B       = 1점 mine 감지 특화 / mine_prob, gate 제공
V6/V7       = V11 Mission 오답 rescue 후보
V11 Student = Teacher attention distillation 신호
V9          = 가능하면 추가 prediction 생성 후 ranking/balance 후보
```

V12는 이 모델들의 예측값과 보조 feature를 입력으로 받아 최종 1~5점 점수를 예측하는 meta ensemble이다.

---

## 2. 핵심 목표

V12는 다음 조건을 만족해야 한다.

```text
필수 조건:
- severe_fn = 0

주요 목표:
- V11 Mission보다 AScore 또는 95_score 상승
- QWK 상승 또는 최소 유지
- Spearman 상승 또는 최소 유지
- mine_recall 유지 또는 상승
- MAE 하락
```

가장 중요한 안전 기준:

```text
실제 4~5점 이미지가 예측 1~2점으로 떨어지면 안 됨.
```

따라서 `severe_fn=0`은 hard gate다.

---

## 3. 사용 DB

기본 DB:

```text
D:\Cos\taste_model\aurarate.db
D:\Cos\taste_model\test_result.db
D:\Cos\taste_model\heatmap.db
```

주요 테이블:

```text
aurarate.db
- labels
- predictions_v6
- predictions_v6_view
- predictions_v7_view
- predictions_v8_view
- prediction_v10b
- predictions_v10b_view
- predictions_v11
- predictions_v11_view
- predictions_v11_mission
- predictions_v11_mission_view

test_result.db
- results_v6
- results_v7
- results_v8
- results_v9
- results_v10
- v11_runs
- v11_full_runs
- ascore_leaderboard
```

주의:

```text
최종 V12 학습/평가는 user rating 기준만 사용한다.
folder label/internal split aggregate는 참고용으로만 사용하고 직접 비교하지 않는다.
```

---

## 4. 학습 기준 라벨

라벨 기준은 다음으로 고정한다.

```text
aurarate.db / labels.user_rating
```

`labels.filepath`를 기준으로 각 prediction view와 join한다.

가능하면 다음을 모두 포함한다.

```text
label = user_rating
key 또는 file_hash
filepath
```

---

## 5. V12 Feature Table 생성

먼저 `v12_feature_table`을 만든다.

권장 저장 위치:

```text
D:\Cos\taste_model\aurarate.db
```

또는 실험 분리를 원하면:

```text
D:\Cos\taste_model\v12_experiment.db
```

### 5.1 기본 feature

각 labeled image마다 다음 feature를 만든다.

```text
label

v6_score
v6_pred

v7_score
v7_pred

v8_score
v8_pred

v10b_score
v10b_pred
v10b_gate
v10b_mine_prob
v10b_p_ge_1
v10b_p_ge_2
v10b_p_ge_3
v10b_p_ge_4
v10b_p_ge_5

v11_student_score
v11_student_pred

v11_mission_score
v11_mission_pred
```

단, 일부 prediction table이 class rating만 갖고 있고 연속 score가 없으면 class rating을 score로 사용한다.

---

### 5.2 파생 feature

다음 파생 feature를 반드시 만든다.

```text
score_mean
score_median
score_min
score_max
score_spread = score_max - score_min

high_vote_count = count(pred >= 3)
very_high_vote_count = count(pred >= 4)

mine_vote_count = count(pred == 1)
low_vote_count = count(pred <= 2)

v8_minus_mission = v8_score - v11_mission_score
v10b_minus_mission = v10b_score - v11_mission_score
v6_minus_mission = v6_score - v11_mission_score

max_minus_mission = score_max - v11_mission_score
mission_minus_min = v11_mission_score - score_min

v10b_conf_high = p_ge_4 또는 p_ge_5 기반 feature
v10b_conf_low = 1 - p_ge_2 또는 mine_prob 기반 feature
```

모델 간 disagreement가 큰 이미지는 ensemble에서 중요한 케이스이므로 `score_spread`는 필수다.

---

## 6. Baseline

V12를 평가하기 전에 반드시 다음 baseline을 같은 labeled set에서 재계산한다.

```text
Baseline A: V11 Mission deployed
Baseline B: V8 deployed
Baseline C: V10-B deployed
Baseline D: 단순 평균 ensemble
Baseline E: 현재 V11 Mission 수동 공식
```

각 baseline에 대해 다음 지표를 계산한다.

```text
QWK
Spearman
MAE
accuracy
adjacent_accuracy
mine_recall
severe_fn
severe_fp
high_recall_safe
high_recall_strict
class별 precision/recall/f1
AScore
95_score
```

---

## 7. 실험할 V12 모델

### 7.1 V12-A: Linear Weighted Ensemble

가장 먼저 단순 선형 조합을 실험한다.

```text
final_score =
b
+ w1 * v6_score
+ w2 * v7_score
+ w3 * v8_score
+ w4 * v10b_score
+ w5 * v11_student_score
+ w6 * v11_mission_score
+ w7 * v10b_mine_prob
+ w8 * v10b_gate
+ w9 * score_spread
```

모델 후보:

```text
Ridge
ElasticNet
HuberRegressor
```

권장 시작:

```text
Ridge(alpha grid)
```

이유:

```text
데이터가 5~6천 장 수준이라 복잡한 meta model은 과적합 위험이 크다.
```

---

### 7.2 V12-B: Linear Ensemble + Guard

V12-A 결과에 safety guard를 추가한다.

예시 guard:

```text
if high_vote_count >= 2:
    final_score = max(final_score, 3)

if v8_score >= 3 or v11_mission_score >= 3:
    final_score = max(final_score, 3)

if v10b_mine_prob >= threshold and mine_vote_count >= 1:
    final_score = min(final_score, 2)
```

단, mine guard는 너무 강하게 걸면 4~5점 보호를 망칠 수 있으므로 반드시 severe_fn을 확인한다.

추천 guard search:

```text
high_vote_count threshold: 1, 2, 3
high_score floor: 2.5, 3.0, 3.25
mine_prob threshold: 0.6, 0.7, 0.8, 0.9
mine floor/ceiling 적용 여부
```

---

### 7.3 V12-C: Residual Correction

V11 Mission을 anchor로 두고, 다른 모델들이 보정값만 예측하게 한다.

```text
residual = label - v11_mission_score

residual_pred = meta_model(features)

final_score = v11_mission_score + residual_pred
```

장점:

```text
V11 Mission의 severe_fn=0 안정성을 유지하기 쉽다.
```

이 방식은 V12에서 가장 유망할 가능성이 높다.

추천 모델:

```text
Ridge residual
ElasticNet residual
Huber residual
```

---

### 7.4 V12-D: Specialist Router

특정 상황에서 specialist를 더 신뢰하는 rule-based router를 실험한다.

예시:

```text
if v10b_mine_prob high and v8_score low:
    mine specialist 가중치 증가

if v8_score high and mission_score lower:
    V8 rescue 가중치 증가

if score_spread high:
    review flag 또는 conservative score 사용

if high_vote_count high:
    high protection guard 적용
```

이 모델은 ML 모델이라기보다 rule ensemble이다.

목표:

```text
V11 Mission 오답 중 V8/V6/V10-B가 맞히는 rescue case를 최대한 회수한다.
```

---

### 7.5 V12-E: LightGBM / Tree Model

선택 실험으로만 수행한다.

```text
LightGBM Regressor
XGBoost
CatBoost
sklearn HistGradientBoostingRegressor
```

주의:

```text
데이터가 적고 feature가 강한 예측값들이라 tree model이 validation에 과적합할 수 있다.
```

따라서 V12-E가 가장 높게 나오더라도, fold별 안정성과 severe_fn=0을 반드시 확인한다.

---

## 8. Cross Validation 설계

V12는 user rating 5~6천 장 기준으로 평가한다.

권장:

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=20260620)
```

stratify 기준:

```text
label
```

각 fold에서:

```text
1. train fold로 meta model 학습
2. train fold 내부에서 threshold/guard 튜닝
3. validation fold에서 성능 평가
4. 모든 fold 예측을 합쳐 OOF prediction 생성
```

절대 하면 안 되는 것:

```text
전체 validation label을 보고 threshold를 맞춘 뒤 같은 데이터에 평가
```

threshold tuning은 fold 내부 train에서만 수행해야 한다.

---

## 9. Threshold Tuning

meta model은 연속 score를 낸다.

최종 class는 threshold로 변환한다.

기본 방식:

```text
score → 1~5 class
```

threshold 후보:

```text
t12, t23, t34, t45
```

최적화 목표:

```text
primary: severe_fn=0
secondary: AScore 또는 95_score 최대화
tertiary: QWK, Spearman, MAE
```

threshold search는 다음을 포함한다.

```text
기본 round threshold:
1.5, 2.5, 3.5, 4.5

random search:
각 threshold 범위에서 5000~20000회

constraint:
t12 < t23 < t34 < t45
```

---

## 10. 평가 지표

반드시 다음 지표를 계산한다.

```text
QWK
Spearman
MAE
accuracy
adjacent_accuracy

mine_recall
severe_fn
severe_fp

high_recall_safe
high_recall_strict

class_1_precision
class_1_recall
class_1_f1
class_2_precision
class_2_recall
class_2_f1
class_3_precision
class_3_recall
class_3_f1
class_4_precision
class_4_recall
class_4_f1
class_5_precision
class_5_recall
class_5_f1

AScore
95_score

rescue_count_vs_mission
rescue_rate_vs_mission
unique_correct_vs_mission
error_overlap_vs_mission
```

---

## 11. 선택 기준

V12 후보는 다음 조건을 통과해야 한다.

### 11.1 필수 통과

```text
OOF severe_fn = 0
```

### 11.2 강한 채택 조건

```text
V11 Mission 대비:
AScore 상승
또는 95_score 상승
또는 QWK 상승

그리고 Spearman/MAE가 크게 악화되지 않아야 함.
```

권장 최소 조건:

```text
QWK >= V11 Mission - 0.002
Spearman >= V11 Mission - 0.002
mine_recall >= V11 Mission - 0.003
MAE <= V11 Mission + 0.02
severe_fn = 0
```

### 11.3 보류 조건

```text
점수는 올랐지만 severe_fn > 0
fold마다 성능 편차가 큼
특정 class 성능이 크게 무너짐
user rating이 아닌 split에서만 좋은 결과
```

---

## 12. Output DB / 파일

### 12.1 DB 테이블

권장 DB:

```text
D:\Cos\taste_model\v12_experiment.db
```

테이블:

```text
v12_feature_table
v12_runs
v12_oof_predictions
v12_fold_metrics
v12_candidate_summary
```

### 12.2 CSV / JSON / MD

산출물:

```text
D:\Cos\taste_model\v12_results\v12_experiment_report.md
D:\Cos\taste_model\v12_results\v12_candidate_summary.csv
D:\Cos\taste_model\v12_results\v12_candidate_summary.json
D:\Cos\taste_model\v12_results\v12_oof_predictions.csv
D:\Cos\taste_model\v12_results\v12_confusion_matrices.json
```

---

## 13. 보고서에 반드시 포함할 내용

V12 보고서는 다음 구조로 작성한다.

```text
# V12 Specialist Ensemble Experiment Report

## 1. Data / label 기준
## 2. 사용한 base model predictions
## 3. Feature 목록
## 4. Baseline 성능
## 5. V12-A Linear 결과
## 6. V12-B Guard 결과
## 7. V12-C Residual 결과
## 8. V12-D Router 결과
## 9. V12-E Tree 결과
## 10. OOF 종합 비교
## 11. Class별 성능
## 12. V11 Mission 오답 rescue 분석
## 13. Severe case 분석
## 14. 최종 추천 모델
## 15. 다음 실험 제안
```

---

## 14. 예상되는 유망 방향

현재 분석 기준으로 가장 유망한 방향은 다음이다.

```text
1순위:
V12-C Residual Correction
anchor = V11 Mission
residual features = V8, V10-B, V6/V7, V11 Student, score_spread

2순위:
V12-B Linear + High Guard
V8/V10-B를 강하게 쓰되 V11 Mission으로 severe_fn guard

3순위:
V12-D Router
V11 Mission이 틀릴 가능성이 큰 disagreement case에서 V8/V6 rescue 활용
```

---

## 15. V12에 반드시 넣을 feature

필수 feature:

```text
v11_mission_score
v11_mission_pred

v8_score
v8_pred

v10b_score
v10b_pred
v10b_gate
v10b_mine_prob

v11_student_score
v11_student_pred

score_spread
high_vote_count
mine_vote_count
```

권장 feature:

```text
v6_score
v6_pred
v7_score
v7_pred

v8_minus_mission
v10b_minus_mission
v6_minus_mission

score_max
score_min
score_mean
score_median
```

---

## 16. 주의사항

1. V11 Mission을 무조건 버리지 말 것.
   현재 가장 안전한 anchor다.

2. V8은 단독 severe_fn 때문에 위험하지만 feature로는 매우 중요하다.
   QWK, Spearman, MAE, rescue가 강하다.

3. V10-B는 mine specialist로 사용한다.
   특히 `mine_prob`, `pred==1`, `gate`는 반드시 feature로 넣는다.

4. V6는 rescue가 높지만 severe_fn이 커서 단독 사용 금지다.
   feature 또는 router 신호로만 사용한다.

5. folder/internal split aggregate 결과는 user rating 결과와 섞지 않는다.

6. 최종 모델은 OOF 기준으로만 판단한다.

7. severe_fn이 1건이라도 나오면 guard를 추가하거나 후보에서 제외한다.

---

## 17. 최종 목표

V12의 최종 목표는 다음이다.

```text
V11 Mission의 severe_fn=0 안정성을 유지하면서,
V8/V10-B/V6/V11 Student의 specialist 신호를 활용해
AScore, 95_score, QWK, Spearman 중 최소 하나 이상을 안정적으로 끌어올리는 것.
```

최종 채택 모델명 예시:

```text
v12_residual_specialist_ensemble
```
