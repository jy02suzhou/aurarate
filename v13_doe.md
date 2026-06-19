# V13 실험계획서: Narrow Manifold / Bottleneck 모델 실험

## 1. 실험 목적

V13의 목적은 기존 2304차원 입력 벡터가 실제로는 훨씬 작은 취향 매니폴드 위에 있을 가능성을 검증하는 것이다.

현재 입력은 다음 구조다.

```text
org_emb: 1152-dim
seg_emb: 1152-dim
concat input: 2304-dim
```

기존 V8/V9/V10 계열은 2304차원 입력을 비교적 큰 MLP로 처리했다. 하지만 user-rating 데이터는 약 5~6천 장 수준이므로, 첫 은닉층을 크게 두면 노이즈까지 외울 가능성이 있다.

V13에서는 첫 단계부터 강한 bottleneck을 걸어 다음 가설을 검증한다.

```text
가설:
사용자 취향 판단에 필요한 실제 유효 차원은 2304보다 훨씬 작다.
따라서 64~256차원 bottleneck 모델이 더 잘 일반화할 수 있다.
```

---

## 2. 절대 주의사항

V13은 실험적이므로 기존 중요 DB를 직접 수정하지 않는다.

### 2.1 절대 수정 금지

다음 DB는 읽기 전용으로만 사용한다.

```text
D:\Cos\taste_model\aurarate.db
D:\Cos\taste_model\test_result.db
D:\Cos\taste_model\heatmap.db
```

금지 작업:

```text
UPDATE
DELETE
ALTER TABLE
DROP TABLE
VACUUM
REINDEX
CREATE TABLE inside source DB
```

특히 `aurarate.db`에는 절대 실험 테이블을 만들지 않는다.

---

## 3. V13 전용 워크벤치

모든 실험은 새 폴더와 새 DB에서만 수행한다.

```text
D:\Cos\taste_model\v13_workbench\
```

생성할 파일 구조:

```text
D:\Cos\taste_model\v13_workbench\
├─ v13_workbench.db
├─ v13_dataset.npz
├─ v13_manifest.json
├─ v13_results\
│  ├─ v13_experiment_report.md
│  ├─ v13_candidate_summary.csv
│  ├─ v13_candidate_summary.json
│  ├─ v13_oof_predictions.csv
│  ├─ v13_holdout_predictions.csv
│  ├─ v13_confusion_matrices.json
│  └─ checkpoints\
└─ scripts\
   ├─ build_v13_dataset.py
   ├─ run_v13_bottleneck_experiments.py
   ├─ evaluate_v13.py
   └─ export_v13_candidate.py
```

---

## 4. Source DB 읽기 방식

가능하면 SQLite read-only URI로 연다.

```python
sqlite3.connect(
    "file:D:/Cos/taste_model/aurarate.db?mode=ro",
    uri=True
)
```

읽기 전용으로 다음만 가져온다.

```text
labels.user_rating
embedding_cache.org_emb
embedding_cache.seg_emb
predictions_v6
prediction_v10b
predictions_v11_view
predictions_v11_mission_view
```

V13 학습용 데이터는 `v13_dataset.npz`와 `v13_workbench.db`에 복사해서 사용한다.

---

## 5. 학습 데이터

라벨 기준:

```text
aurarate.db / labels.user_rating
```

최종 학습 샘플은 다음 조건을 만족해야 한다.

```text
label 존재
org_emb 존재
seg_emb 존재
org_emb dim = 1152
seg_emb dim = 1152
label in {1,2,3,4,5}
```

저장할 배열:

```text
keys
filepaths
y
org_emb: [N, 1152], float32
seg_emb: [N, 1152], float32
concat_emb: [N, 2304], float32
```

추가로 V12 비교용 feature도 가능하면 저장한다.

```text
v8_score
v10b_score
v10b_gate
v10b_mine_prob
v11_mission_score
v12_score, 있으면
```

---

## 6. Baseline

V13 결과를 평가하기 전 같은 데이터 split에서 다음 baseline을 계산한다.

```text
Baseline A: V8 deployed
Baseline B: V10-B deployed
Baseline C: V11 Mission deployed
Baseline D: V12 Selected Ridge
Baseline E: simple round/threshold baseline
```

V13은 반드시 V12와 비교해야 한다.

V12가 현재 서버 적용 후보이므로, V13은 다음 중 하나를 만족해야 의미가 있다.

```text
1. V12보다 성능이 높음
2. V12보다 낮지만 단일 embedding model로 충분히 단순함
3. V12 ensemble에 넣을 새로운 specialist feature로 가치 있음
```

---

## 7. V13 실험군

## 7.1 V13-PCA: Manifold Probe

먼저 PCA/SVD로 실제 유효 차원을 확인한다.

입력:

```text
concat_emb = concat(org_emb, seg_emb)
```

실험 차원:

```text
PCA dim:
16, 32, 64, 128, 256, 512
```

각 차원에서 다음 모델을 학습한다.

```text
Ridge regression
Logistic ordinal
Small MLP: dim → 32 → 5
```

기록할 것:

```text
explained_variance_ratio cumulative
QWK
Spearman
MAE
mine_recall
severe_fn
high_recall_safe
```

목적:

```text
PCA 64~128만으로 V8/V10에 근접하면 취향 매니폴드가 작다는 강한 증거다.
```

---

## 7.2 V13-A: Direct Narrow Bottleneck MLP

2304차원 입력을 첫 층에서 바로 좁힌다.

후보 구조:

```text
A1: 2304 → 32 → 5
A2: 2304 → 64 → 5
A3: 2304 → 128 → 5
A4: 2304 → 256 → 5
```

2-layer 후보:

```text
A5: 2304 → 64 → 32 → 5
A6: 2304 → 128 → 64 → 5
A7: 2304 → 256 → 64 → 5
A8: 2304 → 256 → 128 → 5
```

Loss:

```text
ordinal BCE
```

선택 auxiliary loss:

```text
mine BCE: label == 1
high BCE: label >= 4
```

---

## 7.3 V13-B: Branch Bottleneck

org와 seg를 각각 따로 압축한 뒤 합친다.

추천 핵심 구조:

```text
org_emb 1152 → 128
seg_emb 1152 → 128
concat 256
→ 64
→ ordinal logits 5
```

실험 후보:

```text
B1: org 32 + seg 32 → 32 → 5
B2: org 64 + seg 64 → 32 → 5
B3: org 64 + seg 64 → 64 → 5
B4: org 128 + seg 128 → 64 → 5
B5: org 128 + seg 128 → 128 → 5
B6: org 256 + seg 256 → 128 → 5
```

이 실험이 가장 중요하다.

이유:

```text
concat 2304를 바로 줄이는 것보다 org/seg 역할을 유지하면서 압축할 수 있다.
```

---

## 7.4 V13-C: Bottleneck + Gate

V10-B처럼 gate를 두되, gate도 작게 만든다.

구조:

```text
org_proj = MLP(org_emb) → d
seg_proj = MLP(seg_emb) → d

gate_input = concat(org_proj, seg_proj, abs(org_proj - seg_proj), org_proj * seg_proj)
g = 0.6 + 0.1 * tanh(gate_net(gate_input))

fused = concat(g * org_proj, (1-g) * seg_proj)
head → ordinal logits
```

후보:

```text
d = 32, 64, 128
gate_hidden = 16, 32, 64
```

목적:

```text
V10-B의 dynamic gate 장점을 유지하면서 전체 모델 크기를 크게 줄인다.
```

---

## 7.5 V13-D: V12 Distillation Student

V12 selected Ridge를 teacher로 보고, embedding만으로 V12의 점수를 따라가는 작은 student를 만든다.

입력:

```text
org_emb + seg_emb
```

Target:

```text
user_rating label
V12 score
V12 pred
```

Loss:

```text
loss = ordinal_BCE(label)
     + λ_score * MSE(student_score, v12_score)
     + λ_mine * BCE(mine_label)
     + λ_high * BCE(high_label)
```

후보 λ:

```text
λ_score = 0.1, 0.25, 0.5, 1.0
λ_mine = 0.05
λ_high = 0.05
```

목적:

```text
V12는 여러 모델 점수를 feature로 쓰기 때문에 서버 내부가 복잡하다.
V13-D는 V12의 판단을 단일 embedding model로 압축할 수 있는지 확인한다.
```

이 실험은 성공하면 서버 단순화에 매우 중요하다.

---

## 7.6 V13-E: Specialist Bottleneck Models

특정 역할만 잘하는 작은 모델을 만든다.

### Mine Specialist

```text
target = label == 1
input = org_emb + seg_emb
output = mine_prob
```

목표:

```text
mine_recall 극대화
severe_fn 악화 없는지 확인
```

### High Protection Specialist

```text
target = label >= 4
```

목표:

```text
실제 4~5점을 1~2점으로 떨어뜨리는 위험 방지
high_recall_safe 상승
```

### Five Specialist

```text
target = label == 5
```

목표:

```text
V12의 5점 recall 하락을 보완
```

이 specialist들은 단독 최종 모델이 아니라 V12/V13 ensemble feature로 사용할 수 있다.

---

## 8. 학습 방식

공통 split:

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=20260620)
```

추가 holdout:

```text
holdout_size = 0.2
holdout_seed = 20260621
stratified by user rating
```

원칙:

```text
OOF 성능으로 1차 판단
holdout 성능으로 최종 확인
```

절대 금지:

```text
holdout label을 보고 threshold 튜닝
전체 데이터를 보고 threshold 튜닝 후 같은 데이터에 성능 보고
```

---

## 9. Threshold Tuning

모든 ordinal model은 연속 score 또는 ordinal probability를 만든 뒤 threshold로 1~5 class를 만든다.

기본 threshold:

```text
[1.5, 2.5, 3.5, 4.5]
```

튜닝 방식:

```text
fold train 내부에서만 random search
validation fold에는 고정 threshold 적용
```

최적화 우선순위:

```text
1. severe_fn = 0
2. 95_score 최대화
3. QWK 최대화
4. mine_recall 유지
5. MAE 최소화
```

---

## 10. 평가 지표

모든 후보에 대해 다음 지표를 기록한다.

```text
QWK
Spearman
MAE
accuracy
adjacent_accuracy

mine_recall
severe_fn
severe_fn_count
severe_fp
severe_fp_count

high_recall_safe
high_recall_strict

class_1_precision / recall / f1
class_2_precision / recall / f1
class_3_precision / recall / f1
class_4_precision / recall / f1
class_5_precision / recall / f1

95_score
AScore

rescue_count_vs_v12
rescue_count_vs_v11_mission
unique_correct_vs_v12
error_overlap_vs_v12
```

---

## 11. 모델 선택 기준

### 11.1 최종 모델 후보 조건

```text
OOF severe_fn_count = 0
holdout severe_fn_count = 0
```

### 11.2 V12 대체 후보 조건

V13이 V12를 대체하려면 다음 중 하나를 만족해야 한다.

```text
1. 95_score >= V12 selected - 1.0
   and severe_fn = 0
   and 모델 구조가 훨씬 단순함

2. QWK >= V12 selected
   and severe_fn = 0

3. Spearman >= V12 selected
   and MAE <= V12 selected
   and severe_fn = 0
```

### 11.3 V12 feature 추가 후보 조건

V13이 V12보다 낮아도 다음이면 가치가 있다.

```text
mine_recall 특출
class_5_recall 특출
V12 오답 rescue가 많음
V12와 error_overlap이 낮음
high_recall_safe가 높음
```

---

## 12. 저장 DB 스키마

V13 워크벤치 DB:

```text
D:\Cos\taste_model\v13_workbench\v13_workbench.db
```

테이블:

```sql
CREATE TABLE IF NOT EXISTS v13_dataset_index (
    key TEXT PRIMARY KEY,
    filepath TEXT,
    label INTEGER,
    source_db TEXT,
    has_org_emb INTEGER,
    has_seg_emb INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS v13_runs (
    run_id TEXT PRIMARY KEY,
    model_family TEXT,
    model_name TEXT,
    config_json TEXT,
    seed INTEGER,
    fold_count INTEGER,
    status TEXT,
    created_at TEXT,
    completed_at TEXT,
    checkpoint_path TEXT,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS v13_fold_metrics (
    run_id TEXT,
    fold INTEGER,
    qwk REAL,
    spearman REAL,
    mae REAL,
    mine_recall REAL,
    severe_fn REAL,
    severe_fn_count INTEGER,
    severe_fp REAL,
    severe_fp_count INTEGER,
    high_recall_safe REAL,
    score95 REAL,
    thresholds_json TEXT,
    PRIMARY KEY (run_id, fold)
);

CREATE TABLE IF NOT EXISTS v13_oof_predictions (
    run_id TEXT,
    key TEXT,
    filepath TEXT,
    y_true INTEGER,
    score REAL,
    y_pred INTEGER,
    fold INTEGER,
    prob_1 REAL,
    prob_2 REAL,
    prob_3 REAL,
    prob_4 REAL,
    prob_5 REAL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS v13_candidate_summary (
    run_id TEXT PRIMARY KEY,
    model_family TEXT,
    model_name TEXT,
    qwk REAL,
    spearman REAL,
    mae REAL,
    mine_recall REAL,
    severe_fn REAL,
    severe_fn_count INTEGER,
    severe_fp REAL,
    severe_fp_count INTEGER,
    high_recall_safe REAL,
    class_5_recall REAL,
    score95 REAL,
    rescue_count_vs_v12 INTEGER,
    error_overlap_vs_v12 REAL,
    usable_as_final INTEGER,
    usable_as_feature INTEGER,
    warning TEXT
);
```

---

## 13. 산출물

필수 산출물:

```text
D:\Cos\taste_model\v13_workbench\v13_results\v13_experiment_report.md
D:\Cos\taste_model\v13_workbench\v13_results\v13_candidate_summary.csv
D:\Cos\taste_model\v13_workbench\v13_results\v13_candidate_summary.json
D:\Cos\taste_model\v13_workbench\v13_results\v13_oof_predictions.csv
D:\Cos\taste_model\v13_workbench\v13_results\v13_holdout_predictions.csv
D:\Cos\taste_model\v13_workbench\v13_results\v13_confusion_matrices.json
```

후보 모델이 있으면 export:

```text
D:\Cos\taste_model\v13_workbench\v13_results\v13_selected_model.json
D:\Cos\taste_model\v13_workbench\v13_results\checkpoints\v13_selected.pt
```

---

## 14. 보고서 구성

보고서는 다음 구조로 작성한다.

```text
# V13 Narrow Manifold Experiment Report

## 1. 실험 목적
## 2. Source DB read-only 확인
## 3. Dataset 구성
## 4. PCA manifold probe 결과
## 5. V13-A Direct Bottleneck 결과
## 6. V13-B Branch Bottleneck 결과
## 7. V13-C Bottleneck Gate 결과
## 8. V13-D V12 Distillation 결과
## 9. V13-E Specialist 결과
## 10. V12/V11/V8 baseline 비교
## 11. OOF 종합 비교
## 12. Holdout 검증
## 13. Severe case 분석
## 14. V12 오답 rescue 분석
## 15. 최종 후보
## 16. 서버 적용 가능성
## 17. 다음 실험 제안
```

---

## 15. 우선순위

실험 순서는 다음으로 진행한다.

```text
1. build_v13_dataset.py 작성
2. PCA 16/32/64/128/256/512 probe
3. V13-B Branch Bottleneck 64/128 중심 실험
4. V13-A Direct Bottleneck 비교
5. V13-D V12 Distillation
6. V13-E Specialist mine/high/five
7. 가장 좋은 후보만 holdout 검증
```

가장 먼저 볼 핵심 후보:

```text
V13-B4:
org 128 + seg 128 → 64 → ordinal

V13-D:
V12 distillation + branch bottleneck
```

---

## 16. 성공 기준

V13 실험이 성공했다고 볼 수 있는 경우:

```text
Case A:
V13 단일 모델이 V12에 근접하면서 구조가 훨씬 단순하다.

Case B:
V13이 V12보다 종합 성능은 낮지만 V12 오답을 많이 구제한다.

Case C:
PCA 64~128만으로 강한 성능이 나와 작은 매니폴드 가설이 확인된다.

Case D:
V13-D가 V12의 판단을 단일 embedding model로 압축하는 데 성공한다.
```

---

## 17. 최종 주의

V13은 매우 실험적인 단계다.

```text
서버 적용 금지
aurarate.db 수정 금지
test_result.db 수정 금지
heatmap.db 수정 금지
결과는 v13_workbench 안에만 저장
```

최종 채택 전에는 반드시 다음을 통과해야 한다.

```text
OOF severe_fn_count = 0
holdout severe_fn_count = 0
V12 대비 장점 명확
5점 recall 하락 여부 확인
severe_fp case 이미지 확인
```

최종 한 줄 목표:

```text
V13은 2304차원 embedding을 작은 취향 매니폴드로 압축할 수 있는지 검증하는 독립 워크벤치 실험이다.
```
