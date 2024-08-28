# ============================================================================================================================
# 로지스틱 회귀 모델을 위한 하이퍼파라미터 그리드 설정
param_grid_lr = {
    "C": [0.1, 1, 10],  # 'C': 정규화 강도를 조절
    "penalty": ["l1", "l2"],  # 'penalty': L1 정규화와 L2 정규화
    "solver": [
        "sag",
        "saga",
    ],  # 'solver': 최적화 알고리즘 선택. 'liblinear'와 'saga' 두 가지 사용
    "max_iter": [80, 90],
}

# 그리드 서치 객체 생성
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5)


# ============================================================================================================================
# 랜덤 포레스트 모델을 위한 하이퍼파라미터 그리드 설정
param_grid_rf = {
    "n_estimators": [100, 200, 300],  # 'n_estimators': 사용할 트리의 수
    "max_depth": [10, 20, 30],  # 'max_depth': 트리의 최대 깊이
    "min_samples_split": [
        2,
        5,
        10,
    ],  # 'min_samples_split': 내부 노드를 분할하기 위한 최소 샘플 수
    "max_features": [
        "auto",
        "sqrt",
        "log2",
    ],  # 'max_features': 분할 시 고려할 최대 피처 수
}

# 그리드 서치 객체 생성
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)


# ============================================================================================================================
# XGBoost 모델을 위한 하이퍼파라미터 그리드 설정
param_grid_xgb = {
    "n_estimators": [100, 200, 300],  # 'n_estimators': 부스팅 단계 수
    "max_depth": [3, 6, 9],  # 'max_depth': 각 트리의 최대 깊이
    "learning_rate": [0.01, 0.1, 0.2],  # 'learning_rate': 학습률
    "subsample": [0.6, 0.8, 1.0],  # 'subsample': 각 트리에서 사용할 데이터 샘플의 비율
}

# 그리드 서치 객체 생성
grid_search_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5)


# ============================================================================================================================
# LightGBM 모델을 위한 하이퍼파라미터 그리드 설정
param_grid_lgb = {
    "n_estimators": [100, 200, 300],  # 'n_estimators': 부스팅 단계 수
    "max_depth": [10, 20, -1],  # 'max_depth': 각 트리의 최대 깊이. -1은 깊이 제한 없음
    "learning_rate": [0.01, 0.1, 0.2],  # 'learning_rate': 학습률
    "num_leaves": [
        30,
        60,
        120,
    ],  # 'num_leaves': 하나의 트리에서 사용할 수 있는 최대 리프 수
}

# 그리드 서치 객체 생성
grid_search_lgb = GridSearchCV(LGBMClassifier(), param_grid_lgb, cv=5)


# ============================================================================================================================
# CatBoost 모델을 위한 하이퍼파라미터 그리드 설정
param_grid_cb = {
    "iterations": [100, 200, 300],  # 'iterations': 부스팅 반복 수
    "depth": [4, 6, 8],  # 'depth': 각 트리의 깊이
    "learning_rate": [0.01, 0.1, 0.2],  # 'learning_rate': 학습률
    "l2_leaf_reg": [1, 3, 5],  # 'l2_leaf_reg': L2 정규화 강도
}

# 그리드 서치 객체 생성
grid_search_cb = GridSearchCV(CatBoostClassifier(), param_grid_cb, cv=5)


# ============================================================================================================================
# MLPClassifier에 대한 하이퍼파라미터 그리드 설정(데이터 셋 사이즈를 고려한 설정)
param_grid_mlp = {
    "hidden_layer_sizes": [(100,), (100, 50), (100, 100, 50)],  # 은닉층의 구조 설정
    "activation": ["relu", "tanh"],  # 활성화 함수 선택
    "solver": ["adam", "lbfgs"],  # 최적화 알고리즘 선택
    "alpha": [0.0001, 0.001, 0.01],  # L2 정규화 강도
    "learning_rate": ["constant", "adaptive"],  # 학습률 스케줄 선택
}

# MLPClassifier에 대한 하이퍼파라미터 그리드 설정(돌리는 시간까지 고려한 설정)
param_grid_mlp = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)],  # 은닉층의 구조를 단순하게 설정
    "activation": [
        "relu"
    ],  # 활성화 함수는 가장 일반적으로 좋은 성능을 보이는 'relu'로 고정
    "solver": [
        "adam"
    ],  # 최적화 알고리즘은 대규모 데이터셋에서 효율적인 'adam'으로 고정
    "alpha": [0.0001, 0.001],  # L2 정규화 강도 범위를 줄임
}

# MLPClassifier 모델과 그리드 서치 객체 생성
mlp = MLPClassifier(random_state=1)
grid_search_mlp = GridSearchCV(
    mlp, param_grid_mlp, cv=5, n_jobs=-1
)  # 교차 검증을 3-폴드로 설정하고, 병렬 처리 사용
