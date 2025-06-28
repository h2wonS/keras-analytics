# Tabular 데이터셋 분석 - BrainStroke
import kagglehub

# Download latest version
path = kagglehub.dataset_download("zzettrkalpakbal/full-filled-brain-stroke-dataset")

print("Path to dataset files:", path)

# 기본 라이브러리 및 그래프 관련 임포트
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 케라스 관련 라이브러리 임포트
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Training 데이터 불러오기
df = pd.read_csv(f'{path}/full_data.csv')

df

df.info()
# 처음에 그냥 df를 쳐서 실제 데이터를 일부 확인하고, info()를 통해 값과의 데이터타입 매칭 여부 확인
# 결측치 및 컬럼 용도 확인

# 데이터 전처리 >> 라벨 인코딩
df['gender'] = np.where(df['gender']=='Male', 1, 0)
df['ever_married'] = np.where(df['ever_married']=='Yes', 1, 0)

# 결측치 확인
df.isnull().sum()

"""현재는 결측치 있는 컬럼이 전혀 없음

object 타입의 변수는 일반적으로 문자열(범주형)을 의미함

-> 원-핫 인코딩은 범주형 데이터를 0/1로 이루어진 dummy 변수로 변환하는 방식

-> 수치형 변수는 원핫인코딩이 필요하지 않음

=> pandas의 get_dummies는 지정된 범주형변수를 더미변수(0/1)으로 치환해줌
"""

# 데이터 전처리 >> 원 핫 인코딩 (object인 데이터만 분류해서)
obj_cols = df.select_dtypes('object').columns

df = pd.get_dummies(data=df, columns=obj_cols, drop_first=True)
# 이때, drop_first=True로 설정하면 각 범주형 변수에서 첫번째 범주에 해당하는 더미 컬럼을 제거
# >> 다중공선성(multicollinearity) 방지하기 위함
# N개의 범주가 있으면 N개의 더미 컬럼이 만들어지는데 사실 N-1개만 있어도 나머지는 자동으로 알 수 있기 때문에 불필요한 정보중복을 방지

df.info()

# 데이터의 feature(설명변수, 독립변수, x)와 label(종속변수, y)을 분류
features = df.drop('stroke', axis=1) # stroke 컬럼을 제외한 나머지 모든 컬럼 선택
labels = df['stroke'] # stroke 컬럼이 결국 정답 컬럼인 것

labels.value_counts()

# Train/Valid 데이터셋 분할
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
# 전체 데이터의 20%를 valid set으로 사용하겠다 (일반적으로 20-30% 사용)
# random_state는 관습적으로 42를 씀 (랜덤 시드 고정 안하면 결과 재현 불가능)
# labels(종속변수, 정답값)의 비율을 train/valid 세트에 동일하게 유지
# (stratify를 쓰지 않으면 valid 세트에 0만 과도하게 몰릴 수도 있음. 위에서 value_counts를 볼 때 불균형 데이터이기 때문에 stratify를 꼭 써줘야 함)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape

"""ML 분류 모델
1. LogisticRegression (연산속도 빠름)
2. RandomForestClassifier (1번이 성능 부족하면 전)
3. DecisionTreeClassifier

ML 회귀 모델
1. DecisionTreeRegressor
2. RandomForestRegressor
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression 훈련/검증
lgr = LogisticRegression()
lgr.fit(x_train, y_train)
lgr.score(x_valid, y_valid)

# SGD Classifier 훈련/검증
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc.score(x_valid, y_valid)

# DecisionTree Classifier 훈련/검증
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc.score(x_valid, y_valid)

# RandomForestClassifier 훈련/검증
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.score(x_valid, y_valid)

# 딥러닝 CNN 분류 모델 정의
model = Sequential() # Keras 기본 신경망 모델 구조 (입력층-hiddenLayers-출력층)
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[-1], ))) # Fully Connected Layer (128개 뉴런, 비선형성을 위한 relu, x_train.shape[-1]은 즉 feature 개수만큼의 입력 차원이라는 뜻 )
model.add(Dense(64, activation='relu')) # 64개 뉴런 FC Layer (점진적으로 뉴런 수 줄여가면서 정보의 추상화/압축 유도)
model.add(Dropout(0.3)) # 학습 시 랜덤하게 30%의 뉴런을 무시하겠다는 의미... 0.2-0.5 사이에서 값 조정 가능 (overfitting 방지, local minima에 수렴되지 않도록)
model.add(Dense(32, activation='relu')) # 32개 뉴런 FC Layer
model.add(BatchNormalization()) # relu전에 놓아도 됨. 각 배치마다 입력 정규화 (~N(0,1)으로 맞춤에 따라 학습 안정도 향상, internal covariate shift 감소, dropout이랑 같이 쓰면 더 강력한 regularization 효과)
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) # (출력값이 2개) 이진분류, 각 클래스에 속할 확률을 예측하기 위함 (softmax는 두 클래스에 대한 확률값을 출력)
# model.add(Dense(1, activation='sigmoid')) ## 사실 softmax - categorical_crossentropy는 다중분류에 더 적합하며, 이진분류에서 sigmoid는 0-1사이의 확률값을 직접 반환함, binary_crossentropy와 잘 맞음.

# 모델 compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), # learning_rate를 0.001으로 설정 (보통 Adam 옵티마이저 잘 씀)
    loss='sparse_categorical_crossentropy', # 모델예측값과 실제값 차이 계산하는 손실함수로써 sparse_catrgorical_crossentropy 사용. 다중클래스에서 정답이 정수일때 사용 (만약 완전한 이진분류라면 categorical_crossentropy)
    metrics=['accuracy']
)

# 모델 훈련
# Tabular는 es, mc, history에 숫자를 붙이지 않고 사용했습니다.
# 같은 코드 안에서 3개의 모델을 만들어서 비슷한 변수명에 숫자나, text, img를 붙였습니다.

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=[es, mc],
    verbose=1
)

# Test 데이터 불러오기
df_test = pd.read_csv(f'{path}/full_filled_stroke_data (1).csv')

df_test

"""Test 데이터셋도 동일하게 전처리를 수행

Train 데이터셋에서는 중복/결측치 제거하여 모델 분류 성능 자체를 올려야 함

단, Test 데이터셋의 개수를 줄이는 것은지양해야함 (중복 삭제, 결측 제거 금지)
"""

# 데이터 전처리 >> 라벨 인코딩
df_test['gender'] = np.where(df['gender']=='Male', 1, 0)
df_test['ever_married'] = np.where(df['ever_married']=='Yes', 1, 0)
df_test.isnull().sum() # 결측치 확인

# 데이터 전처리 >> 원-핫 인코딩
obj_cols_test = df_test.select_dtypes('object').columns
df_test = pd.get_dummies(data=df_test, columns=obj_cols_test, drop_first=True)

# 데이터 features, labels 분류
x_test = df_test.drop('stroke', axis=1)
y_test = df_test['stroke']

lgr_y_pred = lgr.predict(x_test)
sgdc_y_pred = sgdc.predict(x_test)
dtc_y_pred = dtc.predict(x_test)
rfc_y_pred = rfc.predict(x_test)
model_y_pred = model.predict(x_test) # 테스트데이터 기반 예측 결과 얻기 (이때 반환값은 각 샘플에 대한 클래스별 점수/확률)
model_y_pred = np.argmax(model_y_pred, axis=1) # 각 샘플별로 가장 확률이 높은 클래스의 인덱스(정수)를 반환하여 실제 클래스번호가 결과값

# 모델중에 가장 성능이 좋아보이는 DecisionTreeClassifier의 예측값을 테스트 데이터셋 csv에 추가
df_test['pred'] = rfc_y_pred
df_test

# Tabular 모델 테스트 데이터셋.csv 저장
df_test.to_csv('01012345678_1.csv', index=False, encoding='utf-8') # Tabular 데이터는 '연락처_1.csv' 으로 저장
df_confirm = pd.read_csv('01012345678_1.csv') # 데이터 저장 확인
df_confirm

# Tabular 머신러닝 모델 저장을 위한 joblib 라이브러리 임포트
import joblib
joblib.dump(rfc, '01012345678_1.joblib') # Tabular 데이터는 '연락처_1.joblib' 으로 저장

# Tabular 모델 저장 확인
tab_load_model = joblib.load('01012345678_1.joblib')
