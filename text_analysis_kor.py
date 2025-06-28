
# 그래프 한글 깨지는 문제 해결
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

"""이후 코랩 런타임 재시작"""

import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')

import numpy as np
import pandas as pd

# AI-HUB 감성 대화 말뭉치 활용하여 만든 데이터 읽어오기
final_data = pd.read_csv('https://github.com/ohgzone/file1/raw/main/aihub_coupus.csv' )

final_data.info()

# 데이터 전처리 >> 한글, 공백을 제외한 불용어 확인
final_data[final_data['문장'].str.contains('[^가-힣 ]')].values[:10]

# 데이터 전처리 >> 불용어 삭제
final_data['문장'] = final_data['문장'].str.replace('[^가-힣 ]','', regex=True)

# 확인
final_data['문장'][final_data['문장'].str.contains('[^가-힣 ]')].sum()

# 데이터 전처리 >> 문자열 앞뒤 공백 삭제
final_data['문장'] = final_data['문장'].str.strip()

# 데이터 전처리 >> 결측 확인
final_data['문장'].isnull().sum()

# 만약 결측치 있다면 (훈련일 때)
# df = df.dropna() # 단, 하나의 결측값이라도 있는 행은 모두 사라짐

# 데이터 전처리 >> 중복 확인
final_data['문장'].duplicated().sum()

# 데이터 전처리 >> 중복 제거, 확인
final_data.drop_duplicates(subset=['문장'], inplace=True)
final_data['문장'].duplicated().sum()

# label '감정' 분포 확인
final_data['감정'].value_counts()

# 라벨 인코딩 (문자열 감정라벨을 숫자로 인코)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
final_data['감정'] = encoder.fit_transform(final_data['감정'])
encoder.classes_

# Feature, Label 분리
features = final_data['문장'].values
labels = final_data['감정'].values
features.shape, labels.shape

print('이벤트 문자열 최대 길이 :{}'.format(max(len(l) for l in features)))
print('이벤트 문자열 평균 길이 :{}'.format(sum(map(len, features))/len(features)))

# Train/Test 데이터셋 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels , test_size=0.2, stratify=labels, random_state=41)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

"""TF-IDF & RandomForestClassifier"""

# TF-IDF 변환
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
x_train_v = tfidf.fit_transform(x_train) # 단어 사전 만들고 학습 벡터화
x_test_v = tfidf.transform(x_test) # 학습된 단어사전 기반으로 벡터화

# RandomForestClassifier 분류 >> 가장 잘 나오는듯
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier() # 모델 정의
rfc.fit(x_train_v, y_train) # 모델 학습
rfc.score(x_test_v, y_test)

predict = rfc.predict(x_test_v[:1])
predict, encoder.inverse_transform(predict)

"""keras Tokenizer & LSTM"""

# LSTM 기반의 딥러닝 전처리를 위한 Tokenizer 라이브러리 임포트
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 케라스 관련 라이브러리 임포트
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 문장에 대해 Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

max_words = len(tokenizer.index_word)
print(max_words) # 총 단어 갯수 확인

# 문장을 숫자로 나열
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

print(len(x_train_seq), len(x_test_seq))
maxlen = max(len(line) for line in x_train_seq)

# 모든 문장을 최대 문장 Seq 길이 38에 맞춘다.
x_train_pad =pad_sequences(x_train_seq, maxlen=maxlen)
x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen)

# LSTM 모델링
# Hyper parameters
max_words = max_words+1 #총 단어 갯수 + padding 0 번호
max_len = maxlen # 최대 문장 길이
embedding_dim = 32 # embedding 차원

# Model 구축
model = Sequential()

# 단어를 32차원으로 Vector 변경(Embedding)
model.add(Embedding(max_words, embedding_dim, input_length=max_len))

model.add(LSTM(16, return_sequences=True))
model.add(LSTM(16, return_sequences=False))
model.add(Flatten())
model.add(Dense(128, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(6, activation='softmax'))

# 모델 compile
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ["accuracy"])
# model.summary()

# 조기종료 콜백함수 정의(EarlyStopping)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# 체크포인트 저장(ModelCheckpoint)
checkpoint_path = "tmp_checkpoint.weights.h5"
cp = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

# 모델 학습(fit)
history = model.fit(x_train_pad, y_train, epochs=50, batch_size=512,
                      validation_split=0.2, verbose =1, callbacks=[es, cp])

# 모델 예측
y_pred_prob = model.predict(x_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)

print(y_pred)
