# Text 데이터 분석 - 영어

import kagglehub

# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

print("Path to dataset files:", path)

# Text train 데이터.csv 읽기
# column이 없는 데이터셋이라 header를 None으로 하고 임의로 라벨을 지정해서 사용

text_df = pd.read_csv(f'{path}/twitter_training.csv', header=None)
text_df

# 데이터 전처리 >> 감정 분류에 불필요한 데이터셋 제거
text_df = text_df.drop([0, 1], axis=1) # 0,1번 컬럼 제거

# 데이터 전처리 >> column 이름 지정
text_df = text_df.rename(columns={2: 'label', 3: 'text'})
text_df

# 데이터 전처리 >> 결측치 확인
text_df.isnull().sum()

# 데이터 전처리 >> 결측치 제거
text_df = text_df.dropna()
text_df.isnull().sum()

"""Text 데이터 전처리 - 불용어 여부 확인

'[^a-zA-Z ]' 으로 적으면 소문자, 대문자, 공백 제외 다른 문자나 숫자가 있는지 확인


'[^가-힣 ]' 으로 적으면 한글, 공백 제외 다른 문자나 숫자가 있는지 확인


'[^a-zA-Z가-힣 ]' 으로 적으면 영문, 한글, 공백 제외


[^a-zA-Z가-힣] <- 공백이 없이 만들면 이후 작업에서 띄어쓰기에 필요한 공백을 모두 제거해버려서 문제가 되니 주의

"""

# 데이터 전처리 >> 텍스트

# 영문 데이터셋이므로 대소문자와 공백 제외하고 삭제
# text_df[text_df['text'].str.contains('[^a-zA-Z ]')].head(3) ## 먼저 확인
text_df['text'] = text_df['text'].str.replace('[^a-zA-Z ]', '', regex=True)

# 문자열 앞뒤 공백 제거
text_df['text'] = text_df['text'].str.strip()

# 모두 소문자로 변경
text_df['text'] = text_df['text'].str.lower()

# 데이터 전처리 >> 중복여부 확인
text_df['text'].duplicated().sum()

# 데이터 전처리 >> 중복 제거 및 확인
text_df.drop_duplicates(subset='text', inplace=True)
text_df['text'].duplicated().sum()

text_df.info()

# 라벨 종류 확인
text_df['label'].value_counts()

# label이 모두 문자열이므로 라벨 인코딩 필요
text_df['label'], uniques = pd.factorize(text_df['label'])  # 순서대로 알아서 라벨링됨
# text_class_to_label = {'Negative':0, 'Positive':1, 'Neutral':2, 'Irrelevant':3} # 라벨 순서가 중요하다면 직접 지정 필요
# text_df['label]= text_df['label'].map(text_class_to_label)
text_df['label'].value_counts()

# feature, label 분류
text_features = text_df['text']
text_labels = text_df['label']

# Text train, valid 데이터셋 나누기
text_x_train, text_x_valid, text_y_train, text_y_valid = train_test_split(text_features, text_labels, test_size=0.2, random_state=42, stratify=text_labels)
text_x_train.shape, text_x_valid.shape, text_y_train.shape, text_y_valid.shape

"""TF-IDF ML모델 기반 분류

- 사용방법이 비교적 간단하고 전처리도 거의 없어서 먼저 시도

- 텍스트가 포함된 벡터 특성에 따라 각 단어의 가중치 계산하고 이를 기반으로 텍스트 분류/회귀모델의 입력으로써 사용
"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() # TF*IDF 방식으로 텍스트 데이터를 숫자 벡터로 변환
text_x_train_v = tfidf.fit_transform(text_x_train) # 학습데이터에 대해 단어 사전 만들고(fit) TF-IDF 벡터 변환하여 SparseMatrix(희소행렬)에 TF-IDF값 채워진 형태로 반환
text_x_valid_v = tfidf.transform(text_x_valid) # 학습된 단어 사전 기반으로 TF-IDF 벡터 변환

# Text LogisticRegression 모델 훈련/검증
text_lgr = LogisticRegression()
text_lgr.fit(text_x_train_v, text_y_train) # 학습벡터와 정답으로 학습
text_lgr.score(text_x_valid_v, text_y_valid) # 검증벡터와 정답으로 검증

# Text SGDClassifer 모델 성능 확인
text_sgdc = SGDClassifier()
text_sgdc.fit(text_x_train_v, text_y_train)
text_sgdc.score(text_x_valid_v, text_y_valid)

# Text DecisionTreeClassifier 모델 성능 확인
text_dtc = DecisionTreeClassifier()
text_dtc.fit(text_x_train_v, text_y_train)
text_dtc.score(text_x_valid_v, text_y_valid)

# Text RandomForestClassifier 모델 성능 확인 (굉장히 오래걸리네;;)
text_rfc = RandomForestClassifier()
text_rfc.fit(text_x_train_v, text_y_train)
text_rfc.score(text_x_valid_v, text_y_valid)

# LSTM 기반의 딥러닝 전처리를 위한 Tokenizer 라이브러리 임포트
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Text Tokenizer 작업
tokenizer = Tokenizer() # 텍스트 데이터를 단어(토큰)별 분리, 각 단어에 unique 정수 인덱스 부여
tokenizer.fit_on_texts(text_x_train) # 학습데이터에 등장하는 모든 단어 인덱스 사전 정의

print(tokenizer.word_index) # 학습 단어 확인
print(tokenizer.word_counts) # 학습 단어별 빈도수
max_words = len(tokenizer.word_index)
print(max_words) # 학습한 단어 개수 확인

# 각 문장을 단어 인덱스의 리스트(시퀀스)로 변환
text_x_train_seq = tokenizer.texts_to_sequences(text_x_train)
text_x_valid_seq = tokenizer.texts_to_sequences(text_x_valid)

# 시퀀스 작업 후 가장 긴 문장의 단어수 확인
print(max(len(i) for i in text_x_train_seq))

# 시퀀스 패딩: 각 문장의 길이를 동일하게 맞춤 (166단어) >> 입력 길이가 일정해야 하는 DL모델에서는 필수적인 전처리과정
# 짧은 문장은 0으로 채우고, 긴 문장은 앞부분만 추출(뒤가 잘림)
text_x_train_pad = pad_sequences(text_x_train_seq, maxlen=166)
text_x_valid_pad = pad_sequences(text_x_valid_seq, maxlen=166)

text_labels.value_counts()

# LSTM 모델 정의
text_model = Sequential()
text_model.add(Embedding(input_dim=max_words+1, output_dim=100, input_length=166)) # 단어 인덱스를 임베딩(밀집벡터)로 변환, max_words+1은 패딩용 0인덱스를 추가한 것, output_dim=100은 각 단어를 100차원 벡터로 변환한다는 의미, input_length는 입력시퀀스의 길이
text_model.add(Bidirectional(LSTM(16, return_sequences=True))) # LSTM을 양방향 적용하겠다는 의미 (문장 앞뒤 양방향 문맥 모두 반영하겠다), return_sequences=True: 각 시점의 출력을 모두 반환해서 다음 LSTM 레이어 입력으로~
text_model.add(Dropout(0.3))
text_model.add(Bidirectional(LSTM(16, return_sequences=False))) # return_sequences=False는 마지막 시점의 출력만 반환, 이는 다음 Dense 레이어 입력용임
text_model.add(Dropout(0.3))
text_model.add(Dense(64, activation='swish')) # FC Layer, swish는 최근 좋은 성능을 보이는 activation func.
text_model.add(BatchNormalization())
text_model.add(Dropout(0.3))
text_model.add(Dense(4, activation='softmax')) # 4개 클래스 분류, 확률 출력

# 모델 컴파일
text_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습

# 콜백 정의
# EarlyStopping: Vali_loss가 더이상 개선되지 않으면(감소하지 않으면) 학습 조기종료하여 overfitting 방지
# monitor: 어떤 것 기준으로 모니터링 할 것인지, patience: 연속 N번까지 확인, restore_best_weights=True: 가장 성능이 좋았던 시점의 모델 가중치로 복원, verbose=1: 중단시점 메시지 출력
es_text = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
# ModelCheckpoint: 학습 중 Vali_loss가 가장 낮았던 모델을 파일로 저장하여 나중에 로드
# best_text_model.keras라는 이름으로 저장됨, save_best_only=True: 성능 개선시에만 저장, verbose=1: 저장될 때 메시지출력
mc_text = ModelCheckpoint('best_text_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history_text = text_model.fit(
    text_x_train_pad, text_y_train, # 학습데이터(패딩된 시퀀스), 정답레이블
    epochs=5, # 최대 5번 전체데이터셋 학습 (es에 의해 금방 끝날 수 있음)
    batch_size=512, # 한번에 512개 샘플씩 모델 입력함에 따라 메모리 효율/학습속도 조절
    validation_data=(text_x_valid_pad, text_y_valid),
    callbacks=[es_text, mc_text],# 위에서 정의한 콜백 적용하기
    verbose=1 # 학습과정 출력
)

# Test 데이터셋 불러오기
text_df_t = pd.read_csv(f'{path}/twitter_validation.csv', header=None)
text_df_t

# 데이터 전처리 >> 불필요한 column 삭제, 컬럼명 수정, 결측치 확인
text_df_t = text_df_t.drop([0, 1], axis=1)
text_df_t = text_df_t.rename(columns={2:'label', 3:'text'})
text_df_t.isnull().sum()

# 데이터 전처리 >> 대소문자와 공백 제외한 불용어 여부 확인
text_df_t[text_df_t['text'].str.contains('[^a-zA-Z ]')].head(3)

"""중요) 테스트 데이터셋은 중복과 결측치는 지우지 않는다"""

# 불용어 제거
text_df_t['text'] = text_df_t['text'].str.replace('[^a-zA-Z ]', '', regex=True)
# 문자열 앞뒤 공백 제거
text_df_t['text'] = text_df_t['text'].str.strip()
# 소문자로 변경
text_df_t['text'] = text_df_t['text'].str.lower()
# 중복 여부, 갯수 확인
text_df_t['text'].duplicated().sum()

# 라벨 종류, 갯수 확인
text_df_t['label'].value_counts()

# 문자열 -> 라벨 인코딩 후 확인 (Train 데이터셋하고 동일하게)
text_df_t['label'], uniques = pd.factorize(text_df_t['label'])  # 순서대로 알아서 라벨링됨
# text_class_to_label = {'Negative':0, 'Positive':1, 'Neutral':2, 'Irrelevant':3} # 라벨 순서가 중요하다면 직접 지정 필요
# text_df_t['label]= text_df_t['label'].map(text_class_to_label)
text_df_t['label'].value_counts()

# feature, label 분류
text_x_test = text_df_t['text']
text_y_test = text_df_t['label']

# 데이터 전처리, Tfidf, Tokenizer
text_x_test_v = tfidf.transform(text_x_test)
text_x_test_seq = tokenizer.texts_to_sequences(text_x_test)
text_x_test_pad = pad_sequences(text_x_test_seq)

# 모델 예측
text_rfc_y_pred = text_rfc.predict(text_x_test_v)
text_model_y_pred = text_model.predict(text_x_test_pad)
text_model_y_pred = np.argmax(text_model_y_pred, axis=1)

# 예측한 값을 csv에 추가
text_df_t['pred'] = text_rfc_y_pred
text_df_t

# Text dataset 라벨 인코딩 했던 것 원상태로
text_label_to_class = {0:'Negative', 1:'Positive', 2:'Neutral', 3:'Irrelevant'}
text_df_t['label'] = text_df_t['label'].map(text_label_to_class)
text_df_t['pred'] = text_df_t['pred'].map(text_label_to_class)
text_df_t

# Text Test 데이터셋.csv 저장 및 불러와서 확인
text_df_t.to_csv('01012345678_2.csv', index=False, encoding='utf-8')
text_df_c = pd.read_csv('01012345678_2.csv')
text_df_c

# Text RandomForestClassifier 모델 저장
joblib.dump(text_rfc, '01012345678_2.joblib')

