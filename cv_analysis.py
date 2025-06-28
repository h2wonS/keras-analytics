
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sauravagarwal/flower-classification")

print("Path to dataset files:", path)

# /kaggle/input/flower-classification/flowers/flowers/flower_photos

!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos
root_path = '/kaggle/input/flower-classification/flowers/flowers/flower_photos'

# img jpg, jpeg, png, gif 뭐가 있는지 확인
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/train/daisy | grep jpg | wc -l
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/train/daisy | grep jpeg | wc -l
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/train/daisy | grep png | wc -l
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/train/daisy | grep gif | wc -l

# img valid 데이터셋 폴더내 꽃 종류 확인
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/validation

# img test 데이터셋 폴더내 daisy 갯수 및 이미지 풀네임 확인
!ls -l /kaggle/input/flower-classification/flowers/flowers/flower_photos/test/daisy

# img train, valid 경로 지정
train_img_path = f'{root_path}/train/'
valid_img_path = f'{root_path}/validation/'

"""데이터셋이 Train/Valid 두 개로 제공된다면 train-train/valid-valid를 활용

(이미 이미지데이터셋이 따로 제공됨에 따라 별도의 데이터 분할 필요 없음)

한 개의 데이터셋만 제공된다면 아래 주석과 같이 split 필요

(하나의 dir에서 valid_split과 subset을 지정해주면 됨)
"""

# Train dataset 생성
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_img_path,
    labels='inferred', # 폴더이름으로 클래스 자동추론 하겠다
    label_mode='categorical', # 원핫인코딩 (다중클래스분류)
    color_mode='rgb', # 컬러이미지
    batch_size=32,
    image_size=(224,224), # 일반적으로 많이 사용
    seed=42,
    shuffle=True
)
# Valid dataset 생성
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=valid_img_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=32,
    image_size=(224,224),
    seed=42,
    shuffle=True
)


## 만약 데이터셋이 하나라면,
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=train_img_path,
#     labels='inferred',
#     label_mode='categorical',
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(224,224),
#     seed=42,
#     shuffle=True,
#     validation_split=0.2,
#     subset='training'
# )
# valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=train_img_path,
#     labels='inferred',
#     label_mode='categorical',
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(224,224),
#     seed=42,
#     shuffle=True,
#     validation_split=0.2,
#     subset='validation'
# )

# img dataset의 라벨 갯수 확인용 (5개)
train_dataset.class_names

# MobileNetV3 기반의 모델 생성
# img 사전학습 모델인 MobileNetV3Large 사용 (TransferLearning)
# 이미지 사이즈는 사전 학습된 모델 호환성 – 많은 딥러닝 모델(VGG16, ResNet 등)이
# 224x224 크기의 이미지를 입력으로 사용하도록 학습되어 있어서 shape는 224,224로 고정, RGB라 3채널
# imagenet에서 학습된 가중치를 가져오겠다, include_top=False는 사전학습모델의 마지막 FCLayer(top)을 제외하고 Feature Extraction 부분만 사용하겠다는 의미. 즉 마지막 레이어를 제외함에 따라 새로운 Classifier(출력층)을 위에 쌓아서 TransferLearning이 가능

mobilev3_base = tf.keras.applications.MobileNetV3Large(input_shape=(224,224,3), weights='imagenet', include_top=False)
mobilev3_base.trainable=False # 사전학습모델의 모든 가중치를 Freeze. 학습 중 업데이트되지 않고 새로 쌓는 Classifier만 학습되어 적은 데이터로 빠르고 안정적인 학습 가능(과적합 방지, 수렴 신속)

img_model = Sequential() # 레이어 순차적으로 쌓기
img_model.add(mobilev3_base) # MobilenetV3 사전학습모델 기반으로 이미지 FeatureExtraction
img_model.add(Flatten()) # CNN 다차원텐서 출력값을 1차원 벡터로 변환하여 Dense Layer(FC Layer)에 입력
img_model.add(Dense(256, activation=None)) # 활성화함수는 바로 뒤에 따로 붙여서 사용하겠다
img_model.add(BatchNormalization())
img_model.add(Activation('relu'))
img_model.add(Dropout(0.5))
img_model.add(Dense(5, activation='softmax'))

# 모델 compile
img_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy', # 정답이 원핫인코딩 형태일때 적합
    metrics=['accuracy']
)

# Custom CNN Model
model = Sequential()
model.add(Rescaling(1. / 255))
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))

# Compile Model
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate), # optimization
    loss = "categorical_crossentropy", # loss function
    metrics = ["accuracy"] # metrics
)

# MobileNetV2 기반 모델 생성
print(dir(tf.keras.applications)) # keras.applications에 어떤 종류의 모델이 있는지 확인

# 학습된 모델 가져오기
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
base_model.trainable = False

# base_model.summary() # 모델 요약

# 모델 layer 설계
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x) # # 3차원(7, 7, 1280) --> 1차원(1280)으로 줄이기 : GlobalAveragePooling2D
# x = tf.keras.layers.Flatten()(x) 써도 되는데,

output = tf.keras.layers.Dense(5, activation='softmax')(x) # number_classes = 5니깐

model = tf.keras.Model(inputs=inputs, outputs=output)
model.summary()

# 여러가지 모델 생성 함수

def create_model(model_type="basic"):
    if model_type == "basic":
        model = Sequential([
            Rescaling(1.0 / 255),
            Conv2D(32, kernel_size=(5,5), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif model_type == "VGG16":
        base_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False  # Transfer learning
        model = Sequential([
            Rescaling(1.0 / 255),
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif model_type == "ResNet50":
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False
        model = Sequential([
            Rescaling(1.0 / 255),
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    elif model_type == "MobileNetV2":
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False
        model = Sequential([
            Rescaling(1.0 / 255),
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError("지원되지 않는 모델입니다.")
    return model

# 모델 종류 설정
model_types = ["basic", "VGG16", "ResNet50", "MobileNetV2"]
histories = {}

# 모델 학습 및 성능 기록
for model_type in model_types:
    print(f"\n\n===== Training {model_type} Model =====")
    model = create_model(model_type)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=[es]
    )

    histories[model_type] = history

plt.figure(figsize=(12, 6))
for model_type, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{model_type} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_type} Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Comparison')
plt.show()

# 모델 훈련
es_img = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
mc_img = ModelCheckpoint('best_img_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history_img = img_model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    callbacks=[es_img, mc_img],
    verbose=1
)

# 시험에서는 csv 파일에 image 파일명이 제공되지만,
# 현재의 환경처럼 csv 파일이 없는 경우 시험 환경과 비슷하게 하기 위해 파일 생성
# >> 실제 시험에서는 이 셀 코드 활용 필요 x

img_glob = sorted(glob.glob(f'{root_path}/test/*/*jpg'))
img_file = [os.path.basename(path) for path in img_glob]
img_label = [os.path.basename(os.path.dirname(path)) for path in img_glob]

# img Test 데이터셋을 이용해서 DataFrame 만들기
img_df_t = pd.DataFrame()
img_df_t['image'] = img_glob
img_df_t['file'] = img_file
img_df_t['label'] = img_label
img_df_t.head(10)

# img train dataset 라벨이 어떤게 들어있는지 확인
# imagedataset labels='inferred'를 설정해서 만들었기 때문에 폴더명 a-z순으로 자동 저장됨
# label의 순서를 임의로 조정해야한다면, 위의 imagedataset Train/Valid 정의에서 labels 옵션을 labels=None, Class_names=['cats', 'dogs']을 추가할 것

train_dataset.class_names

# # 1. [이미지 폴더경로 + 이미지명] 방식
# # (예: img_df5['file'] 컬럼 사용, 모든 이미지가 한 폴더에 모여 있음)

# import os
# import pandas as pd

# # 예시: img_folder = 'Test'
# img_folder = 'Test'
# df = pd.read_csv('img_df5.csv')  # 'file' 컬럼에 이미지명만 있음 (예: flower1.jpg)

# # 이미지 전체 경로 만들기
# df['img_path'] = df['file'].apply(lambda x: os.path.join(img_folder, x))

# # 예시: 첫 번째 이미지 열기
# from PIL import Image
# img = Image.open(df.loc[0, 'img_path'])



# # 2. [이미지의 모든 경로] 방식
# # (예: img_df5['image'] 컬럼 사용, 하위 폴더 포함 전체 경로가 이미 들어 있음)

# import pandas as pd

# df = pd.read_csv('img_df5.csv')  # 'image' 컬럼에 전체 경로가 있음 (예: Test/Daisy/flower1.jpg)

# # 예시: 첫 번째 이미지 열기
# from PIL import Image
# img = Image.open(df.loc[0, 'image'])

# 예측값 저장용 변수
img_predictions = []
# img test 데이터셋 폴더 경로
img_folder = f'{root_path}/test/'
# test 데이터셋 라벨 인코딩용 변수
img_label_to_class = {
    0:'daisy',
    1:'dandelion',
    2:'roses',
    3:'sunflowers',
    4:'tulips'
}

# 이미지를 한개씩 띄우고 예측한 값과 이미지명을 확인
# img_df_t.csv 기준 위에서 부터 순서대로 모델을 예측하고 분류
for i, row in img_df_t.iterrows():
  img_path = row['image'] # 이미지 전체 경로를 활용 (csv에 그대로 있다면)
  #img_path = os.path.join(img_folder, row['file']) # 만약 파일명만 있다면 폴더 경로와 합쳐 전체 경로 만들기
  img = tf.keras.utils.load_img(img_path, target_size=(224,224)) # 이미지 로드, 사전학습 입력크기와 맞춤
  img_array = tf.keras.utils.img_to_array(img) # 이미지를 np배열로 변환
  img_array = np.expand_dims(img_array, axis=0) # 차원확장해서 배치단위로 변환 (1,224,224,3) 형태. keras는 항상 배치 단위로 입력을 받기 때문에 단일 이미지라도 (1, height, weight, channels) 형태로 변환 필요

  img_pred_t = img_model.predict(img_array, verbose=0) # 예측값 받아옴
  img_pred_label = np.argmax(img_pred_t) # 가장 확률높은 클래스 선택
  img_pred_class = img_label_to_class.get(img_pred_label, 'None') # 라벨인덱스를 클래스이름으로 변환
  img_predictions.append(img_pred_label) # 리스트에 저장

  plt.imshow(img)
  plt.title(f'Predict : {img_pred_label}({img_pred_class}) \n {os.path.basename(img_path)}')
  plt.axis('off')
  plt.show()

# img 모델의 예측값을 csv파일에 추가해줍니다.
img_df_t['pred'] = img_predictions
img_df_t.head(10)

# img 예측한 값의 라벨을 수정해줍니다.
img_df_t['pred'] = img_df_t['pred'].map(img_label_to_class)
img_df_t.head(10)

# img 테스트 데이터셋.csv를 저장합니다.
img_df_t.to_csv('01012345678_3.csv', index=False, encoding='utf-8')
img_df_c = pd.read_csv('01012345678_3.csv')
img_df_c.head(10)

# img 딥러닝 모델을 저장해줍니다.
img_model.save('01012345678_3.h5')
