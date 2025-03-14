from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications import MobileNet, MobileNetV2, EfficientNetV2B0
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from config import MODEL_TYPE

def build_mobilenet():
    """ 建立 MobileNet 模型 """
    base_model = MobileNet(input_shape=(224, 224, 3), alpha=1.0, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation='linear', name='ph_value')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def build_mobilenet_CustomV1(freeze_layers=80):
    """ 建立 MobileNet 模型，並凍結部分層數 """
    base_model = MobileNet(input_shape=(224, 224, 3), alpha=1.0, weights='imagenet', include_top=False)

    # **凍結前 freeze_layers 層**
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)  # ✅ 防止過擬合
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation='linear', name='ph_value')(x)  # Regression Output

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error", metrics=["mae"])

    return model

def build_mobilenetv2():
    """ 建立 MobileNetV2 模型 """
    base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)  # 這裡可以修改 layer 結構
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation='linear', name='ph_value')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def build_efficientnet():
    """ 建立 EfficientNetV2B0 模型 """
    base_model = EfficientNetV2B0(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)  # 這裡可以修改 layer 結構
    x = Dense(256, activation="relu")(x)
    output = Dense(1, activation='linear', name='ph_value')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def build_lightweight_cnn():
    """ 建立一個適合小型資料集的 CNN 模型 """
    inputs = Input(shape=(224, 224, 3))

    # 第一層卷積 + BatchNorm + ReLU
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第二層卷積 + BatchNorm + ReLU
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 全局平均池化（比 Flatten 更適合小數據集）
    x = GlobalAveragePooling2D()(x)

    # Dropout 防止 overfitting
    x = Dropout(0.3)(x)

    # 全連接層
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear', name='ph_value')(x)  # Regression Output

    model = Model(inputs, output)

    # ✅ 使用 AdamW（比 Adam 更穩定）
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss="mean_squared_error",
                  metrics=["mae"])

    return model

def create_model():
    """ 根據 `MODEL_TYPE` 選擇對應的模型 """
    if MODEL_TYPE == "MobileNet":
        return build_mobilenet()
    elif MODEL_TYPE == "MobileNetCustomV1":
        return build_efficientnet()
    elif MODEL_TYPE == "MobileNetV2":
        return build_mobilenetv2()
    elif MODEL_TYPE == "EfficientNetV2B0":
        return build_efficientnet()
    elif MODEL_TYPE == "MineLiteModelV1":
        return build_efficientnet()
    else:
        raise ValueError(f"❌ 無效的 MODEL_TYPE: {MODEL_TYPE}, 可選值: MobileNet, MobileNetV2, EfficientNetV2B0, MineLiteModelV1, MobileNetCustomV1")
