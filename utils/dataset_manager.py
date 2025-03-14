import os
import re
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

class DatasetManager:
    def __init__(self):
        """ 初始化 DatasetManager，確保數據集可用 """
        self.dataset_csv = CSV_PATH
        self.dataset_dir = DATASET_DIR
        self.train_df = None
        self.val_df = None
        self.num_train_samples = 0
        self.num_valid_samples = 0

        # **確保數據集 CSV 存在，否則生成**
        if not os.path.exists(self.dataset_csv):
            print("⚠️ 找不到 `ph_dataset.csv`，正在重新生成...")
            self.generate_csv()
        else:
            print(f"✅ 發現現有的 `{self.dataset_csv}`，直接使用！")

        self.load_and_split_data()

    def generate_csv(self):
        """ 從資料夾讀取圖片檔案，提取 PH 值，並存為 `ph_dataset.csv` """
        ph_pattern = re.compile(r'PH(\d+\.?\d*)')
        file_list = []
        ph_values = []

        for root, _, files in os.walk(self.dataset_dir):
            for filename in files:
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    match = ph_pattern.search(filename)
                    if match:
                        ph_value = float(match.group(1))
                        file_path = os.path.join(root, filename)
                        file_list.append(file_path)
                        ph_values.append(ph_value)

        df = pd.DataFrame({"filename": file_list, "ph_value": ph_values})
        df.to_csv(self.dataset_csv, index=False)
        print(f"✅ `ph_dataset.csv` 生成完成，總共 {len(df)} 筆資料！")

    def load_and_split_data(self):
        """ 讀取 `ph_dataset.csv` 並切分訓練/驗證集 """
        df = pd.read_csv(self.dataset_csv)
        self.train_df = df.sample(frac=TRAIN_SPLIT, random_state=42)
        self.val_df = df.drop(self.train_df.index)

        self.num_train_samples = len(self.train_df)
        self.num_valid_samples = len(self.val_df)

        print(f"📊 訓練樣本: {self.num_train_samples}，驗證樣本: {self.num_valid_samples}")

        # **建立訓練與驗證數據生成器**
        self.train_generator = self.create_train_generator(self.train_df, shuffle=True)
        self.val_generator = self.create_valid_generator(self.val_df, shuffle=False)

    def create_train_generator(self, dataframe, shuffle):
        """ 建立訓練資料的 Data Augmentation """
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,        # ✅ 隨機旋轉 15 度
            width_shift_range=0.1,    # ✅ 隨機水平平移 10%
            height_shift_range=0.1,   # ✅ 隨機垂直平移 10%
            zoom_range=0.2,           # ✅ 隨機縮放 20%
            horizontal_flip=True      # ✅ 隨機水平翻轉
        )
        return train_datagen.flow_from_dataframe(
            dataframe,
            x_col="filename",
            y_col="ph_value",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='raw',
            shuffle=shuffle
        )

    def create_valid_generator(self, dataframe, shuffle):
        """ 建立驗證資料的生成器（不做 Data Augmentation） """
        valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
        return valid_datagen.flow_from_dataframe(
            dataframe,
            x_col="filename",
            y_col="ph_value",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='raw',
            shuffle=shuffle
        )
