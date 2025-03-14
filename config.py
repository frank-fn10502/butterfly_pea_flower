
# **通用參數**
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.9  # 90% 訓練, 10% 驗證
TRAIN_RUNS = 3  # 訓練次數
MODEL_TYPE = 'MobileNetCustomV1'

# **路徑設定**
DATASET_DIR = "dataset"
CSV_PATH = "dataset/ph_dataset.csv"
PRED_IMG_DIR = "pred_imgs"  # 預測圖片資料夾
