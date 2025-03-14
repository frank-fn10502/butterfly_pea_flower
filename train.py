import os
import datetime
from utils.dataset_manager import DatasetManager
from utils.trainer import Trainer
from utils.training_summary_writer import TrainingSummaryWriter
from config import TRAIN_RUNS, BATCH_SIZE, EPOCHS, LEARNING_RATE

# **建立時間戳記 & 目錄**
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", TIMESTAMP)
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
TRAININGS_DIR = os.path.join(RESULTS_DIR, "trainings")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(TRAININGS_DIR, exist_ok=True)

# **準備數據**
dataset_manager = DatasetManager()

# **執行訓練**
trainer = Trainer(WEIGHTS_DIR, TRAININGS_DIR, dataset_manager,
                  batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)  # ✅ 傳入 config 設定
training_results = trainer.train(TRAIN_RUNS)

# **寫入 Markdown**
md_writer = TrainingSummaryWriter(TRAININGS_DIR)
md_writer.write_summary(training_results)

print("\n🔄 訓練完成，開始測試該批模型...")
os.system(f"python pred.py {RESULTS_DIR}")
