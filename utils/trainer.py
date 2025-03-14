import os
import time
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from models.models import create_model
from utils.dataset_manager import DatasetManager
from utils.training_summary_writer import TrainingResults
from utils.plotter import Plotter
from config import MODEL_TYPE  # ✅ 確保 `MODEL_TYPE` 被引入

class Trainer:
    def __init__(self, weights_dir, trainings_dir, dataset: DatasetManager, batch_size=32, epochs=50, learning_rate=0.001):
        """ 初始化 `Trainer` """
        self.weights_dir = weights_dir
        self.trainings_dir = trainings_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_type = MODEL_TYPE  # ✅ 記錄當前使用的模型類型

    def train(self, num_runs):
        """ 執行多次訓練，回傳 `TrainingResults` """
        start_time = time.time()

        train_results = []
        best_model, best_loss, best_mae = None, float("inf"), float("inf")

        for run in range(1, num_runs + 1):
            print(f"\n🚀 開始第 {run}/{num_runs} 次訓練 (使用模型: {self.model_type})...")  # ✅ 顯示模型名稱
            model_path, val_loss, val_mae, duration = self._train_single_run(run)
            train_results.append((run, val_loss, val_mae, duration))

            # **選擇最佳模型**
            if val_loss < best_loss:
                best_loss, best_mae, best_model = val_loss, val_mae, model_path

        total_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        return TrainingResults(
            timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            total_training_time=total_training_time,
            num_runs=num_runs,
            train_samples=self.dataset.num_train_samples,
            valid_samples=self.dataset.num_valid_samples,
            train_results=train_results,
            best_model=best_model,
            best_loss=best_loss,
            best_mae=best_mae,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            model_type=self.model_type  # ✅ 新增 `MODEL_TYPE`
        )

    def _train_single_run(self, run_id):
        """ 執行單次訓練，回傳 `(model_path, val_loss, val_mae, duration)` """
        # **開始計時**
        start_time = time.time()

        # **建立模型**
        model = create_model()
        # model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

        # **設置最佳模型儲存**
        model_path = os.path.join(self.weights_dir, f"best_ph_model_run{run_id}.h5")
        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]

        # **執行訓練**
        history = model.fit(
            self.dataset.train_generator,
            validation_data=self.dataset.val_generator,
            epochs=self.epochs,
            callbacks=callbacks
        )

        # 產生 training loss & mae 圖
        plot_filename = f"training_run{run_id}.png"
        plot_path = os.path.join(self.trainings_dir, plot_filename)
        Plotter.draw_training_plot(history, plot_path)  # 呼叫 `Plotter`

        # **取得最佳 val_loss & val_mae**
        best_epoch = history.history["val_loss"].index(min(history.history["val_loss"]))
        val_loss = history.history["val_loss"][best_epoch]
        val_mae = history.history["val_mae"][best_epoch]

        # **計算訓練時間**
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

        print(f"✅ 訓練 {run_id} 完成！`val_loss = {val_loss:.4f}`, `val_mae = {val_mae:.4f}`, ⏱ **訓練時長**: {duration}")

        return model_path, val_loss, val_mae, duration
