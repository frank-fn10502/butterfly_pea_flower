import os

class TrainingResults:
    def __init__(self, timestamp, total_training_time, num_runs, train_samples, valid_samples,
                 train_results, best_model, best_loss, best_mae, batch_size, epochs, learning_rate, model_type):
        """ 統一訓練結果的資料結構 """
        self.timestamp = timestamp
        self.total_training_time = total_training_time
        self.num_runs = num_runs
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.train_results = train_results  # List of (run_id, val_loss, val_mae, duration)
        self.best_model = best_model
        self.best_loss = best_loss
        self.best_mae = best_mae
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_type = model_type  # ✅ 新增 `MODEL_TYPE`


class TrainingSummaryWriter:
    def __init__(self, save_dir):
        """ 初始化 `TrainingSummaryWriter`，設定輸出目錄 """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, "summary.md")

    def write_summary(self, training_results: TrainingResults):
        """ 產生 `trainings/summary.md` """
        summary_content = [
            f"# 訓練紀錄 - {training_results.timestamp}",
            f"\n📅 **訓練時間**: {training_results.timestamp}",
            f"🧠 **使用模型**: `{training_results.model_type}`",  # ✅ 新增 `MODEL_TYPE`
            f"⏳ **總訓練時間**: {training_results.total_training_time}",
            f"🔄 **訓練批次**: {training_results.num_runs} 次",
            "\n## 📊 資料集資訊",
            f"- **訓練集樣本數**: `{training_results.train_samples}`",
            f"- **驗證集樣本數**: `{training_results.valid_samples}`",
            "\n## 訓練結果"
        ]

        # **列出所有訓練批次結果**
        for run_id, val_loss, val_mae, duration in training_results.train_results:
            summary_content.append(f"- **Run {run_id}**: `val_loss = {val_loss:.4f}`, `val_mae = {val_mae:.4f}`, ⏱ **訓練時長**: {duration}")

        # **最佳模型資訊**
        summary_content.append("\n## 最佳模型")
        summary_content.append(f"🏆 **最佳模型**: `{training_results.best_model}`")
        summary_content.append(f"📉 **最低驗證 Loss**: `{training_results.best_loss:.4f}`")
        summary_content.append(f"📊 **最低驗證 MAE**: `{training_results.best_mae:.4f}`")

        # **訓練參數**
        summary_content.append("\n## 訓練超參數")
        summary_content.append(f"- **Batch Size**: `{training_results.batch_size}`")
        summary_content.append(f"- **Epochs**: `{training_results.epochs}`")
        summary_content.append(f"- **Learning Rate**: `{training_results.learning_rate}`")

        self._write_to_file(summary_content)

    def _write_to_file(self, content):
        """ 內部函數，負責寫入 Markdown 檔案 """
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"📜 訓練紀錄已更新: {self.file_path}")