import os
import datetime

class PredictionSummaryWriter:
    def __init__(self, save_dir):
        """ 初始化 `PredictionSummaryWriter`，設定輸出目錄 """
        self.save_dir = os.path.join(save_dir, "predictions")
        os.makedirs(self.save_dir, exist_ok=True)
        self.file_path = os.path.join(self.save_dir, "summary.md")

    def write_summary(self, prediction_results):
        """ 產生 `predictions/summary.md`，格式完全符合使用者需求 """
        summary_content = [
            f"# 預測紀錄 - {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            f"\n- **總預測時間**: `{prediction_results.total_pred_time}`",
            f"- **預測圖片張數**: `{prediction_results.dataset.num_pred_images} 張`",
        ]
        
        if prediction_results.dataset.num_pred_images > 0:
            summary_content.append("    - " + "\n    - ".join(prediction_results.dataset.pred_image_files))

        summary_content.append("\n## 預測結果")
        for i, result in enumerate(prediction_results.prediction_runs, start=1):
            summary_content.append(f"- **Run {i}**: `平均誤差 = {result.avg_mae:.4f}`, ⏱ **預測時長**: {prediction_results.total_pred_time}")

        summary_content.append("\n## 最佳預測模型")
        summary_content.append(f"- **🏆 最佳模型**: `{prediction_results.best_pred_model}`")
        summary_content.append(f"- **📊 最低 平均誤差**: `{prediction_results.best_pred_mae:.4f}`")

        self._write_to_file(summary_content)

    def _write_to_file(self, content):
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"📜 預測紀錄已更新: {self.file_path}")
