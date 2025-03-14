import os
import time
from utils.predictor import Predictor
from utils.prediction_dataset_manager import PredictionDatasetManager

class PredictionResults:
    def __init__(self, total_pred_time, best_pred_model, best_pred_mae, prediction_runs, dataset):
        """ 儲存完整預測結果 """
        self.total_pred_time = total_pred_time
        self.best_pred_model = best_pred_model
        self.best_pred_mae = best_pred_mae
        self.prediction_runs = prediction_runs  # 每次預測的詳細結果
        self.dataset = dataset  # `PredictionDatasetManager` 實例

class PredictionRunner:
    @staticmethod
    def run(results_dir):
        """ 執行預測，回傳 `PredictionResults` """
        weights_dir = os.path.join(results_dir, "weights")

        # **讀取預測圖片**
        dataset = PredictionDatasetManager()

        prediction_runs = []
        best_pred_model = None
        best_pred_mae = float("inf")

        pred_start_time = time.time()

        for i, model_name in enumerate(sorted(os.listdir(weights_dir)), start=1):
            model_path = os.path.join(weights_dir, model_name)
            predictor = Predictor(model_path, results_dir)
            prediction_result = predictor.predict_and_plot(dataset.pred_image_files, i)

            # **紀錄最佳模型**
            if prediction_result.avg_mae < best_pred_mae:
                best_pred_model = prediction_result.model_path
                best_pred_mae = prediction_result.avg_mae

            prediction_runs.append(prediction_result)

        pred_end_time = time.time()
        total_pred_time = time.strftime("%H:%M:%S", time.gmtime(pred_end_time - pred_start_time))

        return PredictionResults(total_pred_time, best_pred_model, best_pred_mae, prediction_runs, dataset)
