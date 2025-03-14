import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from utils.plotter import Plotter

class PredictionResults:
    def __init__(self, model_path, avg_mae, prediction_plot):
        """ 儲存單次模型的預測結果 """
        self.model_path = model_path
        self.avg_mae = avg_mae
        self.prediction_plot = prediction_plot

class Predictor:
    def __init__(self, model_path, results_dir):
        """ 初始化 `Predictor`，載入模型 """
        self.model_path = model_path
        self.results_dir = results_dir
        self.predictions_dir = os.path.join(results_dir, "predictions")  # ✅ 確保 predictions 目錄
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ 成功載入模型: {model_path}")

    def predict_and_plot(self, image_files, run_id):
        """ 對 `image_files` 內的圖片進行預測，並產生結果圖 """
        pred_values = []
        actual_values = []
        filenames = [os.path.basename(img) for img in image_files]  # ✅ 確保 `filename` 被正確傳遞

        for img_path in image_files:
            pred_ph, actual_ph = self._predict_image(img_path)
            if actual_ph is not None:
                pred_values.append(pred_ph)
                actual_values.append(actual_ph)

        # ✅ 計算 MAE (平均誤差)
        avg_mae = np.mean(np.abs(np.array(pred_values) - np.array(actual_values))) if actual_values else None

        # ✅ 產生預測圖
        plot_filename = f"pred_run{run_id}.png"
        plot_path = os.path.join(self.predictions_dir, plot_filename)  # ✅ 修正 `plot_path`
        Plotter.draw_prediction_plot(image_files, pred_values, actual_values, self.predictions_dir, plot_filename)  # ✅ 確保傳遞正確

        return PredictionResults(self.model_path, avg_mae, plot_path)

    def _predict_image(self, img_path):
        """ 預測單張圖片的 PH 值 """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred_ph = self.model.predict(img_array)[0][0]
        actual_ph = self._extract_ph_from_filename(img_path)
        return pred_ph, actual_ph

    def _extract_ph_from_filename(self, filename):
        """ 從檔名擷取 PH 值 """
        match = re.search(r'PH([\d.]+)', filename)
        return float(match.group(1)) if match else None
