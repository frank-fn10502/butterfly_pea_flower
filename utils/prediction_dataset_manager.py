import os
from config import PRED_IMG_DIR

class PredictionDatasetManager:
    def __init__(self):
        """ 初始化，讀取所有預測圖片 """
        self.pred_image_files = sorted([
            os.path.join(PRED_IMG_DIR, f) for f in os.listdir(PRED_IMG_DIR) 
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.num_pred_images = len(self.pred_image_files)
