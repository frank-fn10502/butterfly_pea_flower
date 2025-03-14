import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from tensorflow.keras.preprocessing import image

class Plotter:
    @staticmethod
    def draw_training_plot(history, save_path):
        """ 繪製 Loss & MAE 訓練變化圖，並儲存至 `trainings/` """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']

        epochs_range = range(1, len(loss) + 1)

        plt.figure(figsize=(12, 5))

        # Loss 圖
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, 'b', label='Training Loss')
        plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid()

        # MAE 圖
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, mae, 'b', label='Training MAE')
        plt.plot(epochs_range, val_mae, 'r', label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training & Validation MAE')
        plt.legend()
        plt.grid()

        plt.savefig(save_path)
        plt.close()

        print(f"📊 訓練圖表已儲存至: {save_path}")

    @staticmethod
    def draw_prediction_plot(image_files, predicted_ph, actual_ph, save_dir, filename):
        """ 繪製預測結果，並儲存至 `predictions/` """
        num_images = len(image_files)
        cols = min(num_images, 3)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 6))
        axes = np.array(axes).reshape(-1) if num_images > 1 else [axes]
        chinese_font = Plotter.get_chinese_font()

        for i, img_path in enumerate(image_files):
            img = image.load_img(img_path)
            img_width, img_height = img.size  # ✅ 取得圖片尺寸
            axes[i].imshow(img)

            # ✅ **顯示圖片尺寸的刻度**
            axes[i].set_xticks([0, img_width // 2, img_width])
            axes[i].set_xticklabels([0, img_width // 2, img_width], fontsize=10)
            axes[i].set_yticks([0, img_height // 2, img_height])
            axes[i].set_yticklabels([0, img_height // 2, img_height], fontsize=10)

            # **標籤顯示在圖片下方**
            label_text = f"{os.path.basename(img_path)}\nPred: {predicted_ph[i]:.2f}"
            if actual_ph[i] is not None:
                label_text += f"\nActual: {actual_ph[i]:.2f}"

            if chinese_font:
                axes[i].set_xlabel(label_text, fontsize=10, color="blue", fontproperties=chinese_font)
            else:
                axes[i].set_xlabel(label_text, fontsize=10, color="blue")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.subplots_adjust(hspace=0.5)  # ✅ 調整 `subplot` 間距，防止標籤被裁切

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")  # ✅ 確保標籤可見
        plt.close()

        print(f"📊 預測結果已儲存至: {plot_path}")
        return plot_path

    @staticmethod
    def get_chinese_font():
        """ 載入中文字體 (若有) """
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return fm.FontProperties(fname=font_path)
        return None
