# 📌 AI PH 值預測專案 - 報告與技術簡介

本報告針對本專案所涉及的 **深度學習 (Deep Learning)**、**Keras & TensorFlow**、**卷積神經網路 (CNN)** 及 **程式碼介紹** 進行詳細講解，讓讀者能夠更好地理解 AI 如何預測蝶豆花水的 PH 值。

---

## **🌍 Deep Learning & AI 簡介**
### **1️⃣ AI、Machine Learning、Deep Learning 之間的關係**
- **人工智慧 (AI)**：指的是讓電腦執行需要「智慧」的任務，如語音辨識、影像分類、圍棋等。
- **機器學習 (Machine Learning, ML)**：是 AI 的一個子領域，透過「資料訓練」讓電腦自行學習規律，不需要人工編寫規則。
- **深度學習 (Deep Learning, DL)**：是機器學習的進階分支，**使用人工神經網路 (Neural Networks)** 來模仿人類大腦進行學習與推論。

### **2️⃣ 深度學習的核心概念**
- **神經網路 (Neural Network)**：包含多層「神經元 (Neuron)」，每個神經元會學習不同的特徵。
- **卷積神經網路 (CNN, Convolutional Neural Network)**：專門用來分析影像，擁有 **卷積層 (Conv2D)** 和 **池化層 (Pooling)**，可以自動學習圖片特徵。
- **回歸 (Regression)**：本專案預測 PH 值，數值是連續的，因此使用 **回歸模型** 而非分類模型。

---

## **📌 什麼是卷積神經網路 (CNN)?**
### **1️⃣ 為什麼要使用 CNN？**
傳統的神經網路（如 `Dense` 層）適合處理 **表格數據**，但對於 **影像數據** 來說效果較差。  
CNN **能夠自動學習影像中的重要特徵**，例如：
✅ 邊緣  
✅ 顏色變化  
✅ 紋理  

因此，CNN 特別適合 **影像分類、目標檢測、影像回歸** 等應用。

---

### **2️⃣ CNN 的主要組成**
| **層 (Layer)** | **功能** |
|--------------|--------|
| **卷積層 (Conv2D)** | 提取圖片的局部特徵，如邊緣、顏色變化 |
| **池化層 (MaxPooling2D)** | 降低圖片尺寸，減少運算量，保留主要特徵 |
| **展平層 (Flatten)** | 將 2D 圖片轉換為 1D 向量，以便輸入到全連接層 |
| **全連接層 (Dense)** | 將提取到的特徵組合，輸出最終的預測值 |
| **Dropout** | 隨機關閉一部分神經元，防止過擬合 |

---

### **3️⃣ CNN 如何運作？**
當圖片輸入 CNN 時，它會經過多個 **卷積層 (Conv2D)** 來提取特徵：
```
輸入圖片 → 卷積層 (Conv2D) → 池化層 (MaxPooling) → 卷積層 (Conv2D) → 池化層 (MaxPooling) → 展平層 (Flatten) → 全連接層 (Dense) → 輸出 (PH 值)
```

---

## **📌 Keras & TensorFlow 簡介**
本專案使用 **TensorFlow 2.x** 搭配 **Keras API** 來實作 AI 模型。

- **TensorFlow**：Google 開發的深度學習框架，能高效執行神經網路訓練與推論。
- **Keras**：TensorFlow 內建的高階 API，簡化了深度學習的開發流程。

**TensorFlow + Keras 的優勢**
✅ 易於使用：Keras 提供高層 API，可以快速搭建模型  
✅ 強大的 GPU 加速：TensorFlow 支援 GPU 訓練，加速模型運算  
✅ 靈活的模型設計：可透過 Functional API 設計複雜網路  

---

## **📌 結論**
- 我們利用 **CNN（卷積神經網路）** 來學習 **蝶豆花水的顏色與 PH 值的關係**
- 透過 **MobileNetV2**，在小型資料集上獲得良好的效能
- 使用 **Data Augmentation**、**Learning Rate 調整** 來提升模型準確率
- **自訂 CNN（CustomCNN）** 提供另一種輕量級解法

本專案展示了如何用 **深度學習 (Deep Learning) + TensorFlow/Keras** 來解決 **回歸問題**，並使用 **影像數據** 進行科學預測。
