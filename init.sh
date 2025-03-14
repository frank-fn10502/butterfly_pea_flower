#!/bin/bash

# 定義要建立的目錄
DIRS=("dataset" "pred_imgs" "results" )

# 初始化 README.md 內容
README_CONTENT=(
    "## 📂 Dataset 資料夾\n此資料夾用於存放訓練數據集，請將標註好 PH 值的圖片放在這裡。"
    "## 📂 Pred_imgs 資料夾\n此資料夾用於存放要預測的圖片，AI 會讀取這裡的圖片並輸出預測結果。"
    "## 📂 Results 資料夾\n此資料夾存放 AI 訓練與預測結果，包括 \`weights\`、\`trainings\`、\`predictions\` 等。"
)

# 建立目錄並寫入 README.md
for i in "${!DIRS[@]}"; do
    if [ ! -d "${DIRS[i]}" ]; then
        mkdir -p "${DIRS[i]}"
        echo -e "${README_CONTENT[i]}" > "${DIRS[i]}/README.md"
        echo "✅ 已建立資料夾: ${DIRS[i]}"
    else
        echo "⚠️ 資料夾已存在: ${DIRS[i]}，跳過建立。"
    fi
done

echo "🎉 專案初始化完成！"