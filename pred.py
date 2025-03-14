import sys
from utils.prediction_runner import PredictionRunner
from utils.prediction_summary_writer import PredictionSummaryWriter

if len(sys.argv) < 2:
    print("⚠️ 請提供要測試的結果目錄，例如: `python pred.py results/20250320_153000`")
    sys.exit(1)

results_dir = sys.argv[1]

# **執行預測**
prediction_results = PredictionRunner.run(results_dir)

# **寫入 Markdown**
md_writer = PredictionSummaryWriter(results_dir)
md_writer.write_summary(prediction_results)

print("✅ 該批模型預測已完成！📜 記錄已更新")