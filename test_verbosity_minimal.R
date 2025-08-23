# 最小化verbosity测试 - 快速验证
library(newrllama4)

cat("=== 最小化verbosity测试 ===\n\n")

# 测试一个最小的操作，看输出差异
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

cat("🔇 verbosity=0 (应该最安静):\n")
cat("----------------------------------------\n")
model <- model_load(model_path, n_gpu_layers = 10L, verbosity = 0L)  # 少量GPU层减少输出
rm(model)
backend_free()

cat("\n🔊 verbosity=3 (应该最详细):\n") 
cat("----------------------------------------\n")
model <- model_load(model_path, n_gpu_layers = 10L, verbosity = 3L)  # 少量GPU层减少输出
rm(model)
backend_free()

cat("\n✅ 测试完成\n")
cat("💡 如果verbosity修复生效，上面两次加载的输出应该有明显差异\n")