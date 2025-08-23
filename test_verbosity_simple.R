# 简单的verbosity测试 - 检查不同级别的输出差异
library(newrllama4)

cat("=== 测试verbosity级别差异 ===\n\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  cat("❌ 模型文件未找到:", model_path, "\n")
  quit(status = 1)
}

# 首先确保有新版本的后端库
if (!lib_is_installed()) {
  cat("正在安装newrllama后端库...\n")
  install_newrllama()
}

cat("📝 新verbosity逻辑:\n")
cat("  - verbosity=3: 最多输出 (DEBUG + INFO + WARN + ERROR)\n")
cat("  - verbosity=2: 中等输出 (INFO + WARN + ERROR)\n")  
cat("  - verbosity=1: 基本输出 (WARN + ERROR) - 默认\n")
cat("  - verbosity=0: 最少输出 (ERROR only)\n\n")

cat("开始测试不同verbosity级别...\n\n")

# 测试verbosity=0 (最少输出)
cat("🔇 测试verbosity=0 (最少输出 - 仅ERROR):\n")
cat(strrep("-", 50), "\n")
model0 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0L)
ctx0 <- context_create(model0, n_ctx = 512L, verbosity = 0L)
rm(model0, ctx0)
backend_free()

cat("\n\n🔈 测试verbosity=1 (默认级别 - WARN + ERROR):\n")
cat(strrep("-", 50), "\n")
model1 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 1L)
ctx1 <- context_create(model1, n_ctx = 512L, verbosity = 1L)
rm(model1, ctx1)
backend_free()

cat("\n\n🔉 测试verbosity=2 (中等输出 - INFO + WARN + ERROR):\n")
cat(strrep("-", 50), "\n")
model2 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 2L)
ctx2 <- context_create(model2, n_ctx = 512L, verbosity = 2L)
rm(model2, ctx2)
backend_free()

cat("\n\n🔊 测试verbosity=3 (最多输出 - DEBUG + INFO + WARN + ERROR):\n")
cat(strrep("-", 50), "\n")
model3 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 3L)
ctx3 <- context_create(model3, n_ctx = 512L, verbosity = 3L)
rm(model3, ctx3)
backend_free()

cat("\n\n✅ verbosity测试完成！\n")
cat("💡 观察上面的输出，应该能看到:\n")
cat("   - verbosity=0: 输出最少\n")
cat("   - verbosity=3: 输出最多\n")
cat("   - 数字越大，输出越详细！\n")