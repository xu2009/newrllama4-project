# Test script for v1.0.42 parallel generation function fixes
# This script tests the fixes for short outputs and incoherent responses

cat("🧪 Testing newrllama4 v1.0.42 Parallel Generation Fixes\n")
cat("=" %R% 60, "\n")

# Load the library
library(newrllama4)

# Check version
cat("📦 Package version:", packageVersion("newrllama4"), "\n")

# Test parameters
test_prompts <- c(
  "What is the capital of France? Please explain why it's important.",
  "Describe the process of photosynthesis in plants.",
  "What are the benefits of renewable energy sources?"
)

max_tokens_test <- 100  # This should now generate close to 100 tokens

cat("\n🔧 Test Configuration:\n")
cat("  - Number of prompts:", length(test_prompts), "\n")
cat("  - Max tokens per response:", max_tokens_test, "\n")
cat("  - Expected: Each response should be close to", max_tokens_test, "tokens\n")
cat("  - Expected: Responses should be coherent and well-formed\n")

cat("\n📝 Test Prompts:\n")
for (i in seq_along(test_prompts)) {
  cat("  ", i, ": \"", substr(test_prompts[i], 1, 50), "...\"\n", sep = "")
}

# This test requires you to have a model loaded and context created
# The actual test should be run interactively after setting up the backend

cat("\n⚠️  IMPORTANT: To run this test, you need to:\n")
cat("1. Run install_newrllama() to download the backend\n")
cat("2. Load a model using model_load()\n")
cat("3. Create a context with context_create()\n")
cat("4. Then call generate_parallel() with these test prompts\n")

cat("\n📋 Example test commands:\n")
cat("# backend_init()\n")
cat("# model <- model_load('path/to/your/model.gguf')\n")
cat("# ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)\n")
cat("# results <- generate_parallel(ctx, test_prompts, max_tokens = 100)\n")
cat("# \n")
cat("# Expected results:\n")
cat("# - Each result should be substantially longer than before\n")
cat("# - No more random '?' at the beginning\n")
cat("# - Coherent, well-formed responses\n")
cat("# - Close to 100 tokens each (not 10-20 tokens)\n")

cat("\n🎯 Success Criteria:\n")
cat("✅ Response length: Each response should be 80-120 tokens\n")
cat("✅ Response quality: No garbled text or strange prefixes\n")
cat("✅ Response coherence: Answers should directly address the questions\n")
cat("✅ Stability: No crashes or memory errors\n")

cat("\n🚀 Ready for testing! Load your model and run the parallel generation.\n")