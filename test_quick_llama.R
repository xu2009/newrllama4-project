library(newrllama4)

# Test quick_llama function
print("Testing quick_llama...")
result <- quick_llama(
  prompt = "Hello, how are you?",
  model = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  max_tokens = 10,
  temperature = 0.7,
  n_gpu_layers = 0,
  verbosity = 1
)

print("Success!")
print(result)