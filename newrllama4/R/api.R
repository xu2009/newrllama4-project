# --- FILE: newrllama4/R/api.R ---

#' Initialize newrllama backend
#' 
#' Initialize the backend library. This should be called once before using other functions.
#' 
#' @export
backend_init <- function() {
  .ensure_backend_loaded()
  invisible(.Call("c_r_backend_init"))
}

#' Free newrllama backend
#' 
#' Clean up backend resources. Usually called automatically.
#' 
#' @export  
backend_free <- function() {
  if (.is_backend_loaded()) {
    invisible(.Call("c_r_backend_free"))
  }
}

#' Load a language model
#'
#' @param model_path Path to the GGUF model file
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0)
#' @param use_mmap Whether to use memory mapping (default: TRUE)
#' @param use_mlock Whether to use memory locking (default: FALSE)
#' @return A model object (external pointer)
#' @export
model_load <- function(model_path, n_gpu_layers = 0L, use_mmap = TRUE, use_mlock = FALSE) {
  .ensure_backend_loaded()
  if (!file.exists(model_path)) {
    stop("Model file does not exist: ", model_path, call. = FALSE)
  }
  
  .Call("c_r_model_load", 
        as.character(model_path),
        as.integer(n_gpu_layers), 
        as.logical(use_mmap),
        as.logical(use_mlock))
}

#' Create inference context
#'
#' @param model A model object returned by model_load()
#' @param n_ctx Context size (default: 2048)
#' @param n_threads Number of threads (default: 4)  
#' @param n_seq_max Maximum number of sequences (default: 1)
#' @return A context object (external pointer)
#' @export
context_create <- function(model, n_ctx = 2048L, n_threads = 4L, n_seq_max = 1L) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_context_create",
        model,
        as.integer(n_ctx),
        as.integer(n_threads), 
        as.integer(n_seq_max))
}

#' Tokenize text
#'
#' @param model A model object
#' @param text Text to tokenize
#' @param add_special Whether to add special tokens (default: TRUE)
#' @return Integer vector of token IDs
#' @export
tokenize <- function(model, text, add_special = TRUE) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_tokenize",
        model,
        as.character(text),
        as.logical(add_special))
}

#' Detokenize tokens
#'
#' @param model A model object
#' @param tokens Integer vector of token IDs  
#' @return Detokenized text string
#' @export
detokenize <- function(model, tokens) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_detokenize",
        model,
        as.integer(tokens))
}

#' Apply chat template
#'
#' @param model A model object
#' @param messages List of chat messages, each with 'role' and 'content'
#' @param template Optional custom template (default: NULL, use model's template)
#' @param add_assistant Whether to add assistant prompt (default: TRUE)
#' @return Formatted prompt string
#' @export
apply_chat_template <- function(model, messages, template = NULL, add_assistant = TRUE) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_apply_chat_template",
        model,
        template,
        messages,
        as.logical(add_assistant))
}

#' Generate text
#'
#' @param context A context object
#' @param tokens Input tokens or prompt text
#' @param max_tokens Maximum tokens to generate (default: 100)
#' @param top_k Top-k sampling (default: 40)
#' @param top_p Top-p sampling (default: 0.9)
#' @param temperature Sampling temperature (default: 0.8)
#' @param repeat_last_n Repetition penalty last n tokens (default: 64)
#' @param penalty_repeat Repetition penalty strength (default: 1.1)
#' @param seed Random seed (default: -1 for random)
#' @return Generated text
#' @export
generate <- function(context, tokens, max_tokens = 100L, top_k = 40L, top_p = 0.9, 
                     temperature = 0.8, repeat_last_n = 64L, penalty_repeat = 1.1, seed = -1L) {
  .ensure_backend_loaded()
  if (!inherits(context, "newrllama_context")) {
    stop("Expected a newrllama_context object", call. = FALSE)
  }
  
  .Call("c_r_generate",
        context,
        as.integer(tokens),
        as.integer(max_tokens),
        as.integer(top_k),
        as.numeric(top_p),
        as.numeric(temperature),
        as.integer(repeat_last_n),
        as.numeric(penalty_repeat),
        as.integer(seed))
}

#' Generate text in parallel
#'
#' @param context A context object
#' @param prompts Character vector of prompts
#' @param max_tokens Maximum tokens to generate (default: 100)
#' @param top_k Top-k sampling (default: 40)
#' @param top_p Top-p sampling (default: 0.9)
#' @param temperature Sampling temperature (default: 0.8)
#' @param repeat_last_n Repetition penalty last n tokens (default: 64)
#' @param penalty_repeat Repetition penalty strength (default: 1.1)
#' @param seed Random seed (default: -1 for random)
#' @return Character vector of generated texts
#' @export
generate_parallel <- function(context, prompts, max_tokens = 100L, top_k = 40L, top_p = 0.9,
                              temperature = 0.8, repeat_last_n = 64L, penalty_repeat = 1.1, seed = -1L) {
  .ensure_backend_loaded()
  if (!inherits(context, "newrllama_context")) {
    stop("Expected a newrllama_context object", call. = FALSE)
  }
  
  .Call("c_r_generate_parallel",
        context,
        as.character(prompts),
        as.integer(max_tokens),
        as.integer(top_k),
        as.numeric(top_p),
        as.numeric(temperature),
        as.integer(repeat_last_n),
        as.numeric(penalty_repeat),
        as.integer(seed))
}

#' Test tokenize function (debugging)
#'
#' @param model A model object
#' @return Integer vector of tokens for "H"
#' @export
tokenize_test <- function(model) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_tokenize_test", model)
} 