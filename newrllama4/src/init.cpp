// --- FILE: newrllama4/src/init.cpp ---
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// 声明所有需要从interface.cpp导出的函数
extern "C" {
  // Proxy initialization function
  void r_newrllama_api_init(SEXP handle_sexp);
  void r_newrllama_api_reset();
  
  // Core functions
  SEXP r_backend_init();
  SEXP r_backend_free();
  SEXP r_model_load(SEXP model_path, SEXP n_gpu_layers, SEXP use_mmap, SEXP use_mlock);
  SEXP r_context_create(SEXP model_ptr, SEXP n_ctx, SEXP n_threads, SEXP n_seq_max);
  SEXP r_tokenize(SEXP model_ptr, SEXP text, SEXP add_special);
  SEXP r_detokenize(SEXP model_ptr, SEXP tokens);
  SEXP r_apply_chat_template(SEXP model_ptr, SEXP tmpl, SEXP chat_messages, SEXP add_ass);
  SEXP r_generate(SEXP ctx_ptr, SEXP tokens, SEXP max_tokens, SEXP top_k, SEXP top_p, SEXP temperature, SEXP repeat_last_n, SEXP penalty_repeat, SEXP seed);
  SEXP r_generate_parallel(SEXP ctx_ptr, SEXP prompts, SEXP max_tokens, SEXP top_k, SEXP top_p, SEXP temperature, SEXP repeat_last_n, SEXP penalty_repeat, SEXP seed);
  
  // Token functions
  SEXP r_token_get_text(SEXP model_ptr, SEXP token);
  SEXP r_token_bos(SEXP model_ptr);
  SEXP r_token_eos(SEXP model_ptr);
  SEXP r_token_sep(SEXP model_ptr);
  SEXP r_token_nl(SEXP model_ptr);
  SEXP r_token_pad(SEXP model_ptr);
  SEXP r_token_eot(SEXP model_ptr);
  SEXP r_add_bos_token(SEXP model_ptr);
  SEXP r_add_eos_token(SEXP model_ptr);
  SEXP r_token_fim_pre(SEXP model_ptr);
  SEXP r_token_fim_mid(SEXP model_ptr);
  SEXP r_token_fim_suf(SEXP model_ptr);
  SEXP r_token_get_attr(SEXP model_ptr, SEXP token);
  SEXP r_token_get_score(SEXP model_ptr, SEXP token);
  SEXP r_token_is_eog(SEXP model_ptr, SEXP token);
  SEXP r_token_is_control(SEXP model_ptr, SEXP token);
  
  // Test function for debugging
  SEXP r_tokenize_test(SEXP model_ptr);
}

// 定义C例程表
static const R_CallMethodDef CallEntries[] = {
  // Proxy initialization
  {"c_newrllama_api_init", (DL_FUNC) &r_newrllama_api_init, 1},
  {"c_newrllama_api_reset", (DL_FUNC) &r_newrllama_api_reset, 0},
  
  // Core functions
  {"c_r_backend_init", (DL_FUNC) &r_backend_init, 0},
  {"c_r_backend_free", (DL_FUNC) &r_backend_free, 0},
  {"c_r_model_load", (DL_FUNC) &r_model_load, 4},
  {"c_r_context_create", (DL_FUNC) &r_context_create, 4},
  {"c_r_tokenize", (DL_FUNC) &r_tokenize, 3},
  {"c_r_detokenize", (DL_FUNC) &r_detokenize, 2},
  {"c_r_apply_chat_template", (DL_FUNC) &r_apply_chat_template, 4},
  {"c_r_generate", (DL_FUNC) &r_generate, 9},
  {"c_r_generate_parallel", (DL_FUNC) &r_generate_parallel, 9},
  
  // Token functions
  {"c_r_token_get_text", (DL_FUNC) &r_token_get_text, 2},
  {"c_r_token_bos", (DL_FUNC) &r_token_bos, 1},
  {"c_r_token_eos", (DL_FUNC) &r_token_eos, 1},
  {"c_r_token_sep", (DL_FUNC) &r_token_sep, 1},
  {"c_r_token_nl", (DL_FUNC) &r_token_nl, 1},
  {"c_r_token_pad", (DL_FUNC) &r_token_pad, 1},
  {"c_r_token_eot", (DL_FUNC) &r_token_eot, 1},
  {"c_r_add_bos_token", (DL_FUNC) &r_add_bos_token, 1},
  {"c_r_add_eos_token", (DL_FUNC) &r_add_eos_token, 1},
  {"c_r_token_fim_pre", (DL_FUNC) &r_token_fim_pre, 1},
  {"c_r_token_fim_mid", (DL_FUNC) &r_token_fim_mid, 1},
  {"c_r_token_fim_suf", (DL_FUNC) &r_token_fim_suf, 1},
  {"c_r_token_get_attr", (DL_FUNC) &r_token_get_attr, 2},
  {"c_r_token_get_score", (DL_FUNC) &r_token_get_score, 2},
  {"c_r_token_is_eog", (DL_FUNC) &r_token_is_eog, 2},
  {"c_r_token_is_control", (DL_FUNC) &r_token_is_control, 2},
  
  // Test function
  {"c_r_tokenize_test", (DL_FUNC) &r_tokenize_test, 1},
  
  {NULL, NULL, 0}
};

// 包初始化函数
extern "C" void R_init_newrllama4(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, TRUE);
  R_forceSymbols(dll, FALSE);
} 