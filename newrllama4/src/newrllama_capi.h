#ifndef NEWRLLAMA_CAPI_H
#define NEWRLLAMA_CAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #ifdef NEWRLLAMA_BUILD_DLL
    #define NEWRLLAMA_API __declspec(dllexport)
  #else
    #define NEWRLLAMA_API __declspec(dllimport)
  #endif
#else
  #define NEWRLLAMA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_model*  newrllama_model_handle;
typedef struct llama_context* newrllama_context_handle;
typedef enum { NEWRLLAMA_SUCCESS = 0, NEWRLLAMA_ERROR = 1 } newrllama_error_code;
struct newrllama_chat_message { const char* role; const char* content; };
struct newrllama_parallel_params { int max_tokens; int top_k; float top_p; float temperature; int repeat_last_n; float penalty_repeat; int32_t seed; };

NEWRLLAMA_API newrllama_error_code newrllama_backend_init(const char** error_message);
NEWRLLAMA_API void newrllama_backend_free();
NEWRLLAMA_API newrllama_error_code newrllama_model_load(const char* model_path, int n_gpu_layers, bool use_mmap, bool use_mlock, newrllama_model_handle* model_handle_out, const char** error_message);
NEWRLLAMA_API void newrllama_model_free(newrllama_model_handle model);
NEWRLLAMA_API newrllama_error_code newrllama_context_create(newrllama_model_handle model, int n_ctx, int n_threads, int n_seq_max, newrllama_context_handle* context_handle_out, const char** error_message);
NEWRLLAMA_API void newrllama_context_free(newrllama_context_handle ctx);
NEWRLLAMA_API newrllama_error_code newrllama_tokenize(newrllama_model_handle model, const char* text, bool add_special, int32_t** tokens_out, size_t* n_tokens_out, const char** error_message);
NEWRLLAMA_API newrllama_error_code newrllama_detokenize(newrllama_model_handle model, const int32_t* tokens, size_t n_tokens, char** text_out, const char** error_message);
NEWRLLAMA_API void newrllama_free_string(char* str);
NEWRLLAMA_API void newrllama_free_tokens(int32_t* tokens);
NEWRLLAMA_API newrllama_error_code newrllama_apply_chat_template(newrllama_model_handle model, const char* tmpl, const struct newrllama_chat_message* messages, size_t n_messages, bool add_ass, char** result_out, const char** error_message);
NEWRLLAMA_API newrllama_error_code newrllama_generate(newrllama_context_handle ctx, const int32_t* tokens_in, size_t n_tokens_in, int max_tokens, int top_k, float top_p, float temperature, int repeat_last_n, float penalty_repeat, int32_t seed, char** result_out, const char** error_message);
NEWRLLAMA_API newrllama_error_code newrllama_generate_parallel(newrllama_context_handle ctx, const char** prompts, int n_prompts, const struct newrllama_parallel_params* params, char*** results_out, const char** error_message);
NEWRLLAMA_API void newrllama_free_string_array(char** arr, int count);
NEWRLLAMA_API newrllama_error_code newrllama_token_get_text(newrllama_model_handle model, int32_t token, char** text_out, const char** error_message);
NEWRLLAMA_API float newrllama_token_get_score(newrllama_model_handle model, int32_t token);
NEWRLLAMA_API int newrllama_token_get_attr(newrllama_model_handle model, int32_t token);
NEWRLLAMA_API bool newrllama_token_is_eog(newrllama_model_handle model, int32_t token);
NEWRLLAMA_API bool newrllama_token_is_control(newrllama_model_handle model, int32_t token);
NEWRLLAMA_API int32_t newrllama_token_bos(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_eos(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_sep(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_nl(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_pad(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_eot(newrllama_model_handle model);
NEWRLLAMA_API bool newrllama_add_bos_token(newrllama_model_handle model);
NEWRLLAMA_API bool newrllama_add_eos_token(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_fim_pre(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_fim_mid(newrllama_model_handle model);
NEWRLLAMA_API int32_t newrllama_token_fim_suf(newrllama_model_handle model);

#ifdef __cplusplus
}
#endif
#endif 