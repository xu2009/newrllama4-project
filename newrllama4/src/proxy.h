// --- FILE: newrllama4/src/proxy.h ---
#pragma once
#include "newrllama_capi.h"

// 定义一个结构体来存放所有C-API的函数指针
struct newrllama_api_ptrs {
    // Core functions
    decltype(&newrllama_backend_init) backend_init;
    decltype(&newrllama_backend_free) backend_free;
    decltype(&newrllama_model_load) model_load;
    decltype(&newrllama_model_load_safe) model_load_safe;
    decltype(&newrllama_model_free) model_free;
    decltype(&newrllama_context_create) context_create;
    decltype(&newrllama_context_free) context_free;
    
    // Text processing functions
    decltype(&newrllama_tokenize) tokenize;
    decltype(&newrllama_detokenize) detokenize;
    decltype(&newrllama_apply_chat_template) apply_chat_template;
    decltype(&newrllama_generate) generate;
    decltype(&newrllama_generate_parallel) generate_parallel;
    
    // Memory management functions
    decltype(&newrllama_free_tokens) free_tokens;
    decltype(&newrllama_free_string) free_string;
    decltype(&newrllama_free_string_array) free_string_array;
    
    // Token functions
    decltype(&newrllama_token_get_text) token_get_text;
    decltype(&newrllama_token_bos) token_bos;
    decltype(&newrllama_token_eos) token_eos;
    decltype(&newrllama_token_sep) token_sep;
    decltype(&newrllama_token_nl) token_nl;
    decltype(&newrllama_token_pad) token_pad;
    decltype(&newrllama_token_eot) token_eot;
    decltype(&newrllama_add_bos_token) add_bos_token;
    decltype(&newrllama_add_eos_token) add_eos_token;
    decltype(&newrllama_token_fim_pre) token_fim_pre;
    decltype(&newrllama_token_fim_mid) token_fim_mid;
    decltype(&newrllama_token_fim_suf) token_fim_suf;
    decltype(&newrllama_token_get_attr) token_get_attr;
    decltype(&newrllama_token_get_score) token_get_score;
    decltype(&newrllama_token_is_eog) token_is_eog;
    decltype(&newrllama_token_is_control) token_is_control;
    
    // Model download functions
    decltype(&newrllama_download_model) download_model;
    decltype(&newrllama_resolve_model) resolve_model;
    
    // Memory checking functions
    decltype(&newrllama_estimate_model_memory) estimate_model_memory;
    decltype(&newrllama_check_memory_available) check_memory_available;
};

// 声明一个全局的函数指针结构体实例
extern struct newrllama_api_ptrs newrllama_api;

// 声明一个初始化函数，用于在R中加载符号
bool newrllama_api_init(void* handle);

// 声明一个检查函数，用于确保符号已加载
bool newrllama_api_is_loaded();

// 声明一个重置函数，用于清理函数指针
void newrllama_api_reset(); 