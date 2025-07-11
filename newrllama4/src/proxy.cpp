#include "proxy.h"
#include <stdexcept>
#include <dlfcn.h>
#include <cstring>
#include <string>

// 定义全局的函数指针实例
struct newrllama_api_ptrs newrllama_api;

// 宏定义来简化符号加载过程
#define LOAD_SYMBOL(handle, F) \
    *(void**)(&newrllama_api.F) = dlsym(handle, "newrllama_" #F); \
    if (newrllama_api.F == NULL) { \
        /* Try with underscore prefix (macOS) */ \
        *(void**)(&newrllama_api.F) = dlsym(handle, "_newrllama_" #F); \
    } \
    if (newrllama_api.F == NULL) { \
        const char* error = dlerror(); \
        throw std::runtime_error(std::string("Failed to load symbol: newrllama_" #F) + \
                                (error ? std::string(" - ") + error : "")); \
    }

// 初始化函数实现
bool newrllama_api_init(void* handle) {
    try {
        // 加载核心函数
        LOAD_SYMBOL(handle, backend_init);
        LOAD_SYMBOL(handle, backend_free);
        LOAD_SYMBOL(handle, model_load);
        LOAD_SYMBOL(handle, model_free);
        LOAD_SYMBOL(handle, context_create);
        LOAD_SYMBOL(handle, context_free);
        
        // 加载文本处理函数
        LOAD_SYMBOL(handle, tokenize);
        LOAD_SYMBOL(handle, detokenize);
        LOAD_SYMBOL(handle, apply_chat_template);
        LOAD_SYMBOL(handle, generate);
        LOAD_SYMBOL(handle, generate_parallel);
        
        // 加载内存管理函数
        LOAD_SYMBOL(handle, free_tokens);
        LOAD_SYMBOL(handle, free_string);
        LOAD_SYMBOL(handle, free_string_array);
        
        // 加载token函数
        LOAD_SYMBOL(handle, token_get_text);
        LOAD_SYMBOL(handle, token_bos);
        LOAD_SYMBOL(handle, token_eos);
        LOAD_SYMBOL(handle, token_sep);
        LOAD_SYMBOL(handle, token_nl);
        LOAD_SYMBOL(handle, token_pad);
        LOAD_SYMBOL(handle, token_eot);
        LOAD_SYMBOL(handle, add_bos_token);
        LOAD_SYMBOL(handle, add_eos_token);
        LOAD_SYMBOL(handle, token_fim_pre);
        LOAD_SYMBOL(handle, token_fim_mid);
        LOAD_SYMBOL(handle, token_fim_suf);
        LOAD_SYMBOL(handle, token_get_attr);
        LOAD_SYMBOL(handle, token_get_score);
        LOAD_SYMBOL(handle, token_is_eog);
        LOAD_SYMBOL(handle, token_is_control);
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

// 检查符号是否已加载
bool newrllama_api_is_loaded() {
    // 检查几个关键函数是否已加载
    return (newrllama_api.backend_init != nullptr && 
            newrllama_api.model_load != nullptr && 
            newrllama_api.context_create != nullptr);
}

// 重置所有函数指针（用于卸载时清理）
void newrllama_api_reset() {
    memset(&newrllama_api, 0, sizeof(newrllama_api));
} 