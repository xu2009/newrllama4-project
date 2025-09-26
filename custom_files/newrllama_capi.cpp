#define NEWRLLAMA_BUILD_DLL
#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#define _CRT_SECURE_NO_WARNINGS  // Disable MSVC security warnings
#include <windows.h>
#endif
#include "newrllama_capi.h"
#include "llama.h"
#include "ggml.h"
#include "common/common.h"
#include "common/sampling.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdio>
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

static thread_local std::string last_error_message;

void set_error(const char** error_message, const std::string& msg) { 
    if (error_message) { 
        last_error_message = msg; 
        *error_message = last_error_message.c_str(); 
    } 
}

// Verbosity-controlled log callback
static thread_local int current_verbosity = 1; // Default verbosity level
static thread_local ggml_log_callback original_log_callback = nullptr;
static thread_local void* original_log_user_data = nullptr;

static void verbosity_log_callback(ggml_log_level level, const char* text, void* user_data) {
    int verbosity = current_verbosity;
    
    // Filter messages based on verbosity level (3=most, 2,1,0=decreasing)
    bool should_log = false;
    switch (verbosity) {
        case 3: // Show all (DEBUG + INFO + WARN + ERROR) - Most verbose
            should_log = true;
            break;
        case 2: // Show important info (INFO + WARN + ERROR)
            should_log = (level >= GGML_LOG_LEVEL_INFO);
            break;
        case 1: // Show warnings and errors only (WARN + ERROR) - Default
            should_log = (level >= GGML_LOG_LEVEL_WARN);
            break;
        case 0: // Show errors only - Least verbose
            should_log = (level >= GGML_LOG_LEVEL_ERROR);
            break;
        default:
            should_log = (level >= GGML_LOG_LEVEL_WARN); // Fallback to level 1 (default)
            break;
    }
    
    if (should_log) {
        if (original_log_callback) {
            original_log_callback(level, text, original_log_user_data);
        } else {
            // Default behavior: print to stderr
            fprintf(stderr, "%s", text);
            fflush(stderr);
        }
    }
}

// Set verbosity for logging
static void set_log_verbosity(int verbosity) {
    current_verbosity = verbosity;
    
    // Store original callback if not already stored
    if (original_log_callback == nullptr) {
        // Note: We can't easily get the current callback, so we'll assume default behavior
        original_log_callback = nullptr; // Will use default stderr output
        original_log_user_data = nullptr;
    }
    
    // Set our custom callback
    llama_log_set(verbosity_log_callback, nullptr);
}

// Restore original logging
static void restore_log_callback() {
    if (original_log_callback) {
        llama_log_set(original_log_callback, original_log_user_data);
    } else {
        llama_log_set(nullptr, nullptr); // Restore default behavior
    }
}

static char* string_to_c_str(const std::string& s) { 
    char* cstr = new char[s.length() + 1]; 
    std::strcpy(cstr, s.c_str()); 
    return cstr; 
}

static std::vector<llama_token> helper_tokenize(const llama_model* model, const std::string& text, bool add_special) { 
    const struct llama_vocab* vocab = llama_model_get_vocab(model); 
    int max_tokens = text.size() + 2; 
    std::vector<llama_token> tokens(max_tokens); 
    int32_t n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), max_tokens, add_special, false); 
    if (n < 0) { 
        throw std::runtime_error("Tokenization failed in helper."); 
    } 
    tokens.resize(n); 
    return tokens; 
}

NEWRLLAMA_API newrllama_error_code newrllama_backend_init(const char** error_message) { 
    try { 
        ggml_backend_load_all(); 
        llama_backend_init(); 
        return NEWRLLAMA_SUCCESS; 
    } catch (const std::exception& e) { 
        set_error(error_message, std::string("Backend init failed: ") + e.what()); 
        return NEWRLLAMA_ERROR; 
    } 
}

NEWRLLAMA_API void newrllama_backend_free() { 
    llama_backend_free(); 
}

NEWRLLAMA_API newrllama_error_code newrllama_model_load(const char* model_path, int n_gpu_layers, bool use_mmap, bool use_mlock, newrllama_model_handle* model_handle_out, const char** error_message) { 
    llama_model_params model_params = llama_model_default_params(); 
    model_params.n_gpu_layers = n_gpu_layers; 
    model_params.use_mmap = use_mmap; 
    model_params.use_mlock = use_mlock; 
    llama_model* model = llama_model_load_from_file(model_path, model_params); 
    if (model == nullptr) { 
        set_error(error_message, std::string("Failed to load model from path: ") + model_path); 
        return NEWRLLAMA_ERROR; 
    } 
    *model_handle_out = model; 
    return NEWRLLAMA_SUCCESS; 
}

// Enhanced model loading with memory checking
NEWRLLAMA_API newrllama_error_code newrllama_model_load_safe(const char* model_path, int n_gpu_layers, bool use_mmap, bool use_mlock, bool check_memory, int verbosity, newrllama_model_handle* model_handle_out, const char** error_message) {
    // Set verbosity level for this function call
    set_log_verbosity(verbosity);
    
    try {
        // Check if file exists and is valid
        std::ifstream file(model_path, std::ios::binary);
        if (!file.is_open()) {
            restore_log_callback();
            set_error(error_message, std::string("Cannot open model file: ") + model_path);
            return NEWRLLAMA_ERROR;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        
        // Check GGUF magic number
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
            restore_log_callback();
            set_error(error_message, "Invalid GGUF file format");
            return NEWRLLAMA_ERROR;
        }
        file.close();
        
        // Estimate memory requirements if requested
        if (check_memory) {
            size_t estimated_memory = static_cast<size_t>(file_size * 1.5); // Conservative estimate
            if (use_mmap) {
                estimated_memory = static_cast<size_t>(file_size * 0.1); // Much less with mmap
            }
            
            bool memory_ok = newrllama_check_memory_available(estimated_memory, error_message);
            if (!memory_ok) {
                restore_log_callback();
                set_error(error_message, "Insufficient memory for model loading");
                return NEWRLLAMA_ERROR;
            }
        }
        
        // Load the model with enhanced error handling
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        model_params.use_mmap = use_mmap;
        model_params.use_mlock = use_mlock;
        
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (model == nullptr) {
            restore_log_callback();
            set_error(error_message, std::string("Failed to load model from path: ") + model_path + ". This may be due to insufficient memory, corrupted file, or unsupported model format.");
            return NEWRLLAMA_ERROR;
        }
        
        *model_handle_out = model;
        restore_log_callback();
        return NEWRLLAMA_SUCCESS;
        
    } catch (const std::exception& e) {
        restore_log_callback();
        set_error(error_message, std::string("Exception during model loading: ") + e.what());
        return NEWRLLAMA_ERROR;
    } catch (...) {
        restore_log_callback();
        set_error(error_message, "Unknown exception during model loading");
        return NEWRLLAMA_ERROR;
    }
}

NEWRLLAMA_API void newrllama_model_free(newrllama_model_handle model) { 
    if (model) llama_model_free(model); 
}

NEWRLLAMA_API newrllama_error_code newrllama_context_create(newrllama_model_handle model, int n_ctx, int n_threads, int n_seq_max, int verbosity, newrllama_context_handle* context_handle_out, const char** error_message) { 
    // Set verbosity level for this function call
    set_log_verbosity(verbosity);
    
    if (!model) { 
        restore_log_callback();
        set_error(error_message, "Model handle is null."); 
        return NEWRLLAMA_ERROR; 
    } 
    llama_context_params ctx_params = llama_context_default_params(); 
    ctx_params.n_ctx = n_ctx; 
    ctx_params.n_threads = n_threads; 
    ctx_params.n_seq_max = n_seq_max; 
    llama_context* ctx = llama_init_from_model(model, ctx_params); 
    if (ctx == nullptr) { 
        restore_log_callback();
        set_error(error_message, "Failed to create context from model."); 
        return NEWRLLAMA_ERROR; 
    } 
    *context_handle_out = ctx; 
    restore_log_callback();
    return NEWRLLAMA_SUCCESS; 
}

NEWRLLAMA_API void newrllama_context_free(newrllama_context_handle ctx) { 
    if (ctx) llama_free(ctx); 
}

NEWRLLAMA_API newrllama_error_code newrllama_tokenize(newrllama_model_handle model, const char* text, bool add_special, int32_t** tokens_out, size_t* n_tokens_out, const char** error_message) { 
    try { 
        std::vector<llama_token> tokens_vec = helper_tokenize(model, std::string(text), add_special); 
        *n_tokens_out = tokens_vec.size(); 
        *tokens_out = new int32_t[*n_tokens_out]; 
        std::copy(tokens_vec.begin(), tokens_vec.end(), *tokens_out); 
        return NEWRLLAMA_SUCCESS; 
    } catch (const std::exception& e) { 
        set_error(error_message, e.what()); 
        return NEWRLLAMA_ERROR; 
    } 
}

NEWRLLAMA_API newrllama_error_code newrllama_detokenize(newrllama_model_handle model, const int32_t* tokens, size_t n_tokens, char** text_out, const char** error_message) { 
    const struct llama_vocab* vocab = llama_model_get_vocab(model); 
    size_t max_len = n_tokens * 8 + 1; 
    std::vector<char> buf(max_len); 
    int n_chars = llama_detokenize(vocab, tokens, n_tokens, buf.data(), buf.size(), false, false); 
    if (n_chars < 0) { 
        set_error(error_message, "Detokenization failed."); 
        return NEWRLLAMA_ERROR; 
    } 
    *text_out = string_to_c_str(std::string(buf.data(), n_chars)); 
    return NEWRLLAMA_SUCCESS; 
}

NEWRLLAMA_API void newrllama_free_string(char* str) { 
    if(str) delete[] str; 
}

NEWRLLAMA_API void newrllama_free_tokens(int32_t* tokens) { 
    if(tokens) delete[] tokens; 
}

NEWRLLAMA_API newrllama_error_code newrllama_apply_chat_template(newrllama_model_handle model, const char* tmpl, const struct newrllama_chat_message* messages_in, size_t n_messages, bool add_ass, char** result_out, const char** error_message) { 
    if (!model) {
        set_error(error_message, "Model handle is null.");
        return NEWRLLAMA_ERROR;
    }

    std::vector<llama_chat_message> messages_vec(n_messages); 
    size_t total_length = 0; 
    for(size_t i = 0; i < n_messages; ++i) { 
        messages_vec[i] = {messages_in[i].role, messages_in[i].content}; 
        total_length += (messages_in[i].content ? strlen(messages_in[i].content) : 0); 
    } 
    size_t buffer_size = total_length * 2 + 2048; 
    std::vector<char> buffer(buffer_size, 0); 
    
    // 🔧 修复：正确的两步法实现模型内置template支持
    // 第一步：确定使用哪个template
    const char* effective_tmpl = tmpl; // 用户提供的template
    if (!effective_tmpl) {
        // 如果用户没有提供template，从模型获取默认template
        effective_tmpl = llama_model_chat_template(model, nullptr);
    }

    // 第二步：调用llama_chat_apply_template（标准6参数版本）
    int32_t res = llama_chat_apply_template(effective_tmpl, messages_vec.data(), n_messages, add_ass, buffer.data(), buffer.size()); 
    
    if (res < 0) { 
        // 提供更详细的错误信息
        std::string error_msg = "Failed to apply chat template. Error code: " + std::to_string(res);
        if (res == -1) {
            error_msg += " (template not found or invalid)";
        } else if (res == -2) {
            error_msg += " (buffer too small)";
        }
        if (tmpl) {
            error_msg += ". Custom template used: " + std::string(tmpl, 0, 100) + "...";
        } else {
            error_msg += ". Using model's built-in template.";
        }
        set_error(error_message, error_msg); 
        return NEWRLLAMA_ERROR; 
    } 
    
    *result_out = string_to_c_str(std::string(buffer.data(), res)); 
    return NEWRLLAMA_SUCCESS; 
}

NEWRLLAMA_API newrllama_error_code newrllama_generate(newrllama_context_handle ctx, const int32_t* tokens_in, size_t n_tokens_in, int max_tokens, int top_k, float top_p, float temperature, int repeat_last_n, float penalty_repeat, int32_t seed, char** result_out, const char** error_message) { 
    if (!ctx) { 
        set_error(error_message, "Context handle is null."); 
        return NEWRLLAMA_ERROR; 
    } 
    
    // 🔧 修复：清除KV缓存确保可重复性
    llama_kv_self_clear(ctx);
    
    const llama_model* model = llama_get_model(ctx); 
    const struct llama_vocab* vocab = llama_model_get_vocab(model); 
    llama_token eos_token = llama_vocab_eos(vocab); 
    llama_batch batch = llama_batch_get_one((llama_token*)tokens_in, n_tokens_in); 
    if (llama_decode(ctx, batch) != 0) { 
        set_error(error_message, "Failed to decode input tokens."); 
        return NEWRLLAMA_ERROR; 
    } 
    
    // 🔧 修复：使用与官方examples一致的采样器链配置
    struct llama_sampler_chain_params sparams_chain = llama_sampler_chain_default_params(); 
    struct llama_sampler* sampler_chain = llama_sampler_chain_init(sparams_chain); 
    
    // 按照官方batched.cpp的标准顺序和参数配置采样器
    // 注意：移除了penalties采样器，因为官方标准不包含此采样器
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(top_k)); 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(top_p, 1)); // min_keep=1是标准值
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(temperature)); 
    uint32_t final_seed = (seed < 0) ? time(NULL) : seed; 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(final_seed)); 
    
    std::string generated_text; 
    std::vector<llama_token> recent_tokens; // 维护token历史用于序列检测
    
    for (int i = 0; i < max_tokens; ++i) { 
        llama_token new_token = llama_sampler_sample(sampler_chain, ctx, -1); 
        llama_sampler_accept(sampler_chain, new_token); 
        
        // 🔧 修复：结合标准EOG检测 + multi-token序列检测
        
        // 1. 标准单token EOG检测（保留官方逻辑）
        if (llama_vocab_is_eog(vocab, new_token)) {
            break; // 直接停止生成，不添加到输出
        }
        
        // 2. Multi-token EOG序列检测（基于诊断脚本的准确token序列）
        recent_tokens.push_back(new_token);
        if (recent_tokens.size() > 7) {  // 保持最近7个tokens的窗口
            recent_tokens.erase(recent_tokens.begin());
        }
        
        // 检测完整的EOG序列，在它们完整形成时停止（预检测）
        if (recent_tokens.size() >= 7) {
            // <|eot_id|> 序列: [27, 91, 68, 354, 851, 91, 29]
            std::vector<llama_token> eot_sequence = {27, 91, 68, 354, 851, 91, 29};
            bool is_eot_complete = true;
            for (size_t j = 0; j < 7; ++j) {
                if (recent_tokens[recent_tokens.size() - 7 + j] != eot_sequence[j]) {
                    is_eot_complete = false;
                    break;
                }
            }
            
            if (is_eot_complete) {
                // 序列已完成，移除已添加的前6个tokens并停止
                std::string tokens_to_remove;
                for (size_t k = 0; k < 6; ++k) { // 移除前6个tokens
                    llama_token tok = recent_tokens[recent_tokens.size() - 7 + k];
                    tokens_to_remove += common_token_to_piece(ctx, tok);
                }
                // 安全移除
                if (generated_text.length() >= tokens_to_remove.length() && 
                    generated_text.substr(generated_text.length() - tokens_to_remove.length()) == tokens_to_remove) {
                    generated_text.erase(generated_text.length() - tokens_to_remove.length());
                }
                break; // 停止生成，不添加第7个token
            }
            
            // <|end_header_id|> 序列: [27, 91, 408, 8932, 851, 91, 29]  
            std::vector<llama_token> end_header_sequence = {27, 91, 408, 8932, 851, 91, 29};
            bool is_end_header_complete = true;
            for (size_t j = 0; j < 7; ++j) {
                if (recent_tokens[recent_tokens.size() - 7 + j] != end_header_sequence[j]) {
                    is_end_header_complete = false;
                    break;
                }
            }
            
            if (is_end_header_complete) {
                // 序列已完成，移除已添加的前6个tokens并停止  
                std::string tokens_to_remove;
                for (size_t k = 0; k < 6; ++k) {
                    llama_token tok = recent_tokens[recent_tokens.size() - 7 + k];
                    tokens_to_remove += common_token_to_piece(ctx, tok);
                }
                if (generated_text.length() >= tokens_to_remove.length() && 
                    generated_text.substr(generated_text.length() - tokens_to_remove.length()) == tokens_to_remove) {
                    generated_text.erase(generated_text.length() - tokens_to_remove.length());
                }
                break; // 停止生成
            }
        }
        
        // 只有非EOG token才添加到输出
        const std::string token_str = common_token_to_piece(ctx, new_token);
        generated_text += token_str;
        
        llama_batch next_batch = llama_batch_get_one(&new_token, 1); 
        if (llama_decode(ctx, next_batch) != 0) { 
            llama_sampler_free(sampler_chain); 
            set_error(error_message, "Failed to decode generated token."); 
            return NEWRLLAMA_ERROR; 
        } 
    }
    
    llama_sampler_free(sampler_chain); 
    *result_out = string_to_c_str(generated_text); 
    return NEWRLLAMA_SUCCESS; 
}

NEWRLLAMA_API newrllama_error_code newrllama_generate_parallel(
    newrllama_context_handle ctx,
    const char** prompts,
    int n_prompts,
    const struct newrllama_parallel_params* params,
    char*** results_out,
    const char** error_message) {

    if (!ctx || !prompts || !params || !results_out || n_prompts <= 0) {
        set_error(error_message, "Invalid parameters: null pointers or invalid prompt count");
        return NEWRLLAMA_ERROR;
    }

    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_token eos_token = llama_vocab_eos(vocab);
    const int n_ctx = llama_n_ctx(ctx);
    const int seq_cap_raw = (int)llama_n_seq_max(ctx);
    const int seq_capacity = std::max(1, seq_cap_raw > 0 ? seq_cap_raw : 1);
    const int32_t batch_cap_init = std::max<int32_t>(1, std::min<int32_t>(512, (int32_t)llama_n_batch(ctx)));
    const int total_prompts = n_prompts;
    const bool show_progress_bar = params->show_progress;
    int prompts_completed = 0;
    int spinner_index = 0;
    const char spinner_chars[] = "|/-\\";

    const auto t_start = ggml_time_us();
    int32_t n_total_prompt = 0;
    int32_t n_total_gen = 0;
    int32_t dynamic_cache_miss = 0;

    std::vector<std::string> final_responses(n_prompts);

    // 预先分词，寻找公共前缀
    std::vector<std::vector<llama_token>> prompt_tokens_all(n_prompts);
    size_t shared_prefix_len = 0;
    for (int i = 0; i < n_prompts; ++i) {
        prompt_tokens_all[i] = helper_tokenize(model, std::string(prompts[i]), true);
        if (i == 0) {
            shared_prefix_len = prompt_tokens_all[i].size();
        } else {
            size_t common = 0;
            size_t limit = std::min(shared_prefix_len, prompt_tokens_all[i].size());
            while (common < limit && prompt_tokens_all[i][common] == prompt_tokens_all[0][common]) {
                ++common;
            }
            shared_prefix_len = common;
        }
    }

    llama_kv_self_clear(ctx);
    bool prefix_ready = false;
    std::vector<llama_token> shared_prefix_tokens;
    if (shared_prefix_len > 0) {
        shared_prefix_tokens.assign(prompt_tokens_all[0].begin(), prompt_tokens_all[0].begin() + shared_prefix_len);
        llama_batch prefix_batch = llama_batch_init(shared_prefix_tokens.size(), 0, 1);
        for (size_t j = 0; j < shared_prefix_tokens.size(); ++j) {
            common_batch_add(prefix_batch, shared_prefix_tokens[j], j, {0}, j == shared_prefix_tokens.size() - 1);
        }

        int32_t local_cap = batch_cap_init;
        bool prefix_ok = true;
        for (int32_t start = 0; start < (int32_t)prefix_batch.n_tokens; ) {
            int32_t n_tokens = std::min(local_cap, (int32_t)(prefix_batch.n_tokens - start));
            llama_batch view = {
                n_tokens,
                prefix_batch.token + start,
                nullptr,
                prefix_batch.pos + start,
                prefix_batch.n_seq_id + start,
                prefix_batch.seq_id + start,
                prefix_batch.logits + start,
            };

            int ret = llama_decode(ctx, view);
            if (ret != 0) {
                if (ret > 0 && local_cap > 1) {
                    local_cap = std::max<int32_t>(1, local_cap / 2);
                    dynamic_cache_miss++;
                    continue;
                }
                prefix_ok = false;
                break;
            }
            start += n_tokens;
        }
        llama_batch_free(prefix_batch);

        if (!prefix_ok) {
            llama_kv_self_clear(ctx);
        } else {
            prefix_ready = true;
        }
    }

    struct Slot {
        bool active = false;
        bool finished = false;
        bool failed = false;
        llama_seq_id seq_id = 0;
        int global_index = -1;
        const std::vector<llama_token>* full_tokens = nullptr;
        std::vector<llama_token> suffix_tokens;
        int32_t prefix_len = 0;
        int32_t n_past = 0;
        int32_t n_prompt = 0;
        int32_t n_decoded = 0;
        int32_t i_batch = -1;
        llama_token sampled = 0;
        common_sampler* smpl = nullptr;
        std::string response;
        std::string error_msg;
    };

    auto clean_response_text = [](std::string text) {
        const std::vector<std::string> stop_markers = {
            "<|im_end|>", "<|im_start|>", "<end_of_turn>", "<start_of_turn>",
            "</s>", "<s>", "<|endoftext|>", "<|end|>", "<|start|>",
            "<eos>", "<bos>", "\n<|im_end|>", "\n<end_of_turn>", "\n</s>"
        };

        bool found_marker = true;
        int cleanup_rounds = 0;
        while (found_marker && cleanup_rounds < 5) {
            found_marker = false;
            cleanup_rounds++;
            for (const auto& marker : stop_markers) {
                size_t pos = 0;
                while ((pos = text.find(marker, pos)) != std::string::npos) {
                    text.erase(pos, marker.length());
                    found_marker = true;
                }
            }
        }

        while (!text.empty() && (text.front() == '?' || text.front() < 32 || text.front() > 126)) {
            text = text.substr(1);
        }
        while (!text.empty() && isspace(text.back())) {
            text.pop_back();
        }
        while (!text.empty() && isspace(text.front())) {
            text = text.substr(1);
        }

        size_t pos = text.find("\n\nUser:");
        if (pos != std::string::npos) {
            text = text.substr(0, pos);
        }
        return text;
    };

    auto release_slot = [](Slot& slot) {
        if (slot.smpl) {
            common_sampler_free(slot.smpl);
            slot.smpl = nullptr;
        }
        slot.active = false;
        slot.finished = false;
        slot.failed = false;
        slot.seq_id = 0;
        slot.global_index = -1;
        slot.full_tokens = nullptr;
        slot.suffix_tokens.clear();
        slot.prefix_len = 0;
        slot.n_past = 0;
        slot.n_prompt = 0;
        slot.n_decoded = 0;
        slot.i_batch = -1;
        slot.sampled = 0;
        slot.response.clear();
        slot.error_msg.clear();
    };

    common_params_sampling sparams{};
    sparams.top_k = params->top_k;
    sparams.top_p = params->top_p;
    sparams.temp = params->temperature;
    sparams.penalty_last_n = params->repeat_last_n;
    sparams.penalty_repeat = params->penalty_repeat;
    sparams.seed = (params->seed < 0) ? time(nullptr) : params->seed;

    auto decode_prompt_tokens = [&](Slot& slot) -> bool {
        if (slot.n_prompt <= 0) {
            slot.failed = true;
            slot.error_msg = "Prompt resulted in zero tokens";
            return false;
        }

        if (slot.suffix_tokens.empty()) {
            slot.n_past = slot.n_prompt;
            return true;
        }

        llama_batch batch = llama_batch_init(slot.suffix_tokens.size(), 0, 1);
        for (size_t j = 0; j < slot.suffix_tokens.size(); ++j) {
            const int position = slot.prefix_len + static_cast<int>(j);
            common_batch_add(batch, slot.suffix_tokens[j], position, {slot.seq_id}, j == slot.suffix_tokens.size() - 1);
        }

        int32_t local_cap = batch_cap_init;
        for (int32_t start = 0; start < (int32_t)batch.n_tokens;) {
            int32_t n_tokens = std::min(local_cap, (int32_t)(batch.n_tokens - start));
            llama_batch view = {
                n_tokens,
                batch.token + start,
                nullptr,
                batch.pos + start,
                batch.n_seq_id + start,
                batch.seq_id + start,
                batch.logits + start,
            };

            int ret = llama_decode(ctx, view);
            if (ret == 0) {
                start += n_tokens;
                continue;
            }

            if (ret > 0 && local_cap > 1) {
                local_cap = std::max<int32_t>(1, local_cap / 2);
                dynamic_cache_miss++;
                continue;
            }

            llama_batch_free(batch);
            slot.failed = true;
            slot.error_msg = "Failed to decode prompt tokens";
            return false;
        }

        slot.n_past = slot.n_prompt;
        llama_batch_free(batch);
        return true;
    };

    std::vector<Slot> slots(seq_capacity);
    size_t next_prompt_idx = 0;
    int active_clients = 0;
    std::vector<int> slots_pending_reassign;
    slots_pending_reassign.reserve(seq_capacity);

    auto finalize_slot = [&](int slot_idx, bool success) {
        Slot& slot = slots[slot_idx];
        if (!slot.active) {
            release_slot(slot);
            return;
        }

        if (slot.seq_id > 0) {
            llama_kv_self_seq_rm(ctx, slot.seq_id, 0, -1);
        }

        if (success) {
            final_responses[slot.global_index] = clean_response_text(slot.response);
            n_total_gen += slot.n_decoded;
        } else {
            final_responses[slot.global_index] = "[ERROR] " + (slot.error_msg.empty() ? std::string("Unknown error") : slot.error_msg);
        }

        slot.active = false;
        active_clients--;
        bool needs_reassign = next_prompt_idx < (size_t)n_prompts;
        release_slot(slot);
        if (show_progress_bar) {
            prompts_completed++;
            float percent = (float)prompts_completed / std::max(total_prompts, 1);
            if (percent > 1.0f) {
                percent = 1.0f;
            }
            const int bar_width = 30;
            int filled = (int)(percent * bar_width);
            if (filled > bar_width) {
                filled = bar_width;
            }
            std::string bar(filled, '=');
            bar.append(bar_width - filled, ' ');
            char spinner = spinner_chars[spinner_index];
            spinner_index = (spinner_index + 1) % 4;
            fprintf(stderr, "\r %c [%s] %d/%d (%3.0f%%)", spinner, bar.c_str(), prompts_completed, total_prompts, percent * 100.0f);
            fflush(stderr);
        }
        if (needs_reassign) {
            slots_pending_reassign.push_back(slot_idx);
        }
    };

    auto assign_next_prompt = [&](int slot_idx) -> bool {
        Slot& slot = slots[slot_idx];
        release_slot(slot);

        while (next_prompt_idx < (size_t)n_prompts) {
            const size_t global_idx = next_prompt_idx++;
            slot.seq_id = static_cast<llama_seq_id>(slot_idx + 1);
            slot.global_index = static_cast<int>(global_idx);
            slot.full_tokens = &prompt_tokens_all[global_idx];
            slot.n_prompt = static_cast<int32_t>(slot.full_tokens->size());
            slot.prefix_len = prefix_ready ? std::min<int32_t>((int32_t)shared_prefix_len, slot.n_prompt) : 0;
            slot.suffix_tokens.clear();
            if (slot.n_prompt > slot.prefix_len) {
                slot.suffix_tokens.assign(slot.full_tokens->begin() + slot.prefix_len, slot.full_tokens->end());
            }
            slot.n_past = slot.prefix_len;
            slot.n_decoded = 0;
            slot.i_batch = -1;
            slot.failed = false;
            slot.response.clear();
            slot.error_msg.clear();

            slot.smpl = common_sampler_init(model, sparams);
            if (!slot.smpl) {
                slot.error_msg = "Failed to initialize sampler";
                final_responses[slot.global_index] = "[ERROR] " + slot.error_msg;
                release_slot(slot);
                continue;
            }

            if (slot.n_prompt > n_ctx - 64) {
                slot.error_msg = "Prompt too long for context size";
                final_responses[slot.global_index] = "[ERROR] " + slot.error_msg;
                release_slot(slot);
                continue;
            }

            if (slot.n_prompt == 0) {
                slot.error_msg = "Prompt resulted in zero tokens";
                final_responses[slot.global_index] = "[ERROR] " + slot.error_msg;
                release_slot(slot);
                continue;
            }

            n_total_prompt += slot.n_prompt;

            if (prefix_ready && slot.prefix_len > 0) {
                llama_kv_self_seq_cp(ctx, 0, slot.seq_id, -1, -1);
            }

            if (!decode_prompt_tokens(slot)) {
                if (slot.seq_id > 0) {
                    llama_kv_self_seq_rm(ctx, slot.seq_id, 0, -1);
                }
                final_responses[slot.global_index] = "[ERROR] " + slot.error_msg;
                release_slot(slot);
                continue;
            }

            slot.sampled = slot.full_tokens->empty() ? 0 : slot.full_tokens->back();
            slot.active = true;
            active_clients++;
            return true;
        }

        release_slot(slot);
        return false;
    };

    auto ensure_slots_filled = [&]() {
        for (int i = 0; i < seq_capacity && next_prompt_idx < (size_t)n_prompts; ++i) {
            if (!slots[i].active) {
                assign_next_prompt(i);
            }
        }
    };

    try {
        ensure_slots_filled();

        while (active_clients > 0) {
            ensure_slots_filled();

            std::vector<int> batch_slots;
            batch_slots.reserve(active_clients);
            llama_batch gen_batch = llama_batch_init(n_ctx, 0, active_clients);

            for (int i = 0; i < seq_capacity; ++i) {
                Slot& slot = slots[i];
                if (!slot.active || slot.failed) {
                    continue;
                }

                if (slot.n_decoded == 0 && slot.full_tokens && !slot.full_tokens->empty()) {
                    slot.sampled = slot.full_tokens->back();
                }

                slot.i_batch = gen_batch.n_tokens;
                const int pos = slot.n_past + slot.n_decoded;
                common_batch_add(gen_batch, slot.sampled, pos, {slot.seq_id}, true);
                batch_slots.push_back(i);
            }

            if (gen_batch.n_tokens == 0) {
                llama_batch_free(gen_batch);
                break;
            }

            bool decode_success = true;
            int32_t local_cap = batch_cap_init;
            for (int32_t start = 0; start < (int32_t)gen_batch.n_tokens;) {
                int32_t n_tokens = std::min(local_cap, (int32_t)(gen_batch.n_tokens - start));
                llama_batch view = {
                    n_tokens,
                    gen_batch.token + start,
                    nullptr,
                    gen_batch.pos + start,
                    gen_batch.n_seq_id + start,
                    gen_batch.seq_id + start,
                    gen_batch.logits + start,
                };

                int ret = llama_decode(ctx, view);
                if (ret != 0) {
                    if (ret > 0 && local_cap > 1) {
                        local_cap = std::max<int32_t>(1, local_cap / 2);
                        dynamic_cache_miss++;
                        continue;
                    }
                    decode_success = false;
                    break;
                }

                for (int slot_idx : batch_slots) {
                    Slot& slot = slots[slot_idx];
                    if (!slot.active || slot.failed) {
                        continue;
                    }
                    if (slot.i_batch < start || slot.i_batch >= start + n_tokens) {
                        continue;
                    }

                    const int batch_pos = slot.i_batch - start;

                    try {
                        const llama_token new_token = common_sampler_sample(slot.smpl, ctx, batch_pos);
                        common_sampler_accept(slot.smpl, new_token, true);

                        bool should_stop = false;
                        if (new_token == eos_token || llama_vocab_is_eog(vocab, new_token)) {
                            should_stop = true;
                        }

                        if (params->max_tokens > 0 && slot.n_decoded >= params->max_tokens) {
                            should_stop = true;
                        }

                        if (!should_stop) {
                            const std::string token_str = common_token_to_piece(ctx, new_token);
                            slot.response += token_str;

                            if (slot.n_decoded > 5 &&
                                (slot.response.find("\n\nUser:") != std::string::npos ||
                                 slot.response.find("\n\nHuman:") != std::string::npos)) {
                                should_stop = true;
                            }
                        }

                        slot.sampled = new_token;
                        slot.n_decoded++;
                        slot.i_batch = -1;

                        if (should_stop) {
                            finalize_slot(slot_idx, true);
                        }

                    } catch (const std::exception& e) {
                        Slot& err_slot = slots[slot_idx];
                        err_slot.failed = true;
                        err_slot.error_msg = std::string("Sampling failed: ") + e.what();
                        err_slot.i_batch = -1;
                        finalize_slot(slot_idx, false);
                    }
                }

                start += n_tokens;
            }

            llama_batch_free(gen_batch);

            if (!slots_pending_reassign.empty()) {
                for (int slot_idx : slots_pending_reassign) {
                    assign_next_prompt(slot_idx);
                }
                slots_pending_reassign.clear();
            }

            if (!decode_success) {
                throw std::runtime_error("Fatal decode error during generation batch");
            }
        }

        const auto t_end = ggml_time_us();
        const double total_time = (t_end - t_start) / 1e6;

        if (show_progress_bar) {
            fprintf(stderr, "\r [==============================] %d/%d (100%%)\n", total_prompts, total_prompts);
            fflush(stderr);
        }

        for (int i = 0; i < seq_capacity; ++i) {
            release_slot(slots[i]);
        }

        if (prefix_ready) {
            llama_kv_self_seq_rm(ctx, 0, 0, -1);
        }

        *results_out = new char*[n_prompts];
        for (int i = 0; i < n_prompts; ++i) {
            (*results_out)[i] = string_to_c_str(final_responses[i]);
        }

#ifdef NEWRLLAMA_DEBUG
        std::cout << "=== Parallel Generation Performance ===" << std::endl;
        std::cout << "Total time: " << total_time << "s" << std::endl;
        std::cout << "Prompt tokens: " << n_total_prompt << std::endl;
        std::cout << "Generated tokens: " << n_total_gen << std::endl;
        std::cout << "Cache misses (dynamic throttling): " << dynamic_cache_miss << std::endl;
        std::cout << "Sequence capacity: " << seq_capacity << std::endl;
        if (total_time > 0) {
            std::cout << "Prompt speed: " << n_total_prompt / total_time << " t/s" << std::endl;
            std::cout << "Generation speed: " << n_total_gen / total_time << " t/s" << std::endl;
        }
#endif

        return NEWRLLAMA_SUCCESS;

    } catch (const std::exception& e) {
        if (show_progress_bar) {
            fprintf(stderr, "\r [==============================] %d/%d (100%%)\n", total_prompts, total_prompts);
            fflush(stderr);
        }
        llama_kv_self_clear(ctx);
        set_error(error_message, std::string("Parallel generation failed: ") + e.what());
        return NEWRLLAMA_ERROR;
    }
}

NEWRLLAMA_API void newrllama_free_string_array(char** arr, int count) { 
    if (arr) { 
        for (int i = 0; i < count; ++i) delete[] arr[i]; 
        delete[] arr; 
    } 
}

NEWRLLAMA_API newrllama_error_code newrllama_token_get_text(newrllama_model_handle model, int32_t token, char** text_out, const char** error_message) { 
    const struct llama_vocab* vocab = llama_model_get_vocab(model); 
    const char* text = llama_vocab_get_text(vocab, token); 
    *text_out = text ? string_to_c_str(text) : string_to_c_str(""); 
    return NEWRLLAMA_SUCCESS; 
}

NEWRLLAMA_API float newrllama_token_get_score(newrllama_model_handle model, int32_t token) { 
    return model ? llama_vocab_get_score(llama_model_get_vocab(model), token) : 0.0f; 
}

NEWRLLAMA_API int newrllama_token_get_attr(newrllama_model_handle model, int32_t token) { 
    return model ? llama_vocab_get_attr(llama_model_get_vocab(model), token) : 0; 
}

NEWRLLAMA_API bool newrllama_token_is_eog(newrllama_model_handle model, int32_t token) { 
    return model ? llama_vocab_is_eog(llama_model_get_vocab(model), token) : false; 
}

NEWRLLAMA_API bool newrllama_token_is_control(newrllama_model_handle model, int32_t token) { 
    return model ? llama_vocab_is_control(llama_model_get_vocab(model), token) : false; 
}

NEWRLLAMA_API int32_t newrllama_token_bos(newrllama_model_handle model) { 
    return model ? llama_vocab_bos(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_eos(newrllama_model_handle model) { 
    return model ? llama_vocab_eos(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_sep(newrllama_model_handle model) { 
    return model ? llama_vocab_sep(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_nl(newrllama_model_handle model) { 
    return model ? llama_vocab_nl(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_pad(newrllama_model_handle model) { 
    return model ? llama_vocab_pad(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_eot(newrllama_model_handle model) { 
    return model ? llama_vocab_eot(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API bool newrllama_add_bos_token(newrllama_model_handle model) { 
    return model ? llama_vocab_get_add_bos(llama_model_get_vocab(model)) : false; 
}

NEWRLLAMA_API bool newrllama_add_eos_token(newrllama_model_handle model) { 
    return model ? llama_vocab_get_add_eos(llama_model_get_vocab(model)) : false; 
}

NEWRLLAMA_API int32_t newrllama_token_fim_pre(newrllama_model_handle model) { 
    return model ? llama_vocab_fim_pre(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_fim_mid(newrllama_model_handle model) { 
    return model ? llama_vocab_fim_mid(llama_model_get_vocab(model)) : -1; 
}

NEWRLLAMA_API int32_t newrllama_token_fim_suf(newrllama_model_handle model) { 
    return model ? llama_vocab_fim_suf(llama_model_get_vocab(model)) : -1; 
}

#ifdef LLAMA_USE_CURL
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iostream>
#include <iomanip>

// Helper functions for model downloading (using newrllama_ prefix to avoid conflicts)
static bool newrllama_string_starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

static std::string newrllama_basename(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static int newrllama_rm_until_substring(std::string& model_, const std::string& substring) {
    const std::string::size_type pos = model_.find(substring);
    if (pos == std::string::npos) {
        return 1;
    }
    model_ = model_.substr(pos + substring.size());
    return 0;
}

// Progress callback for curl
struct progress_data {
    std::chrono::steady_clock::time_point start_time;
    size_t file_size;
    bool printed;
};

static size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

static int update_progress(void *ptr, curl_off_t total_to_download, curl_off_t now_downloaded, curl_off_t, curl_off_t) {
    if (!ptr) return 0;
    
    progress_data* data = static_cast<progress_data*>(ptr);
    if (total_to_download <= 0) return 0;
    
    total_to_download += data->file_size;
    const curl_off_t now_downloaded_plus_file_size = now_downloaded + data->file_size;
    const curl_off_t percentage = (now_downloaded_plus_file_size * 100) / total_to_download;
    
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = now - data->start_time;
    const double speed = now_downloaded / elapsed_seconds.count();
    
    if (percentage % 5 == 0 || now_downloaded_plus_file_size == total_to_download) {
        std::cout << "\rDownload progress: " << percentage << "% (" 
                  << now_downloaded_plus_file_size << "/" << total_to_download << " bytes)"
                  << std::flush;
        data->printed = true;
    }
    
    return 0;
}

// HTTP download function
static int download_file(const std::string& url, const std::string& output_file, bool show_progress) {
    CURL* curl = curl_easy_init();
    if (!curl) return 1;
    
    FILE* fp = fopen(output_file.c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        return 1;
    }
    
    progress_data data = {};
    data.start_time = std::chrono::steady_clock::now();
    data.file_size = 0;
    data.printed = false;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    
    // Check for HuggingFace token and add authorization header if needed
    struct curl_slist* headers = nullptr;
    const char* hf_token = std::getenv("HF_TOKEN");
    if (hf_token && strlen(hf_token) > 0 && url.find("huggingface.co") != std::string::npos) {
        std::string auth_header = "Authorization: Bearer " + std::string(hf_token);
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    if (show_progress) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &data);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, update_progress);
    }
    
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    
    // Clean up headers
    if (headers) {
        curl_slist_free_all(headers);
    }
    
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        std::filesystem::remove(output_file);
        return 1;
    }
    
    if (show_progress && data.printed) {
        std::cout << "\nDownload completed!" << std::endl;
    }
    
    curl_easy_cleanup(curl);
    return 0;
}

// Model resolution function
static int resolve_model_url(std::string& model_url, const std::string& output_file) {
    if (newrllama_string_starts_with(model_url, "file://") || std::filesystem::exists(model_url)) {
        newrllama_rm_until_substring(model_url, "://");
        return 0;
    }
    
    if (newrllama_string_starts_with(model_url, "https://") || newrllama_string_starts_with(model_url, "http://")) {
        return download_file(model_url, output_file, true);
    }
    
    // For other protocols (hf://, ollama://, etc.), we need full implementation
    // For now, return error for unsupported protocols
    return 1;
}

#endif // LLAMA_USE_CURL

// API implementations
NEWRLLAMA_API newrllama_error_code newrllama_download_model(const char* model_url, const char* output_path, bool show_progress, const char** error_message) {
#ifdef LLAMA_USE_CURL
    try {
        if (!model_url || !output_path) {
            set_error(error_message, "Invalid parameters: model_url and output_path cannot be null");
            return NEWRLLAMA_ERROR;
        }
        
        std::string url(model_url);
        std::string output(output_path);
        
        // Create output directory if it doesn't exist
        std::filesystem::path output_dir = std::filesystem::path(output).parent_path();
        if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }
        
        if (resolve_model_url(url, output) != 0) {
            set_error(error_message, "Failed to download model from URL: " + std::string(model_url));
            return NEWRLLAMA_ERROR;
        }
        
        return NEWRLLAMA_SUCCESS;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Download error: ") + e.what());
        return NEWRLLAMA_ERROR;
    }
#else
    set_error(error_message, "Model download not supported: built without curl");
    return NEWRLLAMA_ERROR;
#endif
}

NEWRLLAMA_API newrllama_error_code newrllama_resolve_model(const char* model_url, char** resolved_path, const char** error_message) {
    try {
        if (!model_url || !resolved_path) {
            set_error(error_message, "Invalid parameters: model_url and resolved_path cannot be null");
            return NEWRLLAMA_ERROR;
        }
        
        std::string url(model_url);
        
        // Check if it's a local file
        if (newrllama_string_starts_with(url, "file://")) {
            newrllama_rm_until_substring(url, "://");
            *resolved_path = string_to_c_str(url);
            return NEWRLLAMA_SUCCESS;
        }
        
        if (std::filesystem::exists(url)) {
            *resolved_path = string_to_c_str(url);
            return NEWRLLAMA_SUCCESS;
        }
        
        // For URLs, we need to determine cache path
        std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "newrllama_models";
        std::string cache_dir = temp_path.string();
        std::string filename = newrllama_basename(url);
        if (filename.empty()) {
            filename = "model.gguf";
        }
        
        std::string cache_path = cache_dir + "/" + filename;
        
        // Create cache directory
        std::filesystem::create_directories(cache_dir);
        
        *resolved_path = string_to_c_str(cache_path);
        return NEWRLLAMA_SUCCESS;
        
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Model resolution error: ") + e.what());
        return NEWRLLAMA_ERROR;
    }
}

// Memory checking functions
NEWRLLAMA_API size_t newrllama_estimate_model_memory(const char* model_path, const char** error_message) {
    try {
        if (!model_path) {
            set_error(error_message, "Invalid model path");
            return 0;
        }
        
        // Get file size
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            set_error(error_message, std::string("Cannot open model file: ") + model_path);
            return 0;
        }
        
        size_t file_size = static_cast<size_t>(file.tellg());
        file.close();
        
        // Conservative memory estimation:
        // - File size for loading
        // - Additional 50% for processing and overhead
        // - This is a rough estimate, actual usage may vary
        size_t estimated_memory = file_size + (file_size / 2);
        
        return estimated_memory;
        
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Error estimating memory: ") + e.what());
        return 0;
    }
}

NEWRLLAMA_API bool newrllama_check_memory_available(size_t required_bytes, const char** error_message) {
    try {
#ifdef _WIN32
        // Windows memory check
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            size_t available_memory = status.ullAvailPhys;
            return available_memory >= required_bytes;
        }
        return true; // If we can't check, assume it's okay
#elif defined(__APPLE__)
        // macOS memory check
        int64_t physical_memory = 0;
        size_t size = sizeof(physical_memory);
        if (sysctlbyname("hw.memsize", &physical_memory, &size, NULL, 0) == 0) {
            // Simple heuristic: check if required memory is less than 80% of total
            size_t available_estimate = physical_memory * 0.8;
            return available_estimate >= required_bytes;
        }
        return true;
#elif defined(__linux__)
        // Linux memory check
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            size_t mem_available = 0;
            
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") != std::string::npos) {
                    // Extract memory value (in kB)
                    size_t pos = line.find(":") + 1;
                    size_t end = line.find("kB");
                    if (pos != std::string::npos && end != std::string::npos) {
                        std::string mem_str = line.substr(pos, end - pos);
                        mem_available = std::stoull(mem_str) * 1024; // Convert to bytes
                        break;
                    }
                }
            }
            
            return mem_available >= required_bytes;
        }
        return true;
#else
        // Unknown platform, assume memory is available
        return true;
#endif
    } catch (const std::exception& e) {
        set_error(error_message, std::string("Error checking memory: ") + e.what());
        return true; // If we can't check, assume it's okay
    }
} 
