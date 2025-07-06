#define NEWRLLAMA_BUILD_DLL
#include "newrllama_capi.h"
#include "llama.h"
#include "common/common.h"
#include "common/sampling.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <ctime>
#include <algorithm>

static thread_local std::string last_error_message;

void set_error(const char** error_message, const std::string& msg) { 
    if (error_message) { 
        last_error_message = msg; 
        *error_message = last_error_message.c_str(); 
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

NEWRLLAMA_API void newrllama_model_free(newrllama_model_handle model) { 
    if (model) llama_model_free(model); 
}

NEWRLLAMA_API newrllama_error_code newrllama_context_create(newrllama_model_handle model, int n_ctx, int n_threads, int n_seq_max, newrllama_context_handle* context_handle_out, const char** error_message) { 
    if (!model) { 
        set_error(error_message, "Model handle is null."); 
        return NEWRLLAMA_ERROR; 
    } 
    llama_context_params ctx_params = llama_context_default_params(); 
    ctx_params.n_ctx = n_ctx; 
    ctx_params.n_threads = n_threads; 
    ctx_params.n_seq_max = n_seq_max; 
    llama_context* ctx = llama_init_from_model(model, ctx_params); 
    if (ctx == nullptr) { 
        set_error(error_message, "Failed to create context from model."); 
        return NEWRLLAMA_ERROR; 
    } 
    *context_handle_out = ctx; 
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
    std::vector<llama_chat_message> messages_vec(n_messages); 
    size_t total_length = 0; 
    for(size_t i = 0; i < n_messages; ++i) { 
        messages_vec[i] = {messages_in[i].role, messages_in[i].content}; 
        total_length += (messages_in[i].content ? strlen(messages_in[i].content) : 0); 
    } 
    size_t buffer_size = total_length * 2 + 2048; 
    std::vector<char> buffer(buffer_size, 0); 
    int32_t res = llama_chat_apply_template(tmpl, messages_vec.data(), n_messages, add_ass, buffer.data(), buffer.size()); 
    if (res < 0) { 
        set_error(error_message, "Failed to apply chat template. Error code: " + std::to_string(res)); 
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
    const llama_model* model = llama_get_model(ctx); 
    const struct llama_vocab* vocab = llama_model_get_vocab(model); 
    llama_token eos_token = llama_vocab_eos(vocab); 
    llama_batch batch = llama_batch_get_one((llama_token*)tokens_in, n_tokens_in); 
    if (llama_decode(ctx, batch) != 0) { 
        set_error(error_message, "Failed to decode input tokens."); 
        return NEWRLLAMA_ERROR; 
    } 
    struct llama_sampler_chain_params sparams_chain = llama_sampler_chain_default_params(); 
    struct llama_sampler* sampler_chain = llama_sampler_chain_init(sparams_chain); 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_penalties(repeat_last_n, penalty_repeat, 0.0f, 0.0f)); 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(top_k)); 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(top_p, 1)); 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(temperature)); 
    uint32_t final_seed = (seed < 0) ? time(NULL) : seed; 
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(final_seed)); 
    std::string generated_text; 
    for (int i = 0; i < max_tokens; ++i) { 
        llama_token new_token = llama_sampler_sample(sampler_chain, ctx, -1); 
        llama_sampler_accept(sampler_chain, new_token); 
        if (new_token == eos_token || llama_vocab_is_eog(vocab, new_token)) break; 
        generated_text += common_token_to_piece(ctx, new_token); 
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

NEWRLLAMA_API newrllama_error_code newrllama_generate_parallel(newrllama_context_handle ctx, const char** prompts, int n_prompts, const struct newrllama_parallel_params* params, char*** results_out, const char** error_message) { 
    if (!ctx || !params) { 
        set_error(error_message, "Context or params handle is null."); 
        return NEWRLLAMA_ERROR; 
    } 
    const llama_model* model = llama_get_model(ctx); 
    const llama_vocab* vocab = llama_model_get_vocab(model); 
    const llama_token eos_token = llama_vocab_eos(vocab); 
    common_params_sampling sparams{}; 
    sparams.top_k = params->top_k; 
    sparams.top_p = params->top_p; 
    sparams.temp = params->temperature; 
    sparams.penalty_last_n = params->repeat_last_n; 
    sparams.penalty_repeat = params->penalty_repeat; 
    uint32_t final_seed = (params->seed < 0) ? time(NULL) : params->seed; 
    sparams.seed = final_seed; 
    struct Client { 
        int id; 
        llama_seq_id seq_id; 
        std::vector<llama_token> prompt_tokens; 
        size_t n_decoded = 0; 
        std::string response; 
        llama_token sampled = 0; 
        common_sampler* smpl = nullptr; 
        bool finished = false; 
    }; 
    std::vector<Client> clients(n_prompts); 
    try { 
        for (int i = 0; i < n_prompts; ++i) { 
            auto& C = clients[i]; 
            C.id = i; 
            C.seq_id = i; 
            C.prompt_tokens = helper_tokenize(model, std::string(prompts[i]), true); 
            C.smpl = common_sampler_init(model, sparams); 
            if (!C.smpl) throw std::runtime_error("Sampler init failed for client " + std::to_string(i)); 
        } 
        llama_kv_self_clear(ctx); 
        int active = n_prompts; 
        while (active > 0) { 
            llama_batch batch = llama_batch_init(512, 0, active); 
            int batch_idx = 0; 
            for (auto& C : clients) { 
                if (C.finished) continue; 
                if (C.n_decoded == 0) { 
                    for (size_t k = 0; k < C.prompt_tokens.size() && batch.n_tokens < 512; ++k) { 
                        common_batch_add(batch, C.prompt_tokens[k], k, {C.seq_id}, false); 
                    } 
                    C.n_decoded = C.prompt_tokens.size(); 
                    if (batch.n_tokens > 0) batch.logits[batch.n_tokens - 1] = true; 
                } else { 
                    common_batch_add(batch, C.sampled, C.n_decoded, {C.seq_id}, true); 
                    C.n_decoded++; 
                } 
            } 
            if (batch.n_tokens == 0) { 
                llama_batch_free(batch); 
                break; 
            } 
            if (llama_decode(ctx, batch) != 0) { 
                llama_batch_free(batch); 
                throw std::runtime_error("Parallel generation decoding failed."); 
            } 
            int current_batch_idx = 0; 
            for (auto& C : clients) { 
                if (C.finished) continue; 
                if (batch.logits[current_batch_idx]) { 
                    llama_token tok = common_sampler_sample(C.smpl, ctx, current_batch_idx); 
                    common_sampler_accept(C.smpl, tok, true); 
                    if (tok == eos_token || (params->max_tokens > 0 && C.response.length() >= (size_t)params->max_tokens) || llama_vocab_is_eog(vocab, tok)) { 
                        C.finished = true; 
                        active--; 
                    } else { 
                        C.response += common_token_to_piece(ctx, tok); 
                        C.sampled = tok; 
                    } 
                } 
                current_batch_idx++; 
            } 
            llama_batch_free(batch); 
        } 
    } catch (const std::exception& e) { 
        for(auto& C : clients) if (C.smpl) common_sampler_free(C.smpl); 
        set_error(error_message, e.what()); 
        return NEWRLLAMA_ERROR; 
    } 
    for (auto& C : clients) if (C.smpl) common_sampler_free(C.smpl); 
    *results_out = new char*[n_prompts]; 
    for (int i = 0; i < n_prompts; ++i) { 
        (*results_out)[i] = string_to_c_str(clients[i].response); 
    } 
    return NEWRLLAMA_SUCCESS; 
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

// Helper functions for model downloading
static bool string_starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

static std::string basename(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static int rm_until_substring(std::string& model_, const std::string& substring) {
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
    
    if (show_progress) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &data);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, update_progress);
    }
    
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    
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
    if (string_starts_with(model_url, "file://") || std::filesystem::exists(model_url)) {
        rm_until_substring(model_url, "://");
        return 0;
    }
    
    if (string_starts_with(model_url, "https://") || string_starts_with(model_url, "http://")) {
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
        if (string_starts_with(url, "file://")) {
            rm_until_substring(url, "://");
            *resolved_path = string_to_c_str(url);
            return NEWRLLAMA_SUCCESS;
        }
        
        if (std::filesystem::exists(url)) {
            *resolved_path = string_to_c_str(url);
            return NEWRLLAMA_SUCCESS;
        }
        
        // For URLs, we need to determine cache path
        std::string cache_dir = std::filesystem::temp_directory_path() / "newrllama_models";
        std::string filename = basename(url);
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