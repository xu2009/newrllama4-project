// =============================================================================
// 完整改进的并行生成函数 - 高效且安全的实现
// =============================================================================

NEWRLLAMA_API newrllama_error_code newrllama_generate_parallel_improved(
    newrllama_context_handle ctx, 
    const char** prompts, 
    int n_prompts, 
    const struct newrllama_parallel_params* params, 
    char*** results_out, 
    const char** error_message) {
    
    // =============================================================================
    // Phase 1: 参数验证和初始化
    // =============================================================================
    
    if (!ctx || !params || !prompts || !results_out || n_prompts <= 0) {
        set_error(error_message, "Invalid parameters: null pointers or invalid prompt count");
        return NEWRLLAMA_ERROR;
    }
    
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_token eos_token = llama_vocab_eos(vocab);
    const int n_ctx = llama_n_ctx(ctx);
    
    // 性能统计
    const auto t_start = ggml_time_us();
    int32_t n_cache_miss = 0;
    int32_t n_total_prompt = 0;
    int32_t n_total_gen = 0;
    
    // 客户端结构体 - 增强版
    struct EnhancedClient {
        int id;
        llama_seq_id seq_id;
        std::vector<llama_token> prompt_tokens;
        int32_t n_past = 0;
        int32_t n_prompt = 0;
        int32_t n_decoded = 0;
        int32_t i_batch = -1;
        std::string input;
        std::string response;
        llama_token sampled = 0;
        common_sampler* smpl = nullptr;
        bool finished = false;
        bool failed = false;
        std::string error_msg;
        int64_t t_start_prompt = 0;
        int64_t t_start_gen = 0;
        
        ~EnhancedClient() {
            if (smpl) {
                common_sampler_free(smpl);
                smpl = nullptr;
            }
        }
    };
    
    std::vector<EnhancedClient> clients(n_prompts);
    
    // =============================================================================
    // Phase 2: 客户端初始化和令牌化
    // =============================================================================
    
    try {
        // 配置采样参数
        common_params_sampling sparams{};
        sparams.top_k = params->top_k;
        sparams.top_p = params->top_p;
        sparams.temp = params->temperature;
        sparams.penalty_last_n = params->repeat_last_n;
        sparams.penalty_repeat = params->penalty_repeat;
        sparams.seed = (params->seed < 0) ? time(nullptr) : params->seed;
        
        // 清空KV缓存 - 确保干净的开始
        llama_kv_self_clear(ctx);
        
        // 初始化所有客户端
        for (int i = 0; i < n_prompts; ++i) {
            auto& client = clients[i];
            client.id = i;
            client.seq_id = i + 1;  // 序列ID从1开始，避免与系统序列冲突
            client.input = std::string(prompts[i]);
            client.t_start_prompt = ggml_time_us();
            
            // 为每个客户端创建独立的采样器
            client.smpl = common_sampler_init(model, sparams);
            if (!client.smpl) {
                throw std::runtime_error("Failed to initialize sampler for client " + std::to_string(i));
            }
            
            // 令牌化用户输入
            client.prompt_tokens = helper_tokenize(model, client.input, true);
            client.n_prompt = client.prompt_tokens.size();
            n_total_prompt += client.n_prompt;
            
            // 验证提示符长度
            if (client.n_prompt > n_ctx - 64) {  // 预留生成空间
                client.failed = true;
                client.error_msg = "Prompt too long for context size";
                continue;
            }
        }
        
        // =============================================================================
        // Phase 3: 统一批次处理所有提示符 - 关键改进
        // =============================================================================
        
        // 计算最大批次大小
        int max_batch_size = 0;
        for (const auto& client : clients) {
            if (!client.failed) {
                max_batch_size += client.n_prompt;
            }
        }
        
        if (max_batch_size == 0) {
            throw std::runtime_error("All clients failed during initialization");
        }
        
        // 创建统一批次 - 一次性处理所有提示符
        llama_batch batch = llama_batch_init(std::max(max_batch_size, n_ctx), 0, n_prompts);
        
        // 🔑 关键：所有客户端的提示符添加到同一批次，但使用独立的seq_id
        for (auto& client : clients) {
            if (client.failed) continue;
            
            for (size_t j = 0; j < client.prompt_tokens.size(); ++j) {
                // 每个客户端使用独立的seq_id，确保完全隔离
                common_batch_add(batch, 
                               client.prompt_tokens[j], 
                               j,                           // position in sequence
                               {client.seq_id},            // 🔑 独立序列ID
                               j == client.prompt_tokens.size() - 1);  // 只有最后一个token需要logits
            }
            client.n_past = client.n_prompt;
        }
        
        // 分块处理批次 - 处理上下文限制
        const int32_t n_batch_max = 512;  // 最大批次大小
        for (int32_t i = 0; i < (int32_t)batch.n_tokens; i += n_batch_max) {
            const int32_t n_tokens = std::min(n_batch_max, (int32_t)(batch.n_tokens - i));
            
            llama_batch batch_view = {
                n_tokens,
                batch.token + i,
                nullptr,
                batch.pos + i,
                batch.n_seq_id + i,
                batch.seq_id + i,
                batch.logits + i,
            };
            
            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                llama_batch_free(batch);
                
                // 🛡️ 错误处理：如果统一批次失败，回退到独立处理
                if (ret < 0) {
                    throw std::runtime_error("Fatal decode error in prompt processing: " + std::to_string(ret));
                }
                
                // 非致命错误，尝试分别处理每个客户端
                for (auto& client : clients) {
                    if (client.failed) continue;
                    
                    llama_batch individual_batch = llama_batch_init(client.n_prompt, 0, 1);
                    
                    for (size_t j = 0; j < client.prompt_tokens.size(); ++j) {
                        common_batch_add(individual_batch, 
                                       client.prompt_tokens[j], 
                                       j, 
                                       {client.seq_id}, 
                                       j == client.prompt_tokens.size() - 1);
                    }
                    
                    if (llama_decode(ctx, individual_batch) != 0) {
                        client.failed = true;
                        client.error_msg = "Failed to decode individual prompt";
                    }
                    
                    llama_batch_free(individual_batch);
                }
                
                n_cache_miss += 1;
                break;
            }
        }
        
        if (batch.n_tokens > 0) {
            llama_batch_free(batch);
        }
        
        // =============================================================================
        // Phase 4: 生成循环 - 统一批次但保持序列隔离
        // =============================================================================
        
        int active_clients = 0;
        for (const auto& client : clients) {
            if (!client.failed) active_clients++;
        }
        
        if (active_clients == 0) {
            throw std::runtime_error("No clients remain active after prompt processing");
        }
        
        // 生成循环
        while (active_clients > 0) {
            // 创建生成批次
            llama_batch gen_batch = llama_batch_init(n_ctx, 0, active_clients);
            std::vector<int> batch_to_client_map;
            batch_to_client_map.reserve(active_clients);
            
            // 为每个活跃客户端添加待生成的token
            for (auto& client : clients) {
                if (client.finished || client.failed) continue;
                
                // 第一次生成时，从提示符的最后一个token开始
                if (client.n_decoded == 0) {
                    client.sampled = client.prompt_tokens.back();
                    client.t_start_gen = ggml_time_us();
                }
                
                client.i_batch = gen_batch.n_tokens;
                const int pos = client.n_past + client.n_decoded;
                
                // 🔑 关键：每个客户端使用独立的seq_id
                common_batch_add(gen_batch, client.sampled, pos, {client.seq_id}, true);
                batch_to_client_map.push_back(client.id);
            }
            
            if (gen_batch.n_tokens == 0) {
                llama_batch_free(gen_batch);
                break;
            }
            
            // 分块处理生成批次
            const int32_t n_batch_gen = std::min(n_batch_max, (int32_t)gen_batch.n_tokens);
            for (int32_t i = 0; i < (int32_t)gen_batch.n_tokens; i += n_batch_gen) {
                const int32_t n_tokens = std::min(n_batch_gen, (int32_t)(gen_batch.n_tokens - i));
                
                llama_batch batch_view = {
                    n_tokens,
                    gen_batch.token + i,
                    nullptr,
                    gen_batch.pos + i,
                    gen_batch.n_seq_id + i,
                    gen_batch.seq_id + i,
                    gen_batch.logits + i,
                };
                
                const int ret = llama_decode(ctx, batch_view);
                if (ret != 0) {
                    llama_batch_free(gen_batch);
                    
                    if (ret < 0) {
                        throw std::runtime_error("Fatal decode error in generation: " + std::to_string(ret));
                    }
                    
                    // 非致命错误，继续处理
                    n_cache_miss += 1;
                    break;
                }
                
                // 🎯 关键：为每个客户端独立采样
                for (int b = 0; b < (int)batch_to_client_map.size(); ++b) {
                    const int client_id = batch_to_client_map[b];
                    auto& client = clients[client_id];
                    
                    if (client.i_batch < i || client.i_batch >= i + n_tokens) continue;
                    
                    try {
                        // 🔑 关键：使用客户端专属的采样器和正确的批次位置
                        const int batch_pos = client.i_batch - i;
                        const llama_token new_token = common_sampler_sample(client.smpl, ctx, batch_pos);
                        common_sampler_accept(client.smpl, new_token, true);
                        
                        // 检查终止条件
                        bool should_stop = false;
                        
                        // EOS token检查
                        if (new_token == eos_token || llama_vocab_is_eog(vocab, new_token)) {
                            should_stop = true;
                        }
                        
                        // 最大token数检查
                        if (params->max_tokens > 0 && client.n_decoded >= params->max_tokens) {
                            should_stop = true;
                        }
                        
                        // 转换token为文本
                        const std::string token_str = common_token_to_piece(ctx, new_token);
                        client.response += token_str;
                        client.sampled = new_token;
                        client.n_decoded++;
                        
                        // 对话终止检查
                        if (client.n_decoded > 5 && 
                            (client.response.find("\n\nUser:") != std::string::npos || 
                             client.response.find("\n\nHuman:") != std::string::npos)) {
                            should_stop = true;
                        }
                        
                        if (should_stop) {
                            client.finished = true;
                            active_clients--;
                            n_total_gen += client.n_decoded;
                            
                            // 🧹 清理该客户端的KV缓存
                            llama_kv_self_seq_rm(ctx, client.seq_id, 0, -1);
                        }
                        
                    } catch (const std::exception& e) {
                        client.failed = true;
                        client.error_msg = "Sampling failed: " + std::string(e.what());
                        active_clients--;
                    }
                }
            }
            
            llama_batch_free(gen_batch);
        }
        
        // =============================================================================
        // Phase 5: 结果处理和清理
        // =============================================================================
        
        // 清理所有KV缓存
        for (const auto& client : clients) {
            if (client.seq_id > 0) {
                llama_kv_self_seq_rm(ctx, client.seq_id, 0, -1);
            }
        }
        
        // 性能统计
        const auto t_end = ggml_time_us();
        const double total_time = (t_end - t_start) / 1e6;
        
        // 准备结果
        *results_out = new char*[n_prompts];
        for (int i = 0; i < n_prompts; ++i) {
            const auto& client = clients[i];
            
            if (client.failed) {
                // 失败的客户端返回错误信息
                std::string error_result = "[ERROR] " + client.error_msg;
                (*results_out)[i] = string_to_c_str(error_result);
            } else {
                // 清理响应文本
                std::string clean_response = client.response;
                
                // 移除无效字符
                while (!clean_response.empty() && 
                       (clean_response.front() == '?' || 
                        clean_response.front() < 32 || 
                        clean_response.front() > 126)) {
                    clean_response = clean_response.substr(1);
                }
                
                // 移除首尾空白
                while (!clean_response.empty() && isspace(clean_response.back())) {
                    clean_response.pop_back();
                }
                while (!clean_response.empty() && isspace(clean_response.front())) {
                    clean_response = clean_response.substr(1);
                }
                
                // 截断对话终止标记
                const size_t pos = clean_response.find("\n\nUser:");
                if (pos != std::string::npos) {
                    clean_response = clean_response.substr(0, pos);
                }
                
                (*results_out)[i] = string_to_c_str(clean_response);
            }
        }
        
        // 🔍 调试信息（可选）
        #ifdef NEWRLLAMA_DEBUG
        std::cout << "=== Parallel Generation Performance ===" << std::endl;
        std::cout << "Total time: " << total_time << "s" << std::endl;
        std::cout << "Prompt tokens: " << n_total_prompt << std::endl;
        std::cout << "Generated tokens: " << n_total_gen << std::endl;
        std::cout << "Cache misses: " << n_cache_miss << std::endl;
        std::cout << "Prompt speed: " << n_total_prompt / total_time << " t/s" << std::endl;
        std::cout << "Generation speed: " << n_total_gen / total_time << " t/s" << std::endl;
        #endif
        
        return NEWRLLAMA_SUCCESS;
        
    } catch (const std::exception& e) {
        // 🚨 全局错误处理
        for (const auto& client : clients) {
            if (client.seq_id > 0) {
                llama_kv_self_seq_rm(ctx, client.seq_id, 0, -1);
            }
        }
        
        set_error(error_message, std::string("Parallel generation failed: ") + e.what());
        return NEWRLLAMA_ERROR;
    }
}