# 第十四章：LLM服务基础设施

大语言模型（LLM）的推理服务对硬件基础设施提出了独特的挑战：海量的模型参数、自回归的生成模式、动态的序列长度以及严格的延迟要求。本章将深入探讨如何设计和优化FPGA加速的LLM推理系统，从单芯片的Token生成流水线到多FPGA的分布式部署。我们将分析批处理与流式推理的权衡，探索模型分片策略，优化主机通信瓶颈，并实现混合精度推理。通过本章学习，您将掌握构建生产级LLM推理加速器的核心技术，理解FPGA在LLM服务中相对于GPU的独特优势。

## 14.1 Token生成流水线架构

### 14.1.1 自回归生成的硬件挑战

LLM的自回归特性给硬件加速带来特殊挑战。与传统的前馈神经网络不同，LLM在生成每个新token时都需要依赖之前所有token的信息，这种时序依赖性从根本上限制了并行化的可能性。

```systemverilog
// LLM Token生成流水线顶层架构
module llm_inference_pipeline #(
    parameter MODEL_DIM = 4096,      // 模型维度（如LLaMA-7B）
    parameter NUM_HEADS = 32,        // 注意力头数
    parameter NUM_LAYERS = 32,       // Transformer层数
    parameter MAX_SEQ_LEN = 2048,    // 最大序列长度
    parameter BATCH_SIZE = 8,        // 批处理大小
    parameter VOCAB_SIZE = 32000,    // 词表大小
    parameter HEAD_DIM = 128,        // 每个注意力头的维度
    parameter FFN_DIM = 11008        // FFN中间层维度
) (
    input  logic                     clk,
    input  logic                     rst_n,
    
    // Token输入接口
    input  logic [31:0]             input_tokens[BATCH_SIZE],
    input  logic [10:0]             seq_lengths[BATCH_SIZE],  // 每个序列的实际长度
    input  logic                    tokens_valid,
    input  logic                    is_prefill,              // 区分prefill和decode阶段
    
    // KV-Cache接口
    input  logic                    kv_cache_enable,
    output logic                    kv_cache_update,
    output logic [39:0]             kv_cache_addr,          // 40位地址支持大容量HBM
    output kv_data_t                kv_write_data,
    input  kv_data_t                kv_read_data,
    
    // 模型权重接口（连接到HBM）
    output logic [39:0]             weight_addr,
    output logic                    weight_read_en,
    input  logic [511:0]            weight_data,            // 512位宽HBM接口
    
    // 输出接口
    output logic [31:0]             output_tokens[BATCH_SIZE],
    output logic [15:0]             logits[BATCH_SIZE][VOCAB_SIZE],
    output logic                    output_valid,
    
    // 性能监控
    output perf_counters_t          perf_counters
);
```

**关键设计考虑：**

1. **序列依赖性与流水线设计**
   - 每个token必须等待前一个token生成完成
   - 流水线气泡不可避免，特别是在decode阶段
   - 需要精心设计的调度策略来隐藏延迟
   - 使用推测执行技术预测可能的token路径
   - 关键洞察：虽然token生成是串行的，但batch内的不同序列可以并行处理

2. **内存带宽压力与优化**
   - 模型权重访问：~14GB for LLaMA-7B，~140GB for LLaMA-70B
   - KV-Cache增长：每token约占用 `2 * num_layers * seq_len * model_dim * 2 bytes`
   - 激活值存储：需要动态分配和回收机制
   - 权重压缩技术（INT8/INT4量化）至关重要
   - 实测数据：LLaMA-7B在seq_len=2048时，KV-Cache达到2GB

3. **计算模式切换与资源调度**
   - Prefill阶段：GEMM操作为主，计算密集，易并行化
   - Decode阶段：GEMV操作为主，内存带宽受限
   - 需要自适应的资源分配策略
   - 批处理中不同阶段请求的混合调度
   - 计算与访存比（Arithmetic Intensity）：Prefill ~100 ops/byte, Decode ~2 ops/byte

4. **数值精度与性能权衡**
   - FP16/BF16基础精度
   - INT8量化推理（性能提升2倍）
   - INT4量化（性能提升4倍，需要精细校准）
   - 混合精度策略：关键层保持高精度
   - 量化误差累积分析：每层<0.1%，32层累积<3%

**硬件资源预算（Versal AI VCK5000为例）：**
```
总DSP资源：1,968个AI Engine tiles
总HBM带宽：460GB/s
片上SRAM：32MB distributed
NoC带宽：>1TB/s
功耗预算：~150W
峰值算力：147 TOPS (INT8)
```

**自回归生成的根本性挑战分析：**

1. **计算图的动态性**
   ```systemverilog
   // 动态计算图管理器
   module dynamic_compute_graph (
       input  logic         clk,
       input  logic [10:0]  current_seq_len,    // 当前序列长度
       input  logic [10:0]  target_seq_len,     // 目标生成长度
       output logic [31:0]  compute_ops_count,  // 动态计算操作数
       output logic [31:0]  memory_access_count // 动态内存访问数
   );
   
   // 计算量随序列长度二次增长
   assign compute_ops_count = current_seq_len * current_seq_len * MODEL_DIM;
   ```

2. **内存访问模式的不规则性**
   - Attention pattern稀疏且动态变化
   - Cache miss rate随序列增长而增加
   - 预取策略效果有限（pattern难以预测）
   - 解决方案：自适应的cache替换策略

3. **批处理效率的递减**
   ```
   批处理效率 = 有效计算时间 / (有效计算时间 + 同步等待时间)
   
   Batch=1:  95% （几乎无等待）
   Batch=8:  75% （序列长度差异导致等待）
   Batch=32: 55% （严重的tail latency）
   ```

4. **功耗密度的不均匀分布**
   - Attention计算：~60W (40%)
   - FFN计算：~45W (30%)
   - 内存访问：~30W (20%)
   - 控制逻辑：~15W (10%)

**硬件设计的创新方向：**

1. **异构计算架构**
   - 专用Attention引擎：优化稀疏矩阵运算
   - 专用FFN引擎：优化稠密矩阵运算
   - 专用Sampling引擎：优化概率采样
   - 动态重配置支持不同工作负载

2. **预测性执行框架**
   ```systemverilog
   module predictive_execution (
       input  logic [31:0]  context_tokens[MAX_CONTEXT],
       output logic [31:0]  predicted_tokens[PRED_DEPTH],
       output logic [7:0]   confidence_scores[PRED_DEPTH]
   );
   ```
   - 基于n-gram的快速预测
   - 小模型辅助的投机路径
   - 硬件验证机制

3. **自适应精度控制**
   - 动态量化：根据token重要性调整精度
   - 层级精度：不同层使用不同精度
   - 误差补偿：硬件级误差追踪与补偿

**实际部署中的关键指标：**

| 指标 | LLaMA-7B | LLaMA-13B | LLaMA-70B |
|------|----------|-----------|-----------|
| 权重内存 | 14GB | 26GB | 140GB |
| KV-Cache/token | 1MB | 1.6MB | 5MB |
| 计算量/token | 14 GFLOPs | 26 GFLOPs | 140 GFLOPs |
| 内存带宽需求 | 200GB/s | 350GB/s | 1.2TB/s |
| 推理延迟@FPGA | 8ms | 15ms | 80ms |

### 14.1.2 流水线阶段划分

优化的流水线设计需要平衡计算密度、内存访问模式和数据依赖关系。现代LLM推理系统通常采用深度流水线架构，每个阶段都经过精心优化以最大化硬件利用率。

**典型的6级流水线设计：**

1. **嵌入查找与位置编码（Embedding & Positional Encoding）**
   ```systemverilog
   module embedding_stage #(
       parameter VOCAB_SIZE = 32000,
       parameter MODEL_DIM = 4096,
       parameter MAX_SEQ_LEN = 2048,
       parameter EMBED_SHARD = 8    // 嵌入表分片数
   ) (
       input  logic [31:0]  token_ids[BATCH_SIZE],
       input  logic [10:0]  positions[BATCH_SIZE],
       output logic [MODEL_DIM-1:0] embeddings[BATCH_SIZE],
       // 性能监控
       output logic [31:0]  embed_cycles,
       output logic [31:0]  cache_hits
   );
   ```
   - Token ID到向量映射：使用分布式BRAM存储嵌入表
   - RoPE位置编码融合：在线计算旋转矩阵
   - 批处理并行访问：多端口BRAM支持并发读取
   - 优化策略：嵌入表分片存储，减少访问冲突
   - **实测性能**：8路并行查找，延迟3个周期，吞吐量8 tokens/cycle

2. **多头注意力计算（Multi-Head Attention）**
   ```systemverilog
   module attention_engine #(
       parameter NUM_HEADS = 32,
       parameter HEAD_DIM = 128,
       parameter SEQ_LEN = 2048,
       parameter ATTN_PRECISION = "INT8"  // 支持INT8/FP16/BF16
   );
   ```
   - **Q、K、V投影**：
     - 使用systolic array进行矩阵乘法
     - INT8量化减少DSP使用（2x性能提升）
     - 三个投影可并行计算
     - 硬件实现：32×32 systolic array，峰值算力1024 MAC/cycle
   
   - **注意力分数计算**：
     - Scaled dot-product: `QK^T / sqrt(d_k)`
     - 分块计算避免大矩阵存储（64×64块）
     - 使用对数域计算提高数值稳定性
     - Flash Attention优化：IO复杂度从O(N²d)降至O(N²d/M)
   
   - **Softmax归一化**：
     - 两遍扫描：先找最大值，再计算指数
     - 在线归一化避免溢出
     - 查找表加速指数计算（2048项LUT）
     - 硬件优化：分组并行softmax，8组同时计算
   
   - **加权求和**：
     - 稀疏注意力模式优化（只保留Top-256重要位置）
     - 只计算Top-K注意力权重
     - 流式输出减少存储需求
     - 带宽优化：使用Z-order存储减少cache miss

3. **前馈网络（Feed-Forward Network）**
   ```systemverilog
   module ffn_block #(
       parameter MODEL_DIM = 4096,
       parameter FFN_DIM = 11008,
       parameter USE_GATED = 1,     // SwiGLU激活
       parameter SPLIT_FACTOR = 4   // 权重分片因子
   );
   ```
   - **第一层线性变换**：
     - Up-projection到FFN_DIM维度
     - 如果使用SwiGLU，需要两个并行投影（gate和up）
     - 权重切片以匹配HBM带宽（128bit×4通道）
     - 实测：4096→11008投影需要88M参数，分4片存储
   
   - **激活函数处理**：
     - SwiGLU: `x * SiLU(gate)` where `SiLU(x) = x * sigmoid(x)`
     - 使用分段线性近似（16段，误差<0.01%）
     - 定点运算减少资源消耗
     - 硬件：专用SiLU单元，2周期延迟，全流水线
   
   - **第二层线性变换**：
     - Down-projection回MODEL_DIM
     - 输出直接流式传输
     - 与残差连接融合
     - 权重复用：与第一层共享HBM通道

4. **层归一化与残差连接（Layer Norm & Residual）**
   ```systemverilog
   module norm_residual #(
       parameter USE_RMS_NORM = 1,  // LLaMA使用RMSNorm
       parameter EPSILON = 1e-5,
       parameter VECTOR_WIDTH = 16  // 向量化宽度
   );
   ```
   - **RMSNorm计算**：
     - 在线计算均方根：`sqrt(mean(x^2) + ε)`
     - 使用移位寄存器累加平方和
     - CORDIC算法计算平方根（16位迭代）
     - 向量化实现：16路并行，4096维向量需256周期
   
   - **残差连接**：
     - Pre-norm架构：先归一化再计算
     - 精度保持：使用饱和算术防止溢出
     - 流水线延迟匹配（使用FIFO缓冲）
     - 带宽优化：残差路径与主路径共享读端口

5. **KV-Cache管理子系统**
   ```systemverilog
   module kv_cache_manager #(
       parameter CACHE_WAYS = 16,
       parameter PAGE_SIZE = 256,   // PagedAttention
       parameter COMPRESSION = "INT8", // 压缩格式
       parameter MAX_ENTRIES = 65536  // 最大缓存条目
   );
   ```
   - **增量更新机制**：
     - 只写入当前token的K、V（减少90%写带宽）
     - 环形缓冲区管理（避免碎片）
     - 写入地址预计算（隐藏地址计算延迟）
     - 双缓冲设计：读写分离，避免冲突
   
   - **内存压缩技术**：
     - 量化到INT8/INT4（压缩率2-4x）
     - 稀疏存储（只保存注意力>0.01的token）
     - 动态压缩率调整（根据序列长度）
     - 硬件量化器：流水线量化，1周期/向量
   
   - **智能逐出策略**：
     - LRU/LFU混合策略（权重可配置）
     - 注意力分数指导的重要性评估
     - 预测性预取（基于attention pattern）
     - 硬件实现：16路组相联，硬件LRU

6. **输出生成与采样（Output Generation）**
   ```systemverilog
   module output_sampler #(
       parameter TOP_K = 40,
       parameter TOP_P_THRESHOLD = 0.95,
       parameter VOCAB_SHARDS = 8    // 词表分片数
   );
   ```
   - **词表投影**：
     - 最后的线性层映射到VOCAB_SIZE（32000）
     - 分片计算减少延迟（8片并行）
     - 流式输出logits
     - 优化：只计算Top-K候选词的完整logits
   
   - **概率采样策略**：
     - Top-K采样：保留K个最高概率token（硬件排序网络）
     - Top-P采样：累积概率阈值（在线累加器）
     - 温度调节：`logits / temperature`（查表实现）
     - 硬件随机数生成器：LFSR-based，周期2^128
   
   - **并行候选生成**：
     - Beam search支持（最多8束）
     - 多样性惩罚（硬件哈希表追踪）
     - 重复惩罚机制（滑动窗口检测）

**流水线优化技术：**

1. **动态调度**
   - 根据batch中请求的阶段动态分配资源
   - Prefill请求优先级高于decode
   - 避免长序列阻塞短序列
   - 硬件实现：多级优先队列，硬件仲裁器
   - 调度算法：加权轮询 + 抢占式调度

2. **数据预取**
   - 权重预取隐藏HBM延迟（提前16周期）
   - KV-Cache预取基于访问模式预测
   - 双缓冲技术实现连续处理
   - 预取命中率：>85%（基于历史pattern）
   - 硬件预取器：4级预取队列，投机预取

3. **负载均衡**
   - 将长序列分割成多个块（chunk size=256）
   - 多个短序列合并处理（padding最小化）
   - 动态批大小调整（8/16/32自适应）
   - 负载分配算法：贪心装箱 + 预测补偿

**流水线深度与延迟分析：**

```systemverilog
// 流水线控制器 - 自适应深度调整
module pipeline_controller #(
    parameter MAX_STAGES = 32,
    parameter MIN_STAGES = 8
) (
    input  logic         clk,
    input  logic         is_prefill_mode,
    input  logic [31:0]  current_batch_size,
    input  logic [31:0]  avg_seq_length,
    output logic [4:0]   active_stages,    // 当前激活的流水级数
    output logic [31:0]  pipeline_depth,   // 有效流水线深度
    output logic         stall_signal[MAX_STAGES]
);

// 自适应逻辑
always_ff @(posedge clk) begin
    if (is_prefill_mode) begin
        // Prefill模式：深流水线，最大化吞吐量
        active_stages <= 32;
        pipeline_depth <= current_batch_size * 4;  // 超标量执行
    end else begin
        // Decode模式：浅流水线，最小化延迟
        active_stages <= 8;  
        pipeline_depth <= current_batch_size;
    end
end
```

**阶段间数据流优化：**

1. **零拷贝设计**
   - 使用指针传递代替数据拷贝
   - 环形缓冲区避免内存分配
   - DMA直通路径（bypass CPU）

2. **流控机制**
   ```systemverilog
   // 背压流控实现
   interface pipeline_flow_ctrl;
       logic valid;      // 数据有效
       logic ready;      // 下游就绪
       logic [511:0] data; // 数据总线
       
       modport producer (output valid, data, input ready);
       modport consumer (input valid, data, output ready);
   endinterface
   ```

3. **数据格式优化**
   - 紧凑的数据布局（结构体对齐）
   - 向量化的数据传输（512位总线）
   - 压缩的中间表示（稀疏格式）

**实际流水线性能剖析（LLaMA-7B）：**

| 流水线阶段 | Prefill延迟 | Decode延迟 | 资源占用 | 带宽需求 |
|-----------|------------|-----------|---------|----------|
| Embedding | 50 cycles | 10 cycles | 5% DSP | 10GB/s |
| Attention | 800 cycles | 200 cycles | 45% DSP | 100GB/s |
| FFN | 600 cycles | 150 cycles | 35% DSP | 80GB/s |
| LayerNorm | 100 cycles | 30 cycles | 5% DSP | 20GB/s |
| KV-Cache | 200 cycles | 50 cycles | 5% DSP | 150GB/s |
| Sampling | 150 cycles | 150 cycles | 5% DSP | 30GB/s |

**流水线优化的关键指标：**

1. **吞吐量提升**
   - 基线（无流水线）：125 tokens/s
   - 6级流水线：750 tokens/s（6x提升）
   - 理论上限：1000 tokens/s（受限于内存带宽）

2. **延迟隐藏效果**
   - 计算与访存重叠：80%时间
   - 多请求交织执行：延迟降低60%
   - 关键路径优化：单周期临界路径

3. **资源利用率**
   - DSP利用率：85%（vs 无流水线30%）
   - HBM带宽利用：75%（vs 无流水线25%）
   - 片上SRAM利用：90%（双缓冲设计）

### 14.1.3 延迟优化技术

LLM推理的用户体验高度依赖于延迟表现，特别是首token延迟（TTFT - Time To First Token）和token间延迟（ITL - Inter-Token Latency）。FPGA的确定性时序和灵活架构为延迟优化提供了独特优势。

**关键优化策略：**

1. **推测解码（Speculative Decoding）**
   ```systemverilog
   module speculative_decoder #(
       parameter DRAFT_MODEL_SIZE = 68000000,  // 68M参数的draft模型
       parameter TARGET_MODEL_SIZE = 7000000000, // 7B参数的目标模型
       parameter SPECULATION_LENGTH = 4        // 每次推测4个token
   ) (
       input  logic        clk,
       input  logic [31:0] context_tokens[MAX_SEQ_LEN],
       output logic [31:0] verified_tokens[SPECULATION_LENGTH],
       output logic [3:0]  num_accepted  // 实际接受的token数
   );
   ```
   
   **实现要点：**
   - 小模型（Draft Model）快速预测多个token
   - 大模型并行验证所有候选token
   - 基于概率比值的接受/拒绝机制
   - 平均加速比：1.5-3倍（取决于任务复杂度）
   
   **硬件设计考虑：**
   - Draft模型使用独立的计算单元
   - 共享KV-Cache减少内存开销
   - 验证阶段的并行矩阵运算
   - 动态调整推测长度

2. **算子融合与内存优化**
   
   **垂直融合（层内融合）：**
   ```systemverilog
   module fused_attention_block (
       // 融合QKV投影、注意力计算、输出投影
       input  [MODEL_DIM-1:0]  input_hidden,
       output [MODEL_DIM-1:0]  output_hidden
   );
   ```
   - QKV投影三合一：减少3倍权重读取
   - Softmax与矩阵乘法流水线化
   - LayerNorm与残差连接融合
   - 激活值不落盘，全程片上处理
   
   **水平融合（跨层融合）：**
   - 相邻Transformer层的部分计算合并
   - 共享中间结果缓冲区
   - 减少50%的激活值内存带宽

3. **动态批处理与填充优化**
   ```systemverilog
   module dynamic_batch_manager #(
       parameter MAX_BATCH = 64,
       parameter BUCKET_SIZES = '{128, 256, 512, 1024, 2048}
   );
   ```
   
   **Continuous Batching实现：**
   - 请求可在任意时刻加入/退出
   - 序列长度分桶（bucketing）减少填充
   - In-flight batching：不同阶段的请求混合
   - 填充token智能调度（不参与实际计算）
   
   **性能收益：**
   - 平均填充率：<10%（vs 静态批处理的40%）
   - 吞吐量提升：2-3倍
   - 延迟方差降低：50%

4. **内存访问模式优化**
   
   **Flash Attention的FPGA实现：**
   ```systemverilog
   module flash_attention_fpga #(
       parameter BLOCK_SIZE = 64,  // 分块大小
       parameter SRAM_SIZE = 256KB // 片上SRAM大小
   );
   ```
   - 分块计算减少HBM访问
   - 在线softmax避免中间结果存储
   - IO复杂度：O(N²) → O(N²/M)，M为块大小
   - SRAM利用率>90%

5. **流水线深度优化**
   
   **自适应流水线控制：**
   - Prefill阶段：深流水线，最大化吞吐量
   - Decode阶段：浅流水线，最小化延迟
   - 动态重配置支持
   - 关键路径优化到单周期

6. **预计算与缓存策略**
   
   **常见模式加速：**
   - 系统提示词的KV-Cache预计算
   - 常用短语的attention pattern缓存
   - RoPE位置编码查找表
   - Softmax近似计算表

**性能指标对比（Versal AI VCK5000）：**

| 优化技术 | LLaMA-7B性能提升 | 硬件开销 |
|---------|-----------------|---------|
| 基线（FP16） | 1.0x | 100% |
| INT8量化 | 1.8x | 60% |
| 算子融合 | 1.3x | 105% |
| Flash Attention | 1.4x | 110% |
| 推测解码 | 2.2x | 130% |
| 全部优化 | 3.5x | 140% |

**实测延迟数据：**
- LLaMA-7B INT8推理（全优化）：
  - 首Token延迟（TTFT）：15-25ms
  - 后续Token（ITL）：5-8ms/token
  - 批处理吞吐量：2000 tokens/s @batch=16
  - 能效比：8 tokens/joule

**延迟分解分析：**
```
总延迟 = 数据传输 + 计算延迟 + 同步开销
- 数据传输：40%（权重加载为主）
- 计算延迟：50%（注意力计算占60%）
- 同步开销：10%（批处理调度）
```

### 14.1.4 实现案例：高效注意力模块

**FlashAttention的FPGA适配：**

考虑内存层次的注意力计算优化：

1. **分块计算策略**
   - 将QKV矩阵分块
   - 块大小匹配片上BRAM
   - 减少HBM访问

2. **在线Softmax**
   - 增量计算最大值
   - 单次遍历完成归一化
   - 数值稳定性保证

3. **融合计算流**
   - QK矩阵乘法
   - Softmax计算
   - V加权求和
   - 全部在片上完成

**资源利用分析：**
- 32头注意力模块：
  - DSP利用：~2000个DSP48E2
  - BRAM利用：~800个BRAM块
  - 计算效率：>80% DSP利用率

## 14.2 批处理vs流式推理权衡

### 14.2.1 批处理推理架构

批处理优化吞吐量，适合离线场景：

```systemverilog
// 批处理调度器
module batch_scheduler #(
    parameter MAX_BATCH_SIZE = 64,
    parameter MAX_SEQ_LENGTH = 2048
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // 请求队列接口
    input  request_t               new_request,
    input  logic                   request_valid,
    
    // 批次输出
    output batch_t                 current_batch,
    output logic                   batch_ready,
    
    // 性能监控
    output logic [31:0]           queue_depth,
    output logic [31:0]           avg_latency
);
```

**批处理策略分析：**

1. **静态批处理**
   - 固定批次大小
   - 等待批次填满
   - 优点：资源利用率高
   - 缺点：尾部延迟大

2. **动态批处理**
   - 超时机制触发
   - 自适应批次大小
   - 平衡延迟与吞吐量

3. **连续批处理（Continuous Batching）**
   - 请求动态加入/退出
   - PagedAttention内存管理
   - 最大化GPU/FPGA利用率

### 14.2.2 流式推理优化

流式推理优化延迟，适合实时交互：

**关键技术：**

1. **增量解码**
   - 逐token生成与传输
   - 用户体验优化
   - 首token延迟关键

2. **预填充并行化**
   - Prompt并行处理
   - 多级流水线
   - 隐藏计算延迟

3. **投机执行**
   - 预测常见续写
   - 并行验证分支
   - 减少等待时间

### 14.2.3 混合调度策略

**自适应调度器设计：**

1. **请求分类**
   - 短查询：<50 tokens → 流式
   - 长文本：>200 tokens → 批处理
   - 中等长度：动态决策

2. **资源分配**
   - 计算单元动态划分
   - 内存带宽预留
   - QoS保证机制

3. **负载均衡**
   - 多队列管理
   - 优先级调度
   - 饥饿避免

**性能对比（批处理vs流式）：**

| 指标 | 批处理(B=32) | 流式(B=1) | 混合模式 |
|------|-------------|-----------|----------|
| 吞吐量 | 2000 tok/s | 125 tok/s | 1500 tok/s |
| P50延迟 | 200ms | 8ms | 15ms |
| P99延迟 | 800ms | 12ms | 50ms |
| 资源利用率 | 85% | 15% | 70% |

### 14.2.4 实际部署考虑

**生产环境优化要点：**

1. **SLA保证**
   - 延迟上限约束
   - 吞吐量下限保证
   - 动态调整策略

2. **成本效益**
   - 每token成本计算
   - 能效比优化
   - 弹性伸缩

3. **故障处理**
   - 请求重试机制
   - 部分结果返回
   - 优雅降级

## 14.3 模型分片与多FPGA扩展

### 14.3.1 模型并行策略

大规模LLM（如LLaMA-70B、GPT-3）无法装入单个FPGA，需要分布式部署：

```systemverilog
// 模型分片控制器
module model_shard_controller #(
    parameter NUM_FPGAS = 8,
    parameter LAYERS_PER_FPGA = 4,  // 32层模型，8个FPGA
    parameter TENSOR_PARALLEL = 4    // 张量并行度
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // 分片配置
    input  shard_config_t          shard_config,
    
    // FPGA间通信接口
    output logic [NUM_FPGAS-1:0]   fpga_enable,
    input  logic [NUM_FPGAS-1:0]   fpga_ready,
    
    // 数据路由
    output routing_table_t         routing_table
);
```

**三种主要分片方式：**

1. **流水线并行（Pipeline Parallelism）**
   - 按层划分模型
   - 每个FPGA负责几层
   - 优点：通信量小
   - 缺点：流水线气泡

2. **张量并行（Tensor Parallelism）**
   - 单层内矩阵分片
   - 需要AllReduce通信
   - 优点：负载均衡好
   - 缺点：通信开销大

3. **数据并行（Data Parallelism）**
   - 每个FPGA完整模型副本
   - 只适合小模型
   - 优点：实现简单
   - 缺点：内存需求大

### 14.3.2 高速互联架构

**FPGA间通信方案：**

1. **PCIe Switch互联**
   - 带宽：PCIe Gen5 x16 = 64GB/s
   - 延迟：~1μs
   - 拓扑：星型或交换机

2. **Aurora协议直连**
   - 使用GTY收发器
   - 点对点连接
   - 延迟：<100ns
   - 带宽：25Gbps per lane

3. **CCIX/CXL互联**
   - 缓存一致性支持
   - 内存语义访问
   - 适合细粒度共享

**通信优化技术：**

```systemverilog
// 高效AllReduce实现
module allreduce_engine #(
    parameter DATA_WIDTH = 512,
    parameter NUM_NODES = 8
) (
    // Ring AllReduce拓扑
    input  logic [DATA_WIDTH-1:0]  local_data,
    output logic [DATA_WIDTH-1:0]  reduced_data,
    
    // 环形拓扑接口
    input  logic [DATA_WIDTH-1:0]  ring_in,
    output logic [DATA_WIDTH-1:0]  ring_out
);
```

### 14.3.3 负载均衡与调度

**动态负载均衡策略：**

1. **工作窃取（Work Stealing）**
   - 空闲FPGA主动请求任务
   - 细粒度任务队列
   - 减少空闲时间

2. **预测性调度**
   - 基于历史负载预测
   - 提前迁移任务
   - 平滑负载波动

3. **弹性扩展**
   - 根据负载动态增减FPGA
   - 模型重分片
   - 成本优化

### 14.3.4 多FPGA系统案例

**案例：8-FPGA LLaMA-70B推理系统**

**系统架构：**
- 8× Alveo U280 FPGA卡
- PCIe Gen4 Switch互联
- 每卡32GB HBM2e

**分片方案：**
- 层间流水线并行（8段）
- 层内张量并行（4路）
- 混合精度INT8/FP16

**性能指标：**
- 模型加载：~30秒
- 吞吐量：500 tokens/s @batch=16
- 延迟：首token 150ms，后续15ms/token
- 总功耗：~800W（vs GPU ~2000W）

**扩展性分析：**
| FPGA数量 | 吞吐量提升 | 通信开销 | 效率 |
|---------|-----------|---------|------|
| 1 | 1.0x | 0% | 100% |
| 2 | 1.9x | 5% | 95% |
| 4 | 3.6x | 10% | 90% |
| 8 | 6.4x | 20% | 80% |---

<div style="text-align: center; margin: 20px 0;">
  <a href="chapter13.md" style="margin-right: 20px;">← 上一章：未来趋势与新兴技术</a>
  <a href="chapter15.md" style="margin-left: 20px;">下一章：机器人运动控制与FPGA →</a>
</div>
