# 第十一章：AI加速器基础

本章深入探讨FPGA实现AI加速器的基础架构和设计方法。我们将从神经网络的计算特性分析出发，理解如何将算法映射到硬件架构，重点关注量化技术、并行计算模式以及片上互联设计。通过对比脉动阵列和数据流两种主流架构，您将掌握选择合适加速器架构的决策依据。本章还将对比FPGA、GPU和TPU在AI推理中的优劣势，为实际项目的平台选择提供指导。

## 11.1 神经网络计算特征分析

### 11.1.1 计算密集型特征

神经网络的核心计算可以归纳为几种基本操作：

```systemverilog
// 神经网络基本运算单元
module nn_compute_unit #(
    parameter DATA_WIDTH = 16,
    parameter VECTOR_SIZE = 64
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // 输入特征和权重
    input  logic [DATA_WIDTH-1:0]   feature_in[VECTOR_SIZE],
    input  logic [DATA_WIDTH-1:0]   weight_in[VECTOR_SIZE],
    input  logic [DATA_WIDTH-1:0]   bias_in,
    
    // 输出结果
    output logic [DATA_WIDTH*2-1:0] acc_out,
    output logic                    valid_out
);
    // MAC (Multiply-Accumulate) 阵列
    logic [DATA_WIDTH*2-1:0] products[VECTOR_SIZE];
    logic [DATA_WIDTH*2-1:0] acc_reg;
    
    // 并行乘法
    generate
        for (genvar i = 0; i < VECTOR_SIZE; i++) begin : gen_mult
            assign products[i] = feature_in[i] * weight_in[i];
        end
    endgenerate
    
    // 累加树
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_reg <= '0;
        end else begin
            acc_reg <= bias_in;
            for (int i = 0; i < VECTOR_SIZE; i++) begin
                acc_reg <= acc_reg + products[i];
            end
        end
    end
endmodule
```

**计算特征分析：**

1. **矩阵乘法密集**
   - 卷积层：90%+ 计算量
   - 全连接层：大量矩阵-向量乘法
   - 注意力机制：矩阵-矩阵乘法

2. **数据重用模式**
   - 权重重用：批处理推理
   - 输入重用：卷积滑窗
   - 部分和重用：大矩阵分块

3. **内存访问模式**
   - 顺序访问：利于突发传输
   - 可预测性：便于预取
   - 局部性：适合缓存设计

### 11.1.2 并行化机会

```systemverilog
// 多维并行化示例
module parallel_conv_engine #(
    parameter IN_CHANNELS = 64,
    parameter OUT_CHANNELS = 128,
    parameter KERNEL_SIZE = 3,
    parameter PARALLEL_IN = 8,    // 输入通道并行度
    parameter PARALLEL_OUT = 16   // 输出通道并行度
) (
    input  logic clk,
    input  logic rst_n,
    
    // 输入特征图块
    input  logic [15:0] ifmap[PARALLEL_IN][KERNEL_SIZE][KERNEL_SIZE],
    // 权重
    input  logic [15:0] weights[PARALLEL_OUT][PARALLEL_IN][KERNEL_SIZE][KERNEL_SIZE],
    // 输出特征图
    output logic [31:0] ofmap[PARALLEL_OUT],
    output logic        valid
);
    // PE (Processing Element) 阵列
    logic [31:0] partial_sums[PARALLEL_OUT][PARALLEL_IN];
    
    // 实例化PE阵列
    generate
        for (genvar oc = 0; oc < PARALLEL_OUT; oc++) begin : gen_oc
            for (genvar ic = 0; ic < PARALLEL_IN; ic++) begin : gen_ic
                pe_3x3 pe_inst (
                    .clk(clk),
                    .ifmap(ifmap[ic]),
                    .weight(weights[oc][ic]),
                    .psum_out(partial_sums[oc][ic])
                );
            end
        end
    endgenerate
    
    // 规约树
    generate
        for (genvar oc = 0; oc < PARALLEL_OUT; oc++) begin : gen_reduce
            always_ff @(posedge clk) begin
                ofmap[oc] <= '0;
                for (int ic = 0; ic < PARALLEL_IN; ic++) begin
                    ofmap[oc] <= ofmap[oc] + partial_sums[oc][ic];
                end
            end
        end
    endgenerate
endmodule
```

**并行化维度：**
- 批次并行（N维）
- 通道并行（C维）
- 空间并行（H/W维）
- 核并行（K维）

### 11.1.3 数据流分析

```systemverilog
// 数据流调度器
module dataflow_scheduler #(
    parameter TILE_SIZE = 16,
    parameter BUFFER_DEPTH = 1024
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // 外部存储接口
    axi_if.master       ddr_if,
    
    // 计算单元接口
    output logic [15:0] tile_a[TILE_SIZE][TILE_SIZE],
    output logic [15:0] tile_b[TILE_SIZE][TILE_SIZE],
    input  logic [31:0] tile_c[TILE_SIZE][TILE_SIZE],
    
    // 控制信号
    input  logic        start,
    output logic        done
);
    // 双缓冲设计
    logic [15:0] buffer_a[2][TILE_SIZE][TILE_SIZE];
    logic [15:0] buffer_b[2][TILE_SIZE][TILE_SIZE];
    logic        buffer_sel;
    
    // 状态机
    enum logic [2:0] {
        IDLE,
        LOAD_A,
        LOAD_B,
        COMPUTE,
        STORE_C
    } state;
    
    // 流水线控制
    always_ff @(posedge clk) begin
        case (state)
            LOAD_A: begin
                // 从DDR加载tile A到buffer[!buffer_sel]
                // 同时计算使用buffer[buffer_sel]
            end
            
            COMPUTE: begin
                // 触发矩阵乘法
                // 预取下一个tile
            end
        endcase
    end
endmodule
```

## 11.2 量化与定点推理

### 11.2.1 量化基础理论

量化是将浮点数映射到低位宽定点数的过程，对FPGA加速至关重要：

```systemverilog
// 量化器模块
module quantizer #(
    parameter FLOAT_WIDTH = 32,
    parameter FIXED_WIDTH = 8,
    parameter FRAC_BITS = 4      // 小数位数
) (
    input  logic [FLOAT_WIDTH-1:0] float_in,
    input  logic [31:0]            scale,
    input  logic [7:0]             zero_point,
    
    output logic [FIXED_WIDTH-1:0] fixed_out,
    output logic                   overflow
);
    // 量化公式：q = round(x/scale) + zero_point
    logic [63:0] scaled_val;
    logic [31:0] rounded_val;
    
    // 缩放
    assign scaled_val = float_in * (1 << FRAC_BITS) / scale;
    
    // 四舍五入
    assign rounded_val = scaled_val + (1 << (FRAC_BITS-1));
    
    // 饱和处理
    always_comb begin
        if (rounded_val > ((1 << FIXED_WIDTH) - 1)) begin
            fixed_out = (1 << FIXED_WIDTH) - 1;
            overflow = 1'b1;
        end else begin
            fixed_out = rounded_val[FIXED_WIDTH-1:0] + zero_point;
            overflow = 1'b0;
        end
    end
endmodule
```

**量化方案对比：**

1. **均匀量化**
   - INT8：推理精度损失<1%
   - INT4：需要感知训练
   - 二值/三值：极限压缩

2. **非均匀量化**
   - 对数量化
   - 自适应量化
   - 混合精度

### 11.2.2 定点运算单元设计

```systemverilog
// 定点MAC单元
module fixed_point_mac #(
    parameter IN_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    
    input  logic signed [IN_WIDTH-1:0]     input_data,
    input  logic signed [WEIGHT_WIDTH-1:0] weight,
    input  logic signed [ACC_WIDTH-1:0]    bias,
    
    output logic signed [ACC_WIDTH-1:0]    acc_out
);
    // 中间结果
    logic signed [IN_WIDTH+WEIGHT_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0]             acc_reg;
    
    // 乘法器（使用DSP）
    always_ff @(posedge clk) begin
        if (!rst_n)
            product <= '0;
        else if (en)
            product <= input_data * weight;
    end
    
    // 累加器
    always_ff @(posedge clk) begin
        if (!rst_n)
            acc_reg <= '0;
        else if (en) begin
            if (bias != '0)
                acc_reg <= bias + product;
            else
                acc_reg <= acc_reg + product;
        end
    end
    
    assign acc_out = acc_reg;
endmodule
```

### 11.2.3 动态量化策略

```systemverilog
// 动态量化控制器
module dynamic_quantization_controller (
    input  logic        clk,
    input  logic        rst_n,
    
    // 统计输入
    input  logic [31:0] activation_max,
    input  logic [31:0] activation_min,
    input  logic        stats_valid,
    
    // 量化参数输出
    output logic [7:0]  scale_factor,
    output logic [7:0]  zero_point,
    output logic        params_valid
);
    // 滑动窗口统计
    logic [31:0] max_history[8];
    logic [31:0] min_history[8];
    logic [2:0]  hist_ptr;
    
    // EMA (指数移动平均)
    logic [31:0] ema_max, ema_min;
    
    always_ff @(posedge clk) begin
        if (stats_valid) begin
            // 更新历史
            max_history[hist_ptr] <= activation_max;
            min_history[hist_ptr] <= activation_min;
            hist_ptr <= hist_ptr + 1;
            
            // 计算EMA
            ema_max <= (ema_max * 7 + activation_max) >> 3;
            ema_min <= (ema_min * 7 + activation_min) >> 3;
        end
    end
    
    // 计算量化参数
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            scale_factor <= 8'd128;
            zero_point <= 8'd128;
        end else begin
            // 计算scale和zero_point
            scale_factor <= (ema_max - ema_min) >> 8;
            zero_point <= -ema_min / scale_factor;
        end
    end
endmodule
```

### 11.2.4 混合精度推理

```systemverilog
// 混合精度计算单元
module mixed_precision_pe #(
    parameter MAX_WIDTH = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // 精度配置
    input  logic [1:0]  precision_mode, // 00:INT4, 01:INT8, 10:INT16
    
    // 灵活位宽输入
    input  logic [MAX_WIDTH-1:0] a_in,
    input  logic [MAX_WIDTH-1:0] b_in,
    
    // 输出
    output logic [MAX_WIDTH*2-1:0] result
);
    // 不同精度的计算路径
    logic [7:0]  a_int4, b_int4;
    logic [15:0] a_int8, b_int8;
    logic [31:0] result_int4, result_int8, result_int16;
    
    // 精度转换
    assign a_int4 = a_in[3:0];
    assign b_int4 = b_in[3:0];
    assign a_int8 = a_in[7:0];
    assign b_int8 = b_in[7:0];
    
    // 并行计算不同精度
    always_ff @(posedge clk) begin
        result_int4 <= a_int4 * b_int4;
        result_int8 <= a_int8 * b_int8;
        result_int16 <= a_in * b_in;
    end
    
    // 输出选择
    always_comb begin
        case (precision_mode)
            2'b00: result = {24'b0, result_int4[7:0]};
            2'b01: result = {16'b0, result_int8[15:0]};
            2'b10: result = result_int16;
            default: result = '0;
        endcase
    end
endmodule
```

**资源消耗对比（Zynq UltraScale+）：**
- INT4: 1 DSP可计算4个MAC
- INT8: 1 DSP可计算2个MAC  
- INT16: 1 DSP计算1个MAC
- FP16: 需要多个DSP级联

## 11.3 脉动阵列vs数据流架构

### 11.3.1 脉动阵列架构

```systemverilog
// 2D脉动阵列
module systolic_array_2d #(
    parameter ARRAY_SIZE = 16,
    parameter DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,
    
    // 输入数据流
    input  logic [DATA_WIDTH-1:0] a_in[ARRAY_SIZE],  // 从左侧输入
    input  logic [DATA_WIDTH-1:0] b_in[ARRAY_SIZE],  // 从顶部输入
    
    // 输出结果
    output logic [DATA_WIDTH*2-1:0] c_out[ARRAY_SIZE][ARRAY_SIZE]
);
    // PE互联信号
    logic [DATA_WIDTH-1:0] a_wire[ARRAY_SIZE][ARRAY_SIZE+1];
    logic [DATA_WIDTH-1:0] b_wire[ARRAY_SIZE+1][ARRAY_SIZE];
    
    // 边界连接
    generate
        for (genvar i = 0; i < ARRAY_SIZE; i++) begin
            assign a_wire[i][0] = a_in[i];
            assign b_wire[0][i] = b_in[i];
        end
    endgenerate
    
    // PE阵列实例化
    generate
        for (genvar row = 0; row < ARRAY_SIZE; row++) begin : gen_row
            for (genvar col = 0; col < ARRAY_SIZE; col++) begin : gen_col
                systolic_pe #(
                    .DATA_WIDTH(DATA_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .a_in(a_wire[row][col]),
                    .b_in(b_wire[row][col]),
                    .a_out(a_wire[row][col+1]),
                    .b_out(b_wire[row+1][col]),
                    .c_out(c_out[row][col])
                );
            end
        end
    endgenerate
endmodule

// 脉动处理单元
module systolic_pe #(
    parameter DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,
    
    input  logic [DATA_WIDTH-1:0] a_in,
    input  logic [DATA_WIDTH-1:0] b_in,
    output logic [DATA_WIDTH-1:0] a_out,
    output logic [DATA_WIDTH-1:0] b_out,
    output logic [DATA_WIDTH*2-1:0] c_out
);
    logic [DATA_WIDTH*2-1:0] acc_reg;
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_reg <= '0;
            a_out <= '0;
            b_out <= '0;
        end else begin
            // 数据传递
            a_out <= a_in;
            b_out <= b_in;
            // MAC操作
            acc_reg <= acc_reg + (a_in * b_in);
        end
    end
    
    assign c_out = acc_reg;
endmodule
```

**脉动阵列特点：**
- 规则的数据流动
- 高带宽利用率
- 固定的计算模式
- 适合矩阵乘法

### 11.3.2 数据流架构

```systemverilog
// 数据流处理引擎
module dataflow_engine #(
    parameter NUM_PES = 64,
    parameter FIFO_DEPTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    
    // 配置接口
    input  logic [31:0] config_data,
    input  logic        config_valid,
    
    // 数据输入输出
    input  logic [15:0] data_in,
    input  logic        data_in_valid,
    output logic        data_in_ready,
    
    output logic [15:0] data_out,
    output logic        data_out_valid,
    input  logic        data_out_ready
);
    // PE间FIFO连接
    logic [15:0] fifo_data[NUM_PES];
    logic        fifo_valid[NUM_PES];
    logic        fifo_ready[NUM_PES];
    
    // 可配置PE阵列
    generate
        for (genvar i = 0; i < NUM_PES; i++) begin : gen_pe
            configurable_pe pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                
                // 配置
                .config_data(config_data),
                .config_valid(config_valid & (config_data[7:0] == i)),
                
                // 数据路径
                .data_in(i == 0 ? data_in : fifo_data[i-1]),
                .data_in_valid(i == 0 ? data_in_valid : fifo_valid[i-1]),
                .data_in_ready(i == 0 ? data_in_ready : fifo_ready[i-1]),
                
                .data_out(fifo_data[i]),
                .data_out_valid(fifo_valid[i]),
                .data_out_ready(i == NUM_PES-1 ? data_out_ready : fifo_ready[i])
            );
        end
    endgenerate
    
    // 输出连接
    assign data_out = fifo_data[NUM_PES-1];
    assign data_out_valid = fifo_valid[NUM_PES-1];
endmodule
```

### 11.3.3 架构对比与选择

```systemverilog
// 混合架构示例
module hybrid_accelerator (
    input  logic clk,
    input  logic rst_n,
    
    // 控制
    input  logic mode_select, // 0: systolic, 1: dataflow
    
    // 统一接口
    axi_stream_if.slave  s_axis,
    axi_stream_if.master m_axis
);
    // 脉动阵列路径
    systolic_array_2d #(
        .ARRAY_SIZE(16)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),
        .a_in(/* 连接 */),
        .b_in(/* 连接 */),
        .c_out(/* 连接 */)
    );
    
    // 数据流路径
    dataflow_engine #(
        .NUM_PES(64)
    ) dataflow_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(/* 连接 */),
        .data_out(/* 连接 */)
    );
    
    // 路径选择逻辑
    always_comb begin
        if (mode_select == 1'b0) begin
            // 路由到脉动阵列
        end else begin
            // 路由到数据流引擎
        end
    end
endmodule
```

**架构选择指南：**

| 特性 | 脉动阵列 | 数据流 | 混合架构 |
|------|----------|--------|----------|
| 适用场景 | GEMM、卷积 | 不规则计算 | 多样化负载 |
| 数据重用 | 高 | 中 | 可配置 |
| 灵活性 | 低 | 高 | 高 |
| 面积效率 | 高 | 中 | 中 |
| 编程复杂度 | 低 | 高 | 高 |
| 功耗效率 | 优秀 | 良好 | 良好 |
| 内存带宽需求 | 低 | 高 | 中 |

**实际应用案例分析：**

1. **CNN推理加速器（脉动阵列）**
   - ResNet-50: 16×16脉动阵列
   - 数据重用率: 93%
   - DSP利用率: 95%
   - 功耗: 15W @200MHz

2. **Transformer加速器（数据流）**
   - BERT-Base: 可重构数据流
   - 支持动态序列长度
   - 内存带宽: 25.6GB/s
   - 延迟变化: ±15%

3. **通用AI加速器（混合架构）**
   - 支持CNN/RNN/Transformer
   - 模式切换开销: <1μs
   - 资源共享率: 80%
   - 面积开销: +20%

**性能建模与评估：**

```systemverilog
// 性能计数器模块
module performance_monitor (
    input  logic        clk,
    input  logic        rst_n,
    
    // 监控信号
    input  logic        compute_active,
    input  logic        memory_stall,
    input  logic        pipeline_flush,
    
    // 性能指标输出
    output logic [31:0] cycle_count,
    output logic [31:0] active_cycles,
    output logic [31:0] stall_cycles,
    output logic [31:0] mac_operations
);
    // 计数器逻辑
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= '0;
            active_cycles <= '0;
            stall_cycles <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (compute_active) active_cycles <= active_cycles + 1;
            if (memory_stall) stall_cycles <= stall_cycles + 1;
        end
    end
    
    // 计算利用率
    logic [7:0] utilization;
    assign utilization = (active_cycles * 100) / cycle_count;
endmodule
```

**资源预算分析（Zynq UltraScale+ ZU9EG）：**

```
脉动阵列 (16×16):
- LUT: 45K (8%)
- DSP: 256 (10%)
- BRAM: 128 (14%)
- 峰值性能: 51.2 GOPS @200MHz

数据流架构 (64 PE):
- LUT: 85K (15%)
- DSP: 192 (7.5%)
- BRAM: 256 (28%)
- 峰值性能: 38.4 GOPS @200MHz

混合架构:
- LUT: 110K (19%)
- DSP: 384 (15%)
- BRAM: 320 (35%)
- 峰值性能: 76.8 GOPS @200MHz
```

**决策树：**

```
1. 工作负载分析
   ├─ 规则计算为主（>80%）
   │  └─ 脉动阵列
   ├─ 不规则计算为主
   │  └─ 数据流
   └─ 混合负载
      └─ 混合架构

2. 约束条件评估
   ├─ 功耗受限 (<10W)
   │  └─ 优化脉动阵列
   ├─ 延迟敏感 (<1ms)
   │  └─ 并行数据流
   └─ 吞吐量优先
      └─ 深度流水线
```

## 11.4 FPGA vs GPU vs TPU

### 11.4.1 架构特征对比

**计算架构差异：**

```systemverilog
// FPGA特征：细粒度并行
module fpga_compute_unit (
    input  logic        clk,
    input  logic [7:0]  precision_config, // 可变精度
    
    // 自定义数据路径
    input  logic [127:0] custom_data_in,
    output logic [127:0] custom_data_out
);
    // 完全可定制的计算管线
    // 可以实现任意精度、任意运算
endmodule
```

**架构对比表：**

| 特性 | FPGA | GPU | TPU |
|------|------|-----|-----|
| 并行粒度 | 位级 | 线程级 | 矩阵级 |
| 内存层次 | 可定制 | 固定层次 | 专用缓存 |
| 数据精度 | 任意 | FP32/16/INT8 | INT8/BF16 |
| 功耗效率 | 最高 | 中等 | 高 |
| 编程模型 | HDL | CUDA/OpenCL | XLA |
| 部署灵活性 | 高 | 中 | 低 |

### 11.4.2 性能与功耗分析

**推理性能对比（ResNet-50）：**

```
FPGA (Xilinx VU9P):
- 吞吐量: 3,200 images/s
- 延迟: 0.31ms
- 功耗: 75W
- 性能/瓦特: 42.7 img/s/W

GPU (NVIDIA V100):
- 吞吐量: 7,800 images/s  
- 延迟: 0.13ms
- 功耗: 300W
- 性能/瓦特: 26 img/s/W

TPU v3:
- 吞吐量: 12,000 images/s
- 延迟: 0.08ms
- 功耗: 200W
- 性能/瓦特: 60 img/s/W
```

**延迟敏感场景分析：**

```systemverilog
// FPGA低延迟推理管线
module low_latency_inference (
    input  logic        clk,
    input  logic [15:0] sensor_data,
    output logic [7:0]  inference_result,
    output logic        result_valid
);
    // 流水线深度：8级
    // 固定延迟：40ns (8 cycles @ 200MHz)
    
    // 第1级：数据预处理
    logic [15:0] stage1_data;
    always_ff @(posedge clk) begin
        stage1_data <= sensor_data >> 2; // 归一化
    end
    
    // 第2-7级：神经网络层
    // ...省略中间级...
    
    // 第8级：输出
    always_ff @(posedge clk) begin
        inference_result <= final_activation;
        result_valid <= 1'b1;
    end
endmodule
```

### 11.4.3 应用场景选择

**决策矩阵：**

```
场景分析：
1. 边缘AI推理
   - 功耗受限 (<10W) → FPGA
   - 成本敏感 → 专用ASIC
   - 通用性要求 → 嵌入式GPU

2. 数据中心推理
   - 大批量处理 → TPU
   - 混合负载 → GPU
   - 定制算法 → FPGA

3. 实时处理
   - 确定性延迟 → FPGA
   - 高吞吐量 → TPU
   - 灵活部署 → GPU
```

**成本效益分析：**

| 指标 | FPGA | GPU | TPU |
|------|------|-----|-----|
| 硬件成本 | $5,000-20,000 | $10,000-15,000 | $30,000+ |
| 开发成本 | 高 | 中 | 低 |
| 部署时间 | 6-12月 | 1-3月 | 1月 |
| TCO (3年) | 中 | 高 | 低* |

*注：TPU仅在Google Cloud可用

### 11.4.4 混合计算架构

```systemverilog
// CPU+FPGA协同计算
module heterogeneous_compute (
    // PCIe接口到CPU
    pcie_if.slave cpu_if,
    
    // 高带宽内存接口
    hbm_if.master hbm_if,
    
    // 加速器控制
    input logic [31:0] accel_config
);
    // 任务调度器
    task_scheduler scheduler_inst (
        .cpu_tasks(cpu_queue),
        .fpga_tasks(fpga_queue),
        .load_balance_enable(1'b1)
    );
    
    // FPGA加速引擎
    ai_accelerator accel_inst (
        .task_in(fpga_queue),
        .result_out(result_queue)
    );
endmodule
```

**优化策略对比：**

1. **FPGA优化**
   - 算法-硬件协同设计
   - 定制数据通路
   - 近数据计算
   
2. **GPU优化**
   - 核函数融合
   - 张量核心利用
   - 多流并发

3. **TPU优化**
   - 批量大小调优
   - XLA编译优化
   - 模型量化
---

<div style="text-align: center; margin: 20px 0;">
  <a href="chapter10.md" style="margin-right: 20px;">← 上一章：未来趋势与新兴技术</a>
  <a href="chapter12.md" style="margin-left: 20px;">下一章：LLM推理加速 →</a>
</div>
