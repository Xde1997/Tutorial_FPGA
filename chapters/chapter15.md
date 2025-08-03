# 第15章：机器人运动控制与FPGA

机器人运动控制是实时计算的极致体现，特别是在人形机器人领域。本章将深入探讨如何利用FPGA的并行计算能力和确定性时序特性，构建高性能的机器人运动控制系统。我们将从系统架构开始，逐步深入到运动学计算、电机控制、传感器融合等核心技术，最终实现复杂的步态生成和平衡控制。

## 学习目标

完成本章学习后，您将能够：

1. **理解机器人控制系统的实时性要求**：掌握硬实时与软实时的区别，了解FPGA在满足微秒级控制周期中的优势
2. **设计高性能运动学求解器**：实现正向/逆向运动学的并行计算架构，优化浮点运算流水线
3. **构建多轴同步控制系统**：设计支持32+自由度的分布式电机控制网络，实现亚微秒级同步精度
4. **实现传感器融合算法**：整合IMU、力传感器、编码器等多源数据，构建高精度状态估计器
5. **开发力控制器**：实现阻抗控制、导纳控制等柔顺控制策略，处理环境交互
6. **优化步态生成器**：加速ZMP、DCM等平衡控制算法，实现动态步行和跑步

## 为什么选择FPGA进行机器人控制？

### 1. 确定性实时响应
- **控制周期**：1kHz基础控制，10kHz电流环，100kHz PWM生成
- **抖动控制**：<100ns的控制周期抖动，远优于基于CPU的方案
- **中断延迟**：硬件触发，无操作系统调度开销

### 2. 大规模并行计算
- **矩阵运算**：32x32雅可比矩阵计算在单周期内完成
- **多关节并行**：所有关节的运动学同时计算
- **传感器并行处理**：数百个传感器信号同时采集和预处理

### 3. 高速I/O接口
- **EtherCAT主站**：硬件实现的实时以太网协议栈
- **高速编码器接口**：支持BiSS-C、EnDat等高精度协议
- **模拟接口集成**：片上ADC直接采集力传感器信号

### 4. 功能安全特性
- **冗余设计**：关键路径的三模冗余（TMR）
- **看门狗定时器**：硬件级故障检测
- **安全状态机**：独立的安全监控逻辑

## 典型应用场景

### 人形机器人
- Boston Dynamics Atlas：液压驱动控制
- Agility Robotics Digit：电驱动双足机器人
- Tesla Optimus：高集成度执行器控制

### 工业机器人
- KUKA高速装配：亚毫米精度，毫秒级响应
- ABB协作机器人：力控制与碰撞检测
- Fanuc视觉引导：实时轨迹修正

### 特种机器人
- 手术机器人：微米级精度，触觉反馈
- 四足机器人：复杂地形适应
- 外骨骼系统：人机协同控制

## 性能指标参考

以Zynq UltraScale+ MPSoC (ZU9EG)为例：

| 功能模块 | 资源占用 | 性能指标 |
|---------|---------|---------|
| 32-DOF运动学求解器 | 45k LUT, 320 DSP | 10μs/周期 |
| EtherCAT主站(32从站) | 25k LUT, 8 BRAM | 100μs周期 |
| 传感器融合(EKF) | 35k LUT, 180 DSP | 1kHz更新率 |
| 力控制器(6轴) | 15k LUT, 96 DSP | 10kHz控制频率 |
| 步态生成器 | 40k LUT, 240 DSP | 100Hz规划频率 |

## 本章组织结构

1. **人形机器人控制系统架构**：探讨分层控制架构、实时通信网络设计
2. **实时运动学与动力学计算**：并行化DH参数计算、牛顿-欧拉动力学
3. **多轴电机控制与同步**：FOC算法实现、分布式时钟同步
4. **传感器融合与状态估计**：扩展卡尔曼滤波器、互补滤波器设计
5. **力控与柔顺控制实现**：阻抗/导纳控制、主动柔顺策略
6. **步态生成与平衡控制**：ZMP理论、捕获点控制、MPC优化

让我们从构建一个能够支撑复杂机器人行为的控制系统架构开始。

## 15.1 人形机器人控制系统架构

### 15.1.1 分层控制架构概述

现代人形机器人控制系统采用分层架构，每层负责不同时间尺度和抽象级别的控制任务。FPGA在这个架构中扮演关键角色，主要负责底层的高速实时控制。

```
任务规划层 (1-10 Hz)
    ↓
运动规划层 (10-100 Hz)  
    ↓
轨迹生成层 (100-1000 Hz)
    ↓
运动控制层 (1-10 kHz) ← FPGA主要作用域
    ↓
电机驱动层 (10-100 kHz) ← FPGA实现
```

### 15.1.2 FPGA控制器架构设计

#### 硬件平台选择
对于32自由度人形机器人，推荐使用Zynq UltraScale+ MPSoC：
- **PS端(ARM Cortex-A53)**：运行Linux，处理高层规划
- **PL端(FPGA)**：实现实时控制算法
- **RPU(Cortex-R5)**：安全监控和故障处理

#### 核心模块划分

```systemverilog
// 顶层控制器架构
module robot_controller_top #(
    parameter NUM_JOINTS = 32,
    parameter NUM_IMU = 4,
    parameter NUM_FT_SENSORS = 4
)(
    input  clk_200mhz,      // 主时钟
    input  clk_100mhz_sync, // 同步时钟
    
    // EtherCAT接口
    output [3:0] ethercat_tx_p,
    input  [3:0] ethercat_rx_p,
    
    // 高速传感器接口
    input  [NUM_IMU-1:0] imu_spi_miso,
    output [NUM_IMU-1:0] imu_spi_mosi,
    
    // AXI接口到PS
    axi4_if.master m_axi_hp0,
    axi4_lite_if.slave s_axi_gp0
);
```

### 15.1.3 实时通信网络设计

#### EtherCAT主站实现

EtherCAT是机器人控制的事实标准，提供确定性的实时通信：

**关键特性**：
- 分布式时钟同步：<1μs精度
- 循环时间：最快可达31.25μs
- 拓扑灵活：线型、星型、树型

**FPGA实现优势**：
```systemverilog
// EtherCAT处理引擎框架
module ethercat_master_core (
    // 时钟域：200MHz处理，125MHz网络
    input  clk_proc,
    input  clk_net,
    
    // 数据报处理
    output logic [31:0] cyclic_data [0:MAX_SLAVES-1],
    input  logic [31:0] command_data [0:MAX_SLAVES-1],
    
    // 分布式时钟
    output logic [63:0] dc_time,
    output logic dc_sync_pulse
);

// 关键优化点：
// 1. 零拷贝DMA传输
// 2. 硬件CRC计算
// 3. 并行从站数据提取
```

#### 时间同步机制

精确的时间同步是多轴协调控制的基础：

1. **IEEE 1588 PTP硬件时间戳**
   - 纳秒级时间戳生成
   - 硬件辅助的时钟伺服

2. **分布式时钟补偿**
   - 传输延迟测量
   - 本地时钟漂移补偿

### 15.1.4 内存架构与数据流

#### 高带宽内存访问模式

```systemverilog
// 乒乓缓冲区设计
module trajectory_buffer #(
    parameter DEPTH = 1024,
    parameter WIDTH = 32 * NUM_JOINTS
)(
    // 写端口 - 从PS
    input  wr_clk,
    input  wr_en,
    input  [WIDTH-1:0] wr_data,
    
    // 读端口 - 到控制器
    input  rd_clk,
    output [WIDTH-1:0] rd_data,
    output buffer_ready
);

// 实现要点：
// 1. 使用URAM实现深缓冲
// 2. 多端口并行访问
// 3. 预取机制减少延迟
```

#### DMA引擎设计

高效的DMA引擎对于PS-PL数据交换至关重要：

```systemverilog
// 散列-聚集DMA
module sg_dma_engine (
    // AXI4内存接口
    axi4_if.master m_axi,
    
    // 流接口
    axis_if.master m_axis_s2mm,  // Stream to Memory Map
    axis_if.slave  s_axis_mm2s   // Memory Map to Stream
);
```

**优化策略**：
1. **突发传输优化**：最大化AXI突发长度
2. **通道交织**：多通道并行传输
3. **描述符预取**：隐藏描述符获取延迟

### 15.1.5 中断与事件处理

#### 硬件中断控制器

```systemverilog
module interrupt_controller (
    input  [63:0] irq_sources,
    output [7:0]  irq_to_ps,
    
    // 优先级配置
    input  [63:0][2:0] irq_priority,
    input  [63:0]      irq_enable
);

// 关键中断源：
// - 紧急停止
// - 碰撞检测
// - 通信故障
// - 控制周期超时
```

#### 事件时序管理

精确的事件调度确保控制算法按时执行：

```systemverilog
module timing_generator (
    output logic tick_1khz,   // 主控制周期
    output logic tick_10khz,  // 电流环
    output logic tick_100hz   // 轨迹更新
);

// 使用MMCM生成精确时钟
// 相位对齐确保同步
```

### 15.1.6 故障检测与安全机制

#### 硬件看门狗

```systemverilog
module safety_monitor (
    input  clk,
    input  [NUM_JOINTS-1:0] joint_ok,
    input  comm_ok,
    input  power_ok,
    
    output emergency_stop,
    output [3:0] fault_code
);

// 实现特性：
// 1. 独立时钟域
// 2. 三模冗余关键信号
// 3. 故障注入测试接口
```

#### 安全状态机

确保系统在任何情况下都能进入安全状态：

```
状态转换图：
INIT → READY → OPERATIONAL
  ↓      ↓         ↓
  └──→ SAFE ←──────┘
         ↓
      EMERGENCY
```

### 15.1.7 性能监控与调试

#### 硬件性能计数器

```systemverilog
module performance_monitor (
    input  clk,
    input  [31:0] event_vector,
    
    output [63:0] cycle_count,
    output [31:0] event_counts [0:15]
);

// 监控指标：
// - 控制周期时间
// - DMA传输延迟
// - 计算模块利用率
// - 通信错误率
```

#### 集成逻辑分析器(ILA)

在线调试对于复杂系统至关重要：

```tcl
# Vivado ILA配置
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 8192 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
```

### 15.1.8 实际案例：Atlas机器人控制系统

Boston Dynamics Atlas采用的控制架构值得借鉴：

**系统规格**：
- 28个液压执行器
- 1kHz主控制频率
- 每个关节3个传感器（位置、压力、温度）

**FPGA实现要点**：
1. **液压阀控制**：高精度PWM生成，死区补偿
2. **压力反馈处理**：24位ADC采样，数字滤波
3. **泄漏检测**：实时压力曲线分析
4. **多模态切换**：站立/行走/跳跃模式无缝切换

**性能优化结果**：
- 控制延迟：<500μs（传感器到执行器）
- 同步精度：所有关节<10μs
- 功耗：15W（仅FPGA部分）

## 15.2 实时运动学与动力学计算

机器人运动学和动力学计算是控制系统的核心，决定了机器人的运动精度和响应速度。FPGA的并行计算能力使得复杂的矩阵运算能够在微秒级完成。

### 15.2.1 正向运动学并行实现

#### Denavit-Hartenberg (DH)参数方法

DH参数是描述机器人运动链的标准方法：

```systemverilog
// DH参数结构
typedef struct packed {
    logic signed [31:0] a;     // 连杆长度
    logic signed [31:0] alpha; // 连杆扭转
    logic signed [31:0] d;     // 连杆偏移
    logic signed [31:0] theta; // 关节角度
} dh_params_t;

// 单个关节变换矩阵计算
module dh_transform #(
    parameter FIXED_POINT_WIDTH = 32,
    parameter FRACTION_BITS = 16
)(
    input  clk,
    input  dh_params_t params,
    output logic [3:0][3:0][31:0] T_matrix
);
```

#### 并行变换矩阵计算

关键优化：所有关节的变换矩阵同时计算

```systemverilog
module parallel_forward_kinematics #(
    parameter NUM_JOINTS = 32
)(
    input  clk,
    input  dh_params_t joint_params [0:NUM_JOINTS-1],
    output logic [3:0][3:0][31:0] T_matrices [0:NUM_JOINTS-1],
    output logic valid
);

// 实现策略：
// 1. 每个关节配备独立的三角函数计算单元
// 2. 使用CORDIC算法计算sin/cos
// 3. 流水线化矩阵乘法
```

#### CORDIC算法优化

CORDIC算法特别适合FPGA实现三角函数：

```systemverilog
module cordic_sincos #(
    parameter WIDTH = 32,
    parameter ITERATIONS = 16
)(
    input  clk,
    input  [WIDTH-1:0] angle,  // 定点数格式
    output [WIDTH-1:0] sin_out,
    output [WIDTH-1:0] cos_out,
    output valid
);

// CORDIC迭代流水线
// 每个迭代使用独立的加法器和移位器
// 总延迟 = ITERATIONS个时钟周期
```

**资源优化技巧**：
1. **查找表结合CORDIC**：粗精度LUT + 精细CORDIC修正
2. **资源共享**：时分复用CORDIC核心
3. **精度权衡**：根据机器人精度要求选择迭代次数

### 15.2.2 逆向运动学加速

逆向运动学(IK)是从目标位姿计算关节角度，计算复杂度高。

#### 雅可比矩阵方法

```systemverilog
module jacobian_ik_solver #(
    parameter NUM_JOINTS = 7,  // 7自由度机械臂
    parameter MAX_ITERATIONS = 10
)(
    input  clk,
    input  [3:0][3:0][31:0] target_pose,
    input  [31:0] joint_angles [0:NUM_JOINTS-1],
    
    output [31:0] solution [0:NUM_JOINTS-1],
    output logic converged,
    output logic [3:0] iterations_used
);

// 核心计算步骤：
// 1. 计算当前位姿（正向运动学）
// 2. 计算误差
// 3. 计算雅可比矩阵
// 4. 求解 Δq = J^+ * Δx （伪逆）
// 5. 更新关节角度
```

#### 并行矩阵求逆

雅可比伪逆计算是IK的瓶颈，需要高度优化：

```systemverilog
module matrix_pseudoinverse #(
    parameter ROWS = 6,
    parameter COLS = 7
)(
    input  clk,
    input  [31:0] J_matrix [0:ROWS-1][0:COLS-1],
    output [31:0] J_pinv [0:COLS-1][0:ROWS-1],
    output valid
);

// 使用SVD分解：J = U*S*V^T
// J^+ = V*S^+*U^T
// 
// 优化方案：
// 1. 固定点迭代SVD
// 2. 阈值处理小奇异值
// 3. 块矩阵分解减少延迟
```

#### 解析解加速器

对于特定机器人构型，解析IK解更高效：

```systemverilog
module analytical_ik_7dof (
    input  clk,
    input  [3:0][3:0][31:0] target_pose,
    output [31:0] joint_angles [0:6],
    output [2:0] num_solutions,  // 最多8个解
    output valid
);

// 7自由度机械臂典型有8个解
// 并行计算所有可能解
// 基于约束选择最优解
```

### 15.2.3 动力学计算引擎

#### 递归牛顿-欧拉算法

高效的动力学计算对于力控制至关重要：

```systemverilog
module newton_euler_dynamics #(
    parameter NUM_JOINTS = 32
)(
    input  clk,
    // 运动状态
    input  [31:0] q [0:NUM_JOINTS-1],      // 位置
    input  [31:0] qd [0:NUM_JOINTS-1],     // 速度
    input  [31:0] qdd [0:NUM_JOINTS-1],    // 加速度
    
    // 动力学参数
    input  [31:0] mass [0:NUM_JOINTS-1],
    input  [3:0][3:0][31:0] inertia [0:NUM_JOINTS-1],
    
    // 输出力矩
    output [31:0] tau [0:NUM_JOINTS-1],
    output valid
);

// 两遍递归：
// 1. 前向递归：计算速度和加速度
// 2. 后向递归：计算力和力矩
```

#### 并行化策略

动力学计算的并行化需要仔细设计：

```systemverilog
// 前向递归并行化
module forward_recursion_parallel (
    // 使用依赖图调度
    // 同一深度的节点并行计算
);

// 后向递归并行化  
module backward_recursion_parallel (
    // 从叶节点开始
    // 逐层向根节点传播
);
```

**优化技术**：
1. **空间换时间**：预计算常用变换矩阵
2. **定点数优化**：根据精度要求选择位宽
3. **流水线平衡**：确保各级延迟匹配

### 15.2.4 惯性矩阵与科氏力计算

#### 惯性矩阵实时更新

```systemverilog
module mass_matrix_computer #(
    parameter NUM_JOINTS = 32
)(
    input  clk,
    input  [31:0] q [0:NUM_JOINTS-1],
    output [31:0] M_matrix [0:NUM_JOINTS-1][0:NUM_JOINTS-1],
    output valid
);

// M(q)计算优化：
// 1. 利用对称性，只计算上三角
// 2. 复合刚体算法减少计算量
// 3. 增量更新而非完全重算
```

#### 科氏和离心力向量

```systemverilog
module coriolis_computer (
    input  [31:0] q [0:NUM_JOINTS-1],
    input  [31:0] qd [0:NUM_JOINTS-1],
    input  [31:0] M_matrix [0:NUM_JOINTS-1][0:NUM_JOINTS-1],
    
    output [31:0] C_vector [0:NUM_JOINTS-1]
);

// 使用Christoffel符号方法
// C(q,qd) = Σ_j Σ_k c_ijk * qd_j * qd_k
```

### 15.2.5 重力补偿计算

#### 并行重力向量计算

```systemverilog
module gravity_compensation (
    input  clk,
    input  [31:0] q [0:NUM_JOINTS-1],
    input  [2:0][31:0] gravity_vector,  // [0, 0, -9.81]
    
    output [31:0] G_vector [0:NUM_JOINTS-1]
);

// 每个连杆的重力贡献并行计算
// 考虑质心位置和姿态
```

### 15.2.6 实时碰撞检测

#### 几何基元碰撞检测

```systemverilog
module collision_detector #(
    parameter NUM_LINKS = 32,
    parameter NUM_OBSTACLES = 16
)(
    input  clk,
    input  [3:0][3:0][31:0] link_poses [0:NUM_LINKS-1],
    input  obstacle_t obstacles [0:NUM_OBSTACLES-1],
    
    output logic collision_detected,
    output logic [NUM_LINKS-1:0] collision_map,
    output [31:0] min_distance
);

// 并行检测策略：
// 1. 包围盒预筛选
// 2. GJK算法精确检测
// 3. 距离场加速
```

#### 自碰撞检测优化

```systemverilog
// 只检测可能碰撞的连杆对
parameter bit [NUM_LINKS-1:0][NUM_LINKS-1:0] COLLISION_PAIRS = {
    // 预定义的碰撞检测对
};
```

### 15.2.7 性能基准与优化

#### 典型性能指标

| 计算任务 | 关节数 | 延迟 | 吞吐量 | 资源占用 |
|---------|--------|------|---------|----------|
| 正向运动学 | 32 | 2μs | 500kHz | 25k LUT, 128 DSP |
| 逆向运动学(7DOF) | 7 | 10μs | 100kHz | 35k LUT, 96 DSP |
| 动力学计算 | 32 | 15μs | 66kHz | 45k LUT, 256 DSP |
| 碰撞检测 | 32 links | 5μs | 200kHz | 20k LUT, 64 DSP |

#### 资源优化策略

1. **DSP使用优化**
   ```systemverilog
   // 使用DSP48E2的级联特性
   (* use_dsp = "yes" *)
   wire [47:0] p = a * b + c;
   ```

2. **存储器优化**
   - 参数存储在BRAM
   - 中间结果使用分布式RAM
   - 查找表存储在URAM

3. **时钟域优化**
   - 高速计算核心：300MHz
   - 接口逻辑：100MHz
   - 降低跨时钟域传输

### 15.2.8 案例研究：Digit机器人运动学实现

Agility Robotics的Digit双足机器人运动学特点：

**系统配置**：
- 20个自由度（每腿5个，每臂4个，躯干2个）
- 特殊的4连杆腿部设计
- 闭链运动学约束

**FPGA实现亮点**：
1. **闭链求解器**：专门的约束满足模块
2. **动态重配置**：根据步态切换运动学模型
3. **预测控制集成**：运动学与MPC紧密耦合

**优化结果**：
- 全身IK求解：<50μs
- 支持100Hz MPC更新率
- 功耗效率：2W/DOF
---

<div style="text-align: center; margin: 20px 0;">
  <a href="chapter14.md" style="margin-right: 20px;">← 上一章：未来趋势与新兴技术</a>
  <a href="chapter16.md" style="margin-left: 20px;">下一章：激光雷达信号处理与FPGA →</a>
</div>
