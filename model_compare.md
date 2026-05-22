# LeRobot策略模型对比分析

本文档对LeRobot库中实现的各种策略模型进行对比分析，包括它们的优缺点和适用场景。最后更新于代码版本 v0.5.2。

## 目录

- [1. ACT (Action Chunking Transformer)](#1-act-action-chunking-transformer)
- [2. Diffusion (扩散策略)](#2-diffusion-扩散策略)
- [3. TD-MPC (时序差分模型预测控制)](#3-td-mpc-时序差分模型预测控制)
- [4. VQ-BeT (向量量化行为Transformer)](#4-vq-bet-向量量化行为transformer)
- [5. PI0 / PI0.5 / PI0-Fast (Physical Intelligence系列)](#5-pi0--pi05--pi0-fast-physical-intelligence系列)
- [6. SmolVLA (轻量视觉-语言-动作模型)](#6-smolvla-轻量视觉-语言-动作模型)
- [7. Wall-X (跨具身机器人控制)](#7-wall-x-跨具身机器人控制)
- [8. XVLA (扩展视觉-语言-动作模型)](#8-xvla-扩展视觉-语言-动作模型)
- [9. MultiTaskDiT (多任务扩散Transformer)](#9-multitaskdit-多任务扩散transformer)
- [10. EO1](#10-eo1)
- [11. GR00T (NVIDIA)](#11-groot-nvidia)
- [12. GaussianActor / SAC (高斯Actor / 软演员-评论家)](#12-gaussianactor--sac-高斯actor--软演员-评论家)
- [综合对比表](#综合对比表)
- [选择建议](#选择建议)

---

## 1. ACT (Action Chunking Transformer)

**实现文件**: `src/lerobot/policies/act/` | 注册名: `act`

基于论文 *"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"* (2304.13705)。

### 架构
- Transformer编码器-解码器架构
- ResNet视觉骨干网络
- 可选VAE（变分自编码器）目标
- 预测多个未来动作的时间分块（action chunks）
- 推理时使用时间集成（temporal ensembling）

### 优点
- **轻量高效**：在众多策略中参数量最小，推理速度最快
- **训练稳定**：收敛快，超参数不敏感，适合首次训练
- **低显存需求**：batch=4时峰值显存仅约0.94 GB（AdamW下约1.5-2 GB）
- **单任务表现优秀**：对50-100个episode的抓取/放置任务通常可达>70%成功率

### 缺点
- **无语言条件化**：不支持语言指令，无法做多任务
- **单模态**：不支持多模态输入融合
- **单一框架**：仅支持监督学习（行为克隆），不支持强化学习

### 适用场景
- 首次训练、笔记本电脑、入门用户
- 单任务精细操作（抓取、放置、折叠）
- GPU显存 < 8 GB 的首选策略
- 双手操作任务（ALOHA风格）

---

## 2. Diffusion (扩散策略)

**实现文件**: `src/lerobot/policies/diffusion/` | 注册名: `diffusion`

基于论文 *"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"* (2303.04137)。

### 架构
- 1D CNN U-Net去噪网络
- 视觉观测（单/多相机）和状态观测条件化
- DDPM / DDIM噪声调度器可选
- 在动作轨迹上学习去噪扩散过程

### 优点
- **多模态动作分布**：适合建模非高斯、多峰的动作分布
- **动作平滑**：扩散过程自然产生平滑的动作序列
- **样本效率好**：在有限数据下表现良好
- **噪声鲁棒**：对输入噪声具有一定鲁棒性

### 缺点
- **推理速度较慢**：需要多步迭代去噪（默认100步，可减少）
- **显存占用中等**：batch=4时约4.94 GB
- **训练时间较长**：收敛通常比ACT慢2-3倍

### 适用场景
- 视觉运动任务（PushT等）
- 数据量有限但需要高质量多模态动作生成
- GPU显存 4-8 GB 的备选方案

---

## 3. TD-MPC (时序差分模型预测控制)

**实现文件**: `src/lerobot/policies/tdmpc/` | 注册名: `tdmpc`

结合 *"Finetuning Offline World Models in the Real World"* (2310.16029) 和 *"Temporal Difference Learning for Model Predictive Control"* (2203.04955)。

### 架构
- 潜在世界模型（latent world model）
- 模型预测控制（MPC）+ 采样规划
- Q函数 + 策略网络 + 目标网络
- 支持动作重复（action repeats）和多步执行

### 优点
- **规划能力**：使用世界模型进行前瞻性动作规划
- **离线预训练+在线微调**：支持从离线数据集预训练后在线适应
- **不确定性估计**：Q函数集成提供不确定性估计
- **长期视野**：适合需要长期规划的任务

### 缺点
- **组件复杂**：包含世界模型、策略网络、Q函数、MPC规划器等多个组件
- **训练难度高**：需要平衡多个损失函数
- **规划过程计算密集**：MPC采样规划增加推理开销
- **不推荐新手**：调试和超参数调整门槛高

### 适用场景
- 基于模型的控制任务
- 需要长期规划和在线适应的场景
- 有一定强化学习经验的用户

---

## 4. VQ-BeT (向量量化行为Transformer)

**实现文件**: `src/lerobot/policies/vqbet/` | 注册名: `vqbet`

基于论文 *"Behavior Generation with Latent Actions"*。

### 架构
- VQ-VAE将连续动作分块离散化为潜在编码
- GPT风格Transformer预测离散潜在编码序列
- 两阶段训练：先训练Residual VQ，再训练GPT
- 观测条件化（图像+状态）

### 优点
- **离散动作表示**：将连续动作空间转化为离散编码
- **高效训练**：两阶段训练过程简化复杂度
- **Transformer序列建模**：强大的动作序列建模能力
- **代码预测+连续偏移**：结合离散编码和连续值

### 缺点
- **分阶段训练**：需要先训练RVQ再训练GPT
- **组件多**：RVQ、GPT、预测头等多个模块
- **量化误差**：离散化过程可能引入精度损失

### 适用场景
- 行为克隆任务
- 需要序列建模的复杂操作
- 需要使用离散动作表示的研究

---

## 5. PI0 / PI0.5 / PI0-Fast (Physical Intelligence系列)

**实现文件**: `src/lerobot/policies/pi0/`, `pi05/`, `pi0_fast/` | 注册名: `pi0`, `pi05`, `pi0_fast`

Physical Intelligence的视觉-语言-动作策略系列。

### 5.1 PI0

#### 架构
- PaliGemma VLM（`gemma_2b`）作为视觉语言主干
- Gemma 300M 动作专家层（action expert）
- 流匹配（flow matching）连续动作生成
- 动作分块大小50
- 支持RTC（Real-Time Chunking）

#### 优点
- **多模态集成**：视觉 + 语言 + 动作端到端
- **语言条件化**：支持自然语言任务指定
- **通用性强**：适合多任务机器人控制
- **RTC支持**：实时分块推理

#### 缺点
- **模型规模大**：~3B参数，显存需求高（batch=1时~15.5 GB，AdamW下更高）
- **推理速度较慢**：VLM前向+流匹配多步
- **训练复杂**：需要HuggingFace token登录获取gated模型

### 5.2 PI0.5

PI0的精进版本，架构与PI0类似（PaliGemma VLM + Gemma 300M动作专家 + 流匹配），但经过改进优化。占用显存略高于PI0（batch=1时~16.35 GB）。

### 5.3 PI0-Fast

PI0的快速变体：
- 使用PiGemma动作专家替代Gemma 300M
- **离散动作标记化**：支持最多256个动作token，使用IDCT（逆离散余弦变换）
- **相对动作**：支持绝对动作到相对动作的转换

#### PI0-Fast优点
- **更快推理**：离散token化提高处理效率
- **紧凑表示**：动作压缩减少存储需求
- 适合资源受限环境

#### PI0-Fast缺点
- **离散化误差**：token化可能引入精度损失
- **解码复杂**：需要专门的IDCT解码

### 适用场景
- **PI0/PI0.5**: GPU显存 24+ GB、通用机器人控制、需要语言指令的任务
- **PI0-Fast**: 需要更快推理的部署场景

---

## 6. SmolVLA (轻量视觉-语言-动作模型)

**实现文件**: `src/lerobot/policies/smolvla/` | 注册名: `smolvla`

Hugging Face设计的轻量VLA模型，参考论文 (2506.01844)。

### 架构
- 预训练VLM（SmolVLM） + 动作专家层
- 流匹配（flow matching）连续动作生成
- 动作分块大小50
- 图像resize到512x512并padding
- 支持RTC

### 优点
- **紧凑高效**：比PI0系列小得多（batch=1时约3.93 GB）
- **语言条件化**：支持多任务语言指令
- **预训练VLM**：视觉语言骨干已预训练，收敛快
- **解冻视觉编码器**：可以大幅提升特定任务性能（`--policy.freeze_vision_encoder=false`）

### 缺点
- **依赖特定VLM**：依赖SmolVLM模型
- **多模态训练数据需求**：需要图文动作配对数据
- **默认batch小**：大batch训练受限于显存

### 适用场景
- 12-16 GB显存的中间GPU（4070/4080）
- 语言条件化多任务学习
- 需要快速收敛的VLA应用

---

## 7. Wall-X (跨具身机器人控制)

**实现文件**: `src/lerobot/policies/wall_x/` | 注册名: `wall_x`

### 架构
- Qwen2.5-VL视觉语言模型骨干
- **流匹配 + ODE求解器**：使用`torchdiffeq.odeint`进行基于常微分方程的流匹配
- 统一动作表示（20维动作空间）
- LoRA/PEFT微调支持
- 动作分块大小32

### 优点
- **跨具身通用性**：统一动作表示适配多种机器人
- **多模态学习**：视觉+语言+动作数据联合
- **LoRA微调**：高效参数微调，降低训练成本
- **ODE求解**：精确流匹配求解

### 缺点
- **显存需求高**：batch=1时约15.95 GB
- **模型大**：基于Qwen2.5-VL大型VLM
- **依赖多**：需要transformers、peft、torchdiffeq等多个可选包

### 适用场景
- 跨具身机器人控制
- 多模态学习研究
- GPU显存 24+ GB

---

## 8. XVLA (扩展视觉-语言-动作模型)

**实现文件**: `src/lerobot/policies/xvla/` | 注册名: `xvla`

### 架构
- Florence-2视觉语言嵌入
- Soft-Prompted Transformer时序/动作头
- 自动检测动作维度
- 支持二值/离散/连续动作空间
- 动作分块大小32 + 流匹配

### 优点
- **灵活动作空间**：兼容多种动作类型
- **自动维度检测**：适应不同机器人
- **本体感知**：支持proprioception输入
- **Florence-2嵌入**：强大视觉语言特征

### 缺点
- **显存需求较高**：batch=1时约15.52 GB
- **依赖Florence-2**：模型依赖特定VLM
- **架构复杂**：嵌入拼接+Transformer时序头

### 适用场景
- 多动作类型机器人控制
- 需要灵活动作空间的大规模训练
- GPU显存 24+ GB

---

## 9. MultiTaskDiT (多任务扩散Transformer)

**实现文件**: `src/lerobot/policies/multi_task_dit/` | 注册名: `multi_task_dit`

### 架构
- Transformer架构 + 扩散/流匹配双目标
- CLIP文本和视觉编码器（双编码器）
- 支持文本+视觉条件化的多任务学习
- 参考Boston Dynamics "Large Behavior Models"和arxiv 2507.05331

### 优点
- **扩散+流匹配双模式**：灵活选择生成方式
- **多任务学习**：通过CLIP文本编码实现任务条件化
- **Transformer架构**：强大的跨模态融合能力
- **视觉+语言**：双编码器支持多模态条件化

### 缺点
- **双编码器开销**：CLIP文本+视觉编码器增加计算量
- **训练复杂**：扩散和流匹配两种模式需要不同调度
- **CLIP依赖**：依赖transformers和diffusers包

### 适用场景
- 多任务语言条件化策略
- 需要扩散和流匹配灵活切换的研究
- 大规模行为模型训练

---

## 10. EO1

**实现文件**: `src/lerobot/policies/eo1/` | 注册名: `eo1`

### 架构
- Qwen2.5-VL-3B-Instruct VLM骨干
- 流匹配动作头
- 零填充状态/动作维度适配EO1流头
- 余弦衰减+warmup调度器

### 优点
- **原生EO1集成**：专为EO1机器人优化的策略
- **Qwen2.5-VL骨干**：强大的3B视觉语言模型
- **流匹配动作**：连续动作生成

### 缺点
- **硬件特定**：主要为EO1机器人设计
- **VLM依赖**：需要Qwen2.5-VL相关依赖

### 适用场景
- EO1机器人控制
- Qwen VLM生态中的机器人应用

---

## 11. GR00T (NVIDIA)

**实现文件**: `src/lerobot/policies/groot/` | 注册名: `groot`

### 架构
- NVIDIA Isaac-GR00T模型（GR00TN15）包装器
- 自动下载和加载预训练GR00T模型
- 动作维度对齐和零填充
- 跨具身控制

### 优点
- **工业级预训练**：NVIDIA大规模预训练模型
- **跨具身**：支持多种机器人形态
- **即插即用**：从预训练权重加载

### 缺点
- **外部依赖重**：需要NVIDIA GR00T生态（flash-attn、timm等）
- **安装复杂**：当前在`all` extra中被注释掉，需手动安装
- **黑盒程度高**：模型内部细节不透明

### 适用场景
- 有NVIDIA GPU和GR00T环境的用户
- 跨具身通用机器人控制

---

## 12. GaussianActor / SAC (高斯Actor / 软演员-评论家)

**实现文件**: 
- 策略: `src/lerobot/policies/gaussian_actor/` (注册名: `gaussian_actor`)
- 算法: `src/lerobot/rl/algorithms/sac/` (注册名: `sac`)

### 架构
- **GaussianActorPolicy**: TanhMultivariateNormalDiag高斯分布actor
  - 观测编码器（视觉+状态）
  - MLP动作头 + 离散Critic
  - 单步动作预测（无分块）
- **SACAlgorithm**: SAC强化学习算法
  - 双Q函数 + 目标网络
  - 温度/熵自动调节
  - Actor-Learner架构 + 经验回放缓冲区

### 优点
- **最大熵框架**：平衡奖励和熵，提高探索能力
- **在线学习**：支持与环境交互进行训练
- **稳定训练**：目标网络和双Q确保稳定性
- **离线到在线**：支持离线预训练+在线微调

### 缺点
- **需要环境交互**：纯在线RL模式需要仿真环境
- **样本效率低**：相比模仿学习需要更多样本
- **超参数敏感**：actor/critic/temperature多组学习率需调参
- **不支持动作分块**：单步动作预测，不适合需要动作序列的任务

### 适用场景
- 强化学习离线/在线训练
- 需要探索和在线适应的场景
- 有仿真环境（Aloha、PushT等）的用户

---

## 综合对比表

| 策略 | 架构类型 | 动作生成 | 语言支持 | 显存(batch=1) | 推理速度 | 学习范式 | 分块 | 适用场景 |
|------|---------|---------|---------|--------------|---------|---------|------|---------|
| ACT | Transformer+VAE | 直接回归 | ❌ | ~0.94 GB | ★★★★★ | 模仿学习 | 20 | 入门、单任务、低显存 |
| Diffusion | CNN U-Net | 扩散去噪 | ❌ | ~4.94 GB | ★★★ | 模仿学习 | 16 | 多模态动作、视觉运动 |
| TD-MPC | 世界模型+MPC | MPC采样规划 | ❌ | 中 | ★★★ | 模仿+RL | repeater | 长期规划、在线适应 |
| VQ-BeT | VQ-VAE+GPT | 离散token预测 | ❌ | 中 | ★★★★ | 模仿学习 | 5 | 行为克隆、序列建模 |
| PI0 | PaliGemma VLM | 流匹配 | ✅ | ~15.50 GB | ★★ | 模仿学习 | 50 | 通用控制、语言指令 |
| PI0.5 | PaliGemma VLM | 流匹配 | ✅ | ~16.35 GB | ★★ | 模仿学习 | 50 | PI0改进版 |
| PI0-Fast | PaliGemma+PiGemma | 流匹配+离散token | ✅ | 中高 | ★★★ | 模仿学习 | 50 | 快速推理、部署 |
| SmolVLA | SmolVLM | 流匹配 | ✅ | ~3.93 GB | ★★★ | 模仿学习 | 50 | 轻量VLA、多任务 |
| Wall-X | Qwen2.5-VL | 流匹配+ODE | ✅ | ~15.95 GB | ★★ | 模仿学习 | 32 | 跨具身、多模态 |
| XVLA | Florence-2 | 流匹配 | ✅ | ~15.52 GB | ★★ | 模仿学习 | 32 | 多动作类型 |
| MultiTaskDiT | CLIP+Transformer | 扩散/流匹配 | ✅ | 中高 | ★★★ | 模仿学习 | 可配 | 多任务、LBM |
| EO1 | Qwen2.5-VL | 流匹配 | ✅ | 中 | ★★★ | 模仿学习 | 可配 | EO1机器人 |
| GR00T | NVIDIA GR00T | 直接回归 | ✅ | 高 | ★★★ | 模仿学习 | 可配 | NVIDIA生态、跨具身 |
| GaussianActor/SAC | MLP+高斯 | 概率采样 | ❌ | 低 | ★★★★★ | 强化学习 | 1 | RL训练、在线适应 |

> 显存数据来自SGD优化器（batch=1），默认AdamW下实际占用会更高（约1.5-2倍）。

---

## 选择建议

### 按GPU显存选择

| 显存 | 推荐策略 | 备注 |
|------|---------|------|
| < 8 GB (笔记本/3060) | **ACT** | Diffusion可选（需~6-8 GB空闲） |
| 12-16 GB (4070/4080) | **SmolVLA**、ACT、Diffusion | PI0/PI0.5/Wall-X/XVLA可用小batch+梯度累积 |
| 24+ GB (3090/4090) | **任意策略** | SmolVLA解冻视觉编码器；ACT在单任务上ROI最高 |
| 80 GB (A100/H100) | PI0.5、XVLA、Wall-X | 大batch训练舒适 |

### 按任务类型选择

1. **首次训练 / 单任务抓取放置** → **ACT**（最快、最稳定、最省显存）
2. **多模态动作分布 / 复杂视觉运动** → **Diffusion**（多峰动作建模好）
3. **语言指令 / 多任务** → **SmolVLA**（紧凑）+ 解冻视觉编码器；或 **PI0/PI0.5**（大GPU）
4. **跨具身通用控制** → **Wall-X** 或 **XVLA**
5. **长期规划+在线适应** → **TD-MPC**
6. **强化学习训练** → **GaussianActor + SAC**
7. **NVIDIA生态** → **GR00T**
8. **EO1机器人** → **EO1**
9. **研究对比** → **MultiTaskDiT**（扩散/流匹配双模式）、**VQ-BeT**（离散动作）

### 快速决策流程

```
新手? → 显存 < 8GB? → ACT
                     → 显存 12-16GB? → 需要语言指令? → SmolVLA
                                      → 不需要? → ACT (大数据batch) 或 Diffusion
                     → 显存 24+GB?  → 需要语言指令? → PI0/SmolVLA/Wall-X/XVLA
                                      → 不需要?  → ACT (最快收敛)

有RL经验? → 有仿真环境? → SAC (RL训练)
            → 需要世界模型? → TD-MPC

特定机器人? → EO1 → EO1策略
             → NVIDIA → GR00T
```

---

*本文档基于 LeRobot v0.5.2 代码库生成，策略列表与 `src/lerobot/policies/` 目录下的实现一一对应。*
