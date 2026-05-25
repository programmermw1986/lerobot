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

### 设计目的
解决**低成本双手精细操控**的模仿学习问题。核心创新：将动作分块（Action Chunking）与VAE-Transformer混合架构结合，使模型能表示多模态动作分布，并通过时间集成算法实现平滑控制。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, D_state)` | 机器人状态 |
| 输入 | `observation.images.*` | `(B, C, H, W)` | 多相机图像（ResNet骨干） |
| 输入 | `action` | `(B, chunk_size, D_action)` | 训练时需提供（VAE监督） |
| 输出 | forward | `(loss, loss_dict)` | L1损失 + 可选KL散度 |
| 输出 | select_action | `(B, D_action)` | 单步动作（经队列/时间集成） |
| 输出 | predict_action_chunk | `(B, chunk_size, D_action)` | 完整动作分块 |

### 设计原理
- **VAE编码器**（训练时）：BERT风格Transformer编码器，将 [CLS] + 状态 + 动作序列编码为潜在分布 `N(μ, σ²)`，表示专家行为的多个模式。推理时若 `use_vae=False`，latent 设为全零。
- **Transformer编码器**：处理视觉特征（ResNet + 1x1 Conv）+ 状态 + 潜在变量 → 条件化特征图。
- **Transformer解码器**（DETR风格）：用可学习位置编码作为 query，通过 cross-attention 从编码器输出中解码动作序列。默认仅 1 层解码器（匹配原始ACT bug）。
- **时间集成**（Temporal Ensembling）：指数加权滑动平均（`w_i = exp(-coeff * i)`）对重叠的动作预测做平滑。
- **视觉骨干**：ResNet18（预训练ImageNet），Bottleneck特征图经 `FrozenBatchNorm2d` + 1x1 Conv 投影到 `dim_model`。

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

### 设计目的
解决视觉运动策略中的**多模态动作分布**和**时序一致性**问题。将条件去噪扩散过程应用于动作轨迹生成：以视觉观测 + 状态为条件，从随机噪声通过反向扩散生成平滑的动作序列，天然支持多模态输出。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, n_obs_steps, D_state)` | 多步状态历史 |
| 输入 | `observation.images.*` | `(B, n_obs_steps, C, H, W)` | 多相机图像（堆叠为 OBS_IMAGES） |
| 输入 | `action` | `(B, horizon, D_action)` | 训练用真实动作轨迹 |
| 输出 | forward | `(loss, None)` | MSE噪声/样本预测损失 |
| 输出 | select_action | `(B, D_action)` | 单步动作（滑窗+队列） |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 动作分块 |

### 设计原理
- **1D时序U-Net**：1D卷积残差网络在动作序列的时间维度上操作。默认3级下采样（`down_dims=(512,1024,2048)`），每级含 `Conv1d → GroupNorm → Mish + FiLM` 残差块。
- **FiLM条件化**：视觉特征 + 状态 + 扩散时间步编码经MLP注入U-Net各层作为 FiLM（特征线性调制）条件。
- **视觉编码**：每台相机独立 ResNet + SpatialSoftmax（空间软argmax，输出32个关键点坐标），支持 `use_separate_rgb_encoder_per_camera`。
- **噪声调度**：支持 DDPM（余弦beta）或 DDIM；默认 `prediction_type="epsilon"`；推理步数可少于训练步数（DDIM加速）。
- **推理**：滑窗重新规划（receding horizon）：预测 horizon 步，仅执行 n_action_steps，然后重新规划。
- **L1正则化**：可不使用。

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

### 设计目的
将**时序差分学习**与**模型预测控制**结合，学习一个潜在世界模型用于离线预训练 + 在线微调。核心创新：TOLD（Task-Oriented Latent Dynamics）模型在潜在空间做规划，兼顾数据效率与规划精度。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, 1, D_state)` | 机器人状态（含时间维度） |
| 输入 | `observation.image.*` | `(B, 1, C, H, W)` | 单相机（需正方形） |
| 输入 | `action` | `(B, horizon, D_action)` | 训练用动作 |
| 输入 | `reward` | `(B, horizon)` | 训练用奖励 |
| 输出 | forward | `(loss, info_dict)` | 5项损失之和（含consistency/reward/Q/V/pi） |
| 输出 | select_action | `(D_action,)` | 单步动作（无batch） |
| 输出 | predict_action_chunk | `(horizon, B, D_action)` | 规划的动作轨迹 |

### 设计原理
- **TOLD模型**（5组件）：观测编码器（4层CNN+MLP → 统一latent）→ 动力学模型（`latent+action → latent`）→ 奖励模型（`latent+action → scalar`）→ 策略网络（`latent → action`）→ Q函数集成（5个Q网络）→ V值函数。
- **CEM/MPPI规划**：每次采样 `n_gaussian_samples + n_pi_samples` 条轨迹 → 经 `estimate_value()` 打分 → 选Elite → 拟合高斯分布 → 新均值warm-start下轮CEM。
- **目标网络**：EMA（`tau=0.995`）更新完整TOLD模型的副本。
- **5项损失**：consistency_loss（潜在一致性）+ reward_loss + Q_loss + V_loss + policy优势加权回归损失。
- **图像增强**：DrQv2风格 RandomShift 裁剪。

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

### 设计目的
将连续动作生成转化为**离散编码分类 + 偏移回归**问题。通过 VQ-VAE 将动作离散化，再用 GPT 做自回归预测，兼顾离散表示的简洁性和连续动作的精度。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, n_obs_steps, D_state)` | 多步状态历史 |
| 输入 | `observation.images.*` | `(B, n_obs_steps, C, H, W)` | 图像（ResNet编码） |
| 输入 | `action` | `(B, seq_len, D_action)` | Phase 2 中分块后的动作序列 |
| 输出 | forward | `(loss, info_dict)` | 分类 + 偏移 + VQ重建损失 |
| 输出 | select_action | `(D_action,)` | 单步动作 |
| 输出 | predict_action_chunk | `(B, action_chunk_size, D_action)` | 动作分块 |

### 设计原理
- **两阶段训练**：Phase 1 训练两层 Residual VQ-VAE（每层16个码本，嵌入维度256），使用 L1重建 + 承诺损失。Phase 2 冻结VQ，训练 GPT + 动作头。
- **GPT编码**：观测token（ResNet编码后经 SpatialSoftmax）与可学习动作 query token 交错排列 → GPT 因果自回归 → 动作位置输出。
- **VQ-BeT Head**：预测主码本分类（Focal loss）+ 次码本分类 + 连续偏移量（L1 loss）。
- **GPT架构**：minGPT，8层8头512维。

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

#### 设计目的
第一个将**大规模VLM**（PaliGemma 2B）与**流匹配动作头**结合的通用机器人控制模型。核心创新：VLM做高层语义理解，轻量动作专家（Gemma 300M）通过共享注意力从VLM上下文中提取信息生成动作，实现少样本泛化和跨任务迁移。

#### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.images.*` | `(B, 3, H, W)` | float32 [0,1]，内部resize到224并转为[-1,1] |
| 输入 | `observation.state` | `(B, D_state)` | padding至 max_state_dim (32) |
| 输入 | `observation.language.tokens` | `(B, L)` | tokenized任务指令 (max_length=48) |
| 输入 | `action` | `(B, chunk_size, D_action)` | padding至 max_action_dim (32) |
| 输出 | forward | `(loss, {"loss", "loss_per_dim"})` | MSE流匹配损失 |
| 输出 | select_action | `(B, D_action_original)` | 单步动作（队列消费） |
| 输出 | predict_action_chunk | `(B, chunk_size, D_action_original)` | 流匹配降噪后完整分块 |

#### 设计原理
- **Prefix-Suffix架构**：Prefix（图像 + 语言token）→ PaliGemma VLM处理并缓存KV。Suffix（状态 + 噪声动作 + 时间步）→ action expert处理，通过**联合注意力**（concat QKV）访问prefix上下文。
- **流匹配**（非扩散）：线性插值 `x_t = t*noise + (1-t)*action`，预测向量场 `v_t ≈ u_t = noise - action`。推理用Euler积分从 `t=1` 反向到 `t=0`（默认10步）。
- **动作专家**：独立的Gemma 300M模型（无嵌入层），与VLM逐层共享注意力。
- **前缀KV缓存**：推理时Prefix只编码一次，后续降噪步仅处理suffix，利用缓存的prefix KV。
- **时间步采样**：`t ~ Beta(α=1.5, β=1.0)` 缩放到 `[0.001, 0.999]`。

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

PI0的精进版本，架构与PI0类似（PaliGemma VLM + Gemma 300M动作专家 + 流匹配）。**与PI0的关键差异**：移除了状态投影头（`state_proj`），在action expert上使用 **adaRMS归一化**，使用不同的优化器默认参数。训练时默认**冻结视觉编码器和VLM**，仅训练action expert。占用显存略高于PI0（batch=1时~16.35 GB）。

### 5.3 PI0-Fast

#### 设计目的
PI0的离散token快速变体（来自OpenPI），用**自回归离散动作token预测**替代连续流匹配。核心创新：通过BPE风格的FAST动作分词器将连续动作离散化，用PaliGemma VLM自回归预测动作token（交叉熵损失），再通过IDCT解码回连续动作。

#### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.images.*` | `(B, C, H, W)` | [0,1] 转 [-1,1] |
| 输入 | `observation.text_tokens` | `(B, L)` | tokenized指令 |
| 输入 | `action_tokens` | `(B, max_action_tokens)` | 训练用离散动作token ID |
| 输入 | `action_token_mask` | `(B, max_action_tokens)` | padding mask |
| 输出 | forward | `(loss, {"ce_loss"})` | 交叉熵损失（next-token预测） |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 自回归生成 + IDCT解码 |

#### 设计原理
- **单一PaliGemma模型**（无独立action expert）：动作token与文本token共享嵌入层。
- **Prefix-LM注意力掩码**：图像+语言双向注意力，动作token间因果注意力。
- **自回归解码**：支持原始全前向和KV缓存优化两种解码模式。
- **动作token化管线**：PaliGemma token ID → 逆映射到动作token ID → BPE分词器 → DCT系数 → IDCT解码为连续动作。
- **温度控制**：采样温度 configurable（默认0.0 = greedy）。

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

### 设计目的
Hugging Face设计的**轻量VLA**模型，解决大VLA模型部署成本高的问题。核心创新：使用SmolVLM2-500M 小VLM + 交叉注意力动作专家（cross-attention），在保持性能的同时大幅降低计算成本。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.images.*` | `(B, C, H, W)` | 图像resize到512x512 + padding |
| 输入 | `observation.state` | `(B, D_state)` | padding至 max_state_dim (32) |
| 输入 | `observation.language.tokens` | `(B, L)` | tokenized指令 (max_length=48) |
| 输入 | `action` | `(B, chunk_size, D_action)` | padding至 max_action_dim (32) |
| 输出 | forward | `(loss, loss_dict)` | MSE流匹配损失 |
| 输出 | select_action | `(B, D_action)` | 单步动作（队列） |
| 输出 | predict_action_chunk | `(B, chunk_size, D_action)` | 完整动作分块 |

### 设计原理
- **解耦Prefix-Suffix架构**：Prefix（图像+语言+状态）→ VLM编码并缓存KV。Suffix（噪声动作+时间步）→ 轻量动作专家通过 **cross-attention** 读取VLM的prefix KV。
- **注意力模式**：支持 `self_attn`（VLM和专家独立并行）或 `cross_attn`（专家交叉注意力到VLM），默认 `cross_attn`。每隔 `self_attn_every_n_layers` 层用一次self-attn。
- **视觉编码器冻结**：默认冻结SigLIP视觉编码器（`freeze_vision_encoder=True`），仅训练action expert + 投影层（`train_expert_only=True`）。
- **流匹配**：与PI0相同，Beta分布时间步采样 + Euler积分降噪（默认10步）。
- **专家宽度**：`expert_width_multiplier=0.75`，专家隐藏层为VLM的75%。

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

### 设计目的
**跨具身通用机器人控制**，使用Qwen2.5-VL（大MoE VLM）+ 流匹配动作头。支持两种推理模式：**diffusion（ODE流匹配）** 和 **fast（离散token自回归）**。核心创新：统一20维动作空间 + MoE架构 + LoRA微调。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, D_state)` | padding至20 |
| 输入 | `observation.images.*` | `(B, C, H, W)` | 多相机支持 |
| 输入 | `action` | `(B, chunk_size, D_action)` | padding至20 + NaN mask |
| 输入 | `task` | `List[str]` | 语言指令 |
| 输出 | forward | `(loss, output)` | flow_loss + cross_entropy_loss |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 连续动作 |

### 设计原理
- **MoE扩展**：Qwen2.5-VL的MoE解码器层（Qwen2_5_VLMoEForAction）+ 3D RoPE视觉位置编码。
- **双模式推理**："diffusion"模式用 `torchdiffeq.odeint` 做ODE积分；"fast"模式自回归生成离散动作token + detokenization。
- **Token注入**：视觉特征 → masked_scatter 注入 input_embeds；状态替换 `<|propri|>` token；噪声动作替换 `<|action|>` token。
- **ActionHead**：正弦时间步嵌入 + 动作/DOF mask拼接 + MLP → hidden states → 流匹配预测。
- **LoRA**：支持 `q_proj, v_proj` 的PEFT微调。

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

### 设计目的
**扩展VLA模型**，使用Florence-2作为冻结视觉语言骨干 + Soft-Prompted Transformer动作头。核心创新：通过可学习的**领域soft prompt**实现多任务/跨具身泛化，支持多种动作表示（ee6d/absolute/relative/diff）。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, D_state)` | 机器人状态 |
| 输入 | `observation.images.*` | `(B, C, H, W)` | 多视角图像（堆叠后输入） |
| 输入 | `observation.text_tokens` | `(B, L)` | tokenized指令 |
| 输入 | `domain_id` | `(B,)` | 领域ID（可选，默认0） |
| 输入 | `action` | `(B, chunk_size, D_action)` | 训练用动作 |
| 输出 | forward | `(loss, log_dict)` | 流匹配损失 |
| 输出 | predict_action_chunk | `(B, chunk_size, D_action)` | 后处理后的动作分块 |

### 设计原理
- **Florence-2骨干**：DaViT视觉编码器 + BART风格语言编码器（去除解码器），从中提取视觉语言特征。
- **Soft-Prompted Transformer**：可学习的领域soft prompt + VLM特征 + 噪声动作 + 状态 + 时间步 → Transformer → 流匹配速度预测。
- **差分学习率**：VLM参数LR为1/10，Transformer头全LR，soft prompt支持warm-up缩放。
- **动作空间抽象**：支持 `ee6d`（末端6D位姿）、`absolute`、`relative`、`diff`、`auto`（自动检测）。

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

### 设计目的
**多任务扩散Transformer**，使用CLIP（视觉+文本）作为条件编码器 + Diffusion Transformer（DiT）生成动作。核心创新：**AdaLN-Zero**调制（自适应层归一化 + 零初始化残差）结合扩散/流匹配双模式，多任务通过CLIP文本编码实现语言条件化。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, n_obs_steps, D_state)` | 多步状态历史 |
| 输入 | `observation.images.*` | `(B, n_obs_steps, C, H, W)` | 多相机（CLIP ViT编码） |
| 输入 | `observation.text_tokens` | `(B, L)` | CLIP tokenizer处理 |
| 输入 | `action` | `(B, horizon, D_action)` | 训练用动作 |
| 输出 | forward | `(loss, None)` | 扩散/流匹配损失 |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 动作分块 |

### 设计原理
- **CLIP双编码器**：CLIP ViT（视觉）+ 冻结CLIP文本编码器（可学习投影）→ 条件特征。状态向量直接拼接。
- **DiT骨干**（DiffusionTransformer）：正弦时间步嵌入 → 输入投影 → 堆叠 TransformerBlock，每块使用 **AdaLN-Zero**（条件+时间步 → shift/scale/gate，零初始化残差）。
- **双目标**：支持扩散（DDPM/DDIM，epsilon或sample预测）和流匹配（Beta分布时间步，Euler/RK4积分）。
- **RoPE**：可选旋转位置编码替代标准注意力。

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

### 设计目的
**EO1机器人专用策略**，使用Qwen2.5-VL-3B VLM + 流匹配动作头。核心创新：通过**token替换机制**将状态/动作嵌位符替换为投影特征，Qwen骨干处理修改后的token序列，提取动作位置隐状态预测流匹配速度。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `pixel_values` | `(B, C, H, W)` | 图像像素值 |
| 输入 | `input_ids` | `(B, L)` | 含 `<\|state\|>` 和 `<\|action\|>` 占位符的prompt token |
| 输入 | `observation.state` | `(B, D_state)` | padding至 max_state_dim (32)，替换状态占位符 |
| 输入 | `action` | `(B, chunk_size, D_action)` | padding至 max_action_dim (32) |
| 输出 | forward | `(loss, loss_dict)` | 流匹配MSE损失 |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 连续动作 |

### 设计原理
- **Token替换机制**：`<|state|>` → 投影状态特征；`<|action|>` → 融合（噪声动作+时间步）特征。Qwen2.5-VL骨干处理完整序列，动作位置输出经 `action_out_proj` 投影为速度预测。
- **流匹配**：Beta分布时间步采样 + Euler积分降噪（默认10步）。
- **KV缓存推理**：Prefix（图像+文本+状态）编码一次并缓存，降噪步仅处理suffix。
- **精度控制**：流头保持fp32；骨干使用配置dtype（auto/bf16/fp32），支持 `force_fp32_autocast`。

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

### 设计目的
NVIDIA **GR00T N1.5**基础模型的LeRobot包装器，用于人形机器人控制。核心架构（来源Isaac-GR00T）：**Eagle2视觉编码器** + Gemma语言模型 + **流匹配交叉注意力DiT动作头**。LeRobot端通过 `GR00TN15` 代理类封装。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `eagle_*` | Eagle处理器输出 | 图像+文本的预处理tensor |
| 输入 | `state` | `(B, n_obs_steps, max_state_dim)` | 机器人状态（含mask） |
| 输入 | `state_mask` | `(B, n_obs_steps, max_state_dim)` | 状态维度padding mask |
| 输入 | `action` | `(B, chunk_size, max_action_dim)` | 训练用动作（含mask） |
| 输入 | `embodiment_id` | `(B,)` | 机器人形态标识 |
| 输出 | forward | `(loss, loss_dict)` | GR00T模型内部损失 |
| 输出 | predict_action_chunk | `(B, n_action_steps, D_action)` | 连续动作 |

### 设计原理
- **GR00TN15引擎**：Eagle2.5（SigLIP视觉）+ Gemma LLM + 流匹配交叉注意力DiT动作头，在bf16 autocast下运行。
- **微调控制**：可独立控制LLM、视觉编码器、投影器、扩散模型的tuning，支持LoRA。
- **处理器**：Eagle专用tokenizer预处理，使用缓存的Eagle处理器资产文件。
- **包装器模式**：`from_pretrained` 支持两种模式——基础GR00T模型加载和微调LeRobot checkpoint加载。

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

### 设计目的
SAC（软演员-评论家）强化学习算法的 LeRobot 实现。策略部分（`GaussianActorPolicy`）输出对角高斯分布（tanh squashed），用于最大熵连续控制。算法部分（`SACAlgorithm`）管理Q函数、目标网络、温度调节和Bellman更新。

### 输入输出
| 方向 | 键名 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `observation.state` | `(B, D_state)` | 机器人状态 |
| 输入 | `observation.image.*` | `(B, C, H, W)` | 可选图像（CNN/预训练编码） |
| 输入 | `action` | `(B, D_action)` | 训练用动作 |
| 输出 | forward | `dict{"action", "log_prob", "action_mean"}` | 采样动作及其概率 |
| 输出 | select_action | `(B, D_action,)` | **单步动作（无分块！）** |

### 设计原理
- **GaussianActorPolicy**：观测编码器（图像CNN/预训练 + 状态/环境编码，统一投影到 latent_dim）→ MLP骨干 → 分离 `mean_layer` 和 `log_std_layer` → `TanhMultivariateNormalDiag` 分布。
- **共享编码器**：Actor和Critic可共享视觉编码器，Actor分离梯度防止双反向传播。
- **特征缓存**：`get_cached_image_features()` 缓存编码器输出，避免冗余前向（2-4x加速）。
- **Actor-Learner架构**：环境步进与学习分离在不同进程/线程，通过gRPC通信。
- **SACAlgorithm**：双Q + 目标网络 + 温度自动调节 + 经验回放。使用 `SAC` 优化器预设（actor/critic/temperature三组独立学习率）。

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

| 策略 | 架构类型 | 动作生成 | 视觉条件 | 语言支持 | 显存(batch=1) | 推理速度 | 学习范式 | 分块 | 适用场景 |
|------|---------|---------|---------|---------|--------------|---------|---------|------|---------|
| ACT | Transformer+VAE | 直接回归 | ResNet18 | ❌ | ~0.94 GB | ★★★★★ | 模仿学习 | 20-100 | 入门、单任务、低显存 |
| Diffusion | CNN U-Net | 扩散去噪 | ResNet+SS | ❌ | ~4.94 GB | ★★★ | 模仿学习 | 32-64 | 多模态动作、视觉运动 |
| TD-MPC | 世界模型+MPC | MPC采样规划 | 4层CNN | ❌ | 中 | ★★★ | 模仿+RL | 5-10 | 长期规划、在线适应 |
| VQ-BeT | VQ-VAE+GPT | 离散token预测 | ResNet+SS | ❌ | 中 | ★★★★ | 模仿学习 | 5 | 行为克隆、序列建模 |
| PI0 | PaliGemma VLM | 流匹配 | SigLIP | ✅ | ~15.50 GB | ★★ | 模仿学习 | 50 | 通用控制、语言指令 |
| PI0.5 | PaliGemma VLM | 流匹配 | SigLIP | ✅ | ~16.35 GB | ★★ | 模仿学习 | 50 | PI0改进版 |
| PI0-Fast | PaliGemma+PiGemma | 离散token自回归 | SigLIP | ✅ | 中高 | ★★★ | 模仿学习 | 50 | 快速推理、部署 |
| SmolVLA | SmolVLM | 流匹配 | SigLIP | ✅ | ~3.93 GB | ★★★ | 模仿学习 | 50 | 轻量VLA、多任务 |
| Wall-X | Qwen2.5-VL MoE | 流匹配+ODE | 3D RoPE-ViT | ✅ | ~15.95 GB | ★★ | 模仿学习 | 32 | 跨具身、多模态 |
| XVLA | Florence-2 | 流匹配 | DaViT | ✅ | ~15.52 GB | ★★ | 模仿学习 | 32 | 多动作类型 |
| MultiTaskDiT | CLIP+DiT | 扩散/流匹配 | CLIP ViT | ✅ | 中高 | ★★★ | 模仿学习 | 32 | 多任务、LBM |
| EO1 | Qwen2.5-VL | 流匹配 | Qwen ViT | ✅ | 中 | ★★★ | 模仿学习 | 8 | EO1机器人 |
| GR00T | Eagle2+Gemma+DiT | 流匹配 | Eagle2/SigLIP | ✅ | 高 | ★★★ | 模仿学习 | 50 | NVIDIA、跨具身 |
| GaussianActor/SAC | MLP+高斯 | 概率采样 | CNN/MLP | ❌ | 低 | ★★★★★ | 强化学习 | 1 | RL训练、在线适应 |

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

### 核心实现文件索引

| 策略 | 模型文件 | 配置文件 | 处理器文件 |
|------|---------|---------|-----------|
| ACT | `act/modeling_act.py` | `act/configuration_act.py` | `act/processor_act.py` |
| Diffusion | `diffusion/modeling_diffusion.py` | `diffusion/configuration_diffusion.py` | `diffusion/processor_diffusion.py` |
| TD-MPC | `tdmpc/modeling_tdmpc.py` | `tdmpc/configuration_tdmpc.py` | `tdmpc/processor_tdmpc.py` |
| VQ-BeT | `vqbet/modeling_vqbet.py` | `vqbet/configuration_vqbet.py` | `vqbet/processor_vqbet.py` |
| PI0 | `pi0/modeling_pi0.py` | `pi0/configuration_pi0.py` | `pi0/processor_pi0.py` |
| PI0.5 | `pi05/modeling_pi05.py` | `pi05/configuration_pi05.py` | `pi05/processor_pi05.py` |
| PI0-Fast | `pi0_fast/modeling_pi0_fast.py` | `pi0_fast/configuration_pi0_fast.py` | N/A（PaliGemma内建） |
| SmolVLA | `smolvla/modeling_smolvla.py` | `smolvla/configuration_smolvla.py` | `smolvla/processor_smolvla.py` |
| Wall-X | `wall_x/modeling_wall_x.py` | `wall_x/configuration_wall_x.py` | `wall_x/processor_wall_x.py` |
| XVLA | `xvla/modeling_xvla.py` | `xvla/configuration_xvla.py` | `xvla/processor_xvla.py` |
| MultiTaskDiT | `multi_task_dit/modeling_multi_task_dit.py` | `multi_task_dit/configuration_multi_task_dit.py` | `multi_task_dit/processor_multi_task_dit.py` |
| EO1 | `eo1/modeling_eo1.py` | `eo1/configuration_eo1.py` | `eo1/processor_eo1.py` |
| GR00T | `groot/modeling_groot.py` | `groot/configuration_groot.py` | `groot/processor_groot.py` |
| GaussianActor | `gaussian_actor/modeling_gaussian_actor.py` | `gaussian_actor/configuration_gaussian_actor.py` | `gaussian_actor/processor_gaussian_actor.py` |
| SAC (RL算法) | `rl/algorithms/sac/sac_algorithm.py` | `rl/algorithms/sac/configuration_sac.py` | N/A |
