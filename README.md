# Micro

> 基于复合 AI 系统理论的高性能终端编码助手——支持多角色模型协作、自适应上下文工程与纵深安全防御

Micro 是一个轻量级终端 AI 编码助手，核心特色是实现了 **Thinking / Critique / Planner / Action 四角色多模型协作架构**。每个角色可独立配置不同厂商的 LLM，在成本、延迟与质量之间自由权衡。

**技术渊源**：本项目在分析 [OPENDEV](https://arxiv.org/abs/2603.05344)、[MSR 2026 编码智能体实证研究](https://arxiv.org/abs/2601.13597)、[Duke 长上下文智能体](https://arxiv.org/abs/2603.20432)、[13 智能体架构分类学](https://arxiv.org/abs/2604.03515) 等前沿工作的基础上，对开源项目 Pico 进行了 12 项系统性架构增强，将"复合 AI 系统"理论落地为可工程实现的模式语言。

**零外部依赖**，仅需 Python 3.10+ 标准库。

## 架构创新

### 四角色多模型协作

```
用户请求
  │
  ├─ [Planner]   /plan 命令触发，Schema gating 只读探索，输出结构化计划
  │
  └─ 每轮迭代 ──────────────────────────────────────
       │
       ├─ [Thinking]   独立推理模型，无工具压力做纯分析
       │   配置：PICO_THINKING_MODEL
       │
       ├─ [Critique]   审视 Thinking 输出 + 最近工具结果
       │   配置：PICO_CRITIQUE_MODEL（未配 → 回退 Thinking）
       │
       └─ [Action]     实际执行工具调用
           配置：PICO_{PROVIDER}_MODEL
```

### 12 项系统性改进

#### 安全与护栏

| # | 改进 | 关键技术点 |
|---|------|-----------|
| 1 | **死循环检测** | MD5 指纹追踪最近 20 次调用 → 相同 ≥3 次拦截；重复读取拦截（≥4 次）；shell 试错循环检测 |
| 2 | **System Reminders** | 9 个事件检测器，以 `role: user` 在决策点注入提醒（OPENDEV 实验证明合规率显著高于 system 消息） |
| 3 | **审批持久化** | 三级决策链（持久化规则 → 全局策略 → 交互询问）；4 类匹配规则；7 条内置危险命令黑名单 |
| 4 | **大输出 Offloading** | 超 16000 字符自动写入 scratch 文件，对话仅保留预览 + 引用路径 |

#### 上下文工程

| # | 改进 | 关键技术点 |
|---|------|-----------|
| 5 | **5 阶段自适应压缩** | Stage 1(≥70%) 警告 → Stage 2(≥80%) 观察遮蔽 → Stage 2.5(≥85%) 快速修剪 → Stage 3(≥90%) 激进遮蔽 → Stage 4(≥99%) 全量压缩 |
| 6 | **Thinking 分离** | 独立模型做纯推理（无工具定义），避免"匆忙行动"偏差；≥4 步自动启用 |
| 7 | **Self-Critique** | 审视 Thinking 输出 + 工具结果，检查遗漏和误读；≥6 步自动启用 |

#### 编辑与扩展

| # | 改进 | 关键技术点 |
|---|------|-----------|
| 8 | **Fuzzy Patch Matching** | 6 阶段渐进匹配（精确→trim→行→空白行→上下文锚定→编辑距离），解决 LLM "输出差一点点"的固有问题 |
| 9 | **Plan Mode** | Schema gating（Planner 物理上无写工具）；交互确认流程 `[Y/n/e]`；计划自动注入执行上下文 |
| 10 | **工具集扩展** | 新增 file_info / glob / grep_count / git_diff / git_log |

#### 记忆与协作

| # | 改进 | 关键技术点 |
|---|------|-----------|
| 11 | **ACE Playbook** | 4 类信号自动提取（编辑修正/用户偏好/测试命令/核心文件）；跨会话去重；Project Knowledge 自动注入 prompt |
| 12 | **多模型角色配置** | Action / Thinking / Critique / Planner 四角色可独立指定 provider + model + api_key + api_base |

## 设计原则

所有改进遵循 OPENDEV 论文提炼的 5 条跨切面教训：

1. **Context 是预算，不是缓冲区** —— 用 API 报告的 token 数校准，渐进压缩远胜单次紧急压缩
2. **在决策点注入提醒，而非提前布道** —— System prompt 在 30+ 轮后被遗忘，`role: user` 短提醒在关键时刻介入
3. **架构约束实现安全** —— Schema gating（让危险工具不可见）比 Runtime check（事后阻止）更鲁棒
4. **为"近似输出"设计工具** —— 6 阶段 fuzzy matching + 可执行错误信息
5. **懒加载 + 有界增长** —— 每个随会话增长的资源都有硬上限

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/yourname/micro.git && cd micro

# 2. 创建环境（可选）
conda create -n micro python=3.12 -y && conda activate micro

# 3. 安装
pip install -e .

# 4. 配置 .env
cp .env.example .env   # 编辑填入 API key

# 5. 运行
micro "列出当前项目目录结构"
```

## 配置指南

### 基础：单模型运行

```bash
# .env — 只需配一个 provider
PICO_PROVIDER=deepseek
PICO_DEEPSEEK_API_KEY=sk-xxx
PICO_DEEPSEEK_MODEL=deepseek-v4-pro
```

### 进阶：多角色跨厂商

```bash
# Action：主力执行
PICO_PROVIDER=deepseek
PICO_DEEPSEEK_MODEL=deepseek-v4-pro

# Thinking：便宜模型做推理
PICO_THINKING_MODEL=deepseek-chat

# Critique：用 MIMO 做审查
PICO_CRITIQUE_PROVIDER=openai
PICO_CRITIQUE_MODEL=mimo-v2.5-pro
PICO_CRITIQUE_API_KEY=你的MIMO_KEY
PICO_CRITIQUE_API_BASE=https://token-plan-cn.xiaomimimo.com/v1

# Planner：最强模型做规划
PICO_PLANNER_PROVIDER=anthropic
PICO_PLANNER_MODEL=claude-sonnet-4-6
PICO_PLANNER_API_KEY=sk-ant-xxx
```

### 支持的后端

| Provider | API 格式 | 适用模型 |
|----------|---------|---------|
| `deepseek` | Anthropic Messages | DeepSeek V3/V4 |
| `openai` | OpenAI Responses | GPT-4o、MIMO、vLLM 等兼容服务 |
| `anthropic` | Anthropic Messages | Claude Sonnet/Opus |
| `ollama` | Ollama Generate | 本地部署模型 |

## 命令行

### 单次任务

```bash
micro "修复 tests/test_user.py 中的失败测试" --max-steps 8
micro "检查项目安全问题" --provider anthropic --approval auto
```

### 交互模式

```bash
micro                          # 进入 REPL
micro> /plan "重构认证模块"     # Plan Mode：先规划，交互确认后执行
micro> /approve list           # 查看审批规则
micro> /approve add PREFIX "pytest" auto
micro> /memory                 # 查看工作记忆
micro> /reset                  # 清空会话
```

### 参数表

| 参数 | 说明 | 默认 |
|------|------|------|
| `--max-steps` | 最大步数 | 6 |
| `--approval` | 审批策略 auto/ask/never | ask |
| `--provider` | 模型后端 | PICO_PROVIDER |
| `--model` | 模型名 | 按 provider |
| `--thinking-model` | Thinking 模型 | PICO_THINKING_MODEL |
| `--cwd` | 工作目录 | . |

## 工具集（12 个）

| 工具 | 说明 | 风险 |
|------|------|:--:|
| `list_files` | 列出目录 | 低 |
| `read_file` | 按行读取文件 | 低 |
| `search` | 搜索内容（优先 ripgrep） | 低 |
| `file_info` | 文件大小/行数/修改时间 | 低 |
| `glob` | 按模式匹配文件 | 低 |
| `grep_count` | 统计匹配数（不返回内容） | 低 |
| `git_diff` | 工作区变更 | 低 |
| `git_log` | 提交历史 | 低 |
| `run_shell` | 执行 shell 命令 | **高** |
| `write_file` | 写入文件 | **高** |
| `patch_file` | 模糊匹配替换（6 阶段） | **高** |
| `delegate` | 派生子 agent 调查 | 低 |

## 技术指标

```
单元测试:     126 passed（新增 67 个）
回归基准:     12/12 通过
端到端评测:   8/8 任务成功
平均效率:     3.5 步/任务，26s/任务
上下文控制:   46% 预算利用率（12000 字符框架下平均 5530 字符）
模型兼容:     DeepSeek / OpenAI / Anthropic / MIMO / Ollama
外部依赖:     0（仅 Python 3.10+ 标准库）
```

## 项目结构

```
micro/
├── agent_loop.py       # 控制循环（感知→思考→审查→决策→行动）
├── cli.py              # CLI + REPL
├── context_manager.py  # 5 阶段自适应上下文压缩
├── reminders.py        # System Reminders（9 检测器）
├── doom_loop.py        # 死循环检测（MD5 指纹 + 3 层）
├── approval_store.py   # 审批规则持久化
├── plan_mode.py        # Plan Mode（Schema gating）
├── tools.py            # 12 个工具
├── tool_executor.py    # 工具执行器（校验/审批/检测/offload）
├── runtime.py          # Micro 核心类
├── security.py         # 密钥脱敏 + shell env 过滤
├── workspace.py        # Git 工作区快照
├── features/
│   └── memory.py       # 三层记忆 + ACE Playbook
├── providers/
│   └── clients.py      # 模型后端适配
├── evaluation/         # Benchmark 框架（12 个回归任务）
└── benchmarks/         # 性能评测任务 + 报告生成
```

## 参考资料

- Bui, N. D. Q. *Building Effective AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned.* arXiv:2603.05344, 2026.
- Agarwal, S., He, H., & Vasilescu, B. *AI IDEs or Autonomous Agents? Measuring the Impact of Coding Agents on Software Development.* MSR 2026.
- Cao, W., Yin, X., Dhingra, B., & Zhou, S. *Coding Agents are Effective Long-Context Processors.* arXiv:2603.20432, 2026.
- Rombaut, B. *Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures.* arXiv:2604.03515, 2026.
- Ogenrwot, D., & Businge, J. *How AI Coding Agents Modify Code: A Large-Scale Study of GitHub Pull Requests.* MSR 2026.
- Zaharia, M., et al. *The Shift from Models to Compound AI Systems.* BAIR Blog, 2024.
- Mei, K., et al. *A Formal Framework for Context Engineering in LLM-based Systems.* 2025.

## License

MIT. 本项目在 [Pico](https://github.com/htxoffical/pico) 基础上进行了实质性架构改进。
