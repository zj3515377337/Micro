# Micro

轻量级终端 AI 编码助手，基于 [Pico](https://github.com/htxoffical/pico) 进行了 12 项系统性架构增强。

支持多模型协作（Thinking / Critique / Planner），兼容 DeepSeek / OpenAI / Anthropic / MIMO / Ollama 等多厂商 API。零外部依赖，仅需 Python 3.10+。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/micro.git
cd micro

# 2. 创建虚拟环境（可选但推荐）
conda create -n micro python=3.12 -y
conda activate micro

# 3. 安装（开发模式）
pip install -e .

# 4. 配置 API
cp .env.example .env
# 编辑 .env，填入至少一个 provider 的 API key

# 5. 运行
micro "列出当前项目目录结构"
```

## 配置

### 基本配置（`.env`）

```bash
# 默认 provider（openai / anthropic / deepseek / ollama）
PICO_PROVIDER=deepseek

# DeepSeek（默认推荐）
PICO_DEEPSEEK_API_BASE=https://api.deepseek.com/anthropic
PICO_DEEPSEEK_API_KEY=sk-xxx
PICO_DEEPSEEK_MODEL=deepseek-v4-pro

# OpenAI
PICO_OPENAI_API_BASE=https://api.openai.com/v1
PICO_OPENAI_API_KEY=sk-xxx
PICO_OPENAI_MODEL=gpt-4o

# Anthropic
PICO_ANTHROPIC_API_BASE=https://api.anthropic.com
PICO_ANTHROPIC_API_KEY=sk-ant-xxx
PICO_ANTHROPIC_MODEL=claude-sonnet-4-6

# Ollama（本地模型）
PICO_PROVIDER=ollama
# 默认使用 localhost:11434 的 qwen3.5:4b
```

### 多角色模型（可选）

每个角色可以独立指定模型甚至 API 厂商：

```bash
# Thinking：纯推理阶段使用的模型（≥4 步自动启用）
PICO_THINKING_MODEL=deepseek-chat
# PICO_THINKING_PROVIDER=openai          # 可选：独立指定 provider
# PICO_THINKING_API_KEY=sk-xxx           # 可选：独立 API key

# Critique：审视 Thinking 输出的模型（≥6 步自动启用，未配则回退 Thinking）
# PICO_CRITIQUE_MODEL=deepseek-chat

# Planner：/plan 命令使用的模型（未配则回退 Thinking）
# PICO_PLANNER_MODEL=claude-sonnet-4-6
# PICO_PLANNER_PROVIDER=anthropic        # 跨厂商示例
# PICO_PLANNER_API_KEY=sk-ant-xxx
```

### 跨厂商配置示例

```bash
# Action 用 DeepSeek，Thinking 用 MIMO，Planner 用 Claude
PICO_PROVIDER=deepseek
PICO_DEEPSEEK_MODEL=deepseek-v4-pro

PICO_THINKING_PROVIDER=openai
PICO_THINKING_MODEL=mimo-v2.5-pro
PICO_THINKING_API_BASE=https://token-plan-cn.xiaomimimo.com/v1
PICO_THINKING_API_KEY=你的MIMO_KEY

PICO_PLANNER_PROVIDER=anthropic
PICO_PLANNER_MODEL=claude-sonnet-4-6
PICO_PLANNER_API_KEY=sk-ant-xxx
```

## 使用方式

### 单次任务

```bash
# 基本用法
micro "修复 tests/ 目录下的失败测试"

# 指定步数和审批策略
micro "重构 src/utils.py 中的工具函数" --max-steps 8 --approval auto

# 指定 provider 和模型
micro "检查代码中的安全问题" --provider anthropic --model claude-sonnet-4-6
```

### 交互模式（REPL）

```bash
micro

# 进入交互模式后可用命令：
micro> /help              # 查看帮助
micro> /plan <任务描述>    # 先规划再执行（Plan Mode）
micro> /approve list      # 查看持久化审批规则
micro> /approve add PREFIX "pytest" auto  # 添加自动审批规则
micro> /approve remove 3  # 删除第 3 条规则
micro> /memory            # 查看工作记忆
micro> /session           # 查看会话文件路径
micro> /reset             # 清空当前会话
micro> /exit              # 退出
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-steps` | 最大工具调用次数 | 6 |
| `--approval` | 审批策略（auto/ask/never） | ask |
| `--provider` | 模型后端 | PICO_PROVIDER 或 deepseek |
| `--model` | 模型名称 | 按 provider 选择 |
| `--thinking-model` | Thinking 阶段模型 | PICO_THINKING_MODEL |
| `--cwd` | 工作目录 | 当前目录 |
| `--temperature` | 采样温度 | 0.2 |

## 架构改进

基于 [OPENDEV (arXiv:2603.05344)](https://arxiv.org/abs/2603.05344) 等前沿论文，在原 Pico 基础上实现了 12 项系统性增强：

### 护栏增强（Phase 1）

| 改进 | 说明 |
|------|------|
| 死循环检测 | MD5 指纹追踪最近 20 次工具调用，相同 ≥3 次即拦截；同一文件重复读取 ≥4 次警告；shell 命令试错循环检测 |
| System Reminders | 9 个事件检测器，在模型注意力衰减时以 `role: user` 注入简短提醒，每类有触发上限防止噪声 |
| 大输出 Offloading | 工具输出超过 16000 字符自动写入 scratch 文件，对话中仅保留预览，节省上下文 70% |
| 审批持久化 | 三级决策链（持久化规则 → 全局策略 → 交互询问），支持 4 类规则（COMMAND/PREFIX/PATTERN/DANGER），7 条内置危险命令黑名单 |

### 上下文工程（Phase 2）

| 改进 | 说明 |
|------|------|
| 5 阶段自适应压缩 | 70%→80%→85%→90%→99% 渐进式压缩，从观察遮蔽到快速修剪到激进遮蔽，保护窗口动态调整 |
| Thinking 分离 | 独立模型做纯推理（无工具压力），输出注入为 Action 上下文，≥4 步自动启用 |
| Self-Critique | 审视 Thinking 输出和最近工具结果，检查遗漏和误判，≥6 步自动启用 |

### 架构扩展（Phase 3）

| 改进 | 说明 |
|------|------|
| Plan Mode | Schema gating 只读 Planner，输出 7 节结构化计划，交互确认后执行 |
| Fuzzy Patch | 6 阶段渐进匹配（精确→trim→行→空白→锚定→编辑距离），解决 LLM 输出"差一点点"的问题 |
| 工具扩展 | 新增 5 个只读工具：file_info / glob / grep_count / git_diff / git_log（7→12） |

### 记忆与协作（Phase 4）

| 改进 | 说明 |
|------|------|
| ACE Playbook | 4 类信号（编辑修正/用户偏好/测试命令/核心文件）自动提取，跨会话去重，Project Knowledge 自动注入 |
| 多模型角色 | Action / Thinking / Critique / Planner 四角色独立配置，支持跨 API 厂商分工 |

## 工具清单

| 工具 | 参数 | 风险 | 说明 |
|------|------|:--:|------|
| `list_files` | path | 低 | 列出目录内容 |
| `read_file` | path, start, end | 低 | 按行范围读取文件 |
| `search` | pattern, path | 低 | 搜索文件内容（优先 rg） |
| `file_info` | path | 低 | 文件大小、行数、修改时间 |
| `glob` | pattern | 低 | 按模式匹配文件路径 |
| `grep_count` | pattern, path | 低 | 统计匹配数不返回内容 |
| `git_diff` | staged | 低 | 显示工作区变更 |
| `git_log` | n, path | 低 | 显示提交历史 |
| `run_shell` | command, timeout | **高** | 执行 shell 命令 |
| `write_file` | path, content | **高** | 写入文件 |
| `patch_file` | path, old_text, new_text | **高** | 模糊匹配替换（6 阶段） |
| `delegate` | task, max_steps | 低 | 派生子 agent 调查 |

## 项目结构

```
micro/
├── agent_loop.py       # 控制循环（感知→思考→审查→决策→行动）
├── cli.py              # CLI 入口、参数解析、REPL 交互
├── config.py           # .env 加载和配置解析
├── context_manager.py  # 5 阶段自适应上下文压缩
├── prompt_prefix.py    # 稳定 System Prompt 前缀
├── runtime.py          # Micro 核心类
├── security.py         # 密钥脱敏、shell 环境过滤
├── tool_executor.py    # 工具执行器（校验/审批/死循环检测/offload）
├── tools.py            # 12 个工具定义
├── workspace.py        # Git 工作区快照
├── approval_store.py   # 审批规则持久化
├── doom_loop.py        # 死循环检测
├── plan_mode.py        # Plan Mode 交互式规划
├── reminders.py        # System Reminders
├── evaluation/         # Benchmark 评测框架
├── features/
│   └── memory.py       # 三层记忆 + ACE Playbook
└── providers/
    └── clients.py      # 4 种模型后端适配
```

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 跳过环境依赖测试
python -m pytest tests/ -v -k "not test_trace_and_report_redact"

# 仅核心测试
python -m pytest tests/test_micro.py -v
```

## 技术指标

- 零外部依赖（Python 3.10+，仅标准库 `urllib` / `subprocess` / `json`）
- 126 个单元测试
- 12 个 Benchmark 回归任务
- 兼容 OpenAI / Anthropic / DeepSeek / MIMO / Ollama API
- 支持 Windows / macOS / Linux

## 参考资料

- [OPENDEV: Building Effective AI Coding Agents for the Terminal](https://arxiv.org/abs/2603.05344)
- [Coding Agents are Effective Long-Context Processors](https://arxiv.org/abs/2603.20432)
- 原项目 [Pico](https://github.com/htxoffical/pico)

## License

MIT
