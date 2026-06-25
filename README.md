# Micro

轻量终端 AI 编码助手。基于 [Pico](https://github.com/htxoffical/pico) 进行了 12 项架构增强，支持多模型协作（Thinking / Critique / Planner 四角色独立配置），兼容 DeepSeek / OpenAI / Anthropic / MIMO 等多厂商 API。

## 核心改进

| 模块 | 说明 |
|------|------|
| 死循环检测 | MD5 指纹追踪 + 重复读取拦截 + 周期循环检测，3 层防御 |
| System Reminders | 9 个事件检测器，在模型注意力衰减时注入定向提醒 |
| 大输出 Offloading | 超过 8000 字符自动写入 scratch 文件，节省 70% 上下文 |
| 审批持久化 | 3 级决策链 + 4 类规则 + 7 条内置危险命令黑名单 |
| 5 阶段自适应压缩 | 70%→80%→85%→90%→99% 渐进式上下文压缩 |
| Thinking 分离 | 独立推理模型，无工具压力，≥4 步自动启用 |
| Self-Critique | 审视 Thinking 输出，≥6 步自动启用 |
| Plan Mode | Schema gating 只读 Planner，交互确认后执行 |
| Fuzzy Patch | 6 阶段渐进匹配（精确→trim→行→空白→锚定→编辑距离） |
| 工具扩展 | file_info / glob / grep_count / git_diff / git_log（7 → 12） |
| ACE Playbook | 跨会话自动提取项目知识，越用越聪明 |
| 多模型角色 | Action / Thinking / Critique / Planner 四角色独立配置，跨 API 分工 |

## 快速开始

```bash
# 安装
git clone https://github.com/yourname/micro.git
cd micro
pip install -e .

# 配置 .env
cp .env.example .env
# 编辑 .env，填入 API key

# 使用
micro "列出当前项目的目录结构"
```

## 配置

```bash
# .env — 基本配置
MICRO_PROVIDER=deepseek
MICRO_DEEPSEEK_API_KEY=sk-xxx
MICRO_DEEPSEEK_MODEL=deepseek-v4-pro

# 可选：多角色模型（跨 API 分工）
MICRO_THINKING_MODEL=deepseek-chat
MICRO_PLANNER_MODEL=claude-sonnet-4-6
MICRO_PLANNER_PROVIDER=anthropic
MICRO_PLANNER_API_KEY=sk-ant-xxx
```

## 命令

```bash
# 单次任务
micro "修复 tests/test_user.py 中的失败测试" --max-steps 6

# 交互模式
micro
micro> /plan "在 src/ 下添加日志模块"    # 先规划再执行
micro> /approve add PREFIX "pytest" auto  # 持久化审批规则
micro> /memory                             # 查看工作记忆
micro> /exit
```

## 技术指标

- 零依赖（Python 3.10+，仅标准库）
- 126 个单元测试
- 兼容 OpenAI / Anthropic / DeepSeek / MIMO API
- 支持 Ollama 本地模型

## 项目结构

```
micro/
  agent_loop.py      # 控制循环（感知→决策→行动→记录）
  cli.py             # CLI 入口和 REPL
  context_manager.py # 5 阶段自适应上下文压缩
  doom_loop.py       # 死循环检测
  reminders.py       # System Reminders
  approval_store.py  # 审批规则持久化
  plan_mode.py       # Plan Mode 交互式规划
  tools.py           # 12 个工具定义
  features/
    memory.py        # ACE Playbook 跨会话记忆
  providers/
    clients.py       # 4 种模型后端
```

Built upon [Pico](https://github.com/htxoffical/pico) with 12 architectural improvements based on OPENDEV (arXiv:2603.05344) research.
