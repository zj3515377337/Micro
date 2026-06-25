"""持久化审批规则存储。

OPENDEV 论文 Lesson 3 指出：审批规则必须持久化——否则用户疲劳会导致
全部 auto-approve，破坏整个安全体系。

这里提供：
- 4 类匹配规则：PATTERN（正则）、COMMAND（精确）、PREFIX（前缀）、DANGER（黑名单）
- 3 种决议：auto（自动批准）、ask（交互询问）、never（拒绝）
- JSON 文件持久化到 .pico/approvals.json
"""

import json
import re
from pathlib import Path

from .workspace import now

RULE_TYPES = ("PATTERN", "COMMAND", "PREFIX", "DANGER")
DECISIONS = ("auto", "ask", "never")

# 内置危险模式：永远不应自动批准的命令前缀
_BUILTIN_DANGER_PATTERNS = [
    r"rm\s+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r)",
    r"sudo\s",
    r"chmod\s+777",
    r">\s*/dev/[a-z]+d[a-z0-9]",
    r"mkfs\.",
    r"dd\s+if=",
    r":(){ :|:& };:",  # fork bomb
]


class ApprovalStore:
    """加载、保存和匹配持久化审批规则。"""

    def __init__(self, root):
        self.path = Path(root) / ".pico" / "approvals.json"

    def load(self):
        """加载规则列表。文件不存在时返回内置默认规则。"""
        if not self.path.exists():
            return self._default_rules()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            rules = data.get("rules", [])
            # 确保内置危险规则始终存在
            return self._merge_with_defaults(rules)
        except (json.JSONDecodeError, OSError):
            return self._default_rules()

    def save(self, rules):
        """保存规则列表到 JSON 文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rules": rules, "updated_at": now()}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def match(self, tool_name, args):
        """匹配工具调用，返回 'auto' / 'ask' / 'never' / None。

        仅 run_shell 参与规则匹配（按其 command 字符串），
        其他工具走全局 policy。
        """
        if tool_name != "run_shell":
            return None

        command = str(args.get("command", "")).strip()
        if not command:
            return None

        rules = self.load()
        for rule in rules:
            if self._rule_matches(rule, command):
                return rule["decision"]
        return None

    def add(self, rule_type, pattern, decision):
        """添加一条规则，返回更新后的规则列表。"""
        if rule_type not in RULE_TYPES:
            raise ValueError(f"unknown rule type: {rule_type}, must be one of {RULE_TYPES}")
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision: {decision}, must be one of {DECISIONS}")

        rules = self.load()
        # 去重：相同 type + pattern 的规则不重复添加
        for rule in rules:
            if rule["type"] == rule_type and rule["pattern"] == pattern:
                rule["decision"] = decision
                rule["updated_at"] = now()
                self.save(rules)
                return rules

        rules.append({
            "type": rule_type,
            "pattern": pattern,
            "decision": decision,
            "created_at": now(),
        })
        self.save(rules)
        return rules

    def remove(self, index):
        """按索引删除规则。索引从 1 开始（与 list 显示一致）。"""
        rules = self.load()
        if index < 1 or index > len(rules):
            raise IndexError(f"rule index {index} out of range (1-{len(rules)})")
        removed = rules.pop(index - 1)
        self.save(rules)
        return removed

    def list_rules(self):
        """返回规则列表（含索引，从 1 开始）。"""
        rules = self.load()
        result = []
        for i, rule in enumerate(rules, start=1):
            result.append({
                "index": i,
                "type": rule["type"],
                "pattern": rule["pattern"],
                "decision": rule["decision"],
            })
        return result

    def _rule_matches(self, rule, command):
        """检查单条规则是否匹配命令。"""
        rule_type = rule["type"]
        pattern = rule["pattern"]
        if rule_type == "COMMAND":
            return command == pattern
        elif rule_type == "PREFIX":
            return command.startswith(pattern)
        elif rule_type == "PATTERN":
            try:
                return bool(re.search(pattern, command))
            except re.error:
                return False
        elif rule_type == "DANGER":
            try:
                return bool(re.search(pattern, command))
            except re.error:
                return False
        return False

    def _default_rules(self):
        """内置默认规则：危险命令自动拒绝。"""
        rules = []
        for pattern in _BUILTIN_DANGER_PATTERNS:
            rules.append({
                "type": "DANGER",
                "pattern": pattern,
                "decision": "never",
                "created_at": now(),
            })
        return rules

    def _merge_with_defaults(self, user_rules):
        """确保内置危险规则始终存在，用户规则追加在后面。"""
        defaults = self._default_rules()
        default_patterns = {r["pattern"] for r in defaults}
        # 用户如果手动添加了同 pattern 的规则，用用户的覆盖默认
        user_patterns = {r["pattern"] for r in user_rules}
        merged = []
        for rule in defaults:
            if rule["pattern"] not in user_patterns:
                merged.append(rule)
        merged.extend(user_rules)
        return merged
