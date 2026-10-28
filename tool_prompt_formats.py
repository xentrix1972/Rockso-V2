"""Native Direct tool prompt used by BFCL. These helpers never execute calls.

Rendering matches the training-time Direct tooling contract (tool_contract.py, 2026-09-19).
"""
import json
from pathlib import Path

SYSTEM = ('You are a function-calling assistant. You are given tool signatures inside '
          '<tools></tools>. To call tools, emit one or more '
          '<tool_call>{"name": <tool>, "arguments": <args>}</tool_call> blocks. '
          'After receiving tool results, either call more tools or give the user a final '
          'answer. If no tool is needed, answer directly. '
          'Do not invent missing argument values or claim unobserved results.')


def load_tools(path):
    value = json.loads(Path(path).expanduser().read_text())
    tools = value.get('tools') if isinstance(value, dict) else value
    if not isinstance(tools, list) or not tools:
        raise ValueError('Expected a nonempty JSON list of tool schemas, or {"tools": [...]}')
    names = set()
    for tool in tools:
        if (not isinstance(tool, dict) or not isinstance(tool.get('name'), str)
                or not isinstance(tool.get('parameters'), dict)):
            raise ValueError('Each tool needs a name and a parameters JSON schema')
        if tool['name'] in names:
            raise ValueError(f'Duplicate tool name: {tool["name"]}')
        names.add(tool['name'])
    return tools


def render_tool_question(tools, question, policy=None):
    parts = [SYSTEM, '']
    if policy:
        parts.extend(['System: ' + policy, ''])
    parts.extend(['Available tools:', '<tools>'])
    parts.extend(json.dumps(t, ensure_ascii=False, separators=(',', ':'), allow_nan=False) for t in tools)
    parts.extend(['</tools>', '', 'User: ' + question])
    return '\n'.join(parts)
