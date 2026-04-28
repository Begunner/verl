# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test fixture: ``@function_tool`` examples loaded by the agent-loop tests.

Used as the ``function_tool_path`` target by
``test_tool_agent_loop_with_function_tools_on_cpu.py``. Two minimal tools
(``echo``, ``calculator``) are sufficient to exercise both schema inference
and dispatch.

Not intended as a production tool module; copy the pattern into your own
file and reference it via the rollout config field
``actor_rollout_ref.rollout.multi_turn.function_tool_path``.
"""

from verl.tools.utils.function_tool import function_tool


@function_tool("echo")
def echo(text: str) -> str:
    """Echo back the text the caller passed in.

    Args:
        text: The string to echo back verbatim.
    """
    return text


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports +, -, *, /, **, parentheses, and unary minus. Use this for any
    numerical computation instead of doing mental arithmetic.

    Args:
        expression: A Python-style arithmetic expression, e.g. "(3+4)*5".
    """
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):  # noqa: UP038
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported node: {ast.dump(node)}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"ERROR: {e}"
