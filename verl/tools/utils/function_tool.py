# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Lightweight function-based tool registration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# global registry
FUNCTION_TOOL_REGISTRY: dict[str, FunctionTool] = {}


@dataclass
class FunctionTool:
    """Carrier object stored in :data:`FUNCTION_TOOL_REGISTRY`.

    Exposes the minimal interface that the agent loop relies on:

    - ``name``: tool name (matches ``tool_schema.function.name``)
    - ``tool_schema``: ``OpenAIFunctionToolSchema`` for prompt assembly
    - ``fn``: the underlying callable
    """

    name: str
    fn: Callable[..., Any]
    tool_schema: OpenAIFunctionToolSchema
    is_async: bool = False

    async def call(self, parameters: dict[str, Any]) -> Any:
        """Invoke the underlying function with the LLM-supplied parameters."""
        if self.is_async:
            return await self.fn(**parameters)
        return await asyncio.to_thread(self.fn, **parameters)


def function_tool(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    schema: Optional[OpenAIFunctionToolSchema | dict] = None,
):
    """Register a Python function as a verl tool.

    The function's signature drives the JSON schema; type hints and
    Google-style ``Args:`` docstring sections supply types and per-argument
    descriptions. Pass ``schema=`` to override the inferred schema entirely.

    Args:
        name: Tool name exposed to the LLM. Defaults to the function name.
        description: Tool description; defaults to the function's docstring
            summary.
        schema: Override the auto-inferred OpenAI schema. Accepts an
            ``OpenAIFunctionToolSchema`` or a dict matching that shape.

    Example:
        >>> @function_tool("web_search")
        ... def web_search(query: str, top_k: int = 5) -> str:
        ...     '''Search the web for information.
        ...
        ...     Args:
        ...         query: The search query.
        ...         top_k: Maximum number of results.
        ...     '''
        ...     return do_search(query, top_k)
    """

    def decorator(fn: Callable):
        tool_name = name or fn.__name__

        if isinstance(schema, OpenAIFunctionToolSchema):
            built_schema = schema
        elif isinstance(schema, dict):
            built_schema = OpenAIFunctionToolSchema.model_validate(schema)
        else:
            built_schema = _build_schema_from_fn(fn, tool_name, description)

        entry = FunctionTool(
            name=tool_name,
            fn=fn,
            tool_schema=built_schema,
            is_async=inspect.iscoroutinefunction(fn),
        )

        existing = FUNCTION_TOOL_REGISTRY.get(tool_name)
        if existing is not None and existing.fn is not fn:
            raise ValueError(
                f"Function tool '{tool_name}' is already registered to "
                f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                f"refusing to overwrite with {fn.__module__}.{fn.__qualname__}."
            )
        FUNCTION_TOOL_REGISTRY[tool_name] = entry
        logger.info("Registered function tool '%s' from %s.%s", tool_name, fn.__module__, fn.__qualname__)
        return fn

    return decorator


def get_function_tool(name: str) -> FunctionTool:
    """Look up a registered function tool by name. Raises ``KeyError`` if absent."""
    if name not in FUNCTION_TOOL_REGISTRY:
        raise KeyError(
            f"Function tool '{name}' not found in registry. Make sure its defining "
            f"file is referenced via the rollout `function_tool_path` config."
        )
    return FUNCTION_TOOL_REGISTRY[name]


def load_function_tools_from_path(path: str) -> list[FunctionTool]:
    """Execute a Python file at ``path`` and return all registered function tools."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"function_tool_path does not exist: {path}")

    # Use a path-derived synthetic module name so the imported file can
    # `from X import Y` its siblings via sys.modules.
    module_name = "_verl_function_tools_" + abs_path.replace(os.sep, "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for function_tool_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not FUNCTION_TOOL_REGISTRY:
        logger.warning(
            "function_tool_path '%s' loaded but no @function_tool decorators fired; "
            "did you forget to apply the decorator?",
            path,
        )
    else:
        logger.info(
            "Loaded %d function tool(s) from %s: %s",
            len(FUNCTION_TOOL_REGISTRY),
            path,
            sorted(FUNCTION_TOOL_REGISTRY),
        )
    return list(FUNCTION_TOOL_REGISTRY.values())


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

# Best-effort mapping from a Python type hint to a JSON-schema primitive.
# Anything not in this map falls back to "string" so the LLM still gets a
# valid schema even for exotic types.
_PRIMITIVE_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_type(py_type: Any) -> str:
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
    return _PRIMITIVE_TYPE_MAP.get(py_type, "string")


def _parse_args_section(docstring: Optional[str]) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into a (summary, arg_descriptions) pair.

    Recognized argument-section headers: ``Args:``, ``Arguments:``,
    ``Parameters:`` (case-insensitive). Section ends at any of
    ``Returns:`` / ``Raises:`` / ``Yields:`` / ``Examples:`` / ``Notes:`` or at
    a non-indented line.
    """
    if not docstring:
        return "", {}

    lines = inspect.cleandoc(docstring).splitlines()
    summary_lines: list[str] = []
    arg_descs: dict[str, str] = {}
    in_args = False
    current_arg: Optional[str] = None

    arg_headers = {"args", "arguments", "parameters"}
    end_headers = {"returns", "return", "raises", "yields", "examples", "example", "note", "notes"}

    for line in lines:
        stripped = line.strip()
        header = stripped.lower().rstrip(":")
        if not in_args:
            if header in arg_headers:
                in_args = True
                continue
            summary_lines.append(line)
            continue

        if header in end_headers:
            break
        # A non-indented, non-empty line ends the Args section
        if line and not line[0].isspace() and stripped:
            break

        if ":" in stripped:
            arg_part, _, desc = stripped.partition(":")
            # Strip optional "(type)" annotation, e.g. "name (str): ..."
            arg_name = arg_part.split("(", 1)[0].strip()
            if arg_name:
                arg_descs[arg_name] = desc.strip()
                current_arg = arg_name
        elif current_arg and stripped:
            arg_descs[current_arg] = (arg_descs[current_arg] + " " + stripped).strip()

    return "\n".join(summary_lines).strip(), arg_descs


def _build_schema_from_fn(
    fn: Callable, tool_name: str, override_description: Optional[str]
) -> OpenAIFunctionToolSchema:
    signature = inspect.signature(fn)
    summary, arg_descs = _parse_args_section(fn.__doc__)
    description = override_description or summary or f"Tool '{tool_name}'."

    properties: dict[str, OpenAIFunctionPropertySchema] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        # *args / **kwargs are not representable in OpenAI function schemas.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        json_type = _python_type_to_json_type(annotation)

        properties[param_name] = OpenAIFunctionPropertySchema(
            type=json_type,
            description=arg_descs.get(param_name),
        )
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=tool_name,
            description=description,
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties=properties,
                required=required,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Return-value normalization
# ---------------------------------------------------------------------------


def normalize_function_tool_return(ret: Any) -> tuple[ToolResponse, float, dict]:
    """Coerce a function's return value into the ``(ToolResponse, reward, metrics)`` triple.

    Conventions:

    ===========================  =================================================
    Function return              Normalized triple
    ===========================  =================================================
    ``str``                      ``(ToolResponse(text=str), 0.0, {})``
    ``ToolResponse``             ``(ret, 0.0, {})``
    ``dict``                     ``(ToolResponse(text=json.dumps(ret)), 0.0, {})``
    1-tuple ``(resp,)``          ``(coerce(resp), 0.0, {})``
    2-tuple ``(resp, reward)``   ``(coerce(resp), float(reward), {})``
    3-tuple ``(resp, reward, m)``  pass-through with type coercion
    other                        ``(ToolResponse(text=str(ret)), 0.0, {})``
    ===========================  =================================================
    """
    if isinstance(ret, ToolResponse):
        return ret, 0.0, {}
    if isinstance(ret, str):
        return ToolResponse(text=ret), 0.0, {}
    if isinstance(ret, dict):
        return ToolResponse(text=json.dumps(ret, ensure_ascii=False)), 0.0, {}
    if isinstance(ret, tuple):
        if len(ret) == 1:
            return _coerce_response(ret[0]), 0.0, {}
        if len(ret) == 2:
            return _coerce_response(ret[0]), float(ret[1]), {}
        if len(ret) == 3:
            return _coerce_response(ret[0]), float(ret[1]), dict(ret[2])
    return ToolResponse(text=str(ret)), 0.0, {}


def _coerce_response(value: Any) -> ToolResponse:
    if isinstance(value, ToolResponse):
        return value
    if isinstance(value, str):
        return ToolResponse(text=value)
    if isinstance(value, dict):
        return ToolResponse(text=json.dumps(value, ensure_ascii=False))
    return ToolResponse(text=str(value))
