"""
Controlled Python Analysis Tool Registry (Phase 11: Intelligent Analysis Planning & Chart Generation).

Enforces strict execution security:
1. Only explicitly registered analysis tools may be called.
2. Dynamic code execution (eval, exec, subprocess, custom scripts) is strictly prohibited.
3. Validates parameter types and signatures before execution.
4. Generates standard JSON-schema descriptions for LLM routing and UI transparency.
"""

import inspect
import logging
from typing import Dict, Any, Callable, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when an unregistered tool is requested or execution fails."""
    pass


class RegisteredTool:
    """Represents a validated, registered Python analysis tool."""

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        category: str = "general",
        requires_visualization: bool = False,
        default_chart_type: Optional[str] = None,
    ):
        self.name = name
        self.func = func
        self.description = description
        self.category = category
        self.requires_visualization = requires_visualization
        self.default_chart_type = default_chart_type
        self.signature = inspect.signature(func)
        self.docstring = func.__doc__ or description

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON-serializable parameter schema for this tool."""
        parameters = {}
        for param_name, param in self.signature.parameters.items():
            if param_name == "df":
                continue  # DataFrame is injected by the framework, not LLM
            param_type = "string"
            if param.annotation == int or param.annotation == Optional[int]:
                param_type = "integer"
            elif param.annotation == float or param.annotation == Optional[float]:
                param_type = "number"
            elif param.annotation == bool or param.annotation == Optional[bool]:
                param_type = "boolean"
            elif param.annotation == list or param.annotation == Optional[List[str]]:
                param_type = "array"

            is_required = param.default == inspect.Parameter.empty

            parameters[param_name] = {
                "type": param_type,
                "required": is_required,
                "default": None if is_required else param.default,
            }

        return {
            "name": self.name,
            "description": self.description.strip(),
            "category": self.category,
            "requires_visualization": self.requires_visualization,
            "default_chart_type": self.default_chart_type,
            "parameters": parameters,
        }

    def validate_and_execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate arguments against signature and execute tool deterministically.
        Strictly prevents arbitrary script or code execution.
        """
        if not isinstance(df, pd.DataFrame):
            raise ToolExecutionError(f"First argument to tool '{self.name}' must be a pandas DataFrame.")

        # Bind and validate parameters
        bound_args = {}
        for param_name, param in self.signature.parameters.items():
            if param_name == "df":
                bound_args["df"] = df
            elif param_name in kwargs:
                val = kwargs[param_name]
                # Type coercion if needed
                if val is not None:
                    if param.annotation in (int, Optional[int]) and not isinstance(val, int):
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            pass
                    elif param.annotation in (float, Optional[float]) and not isinstance(val, float):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            pass
                    elif param.annotation in (bool, Optional[bool]) and not isinstance(val, bool):
                        if isinstance(val, str):
                            val = val.lower() in ("true", "1", "yes")
                bound_args[param_name] = val
            elif param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default
            else:
                # Required parameter missing
                bound_args[param_name] = None

        logger.info("Executing registered tool '%s' with safe parameters: %s", self.name, {k: v for k, v in bound_args.items() if k != "df"})
        return self.func(**bound_args)


class ToolRegistry:
    """
    Central Registry for all authorized deterministic Python data analysis tools.
    Provides strict authorization, schema inspection, and safe execution boundaries.
    """

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        category: str = "general",
        requires_visualization: bool = False,
        default_chart_type: Optional[str] = None,
    ) -> Callable:
        """Decorator to register a safe Python analysis function."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            desc = description or (func.__doc__ or "").strip().split("\n")[0]
            tool_obj = RegisteredTool(
                name=tool_name,
                func=func,
                description=desc,
                category=category,
                requires_visualization=requires_visualization,
                default_chart_type=default_chart_type,
            )
            self._tools[tool_name] = tool_obj
            return func
        return decorator

    def register_tool_instance(
        self,
        name: str,
        func: Callable,
        description: str,
        category: str = "general",
        requires_visualization: bool = False,
        default_chart_type: Optional[str] = None,
    ) -> RegisteredTool:
        """Explicitly register an analysis tool function."""
        tool_obj = RegisteredTool(
            name=name,
            func=func,
            description=description,
            category=category,
            requires_visualization=requires_visualization,
            default_chart_type=default_chart_type,
        )
        self._tools[name] = tool_obj
        return tool_obj

    def is_registered(self, tool_name: str) -> bool:
        """Check if a tool is registered and authorized for execution."""
        return tool_name in self._tools

    def get_tool(self, tool_name: str) -> Optional[RegisteredTool]:
        """Retrieve registered tool instance."""
        return self._tools.get(tool_name)

    def execute(self, tool_name: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute an authorized analysis tool safely.
        Raises ToolExecutionError if the tool is not registered.
        """
        if not self.is_registered(tool_name):
            raise ToolExecutionError(
                f"Tool '{tool_name}' is not an authorized registered analysis function. "
                f"Arbitrary code execution is strictly prohibited. "
                f"Authorized tools: {list(self._tools.keys())}"
            )

        tool = self._tools[tool_name]
        return tool.validate_and_execute(df, **kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List schemas of all registered analysis tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())


# Global Tool Registry Singleton
registry = ToolRegistry()
