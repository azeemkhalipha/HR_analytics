"""MCP server exposing the deterministic analytics tools so any MCP client (Claude Desktop, agents) can call them.

Run:  python -m wfr.mcp_server        (stdio transport)
"""
from mcp.server.mcpserver import MCPServer

from . import governance, tools

mcp = MCPServer("workforce-reinvention-tools")


@mcp.tool()
def attrition_risk_by_role(role: str | None = None) -> dict:
    """Out-of-fold attrition risk aggregated by role (anonymised; employee IDs only)."""
    return tools.attrition_risk(role).model_dump()


@mcp.tool()
def task_automation_impact(role: str) -> dict:
    """Task-level automation exposure for a role: which tasks to automate, augment or retain."""
    return tools.task_impact(role).model_dump()


@mcp.tool()
def adjacent_roles(role: str, top_k: int = 2) -> dict:
    """Lower-exposure roles with the highest skill overlap, plus the skills to build."""
    return tools.role_adjacency(role, top_k).model_dump()


@mcp.tool()
def employee_skills_gap(employee_id: str, target_role: str) -> dict:
    """Skills an employee (by ID) is missing for a target role."""
    return tools.skills_gap(employee_id, target_role)


@mcp.tool()
def search_roles(query: str, k: int = 3) -> list[dict]:
    """Retrieve the roles in the taxonomy that best match a free-text description."""
    return tools.retrieve_roles(query, k)


@mcp.tool()
def bias_audit() -> dict:
    """Four-fifths-rule audit of the attrition model across gender and age band."""
    return governance.bias_report()


if __name__ == "__main__":
    mcp.run()
