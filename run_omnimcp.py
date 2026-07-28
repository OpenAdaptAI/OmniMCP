#!/usr/bin/env python
"""
Run script for OmniMCP - Model Context Protocol for UI Automation.

This is the main entry point for the OmniMCP application. It provides command-line
interfaces for running the MCP server and other utilities.
"""

import asyncio

import fire
from loguru import logger
from omnimcp.omnimcp import OmniMCP


async def start_server(port=8000, debug=False, parser_url=None):
    """Start the OmniMCP server."""
    mcp = OmniMCP(parser_url=parser_url, debug=debug)
    await mcp.start(port=port)


def server(port=8000, debug=False, parser_url=None):
    """Run OmniMCP in server mode.

    Args:
        port: Port number to serve on (default: 8000)
        debug: Enable debug mode (default: False)
        parser_url: Custom URL for OmniParser service (default: None)
    """
    logger.info(f"Starting OmniMCP server on port {port}")
    asyncio.run(start_server(port=port, debug=debug, parser_url=parser_url))


def debug(port=8000, parser_url=None):
    """Run OmniMCP in debug mode.

    Args:
        port: Port number to serve on (default: 8000)
        parser_url: Custom URL for OmniParser service (default: None)
    """
    logger.info("Starting OmniMCP in debug mode")
    server(port=port, debug=True, parser_url=parser_url)


def main():
    """OmniMCP command line interface."""
    commands = {
        "server": server,
        "debug": debug,
    }
    fire.Fire(commands)


if __name__ == "__main__":
    main()
