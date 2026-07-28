# CLAUDE.md - OmniMCP Implementation Guide

## Overview
This document describes how to implement OmniMCP, a system for UI automation through visual understanding and the Model Context Protocol (MCP).

## Core Architecture

The system consists of these essential components:

1. VisualState - Current screen state
2. MCP Server - Protocol implementation
3. Input Control - UI actions
4. UI Parser Integration - Visual analysis

## Implementation Approach

### 1. Start with VisualState
```python
class VisualState:
    def __init__(self):
        self.elements = []
        self.timestamp = None
        self.screen_dimensions = None
        
    def update(self, screenshot):
        """Update visual state from screenshot.
        
        Critical function that maintains screen state.
        Must handle:
        - Screenshot capture
        - UI element parsing
        - State updates
        - Coordinate normalization
        """
```

### 2. Implement Core MCP Server
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("omnimcp")


@mcp.tool()
async def get_screen_state() -> ScreenState:
    """Get current state of visible UI elements"""


@mcp.tool()
async def click_element(description: str) -> ClickResult:
    """Click UI element matching description"""


@mcp.tool()
async def type_text(text: str) -> TypeResult:
    """Type text"""
```

### 3. Build Element Targeting
```python
def find_element(description: str) -> Element:
    """Find UI element matching description.
    
    Critical for action reliability.
    Consider:
    - Text matching
    - Element type
    - Location/context
    - Confidence scores
    """
```

## Implementation Order

1. Visual State Management
   - Screenshot capture
   - UI parsing
   - State updates
   - Basic caching

2. MCP Protocol
   - Observe endpoint
   - Simple actions
   - Rich responses
   - Error handling

3. Action System
   - Element targeting
   - Input simulation
   - Action verification
   - Error recovery

## Key Considerations

### State Management
- Always update before actions
- Cache intelligently
- Track history when needed
- Clear invalidation

### Error Handling
- Rich error context
- Recovery strategies
- Debug information
- Verification

### Performance
- Minimize updates
- Smart caching
- Async where beneficial
- Efficient targeting

## MCP Protocol Details

### Observe
```python
@dataclass
class UIElement:
    content: str
    type: str
    bounds: Bounds
    confidence: float


@dataclass
class ScreenState:
    elements: List[UIElement]
    dimensions: tuple[int, int]
    timestamp: float


@dataclass
class ActionResult:
    success: bool
    element: Optional[UIElement]
    error: Optional[str] = None
```

## Code Structure

Current implementation:
```
./
├── omnimcp/             # Main package directory
│   ├── omnimcp.py       # Core implementation with OmniMCP class and VisualState
│   ├── input.py         # Input controller for UI interactions
│   ├── types.py         # Type definitions (Bounds, UIElement, etc.)
│   ├── utils.py         # Utilities for screenshots, coordinates, etc.
│   ├── config.py        # Centralized configuration
│   └── omniparser/      # UI parsing functionality
│       ├── client.py    # Parser client and provider
│       └── server.py    # Parser deployment and management
├── tests/               # Test directory
│   ├── test_synthetic_ui.py  # Synthetic UI generation for testing
│   └── test_omnimcp.py       # Core functionality tests
└── run_omnimcp.py       # Command-line entry point
```

Planned expansion:
```
./
├── utils.py              # Core utilities and input control
├── omniparser/          # UI parsing functionality
│   ├── client.py        # Parser client and provider
│   └── server.py        # Parser deployment and management
├── core/               # Future: Core state management
│   ├── visual_state.py
│   └── element.py
└── mcp/                # Future: MCP implementation
    └── server.py
```

## Package Management

OmniMCP uses `uv` for dependency management. When adding new dependencies, use:

```bash
uv add <package-name>       # Add a regular dependency
uv add --dev <package-name> # Add a development dependency
uv pip install -e .         # Install all dependencies
```

This ensures dependencies are properly recorded in pyproject.toml.

## Configuration System

OmniMCP now uses a centralized configuration system with:

- Settings loaded from environment variables and `.env` file
- Default values for all settings
- Support for various configuration types:
  - Claude API settings
  - OmniParser connection settings 
  - AWS deployment configuration
  - Debug and logging settings

To configure OmniMCP, create a `.env` file in the project root with your settings:

## Implementation Notes

### Core Principles
1. Visual state is always current
2. Every action verifies completion
3. Rich error context always available
4. Debug information accessible

### Critical Functions
1. VisualState.update()
2. MCPServer.observe()
3. find_element()
4. verify_action()

### Error Handling
```python
@dataclass
class ToolError:
    message: str
    visual_context: Optional[bytes]  # Screenshot
    attempted_action: str
    element_description: str
    recovery_suggestions: List[str]
```

### Testing Requirements
1. Unit tests for core logic
2. Integration tests for flows
3. Visual verification
4. Performance benchmarks

### Synthetic UI Testing
OmniMCP includes tools for generating synthetic test UIs with:

- Predefined UI elements (buttons, text fields, checkboxes)
- Before/after image pairs for action verification
- Element visualization for debugging

This approach offers several advantages:
- Works across all platforms
- Runs in any environment (including CI)
- Provides deterministic results 
- Doesn't require actual displays
- Enables testing different scenarios

## Example Implementation Flow

1. **Setup Visual State**
```python
visual_state = VisualState()
visual_state.update(take_screenshot())
```

2. **Find Target Element**
```python
element = visual_state.find_element_by_content("Submit")
if not element:
    raise MCPError("Element not found", context=visual_state.to_dict())
```

3. **Take Action**
```python
success = await input_controller.click(element.center)
if not success:
    raise MCPError("Click failed", context={"element": element})
```

4. **Verify Result**
```python
@dataclass
class ActionVerification:
    success: bool
    before_state: bytes  # Screenshot
    after_state: bytes
    changes_detected: List[BoundingBox]
    confidence: float


async def verify_tool_execution(
    action_result: ActionResult, verification: ActionVerification
) -> bool:
    """Verify tool executed successfully"""
```

## Remember

1. Focus on core functionality first
2. Build incrementally
3. Test thoroughly
4. Keep it simple but robust
5. Always verify actions
6. Maintain current state
7. Provide rich error context

This implementation guide focuses on the essential components needed for effective UI automation through visual understanding and action. Follow the implementation order strictly and ensure each component is solid before moving to the next.

===
===

Here's a high-level description of the ideal OmniMCP system:

# OmniMCP System Design

## Core Purpose
OmniMCP is a Model Context Protocol (MCP) server that enables AI models (particularly Claude) to:
1. Understand UI elements on screen through visual analysis
2. Take actions through mouse and keyboard control
3. Get rich visual context about UI elements using Claude's vision capabilities

## Key Components

### 1. MCP Server
```python
class MCPServer:
    """Core MCP server implementing the Model Context Protocol.
    
    Primary interface for AI models to interact with the UI.
    """
    
    async def get_screen_state() -> Dict:
        """Get current screen state with UI elements."""
        
    async def analyze_ui(query: str, max_elements: int = 5) -> Dict:
        """Analyze UI elements matching a natural language query."""
        
    async def click_element(descriptor: str) -> Dict:
        """Click UI element by description."""
        
    async def type_text(text: str) -> Dict:
        """Type text using keyboard."""
        
    async def press_key(key: str) -> Dict:
        """Press a keyboard key."""
```

### 2. Visual Analysis
```python
class VisualState:
    """Represents current screen state with UI elements."""
    
    def update_from_parser(self, parser_result: Dict):
        """Update state from UI parser results."""
        
    def find_element_by_content(self, content: str) -> Optional[Element]:
        """Find UI element by content."""
        
    def to_mcp_description(self) -> Dict:
        """Convert state to MCP format."""
```

### 3. UI Parser Integration 
```python
class OmniParserClient:
    """Client for interacting with the OmniParser API."""

    def parse_image(self, image: Image.Image) -> Dict[str, Any]:
        """Parse an image using the OmniParser service."""

    def check_server_available(self) -> bool:
        """Check if the OmniParser server is available."""


class OmniParserProvider:
    """Provider for OmniParser services with deployment capabilities."""

    def deploy(self) -> bool:
        """Deploy OmniParser if not already running."""

    def is_available(self) -> bool:
        """Check if parser is available."""
```

### 4. Input Control
```python
class InputController:
    """Handles mouse and keyboard input."""
    
    def click(self, x: float, y: float):
        """Click at coordinates."""
        
    def type_text(self, text: str):
        """Type text."""
        
    def press_key(self, key: str):
        """Press keyboard key."""
```

### 5. Claude Vision Integration
```python
class ClaudeVision:
    """Handles visual analysis using Claude."""

    async def describe_elements(
        elements: List[Element], context: Optional[Image] = None
    ) -> List[str]:
        """Get detailed descriptions of UI elements."""

    async def analyze_visual_query(
        query: str, screenshot: Image, elements: List[Element]
    ) -> Dict:
        """Answer questions about UI using Claude's vision."""
```

## MCP Tools Interface

@mcp.tool()
async def get_screen_state() -> ScreenState:
    """Get current state of visible UI elements"""
    state = await visual_state.capture()
    return state

@mcp.tool()
async def find_element(description: str) -> Optional[UIElement]:
    """Find UI element matching natural language description"""
    state = await get_screen_state()
    return semantic_element_search(state.elements, description)

@mcp.tool()
async def click_element(description: str) -> ClickResult:
    """Click UI element matching description"""
    element = await find_element(description)
    if not element:
        return ClickResult(success=False, error="Element not found")
    return await perform_click(element)

@mcp.tool()
async def type_text(text: str) -> TypeResult:
    """Type text using keyboard"""
    try:
        await keyboard.type_text(text)
        return TypeResult(success=True, text_entered=text)
    except Exception as e:
        return TypeResult(success=False, error=str(e))

@mcp.tool()
async def press_key(
    key: str,
    modifiers: List[str] = None
) -> ActionResult:
    """Press keyboard key with optional modifiers"""

## Key Features

1. **Smart UI Analysis**
   - Visual element detection
   - Natural language queries
   - Rich context through Claude vision
   - Element relationships and hierarchy

2. **Robust Actions**
   - Smart element targeting
   - Coordinate normalization
   - Input verification
   - Action confirmation

3. **Development Support**
   - Debug visualizations
   - Action logging 
   - Error diagnostics
   - Performance metrics

4. **Deployment Options**
   - Local parser
   - Remote parser service
   - Auto-deployment
   - Service management


===
===


# OmniMCP Implementation Approach

## Core Design Principles
1. MCP server is the primary interface
2. Visual state is always current
3. Errors are descriptive and actionable
4. Debug information is always available

## Implementation Path

### 1. Foundation (Based on proven code)
```python
class OmniMCP:
    def __init__(self):
        self.visual_state = VisualState()
        self.ui_parser = UIParserProvider()
        self.keyboard = KeyboardController()
        self.mouse = MouseController()

    def update_visual_state(self):
        screenshot = take_screenshot()
        parser_result = self.ui_parser.parse_screenshot(screenshot)
        self.visual_state.update_from_parser(parser_result)
```

### 2. MCP Server First
- Implement core MCP tools based on our working server.py
- Each tool updates visual state before acting
- All tools return structured responses
- Debug screenshots for each action

### 3. Visual Analysis Pipeline
1. Screenshot capture
2. UI element parsing
3. State management 
4. Claude vision integration for rich context

### 4. Action System
1. Element targeting
2. Coordinate handling
3. Input simulation
4. Action verification

### 5. Debug Infrastructure
- Visual state snapshots
- Action logging
- Error context
- Performance metrics

## Key Implementation Details

### MCP Server
- Use FastMCP for protocol compatibility
- Structured responses for all actions
- Visual state always updated before actions
- Rich error context in responses

### Visual State
- Keep normalized and absolute coordinates
- Track element confidence scores
- Maintain element relationships
- Cache recent states for context

### UI Parser Integration
- Start with local parser
- Remote parser as fallback
- Smart deployment management
- Connection recovery

### Input Control
- Use proven pynput implementation
- Coordinate normalization
- Action verification
- Error recovery

## Critical Considerations

1. **Error Handling**
   - Clear error messages
   - Recovery strategies
   - Debug context
   - User feedback

2. **Performance**
   - Minimize visual state updates
   - Cache when possible
   - Async where beneficial
   - Smart retries

3. **Reliability**
   - Verify actions
   - Handle edge cases
   - Recover from failures
   - Maintain state consistency


===
===


# OmniMCP Core Protocol

## Core Concept
MCP for OmniMCP is fundamentally about enabling AI models to:
1. Understand what's on screen through rich context
2. Take actions using natural language descriptions

## Essential Tools

@mcp.tool()
async def get_screen_state() -> ScreenState:
    """Get current state of visible UI elements
    
    Returns:
        ScreenState containing all visible UI elements with their properties
    """

@mcp.tool()
async def find_element(description: str) -> Optional[UIElement]:
    """Find UI element matching natural language description
    
    Args:
        description: Natural language description of element (e.g. "the submit button")
    """

@mcp.tool()
async def click_element(description: str) -> ClickResult:
    """Click UI element matching description
    
    Args:
        description: Natural language description of element to click
    """

@mcp.tool()
async def type_text(text: str) -> TypeResult:
    """Type text using keyboard
    
    Args:
        text: Text to type
    """

@mcp.tool()
async def press_key(
    key: str,
    modifiers: List[str] = None
) -> ActionResult:
    """Press keyboard key with optional modifiers
    
    Args:
        key: Key to press (e.g. "enter", "tab")
        modifiers: Optional modifier keys (e.g. ["ctrl", "shift"])
    """

## Key Design Points

1. **Simplicity**
   - Two core endpoints: observe and act
   - Analysis as enhancement of observation
   - Clear, consistent response structure

2. **Stateful Context**
   - Server maintains current visual state
   - Actions update state automatically
   - Historical context available when needed

3. **Natural Language Interface**
   - Element targeting by description
   - Rich analysis of visual state
   - Error messages in natural language

4. **Verification**
   - Actions confirm completion
   - Visual state updates verify changes
   - Clear error reporting

This represents the minimal, essential MCP interface needed for effective UI automation through visual understanding and action.

### Prompt Templates

Use template utilities for clean, maintainable prompts:

```python
from omnimcp.utils import create_prompt_template, render_prompt

# Create reusable template
analyze_template = create_prompt_template("""
    Analyze this UI element:
    {{ element.description }}
    
    Location: {{ element.bounds }}
    Type: {{ element.type }}
    
    Suggest interactions based on:
    {% for attr in element.attributes %}
    - {{ attr }}
    {% endfor %}
""")

# Render with data
prompt = analyze_template.render(
    element=ui_element
)

# Or one-step helper
prompt = render_prompt("""
    Quick analysis: {{ element.description }}
""", element=ui_element)

## Implementation Status

Note: The current implementation in `omnimcp.py` represents the API design based on MCP specifications but has not been tested with actual MCP server implementations yet. The types and tools are defined but require:

1. Integration testing with MCP SDK
2. Verification of tool definitions
3. Testing with Claude and other MCP clients
4. Implementation of actual tool logic

This design serves as a starting point for implementing a compliant MCP server for UI understanding.

## Testing Strategy

### Synthetic UI Testing

For testing visual understanding without relying on real UIs or displays, we'll use programmatically generated images:

```python
def generate_test_ui():
    """Generate synthetic UI image with known elements."""
    from PIL import Image, ImageDraw
    
    # Create blank canvas
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw UI elements with known positions
    elements = []
    
    # Button
    draw.rectangle([(100, 100), (200, 150)], fill='blue', outline='black')
    draw.text((110, 115), "Submit", fill="white")
    elements.append({
        "type": "button",
        "content": "Submit",
        "bounds": {"x": 100, "y": 100, "width": 100, "height": 50},
        "confidence": 1.0
    })
    
    # Text field
    draw.rectangle([(300, 100), (500, 150)], fill='white', outline='black')
    draw.text((310, 115), "Username", fill="gray")
    elements.append({
        "type": "text_field",
        "content": "Username",
        "bounds": {"x": 300, "y": 100, "width": 200, "height": 50},
        "confidence": 1.0
    })
    
    return img, elements
```

### Action Verification Testing

For testing action verification, we'll generate before/after image pairs:

```python
def generate_action_test_pair(action_type="click"):
    """Generate before/after UI image pair for a specific action."""
    before_img, elements = generate_test_ui()
    after_img = before_img.copy()
    after_draw = ImageDraw.Draw(after_img)

    if action_type == "click":
        # Show button in pressed state
        after_draw.rectangle([(100, 100), (200, 150)], fill="darkblue", outline="black")
        after_draw.text((110, 115), "Submit", fill="white")
        # Add success message
        after_draw.text((100, 170), "Form submitted!", fill="green")

    elif action_type == "type":
        # Show text entered in field
        after_draw.rectangle([(300, 100), (500, 150)], fill="white", outline="black")
        after_draw.text((310, 115), "testuser", fill="black")

    return before_img, after_img, elements
```

### Test Implementation

Testing Claude integration with synthetic images:

```python
async def test_element_finding():
    """Test Claude's ability to find elements in synthetic UI."""
    # Generate test image with known elements
    test_img, elements = generate_test_ui()

    # Mock screenshot capture to return test image
    with patch("omnimcp.utils.take_screenshot", return_value=test_img):
        # Setup OmniMCP with mock parser that returns our elements
        # ...

        # Test with various descriptions
        descriptions = [
            "submit button",
            "blue button",
            "the username field",
            "textbox in the middle",
        ]

        for desc in descriptions:
            # Call find_element with each description
            element = await mcp._visual_state.find_element(desc)
            # Verify the correct element was found
            # ...
```

This testing approach:
- Works across all platforms
- Runs in any environment (including CI)
- Provides deterministic results
- Doesn't require actual displays or UI
- Allows testing a variety of scenarios

For real UI action testing, we'll start with manual verification while developing more sophisticated test environments.

Focus on implementing the core functionality first, then expand the testing framework.

