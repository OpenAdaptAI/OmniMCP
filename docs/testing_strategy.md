# CI Testing Options for OmniMCP

This document outlines potential approaches for testing OmniMCP in CI environments and across different platforms where display access may be limited.

## Challenge

Testing UI automation tools in CI environments presents several challenges:
- No physical display may be available
- Mouse/keyboard control may not be possible
- Cross-platform differences in window management
- Deterministic testing requires controlled environments

## Potential Approaches

### 1. Virtual Display with Headless Browser

Use virtual display technology to simulate a screen:

```python
def setup_virtual_display():
    """Setup virtual display for UI testing."""
    try:
        from pyvirtualdisplay import Display

        display = Display(visible=0, size=(1280, 1024))
        display.start()

        # Use a headless browser
        from selenium import webdriver

        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get("http://localhost:8080/testpage.html")

        return display, driver
    except ImportError:
        # Handle platforms without Xvfb support
        return None, None
```

**Pros:**
- Tests actual UI rendering
- Can work with real browsers in headless mode
- Relatively realistic

**Cons:**
- Platform-specific (Xvfb mainly for Linux)
- May require additional setup in CI
- Can be flaky

### 2. Synthetic Test Images

Generate test images programmatically with known UI elements:

```python
def create_test_images():
    """Generate synthetic UI test images."""
    from PIL import Image, ImageDraw, ImageFont

    # Before image with button
    before = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(before)
    draw.rectangle([(100, 100), (250, 150)], fill="blue")
    draw.text((125, 115), "Test Button", fill="white")

    # After image with success message
    after = before.copy()
    draw = ImageDraw.Draw(after)
    draw.text((100, 170), "Success! Button was clicked.", fill="green")

    return before, after
```

**Pros:**
- Works on any platform
- No display required
- Completely deterministic
- Fast and reliable

**Cons:**
- Not testing actual UI behavior
- Simplified representation of real UIs
- Need to manually specify element positions

### 3. Mock the Visual Pipeline

Mock the screenshot and parsing components to return predefined data:

```python
def mock_visual_pipeline():
    """Patch the visual pipeline components for testing."""
    patches = []

    # Mock screenshot function
    before_img, after_img = create_test_images()
    mock_screenshot = MagicMock(return_value=before_img)
    patches.append(patch("omnimcp.utils.take_screenshot", mock_screenshot))

    # Create predefined elements
    test_elements = [
        {
            "type": "button",
            "content": "Test Button",
            "bounds": {"x": 100, "y": 100, "width": 150, "height": 50},
            "confidence": 1.0,
        }
    ]

    # Mock parser
    mock_parser = MagicMock()
    mock_parser.parse_image.return_value = {"parsed_content_list": test_elements}
    patches.append(
        patch("omnimcp.omniparser.client.OmniParserClient", return_value=mock_parser)
    )

    return patches
```

**Pros:**
- Works everywhere
- Fast and reliable
- No external dependencies
- Easy to control test scenarios

**Cons:**
- Not testing actual UI behavior
- Mocking too much of the system
- May miss integration issues

### 4. HTML Canvas Rendering

Generate UI in HTML canvas and capture it:

```python
def generate_ui_canvas():
    """Generate UI using HTML canvas and capture it."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <canvas id="uiCanvas" width="800" height="600"></canvas>
        <script>
            const canvas = document.getElementById('uiCanvas');
            const ctx = canvas.getContext('2d');
            
            // Draw background
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, 800, 600);
            
            // Draw button
            ctx.fillStyle = 'blue';
            ctx.fillRect(100, 100, 150, 50);
            
            // Draw button text
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText('Test Button', 125, 130);
            
            // Convert to image data
            const imgData = canvas.toDataURL('image/png');
            console.log(imgData);  // This can be captured and converted to PIL Image
        </script>
    </body>
    </html>
    """
    # Method to render this HTML and capture the canvas output
    # would be implemented here
```

**Pros:**
- Cross-platform
- No display needed
- Can be rendered headlessly
- Visual representation without browser

**Cons:**
- Complex implementation
- Doesn't test real UI interaction
- Extra rendering engine dependency

### 5. Hybrid Environment-Aware Testing

Adapt tests based on the environment:

```python
def get_test_environment():
    """Determine test environment and return appropriate testing setup."""
    is_ci = os.environ.get("CI", "0") == "1"
    platform = sys.platform

    if is_ci:
        # In CI, use synthetic images
        return {
            "type": "synthetic",
            "images": create_test_images(),
            "elements": create_test_elements(),
        }
    elif platform == "darwin":  # macOS
        # On macOS developer machine, use real UI
        return {"type": "real", "setup": lambda: start_test_app()}
    elif platform == "win32":  # Windows
        # On Windows, use headless browser
        return {"type": "headless", "setup": lambda: setup_headless_browser()}
    else:  # Linux or other
        # On Linux, use Xvfb
        return {"type": "xvfb", "setup": lambda: setup_virtual_display()}
```

**Pros:**
- Adaptable to different environments
- Best approach for each platform
- Real tests on developer machines
- Synthetic tests in CI

**Cons:**
- More complex to maintain
- Different test behavior in different environments
- May mask environment-specific issues

## Recommended Next Steps

1. Start with simple synthetic images for initial testing
2. Document test limitations clearly
3. Gradually build more sophisticated testing as the project matures
4. Consider developing a test UI application specifically for OmniMCP testing

No single approach is perfect, and the final testing strategy will likely combine elements from multiple approaches based on the specific needs and constraints of the project.
