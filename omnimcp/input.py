# omnimcp/input.py

import os
import sys
import time
from typing import Any, ClassVar, Literal

from loguru import logger

keyboard = None
mouse = None
_pynput_error = None

# Only attempt to import pynput if not on headless Linux
# (Check platform and presence of DISPLAY environment variable)
if sys.platform != "linux" or os.environ.get("DISPLAY"):
    try:
        from pynput import keyboard, mouse

        # Test if backend loaded successfully (might still fail later)
        _kb_test = keyboard.Controller()
        _ms_test = mouse.Controller()
        logger.info("pynput imported successfully.")
    except ImportError as e:
        _pynput_error = f"pynput import failed: {e}"
        logger.error(_pynput_error)
    except Exception as e:  # Catch potential backend errors during test instantiation
        _pynput_error = f"pynput backend failed to load: {e}"
        logger.error(_pynput_error)
        # Ensure keyboard/mouse are reset to None if test instantiation fails
        keyboard = None
        mouse = None
else:
    _pynput_error = "Skipping pynput import in headless Linux environment (no DISPLAY)."
    logger.warning(_pynput_error)

from omnimcp.utils import log_action

# Define Bounds type if not imported from elsewhere
BoundsTuple = tuple[float, float, float, float]  # (norm_x, norm_y, norm_w, norm_h)


class InputController:
    """
    Provides methods for controlling mouse and keyboard actions,
    including parsing key strings using pynput.
    """

    # --- Moved _special_map_definitions to be a Class Attribute ---
    _special_map_definitions: ClassVar[dict[str, str]] = {
        # Alias      : pynput Key attribute name
        "enter": "enter",
        "return": "enter",
        "space": "space",
        "spacebar": "space",
        "tab": "tab",
        "esc": "esc",
        "escape": "esc",
        "backspace": "backspace",
        "delete": "delete",
        "f1": "f1",
        "f2": "f2",
        "f3": "f3",
        "f4": "f4",
        "f5": "f5",
        "f6": "f6",
        "f7": "f7",
        "f8": "f8",
        "f9": "f9",
        "f10": "f10",
        "f11": "f11",
        "f12": "f12",
        "f13": "f13",
        "f14": "f14",
        "f15": "f15",
        "f16": "f16",
        "f17": "f17",
        "f18": "f18",
        "f19": "f19",
        "f20": "f20",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "page_up": "page_up",
        "page_down": "page_down",
        "home": "home",
        "end": "end",
        # Keys that might be missing on some platforms/keyboards:
        "insert": "insert",
        "menu": "menu",
        "num_lock": "num_lock",
        "pause": "pause",
        "print_screen": "print_screen",
        "scroll_lock": "scroll_lock",
    }
    # --- End Class Attribute ---

    def __init__(self):
        """
        Initializes the pynput controllers and defines key mappings.
        Raises ImportError if pynput is not installed.
        """
        if mouse is None or keyboard is None:
            raise ImportError(
                "pynput library is required for InputController but not installed or failed to import."
            )
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self.MouseButton = mouse.Button
        self.Key = keyboard.Key
        self.KeyCode = keyboard.KeyCode
        logger.info("pynput mouse and keyboard controllers initialized.")

        # --- Mappings referencing Class Attribute ---
        self.MODIFIER_MAP: dict[str, Any] = {
            "cmd": self.Key.cmd,
            "command": self.Key.cmd,
            "win": self.Key.cmd,
            "ctrl": self.Key.ctrl,
            "control": self.Key.ctrl,
            "alt": self.Key.alt,
            "option": self.Key.alt,
            "shift": self.Key.shift,
        }
        logger.debug(f"Initialized MODIFIER_MAP with {len(self.MODIFIER_MAP)} keys.")

        # Helper to safely get key attribute
        def _get_key(key_name: str) -> Any | None:
            try:
                return getattr(self.Key, key_name)
            except AttributeError:
                return None
            except Exception as e:
                logger.error(f"Unexpected error getting Key.{key_name}: {e}")
                return None

        # Build the instance's SPECIAL_KEY_MAP safely using the class attribute definitions
        self.SPECIAL_KEY_MAP: dict[str, Any] = {}
        missing_keys = set()
        # Use self._special_map_definitions or InputController._special_map_definitions here
        for alias, key_name in InputController._special_map_definitions.items():
            key_obj = _get_key(key_name)
            if key_obj:
                self.SPECIAL_KEY_MAP[alias] = key_obj
            else:
                missing_keys.add(key_name)

        logger.debug(
            f"Initialized SPECIAL_KEY_MAP with {len(self.SPECIAL_KEY_MAP)} keys. Missing/Skipped: {missing_keys or 'None'}"
        )
        # --- End Mappings ---

    @log_action
    def move(self, x: int, y: int) -> bool:
        """
        Move mouse pointer to ABSOLUTE pixel coordinates (x, y).

        Args:
            x: Target x-coordinate (pixel).
            y: Target y-coordinate (pixel).

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.mouse_controller.position = (int(x), int(y))
            return True
        except Exception as e:
            logger.error(f"Error moving mouse to ({x}, {y}): {e}")
            return False

    @log_action
    def click(
        self,
        x: int,
        y: int,
        click_type: Literal["single", "double", "right"] = "single",
    ) -> bool:
        """
        Move mouse to ABSOLUTE pixel coordinates (x, y) and perform a click.

        Args:
            x: Target x-coordinate (pixel).
            y: Target y-coordinate (pixel).
            click_type: Type of click ('single', 'double', 'right').

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.mouse_controller.position = (int(x), int(y))
            time.sleep(0.05)
            button_to_click = (
                self.MouseButton.right
                if click_type == "right"
                else self.MouseButton.left
            )
            click_count = 2 if click_type == "double" else 1
            self.mouse_controller.click(button_to_click, click_count)
            logger.debug(
                f"Performed {click_type} click with {button_to_click} at ({x}, {y})"
            )
            return True
        except Exception as e:
            logger.error(f"Error performing {click_type} click at ({x}, {y}): {e}")
            return False

    @log_action
    def type_text(self, text: str) -> bool:
        """
        Type the given string using the keyboard controller.

        Args:
            text: The string to type.

        Returns:
            True if successful, False otherwise.
        """
        if not isinstance(text, str):
            logger.error(
                f"Invalid type for text_to_type: {type(text)}. Must be string."
            )
            return False
        try:
            self.keyboard_controller.type(text)
            time.sleep(0.1 + len(text) * 0.01)
            return True
        except self.keyboard_controller.InvalidCharacterException as e:
            logger.error(f"Invalid character encountered while trying to type: {e}")
            return False
        except Exception as e:
            logger.error(f"Error typing text '{text[:50]}...': {e}")
            return False

    @log_action
    def execute_key_string(self, key_info_str: str) -> bool:
        """
        Parses a key string (e.g., "Cmd+Space", "Enter", "a") and executes the
        corresponding keyboard action using pynput controller methods.

        Args:
            key_info_str: The string describing the key action.

        Returns:
            True on success, False on failure (e.g., invalid key string).
        """
        if not key_info_str or not isinstance(key_info_str, str):
            logger.error(f"Invalid or empty key_info_str provided: {key_info_str}")
            return False

        logger.info(f"Attempting to execute key string: '{key_info_str}'")
        key_info_str = key_info_str.strip()
        parts = [
            part.strip().lower() for part in key_info_str.replace("-", "+").split("+")
        ]

        modifiers_to_press: list[keyboard.Key] = []
        primary_key_str: str | None = None

        # 1. Parse the string
        for part in parts:
            if not part:
                continue
            if part in self.MODIFIER_MAP:
                mod_key = self.MODIFIER_MAP[part]
                if mod_key not in modifiers_to_press:
                    modifiers_to_press.append(mod_key)
            elif primary_key_str is None:
                primary_key_str = part
            else:
                logger.error(
                    f"Invalid key combo string: Multiple non-modifier keys ('{primary_key_str}', '{part}') found in '{key_info_str}'"
                )
                return False

        # 2. Determine primary key object
        primary_key_obj: str | keyboard.Key | keyboard.KeyCode | None = None
        if primary_key_str:
            if primary_key_str in self.SPECIAL_KEY_MAP:
                primary_key_obj = self.SPECIAL_KEY_MAP[primary_key_str]
            elif len(primary_key_str) == 1:
                primary_key_obj = primary_key_str
            else:
                # --- Updated Check using Class Attribute ---
                # Check if the key name exists in the original definitions
                is_defined_alias = (
                    primary_key_str in InputController._special_map_definitions
                )
                # --- End Updated Check ---

                if is_defined_alias:
                    # It was defined, but not found in self.SPECIAL_KEY_MAP -> platform issue
                    logger.error(
                        f"Key '{primary_key_str}' is defined but not available on this platform/keyboard. Cannot execute."
                    )
                else:
                    # Truly unknown key name
                    logger.error(
                        f"Unknown primary key name: '{primary_key_str}' in key string '{key_info_str}'"
                    )
                return False

        # 3. Execute action
        try:
            if modifiers_to_press:
                if primary_key_obj:
                    logger.debug(
                        f"Executing combo: Modifiers={modifiers_to_press}, Key={primary_key_obj}"
                    )
                    with self.keyboard_controller.pressed(*modifiers_to_press):
                        self.keyboard_controller.tap(primary_key_obj)
                    time.sleep(0.05)
                else:
                    logger.debug(f"Tapping modifiers only: {modifiers_to_press}")
                    for mod in modifiers_to_press:
                        self.keyboard_controller.tap(mod)
                        time.sleep(0.03)
            elif primary_key_obj:
                if isinstance(primary_key_obj, str):
                    logger.debug(f"Typing character: '{primary_key_obj}'")
                    self.keyboard_controller.type(primary_key_obj)
                else:
                    logger.debug(f"Tapping special key: {primary_key_obj}")
                    self.keyboard_controller.tap(primary_key_obj)
                time.sleep(0.05)
            else:
                logger.error(
                    f"No valid key or modifier identified to execute in '{key_info_str}'"
                )
                return False
            return True
        except (
            ValueError,
            AttributeError,
            self.keyboard_controller.InvalidKeyException,
            self.keyboard_controller.InvalidCharacterException,
        ) as e:
            logger.error(f"Error executing key string '{key_info_str}': {e}")
            return False
        except Exception:
            logger.exception(
                f"Unexpected error during pynput execution for '{key_info_str}'"
            )
            return False

    @log_action
    def scroll(self, dx: int, dy: int) -> bool:
        """
        Scroll the mouse wheel horizontally (dx) and vertically (dy).

        Args:
            dx: Horizontal scroll amount.
            dy: Vertical scroll amount.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.mouse_controller.scroll(int(dx), int(dy))
            logger.debug(f"Scrolled mouse wheel by dx={dx}, dy={dy}")
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Error scrolling mouse (dx={dx}, dy={dy}): {e}")
            return False


# Example Usage (for testing input.py directly)
if __name__ == "__main__":
    logger.info("Testing InputController...")
    try:
        # Define _get_key here only for the test scope if needed, or rely on class instance
        # This is slightly awkward, maybe InputController init should handle this better
        # For now, assume InputController init succeeded.
        controller = InputController()
        logger.info("Controller initialized.")

        print("\n--- Testing Keyboard ---")
        print("Testing keyboard in 3s (will type, press Enter, combos)...")
        print(">>> Please focus a text input field now! <<<")
        time.sleep(3)

        logger.info("Testing simple typing...")
        success = controller.type_text("Test_123!?.")
        logger.info(f"type_text Result: {success}")
        time.sleep(0.5)

        logger.info("Testing special key (Enter)...")
        success = controller.execute_key_string("enter")
        logger.info(f"execute_key_string('enter') Result: {success}")
        time.sleep(0.5)

        logger.info("Testing combination (Shift+A)...")
        success = controller.execute_key_string("shift+a")  # Should type 'A'
        logger.info(f"execute_key_string('shift+a') Result: {success}")
        time.sleep(0.5)

        # Use platform specific modifier name for clarity in test
        modifier_key = "cmd" if sys.platform == "darwin" else "win"
        logger.info(
            f"Testing platform modifier ({modifier_key})... (Will open Spotlight/Start)"
        )
        success = controller.execute_key_string(modifier_key)
        logger.info(f"execute_key_string('{modifier_key}') Result: {success}")
        time.sleep(1)  # Give time to see the effect

        logger.info(
            f"Testing combo ({modifier_key}+Space)... (Will open Spotlight/Input Switcher)"
        )
        success = controller.execute_key_string(f"{modifier_key}+space")
        logger.info(f"execute_key_string('{modifier_key}+space') Result: {success}")
        time.sleep(1)  # Give time to see the effect

        # Test a key known to be missing on Mac (if running on Mac)
        if sys.platform == "darwin":
            logger.info("Testing known missing key ('insert')...")
            success = controller.execute_key_string("insert")
            logger.info(
                f"execute_key_string('insert') Result: {success} (Expected False on Mac)"
            )
            time.sleep(0.5)

        logger.info("Testing truly invalid key name...")
        success = controller.execute_key_string("completely_invalid_key_xyz")
        logger.info(
            f"execute_key_string('completely_invalid_key_xyz') Result: {success} (Expected False)"
        )
        time.sleep(0.5)

        logger.info("Testing invalid combo (multiple primary keys)...")
        success = controller.execute_key_string("ctrl+a+b")
        logger.info(
            f"execute_key_string('ctrl+a+b') Result: {success} (Expected False)"
        )
        time.sleep(0.5)

        logger.info("Testing only modifiers...")
        success = controller.execute_key_string("ctrl+shift")
        logger.info(f"execute_key_string('ctrl+shift') Result: {success}")
        time.sleep(0.5)

        print("\n--- Testing Mouse (Move/Click/Scroll) in 3s ---")
        print(">>> Move mouse away from corners <<<")
        time.sleep(3)
        logger.info("Moving mouse to (100, 150)...")
        success = controller.move(100, 150)
        logger.info(f"move Result: {success}")
        time.sleep(0.5)

        logger.info("Single clicking at (100, 150)...")
        success = controller.click(100, 150)
        logger.info(f"click Result: {success}")
        time.sleep(0.5)

        logger.info("Scrolling down...")
        success = controller.scroll(0, -3)  # Scroll down 3 'units'
        logger.info(f"scroll Result: {success}")
        time.sleep(1)

        logger.info("Scrolling up...")
        success = controller.scroll(0, 3)  # Scroll up 3 'units'
        logger.info(f"scroll Result: {success}")
        time.sleep(0.5)

        logger.info("Input controller testing finished.")

    except ImportError:
        logger.error("pynput is required to run these tests.")
    except Exception:
        logger.exception("An error occurred during testing.")
