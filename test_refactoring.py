"""Test script to verify refactoring progress.

This script tests that the new modules can be imported correctly
and that basic functionality works as expected.
"""

import sys
import traceback
from pathlib import Path

def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core module imports...")

    try:
        from src.core.constants import APP_NAME, DEFAULT_FPS
        print(f"✓ Constants imported successfully. APP_NAME: {APP_NAME}")

        from src.core.exceptions import GSRRGBTError, DeviceError
        print("✓ Exceptions imported successfully")

        from src.core.config import Config, get_config
        print("✓ Config imported successfully")

        # Test config creation
        config = Config()
        print(f"✓ Config created successfully. FPS: {config.fps}")

        return True

    except Exception as e:
        print(f"✗ Core import failed: {e}")
        traceback.print_exc()
        return False

def test_data_capture_imports():
    """Test that data capture modules can be imported."""
    print("\nTesting data capture module imports...")

    try:
        from src.data.capture.base import BaseCapture, BaseCaptureThread, CaptureState
        print("✓ Base capture classes imported successfully")

        # Test enum
        state = CaptureState.IDLE
        print(f"✓ CaptureState enum works: {state.name}")

        return True

    except Exception as e:
        print(f"✗ Data capture import failed: {e}")
        traceback.print_exc()
        return False

def test_config_functionality():
    """Test configuration functionality."""
    print("\nTesting configuration functionality...")

    try:
        from src.core.config import Config
        from src.core.exceptions import ValidationError

        # Test valid config
        config = Config(fps=60, frame_width=1280)
        print(f"✓ Valid config created: FPS={config.fps}, Width={config.frame_width}")

        # Test validation
        try:
            invalid_config = Config(fps=999)  # Should fail validation
            print("✗ Validation should have failed for invalid FPS")
            return False
        except ValidationError:
            print("✓ Validation correctly rejected invalid FPS")

        # Test config update
        config.update(fps=25)
        print(f"✓ Config update works: new FPS={config.fps}")

        return True

    except Exception as e:
        print(f"✗ Config functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_exception_hierarchy():
    """Test custom exception hierarchy."""
    print("\nTesting exception hierarchy...")

    try:
        from src.core.exceptions import GSRRGBTError, DeviceError, CaptureError

        # Test exception creation
        base_error = GSRRGBTError("Base error", "ERR001")
        device_error = DeviceError("Device error", "DEV001")

        print(f"✓ Base error: {base_error}")
        print(f"✓ Device error: {device_error}")

        # Test inheritance
        assert isinstance(device_error, GSRRGBTError)
        print("✓ Exception inheritance works correctly")

        return True

    except Exception as e:
        print(f"✗ Exception hierarchy test failed: {e}")
        traceback.print_exc()
        return False

def test_gui_application():
    """Test GUI application module."""
    print("\nTesting GUI application module...")

    try:
        from src.gui.application import GSRRGBTApplication, ApplicationState, create_application

        # Test enum
        state = ApplicationState.IDLE
        print(f"✓ ApplicationState enum works: {state.name}")

        # Test application creation without PyQt5 (should handle gracefully)
        try:
            app = GSRRGBTApplication()
            print("✓ Application created (PyQt5 available)")
        except RuntimeError as e:
            if "PyQt5 is required" in str(e):
                print("✓ Application correctly requires PyQt5")
            else:
                raise

        # Test factory function
        try:
            app = create_application()
            print("✓ Application factory function works")
        except Exception as e:
            if "PyQt5 is required" in str(e):
                print("✓ Application factory correctly requires PyQt5")
            else:
                raise

        return True

    except Exception as e:
        print(f"✗ GUI application test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GSR-RGBT Project Refactoring Test")
    print("=" * 60)

    tests = [
        test_core_imports,
        test_data_capture_imports,
        test_config_functionality,
        test_exception_hierarchy,
        test_gui_application,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Refactoring is on track.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before continuing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
