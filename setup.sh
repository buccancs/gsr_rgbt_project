#!/bin/bash

# gsr_rgbt_tools.sh - Unified tool for the GSR-RGBT Project
# This script provides a unified interface for various tasks:
# - setup: Setting up the environment and dependencies
# - run: Running the full pipeline or specific components
# - test: Running system validation and synchronization tests
# - clean: Cleaning up temporary files and build artifacts
#
# Usage: ./gsr_rgbt_tools.sh [command] [options]
#   Commands:
#     setup       - Creates a virtual environment and installs dependencies
#     run         - Runs the full pipeline or specific components
#     test        - Runs system validation and synchronization tests
#     clean       - Removes temporary files and build artifacts
#     help        - Displays this help message
#
#   Options:
#     --no-sdk    - Skip FLIR Spinnaker SDK installation check
#     --force     - Force reinstallation of dependencies
#     --verbose   - Display detailed output
#     --component=COMP - Run specific component (with 'run' command)
#                      Valid components: app, pipeline, mock_data

# Don't exit on error - we want to handle errors gracefully
set +e

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Default options
SKIP_SDK=false
FORCE=false
VERBOSE=false
COMPONENT=""

# Parse command line arguments
COMMAND=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-sdk)
            SKIP_SDK=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --component=*)
            COMPONENT="${1#*=}"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BOLD}${BLUE}=========================================${RESET}"
echo -e "${BOLD}${BLUE}  GSR-RGBT Project Tools               ${RESET}"
echo -e "${BOLD}${BLUE}=========================================${RESET}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if virtual environment exists and is activated
check_venv() {
    if [ -d ".venv" ]; then
        echo -e "${GREEN}✓${RESET} Virtual environment exists"

        # Check if venv is activated
        if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
            echo -e "${GREEN}✓${RESET} Virtual environment is activated"
            return 0
        else
            echo -e "${YELLOW}!${RESET} Virtual environment exists but is not activated"
            echo -e "   Activating virtual environment..."
            source .venv/bin/activate
            if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
                echo -e "${GREEN}✓${RESET} Virtual environment activated"
                return 0
            else
                echo -e "${RED}✗${RESET} Failed to activate virtual environment"
                return 1
            fi
        fi
    else
        echo -e "${YELLOW}!${RESET} Virtual environment does not exist"
        return 1
    fi
}

# Function to run a make target and check for success
run_make_target() {
    local target=$1
    local description=$2

    echo -e "\n${BOLD}${BLUE}Running: ${description}${RESET}"
    make $target

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${RESET} ${description} completed successfully"
        return 0
    else
        echo -e "${RED}✗${RESET} ${description} failed"
        return 1
    fi
}

# Function to check for required system dependencies
check_dependencies() {
    echo -e "${BOLD}Checking for required system dependencies...${RESET}"

    # Check for Python
    if command_exists python3; then
        PYTHON="python3"
        echo -e "  ${GREEN}✓${RESET} Python 3 is installed"
    elif command_exists python; then
        PYTHON="python"
        # Check Python version
        PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$PYTHON_VERSION" -lt 3 ]; then
            echo -e "  ${RED}✗${RESET} Python 3 is required, but Python $PYTHON_VERSION is installed"
            echo -e "  ${YELLOW}Please install Python 3 and try again${RESET}"
            return 1
        else
            echo -e "  ${GREEN}✓${RESET} Python 3 is installed"
        fi
    else
        echo -e "  ${RED}✗${RESET} Python 3 is not installed"
        echo -e "  ${YELLOW}Please install Python 3 and try again${RESET}"
        return 1
    fi

    # Check for make
    if ! command_exists make; then
        echo -e "${RED}✗${RESET} make command not found. Please install make and try again."
        return 1
    else
        echo -e "  ${GREEN}✓${RESET} make is installed"
    fi

    return 0
}

# Function to check for pip and venv
check_pip_and_venv() {
    # Check for pip
    if command_exists pip3; then
        PIP="pip3"
        echo -e "  ${GREEN}✓${RESET} pip is installed"
    elif command_exists pip; then
        PIP="pip"
        echo -e "  ${GREEN}✓${RESET} pip is installed"
    else
        echo -e "  ${RED}✗${RESET} pip is not installed"
        echo -e "  ${YELLOW}Please install pip and try again${RESET}"
        return 1
    fi

    # Check for venv module
    if $PYTHON -c "import venv" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} venv module is available"
    else
        echo -e "  ${RED}✗${RESET} venv module is not available"
        echo -e "  ${YELLOW}Please install the Python venv module and try again${RESET}"
        return 1
    fi

    # Check for Numba
    if $PYTHON -c "import numba" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} Numba is installed"
    else
        echo -e "  ${YELLOW}!${RESET} Numba is not installed. Performance optimizations will not be available."
        echo -e "  ${YELLOW}  It will be installed with the requirements.txt file.${RESET}"
    fi

    return 0
}

# Function to check for FLIR Spinnaker SDK
check_flir_sdk() {
    echo -e "\n${BOLD}Checking for FLIR Spinnaker SDK...${RESET}"

    if $PYTHON -c "import PySpin" 2>/dev/null; then
        # Get the SDK version
        SDK_VERSION=$($PYTHON -c "import PySpin; system = PySpin.System.GetInstance(); version = system.GetLibraryVersion(); print(f'{version.major}.{version.minor}.{version.type}.{version.build}'); system.ReleaseInstance()" 2>/dev/null)
        echo -e "  ${GREEN}✓${RESET} FLIR Spinnaker SDK is installed (version: $SDK_VERSION)"
        return 0
    else
        echo -e "  ${RED}✗${RESET} FLIR Spinnaker SDK is not installed"
        echo -e "  ${YELLOW}The FLIR Spinnaker SDK is required for thermal camera support.${RESET}"
        echo -e "  ${YELLOW}Please download and install it from:${RESET}"
        echo -e "  ${YELLOW}https://www.flir.com/products/spinnaker-sdk/${RESET}"
        echo -e "  ${YELLOW}Make sure to select the option to install the Python bindings (PySpin).${RESET}"

        # Ask if the user wants to continue without the SDK
        echo -e "\n${YELLOW}Do you want to continue without the FLIR Spinnaker SDK? (y/n)${RESET}"
        read -p "" -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Exiting...${RESET}"
            return 1
        fi
        echo -e "${YELLOW}Continuing without the FLIR Spinnaker SDK...${RESET}"
        echo -e "${YELLOW}You can still use the application in simulation mode.${RESET}"
        echo -e "${YELLOW}Set THERMAL_SIMULATION_MODE = True in src/config.py${RESET}"
        return 0
    fi
}

# Function to setup the environment
setup_environment() {
    echo -e "${BOLD}${BLUE}Setting up the environment...${RESET}"

    # Check dependencies
    check_dependencies || return 1
    check_pip_and_venv || return 1

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo -e "\n${BOLD}Creating Python virtual environment...${RESET}"
        $PYTHON -m venv .venv
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗${RESET} Failed to create virtual environment"
            return 1
        fi
        echo -e "${GREEN}✓${RESET} Virtual environment created"
    fi

    # Activate virtual environment
    echo -e "\n${BOLD}Activating virtual environment...${RESET}"
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${RESET} Failed to activate virtual environment"
        return 1
    fi
    echo -e "${GREEN}✓${RESET} Virtual environment activated"

    # Install dependencies
    echo -e "\n${BOLD}Installing dependencies...${RESET}"
    $PIP install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${RESET} Failed to install dependencies"
        return 1
    fi
    echo -e "${GREEN}✓${RESET} Dependencies installed"

    # Check for Numba
    echo -e "\n${BOLD}Checking for Numba...${RESET}"
    if $PYTHON -c "import numba" 2>/dev/null; then
        echo -e "${GREEN}✓${RESET} Numba is installed"
    else
        echo -e "${YELLOW}!${RESET} Numba is not installed. It will be installed with the requirements.txt file."
    fi

    # Check for FLIR Spinnaker SDK
    if [ "$SKIP_SDK" = false ]; then
        check_flir_sdk
    fi

    echo -e "\n${GREEN}✓${RESET} Environment setup completed successfully"
    return 0
}

# Function to run the application or pipeline
run_application() {
    echo -e "${BOLD}${BLUE}Running the application...${RESET}"

    # Check if virtual environment is activated
    check_venv || {
        echo -e "${YELLOW}!${RESET} Setting up environment first..."
        setup_environment || return 1
    }

    case "$COMPONENT" in
        "app")
            echo -e "\n${BOLD}Running the data collection application...${RESET}"
            $PYTHON src/main.py
            ;;
        "pipeline")
            echo -e "\n${BOLD}Running the ML pipeline...${RESET}"
            run_make_target "pipeline" "ML Pipeline"
            ;;
        "mock_data")
            echo -e "\n${BOLD}Generating mock data...${RESET}"
            run_make_target "mock_data" "Mock Data Generation"
            ;;
        "")
            # Run everything
            echo -e "\n${BOLD}Running system validation checks...${RESET}"
            run_make_target "test" "System Validation Checks" || {
                echo -e "${YELLOW}!${RESET} System validation checks failed, but continuing..."
            }

            echo -e "\n${BOLD}Running synchronization tests...${RESET}"
            run_make_target "test_sync" "Synchronization Tests" || {
                echo -e "${YELLOW}!${RESET} Synchronization tests failed, but continuing..."
            }

            echo -e "\n${BOLD}Checking for existing data...${RESET}"
            if [ ! -d "data/recordings" ] || [ -z "$(ls -A data/recordings 2>/dev/null)" ]; then
                echo -e "No data found, generating mock data..."
                run_make_target "mock_data" "Generate Mock Data" || {
                    echo -e "${RED}✗${RESET} Failed to generate mock data, cannot continue with ML pipeline"
                    return 1
                }
            else
                echo -e "${GREEN}✓${RESET} Data already exists, skipping mock data generation"
            fi

            echo -e "\n${BOLD}Running the full ML pipeline...${RESET}"
            run_make_target "pipeline" "Full ML Pipeline" || {
                echo -e "${YELLOW}!${RESET} ML pipeline failed or was incomplete"
            }

            # Ask if the user wants to run the data collection application
            echo -e "\n${BOLD}${BLUE}Optional: Run the data collection application?${RESET}"
            read -p "Do you want to run the data collection application? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                run_make_target "run_app" "Data Collection Application"
            fi
            ;;
        *)
            echo -e "${RED}✗${RESET} Unknown component: $COMPONENT"
            echo -e "${YELLOW}Valid components: app, pipeline, mock_data${RESET}"
            return 1
            ;;
    esac

    return 0
}

# Function to run tests
run_tests() {
    echo -e "${BOLD}${BLUE}Running tests...${RESET}"

    # Check if virtual environment is activated
    check_venv || {
        echo -e "${YELLOW}!${RESET} Setting up environment first..."
        setup_environment || return 1
    }

    # Run system validation checks
    echo -e "\n${BOLD}Running system validation checks...${RESET}"
    run_make_target "test" "System Validation Checks" || {
        echo -e "${RED}✗${RESET} System validation checks failed"
        return 1
    }

    # Run synchronization tests
    echo -e "\n${BOLD}Running synchronization tests...${RESET}"
    run_make_target "test_sync" "Synchronization Tests" || {
        echo -e "${RED}✗${RESET} Synchronization tests failed"
        return 1
    }

    echo -e "\n${GREEN}✓${RESET} All tests completed successfully"
    return 0
}

# Function to clean up
clean_up() {
    echo -e "${BOLD}${BLUE}Cleaning up...${RESET}"

    run_make_target "clean" "Clean Up" || {
        echo -e "${RED}✗${RESET} Clean up failed"
        return 1
    }

    echo -e "\n${GREEN}✓${RESET} Clean up completed successfully"
    return 0
}

# Function to display help
show_help() {
    echo -e "${BOLD}${BLUE}GSR-RGBT Project Tools${RESET}"
    echo -e "Usage: ./gsr_rgbt_tools.sh [command] [options]"
    echo -e ""
    echo -e "Commands:"
    echo -e "  setup       - Creates a virtual environment and installs dependencies"
    echo -e "  run         - Runs the full pipeline or specific components"
    echo -e "  test        - Runs system validation and synchronization tests"
    echo -e "  clean       - Removes temporary files and build artifacts"
    echo -e "  help        - Displays this help message"
    echo -e ""
    echo -e "Options:"
    echo -e "  --no-sdk    - Skip FLIR Spinnaker SDK installation check"
    echo -e "  --force     - Force reinstallation of dependencies"
    echo -e "  --verbose   - Display detailed output"
    echo -e "  --component=COMP - Run specific component (with 'run' command)"
    echo -e "                   Valid components: app, pipeline, mock_data"
    echo -e ""
    echo -e "Examples:"
    echo -e "  ./gsr_rgbt_tools.sh setup"
    echo -e "  ./gsr_rgbt_tools.sh run --component=app"
    echo -e "  ./gsr_rgbt_tools.sh test"
    echo -e "  ./gsr_rgbt_tools.sh clean"
    echo -e ""

    return 0
}

# Function to print final status and usage information
print_final_status() {
    echo -e "\nTo activate the virtual environment in the future, run:"
    echo -e "  ${BOLD}source .venv/bin/activate${RESET} (Linux/macOS)"
    echo -e "  ${BOLD}.venv\\Scripts\\activate${RESET} (Windows)"
    echo ""
    echo -e "To run the application:"
    echo -e "  ${BOLD}./gsr_rgbt_tools.sh run --component=app${RESET}"
    echo -e "  or ${BOLD}python src/main.py${RESET}"
    echo -e "  or ${BOLD}make run_app${RESET} (if make is available)"
    echo ""
    echo -e "For more detailed instructions, see the guide.md file."
    echo ""

    return 0
}

echo ""

# Create and activate virtual environment
echo -e "${BOLD}Setting up Python virtual environment...${RESET}"
if [ -d ".venv" ]; then
    echo -e "  ${YELLOW}!${RESET} Virtual environment already exists"
    read -p "  Do you want to recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "  Removing existing virtual environment..."
        rm -rf .venv
        $PYTHON -m venv .venv
        echo -e "  ${GREEN}✓${RESET} Virtual environment created"
    else
        echo -e "  Using existing virtual environment"
    fi
else
    $PYTHON -m venv .venv
    echo -e "  ${GREEN}✓${RESET} Virtual environment created"
fi

# Activate virtual environment
echo -e "  Activating virtual environment..."
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    echo -e "  ${YELLOW}!${RESET} Unsupported operating system: $OSTYPE"
    echo -e "  ${YELLOW}  Please activate the virtual environment manually:${RESET}"
    echo -e "  ${YELLOW}  - On Linux/macOS: source .venv/bin/activate${RESET}"
    echo -e "  ${YELLOW}  - On Windows: .venv\\Scripts\\activate${RESET}"
    exit 1
fi

# Upgrade pip
echo -e "  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo -e "${BOLD}Installing required Python packages...${RESET}"
pip install -r requirements.txt
echo -e "  ${GREEN}✓${RESET} Required packages installed"

# Check for Numba
echo -e "${BOLD}Checking for Numba...${RESET}"
if $PYTHON -c "import numba" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} Numba is installed"
else
    echo -e "  ${YELLOW}!${RESET} Numba is not installed. It will be installed with the requirements.txt file."
    echo -e "  ${YELLOW}  Continuing with setup...${RESET}"
}

# Check for FLIR Spinnaker SDK
echo -e "\n${BOLD}Checking for FLIR Spinnaker SDK...${RESET}"
if $PYTHON -c "import PySpin" 2>/dev/null; then
    # Get SDK version
    SDK_VERSION=$($PYTHON -c "import PySpin; system = PySpin.System.GetInstance(); version = system.GetLibraryVersion(); print(f'{version.major}.{version.minor}.{version.type}.{version.build}'); system.ReleaseInstance()" 2>/dev/null)
    echo -e "  ${GREEN}✓${RESET} FLIR Spinnaker SDK is installed (Version: ${SDK_VERSION})"
else
    echo -e "  ${RED}✗${RESET} FLIR Spinnaker SDK is not installed"
    echo -e "  ${YELLOW}  The FLIR Spinnaker SDK is required for thermal camera operation.${RESET}"

    # Determine OS and provide appropriate instructions
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo -e "  ${BOLD}Installing FLIR Spinnaker SDK for macOS...${RESET}"
        echo -e "  ${YELLOW}  Note: This requires administrative privileges${RESET}"

        # Create a temporary directory for downloads
        TEMP_DIR=$(mktemp -d)
        echo -e "  Created temporary directory: $TEMP_DIR"

        # Download the SDK
        echo -e "  ${YELLOW}  The FLIR Spinnaker SDK requires a user account on the FLIR website.${RESET}"
        echo -e "  ${YELLOW}  Please download the SDK manually from:${RESET}"
        echo -e "  ${YELLOW}  https://www.flir.com/products/spinnaker-sdk/${RESET}"

        read -p "  Have you downloaded the SDK? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "  Please enter the full path to the downloaded .dmg file: " DMG_PATH
            if [ -f "$DMG_PATH" ]; then
                echo -e "  ${GREEN}✓${RESET} Found SDK installer at $DMG_PATH"
                echo -e "  ${YELLOW}  Please follow the installation instructions in the installer.${RESET}"
                echo -e "  ${YELLOW}  Make sure to select the option to install the Python bindings (PySpin).${RESET}"

                # Mount the DMG
                echo -e "  Mounting the DMG file..."
                MOUNT_POINT=$(hdiutil attach "$DMG_PATH" | grep Volumes | cut -f 3)
                echo -e "  Mounted at: $MOUNT_POINT"

                # Find the installer package
                PKG_FILE=$(find "$MOUNT_POINT" -name "*.pkg" | head -n 1)
                if [ -n "$PKG_FILE" ]; then
                    echo -e "  Found installer package: $PKG_FILE"
                    echo -e "  ${YELLOW}  Installing the SDK...${RESET}"
                    sudo installer -pkg "$PKG_FILE" -target /

                    # Unmount the DMG
                    hdiutil detach "$MOUNT_POINT" -force

                    # Check if installation was successful
                    if $PYTHON -c "import PySpin" 2>/dev/null; then
                        echo -e "  ${GREEN}✓${RESET} FLIR Spinnaker SDK installed successfully"
                    else
                        echo -e "  ${RED}✗${RESET} FLIR Spinnaker SDK installation failed"
                        echo -e "  ${YELLOW}  Please install the SDK manually following the instructions in guide.md${RESET}"
                    fi
                else
                    echo -e "  ${RED}✗${RESET} Could not find installer package in the DMG"
                    echo -e "  ${YELLOW}  Please install the SDK manually following the instructions in guide.md${RESET}"
                    hdiutil detach "$MOUNT_POINT" -force
                fi
            else
                echo -e "  ${RED}✗${RESET} File not found: $DMG_PATH"
                echo -e "  ${YELLOW}  Please install the SDK manually following the instructions in guide.md${RESET}"
            fi
        else
            echo -e "  ${YELLOW}  Please install the SDK manually following the instructions in guide.md${RESET}"
        fi

        # Clean up
        rm -rf "$TEMP_DIR"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo -e "  ${YELLOW}  Please install the FLIR Spinnaker SDK for Linux manually.${RESET}"
        echo -e "  ${YELLOW}  Download from: https://www.flir.com/products/spinnaker-sdk/${RESET}"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo -e "  ${YELLOW}  Please install the FLIR Spinnaker SDK for Windows manually.${RESET}"
        echo -e "  ${YELLOW}  Download from: https://www.flir.com/products/spinnaker-sdk/${RESET}"
    else
        echo -e "  ${YELLOW}  Please install the FLIR Spinnaker SDK manually for your platform.${RESET}"
        echo -e "  ${YELLOW}  Download from: https://www.flir.com/products/spinnaker-sdk/${RESET}"
    fi

    # Set a flag to indicate that the SDK is not installed
    SDK_INSTALLED=false
fi

# Check for pyshimmer
echo -e "\n${BOLD}Checking for pyshimmer...${RESET}"
if $PYTHON -c "import pyshimmer" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} pyshimmer is installed"
else
    echo -e "  ${RED}✗${RESET} pyshimmer is not installed"
    echo -e "  ${YELLOW}  Installing pyshimmer...${RESET}"
    pip install pyshimmer
    if $PYTHON -c "import pyshimmer" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} pyshimmer installed successfully"
    else
        echo -e "  ${RED}✗${RESET} pyshimmer installation failed"
        echo -e "  ${YELLOW}  Please install pyshimmer manually: pip install pyshimmer${RESET}"
    fi
fi

# Run system validation checks
echo -e "\n${BOLD}Running system validation checks...${RESET}"
echo -e "  This will check for available devices, ports, and dependencies..."
echo -e "  ${BLUE}Note: This check will detect available serial ports and FLIR cameras${RESET}"
echo -e "  ${BLUE}      and verify if the configured ports and IPs are correct.${RESET}"
echo ""

# Run the check_system.py script
$PYTHON src/scripts/check_system.py || {
    echo -e ""
    echo -e "  ${YELLOW}!${RESET} System validation checks failed"
    echo -e "  ${YELLOW}  This may be because the hardware devices are not connected.${RESET}"
    echo -e "  ${YELLOW}  You can still use the application in simulation mode.${RESET}"
    echo -e "  ${YELLOW}  To use simulation mode, set the following in src/config.py:${RESET}"
    echo -e "  ${YELLOW}    - THERMAL_SIMULATION_MODE = True${RESET}"
    echo -e "  ${YELLOW}    - GSR_SIMULATION_MODE = True${RESET}"
}

# Print hardware setup instructions
echo ""
echo -e "${BOLD}${BLUE}=========================================${RESET}"
echo -e "${BOLD}${BLUE}  Hardware Setup Instructions           ${RESET}"
echo -e "${BOLD}${BLUE}=========================================${RESET}"
echo ""
echo -e "${BOLD}FLIR A65 Thermal Camera:${RESET}"
echo -e "  1. Download and install the FLIR Spinnaker SDK from the FLIR website"
echo -e "     Make sure to select the option to install the Python bindings (PySpin)"
echo -e "  2. Connect the camera via USB"
echo -e "  3. Update the THERMAL_CAMERA_ID in src/config.py if needed"
echo -e "  4. Set THERMAL_SIMULATION_MODE = False in src/config.py"
echo ""
echo -e "${BOLD}Shimmer3 GSR+ Sensor:${RESET}"
echo -e "  1. Install Shimmer Connect or ConsensysPRO from the Shimmer website"
echo -e "  2. Use this software to update firmware and configure the sensor"
echo -e "  3. Connect the sensor via USB dock or pair via Bluetooth"
echo -e "  4. Identify the serial port (e.g., /dev/tty.usbmodem* on macOS)"
echo -e "  5. Update the GSR_SENSOR_PORT in src/config.py"
echo -e "  6. Set GSR_SIMULATION_MODE = False in src/config.py"
echo ""

# Check if all required components are installed
echo -e "\n${BOLD}Checking if all required components are installed...${RESET}"
ALL_COMPONENTS_INSTALLED=true

# Check for PySpin
if ! $PYTHON -c "import PySpin" 2>/dev/null; then
    echo -e "  ${RED}✗${RESET} FLIR Spinnaker SDK (PySpin) is not installed"
    ALL_COMPONENTS_INSTALLED=false
else
    echo -e "  ${GREEN}✓${RESET} FLIR Spinnaker SDK (PySpin) is installed"
fi

# Check for pyshimmer
if ! $PYTHON -c "import pyshimmer" 2>/dev/null; then
    echo -e "  ${RED}✗${RESET} pyshimmer is not installed"
    ALL_COMPONENTS_INSTALLED=false
else
    echo -e "  ${GREEN}✓${RESET} pyshimmer is installed"
fi

# Check for other critical dependencies
for package in numpy pandas cv2 PyQt5 tensorflow sklearn neurokit2; do
    if ! $PYTHON -c "import $package" 2>/dev/null; then
        if [ "$package" = "cv2" ]; then
            # Special case for OpenCV
            if ! $PYTHON -c "import cv2" 2>/dev/null; then
                echo -e "  ${RED}✗${RESET} $package is not installed"
                ALL_COMPONENTS_INSTALLED=false
            else
                echo -e "  ${GREEN}✓${RESET} $package is installed"
            fi
        elif [ "$package" = "sklearn" ]; then
            # Special case for scikit-learn
            if ! $PYTHON -c "import sklearn" 2>/dev/null; then
                echo -e "  ${RED}✗${RESET} $package is not installed"
                ALL_COMPONENTS_INSTALLED=false
            else
                echo -e "  ${GREEN}✓${RESET} $package is installed"
            fi
        elif [ "$package" = "neurokit2" ]; then
            # Special case for neurokit2
            if ! $PYTHON -c "try: import neurokit2; print('neurokit2 imported successfully') except ImportError as e: print(f'Import error: {e}')" 2>/dev/null | grep -q "imported successfully"; then
                echo -e "  ${RED}✗${RESET} $package is not installed"
                ALL_COMPONENTS_INSTALLED=false
            else
                echo -e "  ${GREEN}✓${RESET} $package is installed"
            fi
        else
            echo -e "  ${RED}✗${RESET} $package is not installed"
            ALL_COMPONENTS_INSTALLED=false
        fi
    else
        echo -e "  ${GREEN}✓${RESET} $package is installed"
    fi
done

# Print final status message
if [ "$ALL_COMPONENTS_INSTALLED" = true ]; then
    echo -e "\n${BOLD}${GREEN}Setup complete! All required components are installed.${RESET}"
else
    echo -e "\n${BOLD}${YELLOW}Setup incomplete! Some required components are missing.${RESET}"
    echo -e "${YELLOW}Please install the missing components before running the application.${RESET}"
    echo -e "${YELLOW}See the guide.md file for detailed instructions.${RESET}"
fi

# Main command handling
case "$COMMAND" in
    "setup")
        print_final_status
        ;;
    "run")
        run_application
        ;;
    "test")
        run_tests
        ;;
    "clean")
        clean_up
        ;;
    "help" | "")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${RESET}"
        echo -e "Run './gsr_rgbt_tools.sh help' for usage information."
        exit 1
        ;;
esac

exit 0
