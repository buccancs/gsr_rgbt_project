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

# Handle help with topic
if [ "$COMMAND" = "help" ] && [ $# -gt 0 ]; then
    HELP_TOPIC=$1
    shift
fi

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
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --model=*)
            MODEL_TYPE="${1#*=}"
            shift
            ;;
        --data=*)
            DATA_DIR="${1#*=}"
            shift
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --simulate)
            SIMULATE=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --cv-folds=*)
            CV_FOLDS="${1#*=}"
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

    # Check for C compiler (needed for Cython)
    if command_exists gcc; then
        echo -e "  ${GREEN}✓${RESET} gcc is installed"
    elif command_exists clang; then
        echo -e "  ${GREEN}✓${RESET} clang is installed"
    else
        echo -e "  ${YELLOW}!${RESET} No C compiler detected. Cython extensions may fail to build."
        echo -e "  ${YELLOW}  Please install gcc or clang if you encounter build errors.${RESET}"
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

    # Build Cython extensions
    echo -e "\n${BOLD}Building Cython extensions...${RESET}"
    $PYTHON setup.py build_ext --inplace
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗${RESET} Failed to build Cython extensions"
        return 1
    fi
    echo -e "${GREEN}✓${RESET} Cython extensions built"

    # Check for FLIR Spinnaker SDK
    if [ "$SKIP_SDK" = false ]; then
        check_flir_sdk
    fi

    # Check for pyshimmer
    echo -e "\n${BOLD}Checking for pyshimmer...${RESET}"
    if $PYTHON -c "import pyshimmer" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} pyshimmer is installed"
    else
        echo -e "  ${RED}✗${RESET} pyshimmer is not installed"
        echo -e "  ${YELLOW}Installing pyshimmer...${RESET}"
        $PIP install pyshimmer
        if ! $PYTHON -c "import pyshimmer" 2>/dev/null; then
            echo -e "  ${RED}✗${RESET} pyshimmer installation failed"
            echo -e "  ${YELLOW}  Please install pyshimmer manually: pip install pyshimmer${RESET}"
        fi
    fi

    # Run system validation checks
    echo -e "\n${BOLD}Running system validation checks...${RESET}"
    echo -e "  This will check for available devices, ports, and dependencies..."
    $PYTHON src/scripts/check_system.py || {
        echo -e ""
        echo -e "  ${YELLOW}!${RESET} System validation checks failed"
        echo -e "  ${YELLOW}  This may be because the hardware devices are not connected.${RESET}"
        echo -e "  ${YELLOW}  You can still use the application in simulation mode.${RESET}"
        echo -e "  ${YELLOW}  To use simulation mode, set the following in src/config.py:${RESET}"
        echo -e "  ${YELLOW}    - THERMAL_SIMULATION_MODE = True${RESET}"
        echo -e "  ${YELLOW}    - GSR_SIMULATION_MODE = True${RESET}"
    }

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
    local help_topic=$1

    case "$help_topic" in
        "setup")
            show_setup_help
            ;;
        "run")
            show_run_help
            ;;
        "test")
            show_test_help
            ;;
        "clean")
            show_clean_help
            ;;
        "collect")
            show_collect_help
            ;;
        "train")
            show_train_help
            ;;
        "evaluate")
            show_evaluate_help
            ;;
        "troubleshoot")
            show_troubleshoot_help
            ;;
        *)
            show_general_help
            ;;
    esac

    return 0
}

# Function to display general help
show_general_help() {
    echo -e "${BOLD}${BLUE}GSR-RGBT Project Tools${RESET}"
    echo -e "A unified interface for the GSR-RGBT contactless physiological monitoring project."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh [command] [options]"
    echo -e "  ./gsr_rgbt_tools.sh help [command]  # Get detailed help for a specific command"
    echo -e ""
    echo -e "${BOLD}COMMANDS:${RESET}"
    echo -e "  ${GREEN}setup${RESET}       - Set up the development environment and dependencies"
    echo -e "  ${GREEN}collect${RESET}     - Start the data collection GUI application"
    echo -e "  ${GREEN}train${RESET}       - Train machine learning models on collected data"
    echo -e "  ${GREEN}evaluate${RESET}    - Evaluate trained models and generate reports"
    echo -e "  ${GREEN}test${RESET}        - Run system validation and synchronization tests"
    echo -e "  ${GREEN}clean${RESET}       - Remove temporary files and build artifacts"
    echo -e "  ${GREEN}help${RESET}        - Display help information"
    echo -e ""
    echo -e "${BOLD}GLOBAL OPTIONS:${RESET}"
    echo -e "  --verbose   - Display detailed output during operations"
    echo -e "  --force     - Force operations (e.g., reinstall dependencies)"
    echo -e "  --no-sdk    - Skip FLIR Spinnaker SDK installation check"
    echo -e ""
    echo -e "${BOLD}QUICK START:${RESET}"
    echo -e "  1. ${BOLD}./gsr_rgbt_tools.sh setup${RESET}                    # Set up environment"
    echo -e "  2. ${BOLD}./gsr_rgbt_tools.sh collect${RESET}                  # Collect data"
    echo -e "  3. ${BOLD}./gsr_rgbt_tools.sh train${RESET}                    # Train models"
    echo -e ""
    echo -e "${BOLD}GET DETAILED HELP:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh help setup        # Setup help"
    echo -e "  ./gsr_rgbt_tools.sh help collect      # Data collection help"
    echo -e "  ./gsr_rgbt_tools.sh help train        # Training help"
    echo -e "  ./gsr_rgbt_tools.sh help troubleshoot # Troubleshooting guide"
    echo -e ""
    echo -e "${BOLD}DOCUMENTATION:${RESET}"
    echo -e "  README.md                           - Project overview and quick start"
    echo -e "  docs/user/USER_GUIDE.md            - Complete user guide"
    echo -e "  docs/developer/DEVELOPER_GUIDE.md  - Developer documentation"
    echo -e "  docs/technical/ARCHITECTURE.md     - System architecture"
    echo -e ""
}

# Function to display setup help
show_setup_help() {
    echo -e "${BOLD}${BLUE}Setup Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Sets up the complete development environment for the GSR-RGBT project."
    echo -e "  This includes creating a Python virtual environment, installing dependencies,"
    echo -e "  building Cython extensions, and validating hardware setup."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh setup [options]"
    echo -e ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo -e "  --force     - Force reinstallation of all dependencies"
    echo -e "  --no-sdk    - Skip FLIR Spinnaker SDK installation check"
    echo -e "  --verbose   - Show detailed installation progress"
    echo -e ""
    echo -e "${BOLD}WHAT IT DOES:${RESET}"
    echo -e "  ✓ Creates Python virtual environment (.venv)"
    echo -e "  ✓ Installs Python packages from requirements.txt"
    echo -e "  ✓ Builds Cython extensions for performance"
    echo -e "  ✓ Validates system dependencies"
    echo -e "  ✓ Checks hardware connectivity (cameras, sensors)"
    echo -e "  ✓ Provides guidance for FLIR Spinnaker SDK installation"
    echo -e ""
    echo -e "${BOLD}PREREQUISITES:${RESET}"
    echo -e "  • Python 3.9 or higher"
    echo -e "  • Git"
    echo -e "  • FLIR Spinnaker SDK (for thermal camera support)"
    echo -e ""
    echo -e "${BOLD}EXAMPLES:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh setup                    # Standard setup"
    echo -e "  ./gsr_rgbt_tools.sh setup --force            # Force reinstall"
    echo -e "  ./gsr_rgbt_tools.sh setup --no-sdk --verbose # Skip SDK, verbose output"
    echo -e ""
}

# Function to display collect help
show_collect_help() {
    echo -e "${BOLD}${BLUE}Data Collection Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Launches the data collection GUI application for synchronized capture"
    echo -e "  of RGB video, thermal video, and GSR sensor data."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh collect [options]"
    echo -e ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo -e "  --simulate  - Run in simulation mode (no hardware required)"
    echo -e "  --config=FILE - Use custom configuration file"
    echo -e ""
    echo -e "${BOLD}HARDWARE REQUIREMENTS:${RESET}"
    echo -e "  • RGB Camera (USB/built-in webcam)"
    echo -e "  • FLIR Thermal Camera (requires Spinnaker SDK)"
    echo -e "  • Shimmer3 GSR+ Sensor (Bluetooth or USB dock)"
    echo -e ""
    echo -e "${BOLD}BEFORE YOU START:${RESET}"
    echo -e "  1. Ensure all hardware is connected and powered on"
    echo -e "  2. Pair Shimmer sensor via Bluetooth (if using wireless)"
    echo -e "  3. Run './gsr_rgbt_tools.sh test' to validate setup"
    echo -e ""
    echo -e "${BOLD}DATA COLLECTION WORKFLOW:${RESET}"
    echo -e "  1. Launch application: ./gsr_rgbt_tools.sh collect"
    echo -e "  2. Enter subject ID and session parameters"
    echo -e "  3. Configure camera settings (resolution, frame rate)"
    echo -e "  4. Start recording and follow your experimental protocol"
    echo -e "  5. Stop recording when complete"
    echo -e ""
    echo -e "${BOLD}OUTPUT:${RESET}"
    echo -e "  Data is saved to: data/recordings/[subject_id]/[timestamp]/"
    echo -e "  • RGB video: rgb_video.avi"
    echo -e "  • Thermal video: thermal_video.avi"
    echo -e "  • GSR data: gsr_data.csv"
    echo -e "  • Timestamps: timestamps.csv"
    echo -e ""
}

# Function to display train help
show_train_help() {
    echo -e "${BOLD}${BLUE}Model Training Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Trains machine learning models on collected physiological data."
    echo -e "  Supports multiple architectures: LSTM, CNN, Transformer, ResNet, VAE."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh train [options]"
    echo -e ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo -e "  --config=FILE     - Use specific pipeline configuration"
    echo -e "  --model=TYPE      - Train specific model type (lstm, cnn, transformer, etc.)"
    echo -e "  --data=DIR        - Use data from specific directory"
    echo -e "  --output=DIR      - Save models to specific directory"
    echo -e "  --cv-folds=N      - Use N-fold cross-validation (default: leave-one-out)"
    echo -e ""
    echo -e "${BOLD}SUPPORTED MODELS:${RESET}"
    echo -e "  • LSTM - Long Short-Term Memory networks"
    echo -e "  • CNN - Convolutional Neural Networks"
    echo -e "  • Transformer - Self-attention based models"
    echo -e "  • ResNet - Residual Networks"
    echo -e "  • VAE - Variational Autoencoders"
    echo -e ""
    echo -e "${BOLD}EXAMPLES:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh train                                    # Default training"
    echo -e "  ./gsr_rgbt_tools.sh train --model=lstm --cv-folds=5         # LSTM with 5-fold CV"
    echo -e "  ./gsr_rgbt_tools.sh train --config=configs/custom.yaml      # Custom config"
    echo -e ""
    echo -e "${BOLD}OUTPUT:${RESET}"
    echo -e "  Models saved to: outputs/models/"
    echo -e "  Training logs: outputs/logs/"
    echo -e "  Evaluation metrics: outputs/metrics/"
    echo -e ""
}

# Function to display run help
show_run_help() {
    echo -e "${BOLD}${BLUE}Run Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Runs the full pipeline or specific components of the GSR-RGBT system."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh run [options]"
    echo -e ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo -e "  --component=COMP - Run specific component"
    echo -e "                   Valid components: app, pipeline, mock_data"
    echo -e ""
    echo -e "${BOLD}COMPONENTS:${RESET}"
    echo -e "  ${GREEN}app${RESET}       - Run the data collection GUI application"
    echo -e "  ${GREEN}pipeline${RESET}  - Run the complete ML pipeline (training + evaluation)"
    echo -e "  ${GREEN}mock_data${RESET} - Generate mock data for testing"
    echo -e ""
    echo -e "${BOLD}EXAMPLES:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh run                    # Run everything (tests + pipeline + optional app)"
    echo -e "  ./gsr_rgbt_tools.sh run --component=app    # Run only the GUI application"
    echo -e "  ./gsr_rgbt_tools.sh run --component=pipeline # Run only the ML pipeline"
    echo -e ""
}

# Function to display test help
show_test_help() {
    echo -e "${BOLD}${BLUE}Test Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Runs comprehensive system validation and synchronization tests."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh test [options]"
    echo -e ""
    echo -e "${BOLD}WHAT IT TESTS:${RESET}"
    echo -e "  ✓ Python environment and dependencies"
    echo -e "  ✓ Hardware connectivity (cameras, sensors)"
    echo -e "  ✓ FLIR Spinnaker SDK installation"
    echo -e "  ✓ Shimmer sensor communication"
    echo -e "  ✓ Data synchronization mechanisms"
    echo -e "  ✓ File I/O and storage systems"
    echo -e ""
    echo -e "${BOLD}OUTPUT:${RESET}"
    echo -e "  Detailed test results with pass/fail status for each component."
    echo -e "  Recommendations for fixing any detected issues."
    echo -e ""
}

# Function to display clean help
show_clean_help() {
    echo -e "${BOLD}${BLUE}Clean Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Removes temporary files, build artifacts, and cached data."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh clean"
    echo -e ""
    echo -e "${BOLD}WHAT IT REMOVES:${RESET}"
    echo -e "  • Python bytecode files (*.pyc, __pycache__)"
    echo -e "  • Cython build artifacts"
    echo -e "  • Temporary log files"
    echo -e "  • Build directories"
    echo -e ""
    echo -e "${BOLD}NOTE:${RESET}"
    echo -e "  This does NOT remove:"
    echo -e "  • Virtual environment (.venv)"
    echo -e "  • Collected data (data/recordings)"
    echo -e "  • Trained models (outputs/models)"
    echo -e ""
}

# Function to display evaluate help
show_evaluate_help() {
    echo -e "${BOLD}${BLUE}Evaluate Command Help${RESET}"
    echo -e ""
    echo -e "${BOLD}DESCRIPTION:${RESET}"
    echo -e "  Evaluates trained models and generates comprehensive reports."
    echo -e ""
    echo -e "${BOLD}USAGE:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh evaluate [options]"
    echo -e ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo -e "  --model=FILE      - Evaluate specific model file"
    echo -e "  --data=DIR        - Use specific test data directory"
    echo -e "  --output=DIR      - Save evaluation results to directory"
    echo -e "  --visualize       - Generate visualization plots"
    echo -e ""
    echo -e "${BOLD}WHAT IT GENERATES:${RESET}"
    echo -e "  • Performance metrics (accuracy, precision, recall, F1)"
    echo -e "  • Confusion matrices"
    echo -e "  • ROC curves and AUC scores"
    echo -e "  • Prediction vs. ground truth plots"
    echo -e "  • Model comparison reports"
    echo -e ""
    echo -e "${BOLD}EXAMPLES:${RESET}"
    echo -e "  ./gsr_rgbt_tools.sh evaluate                           # Evaluate all models"
    echo -e "  ./gsr_rgbt_tools.sh evaluate --model=lstm_model.pth    # Evaluate specific model"
    echo -e "  ./gsr_rgbt_tools.sh evaluate --visualize               # Include visualizations"
    echo -e ""
}

# Function to display troubleshoot help
show_troubleshoot_help() {
    echo -e "${BOLD}${BLUE}Troubleshooting Guide${RESET}"
    echo -e ""
    echo -e "${BOLD}COMMON ISSUES AND SOLUTIONS:${RESET}"
    echo -e ""
    echo -e "${YELLOW}Problem:${RESET} Shimmer device not found"
    echo -e "${GREEN}Solutions:${RESET}"
    echo -e "  • Ensure device is charged and powered on"
    echo -e "  • Check Bluetooth pairing in system settings"
    echo -e "  • On Windows: Check Device Manager for COM port"
    echo -e "  • Try: ./gsr_rgbt_tools.sh test"
    echo -e ""
    echo -e "${YELLOW}Problem:${RESET} FLIR camera not detected"
    echo -e "${GREEN}Solutions:${RESET}"
    echo -e "  • Verify Spinnaker SDK installation with Python bindings"
    echo -e "  • Use USB 3.0 port for optimal performance"
    echo -e "  • Close other applications using the camera"
    echo -e "  • Try: python -c \"import PySpin; print('SDK OK')\""
    echo -e ""
    echo -e "${YELLOW}Problem:${RESET} Python import errors"
    echo -e "${GREEN}Solutions:${RESET}"
    echo -e "  • Activate virtual environment: source .venv/bin/activate"
    echo -e "  • Reinstall dependencies: ./gsr_rgbt_tools.sh setup --force"
    echo -e "  • Rebuild extensions: python setup.py build_ext --inplace"
    echo -e ""
    echo -e "${YELLOW}Problem:${RESET} GUI application crashes"
    echo -e "${GREEN}Solutions:${RESET}"
    echo -e "  • Check console output for error messages"
    echo -e "  • Verify hardware connections"
    echo -e "  • Try simulation mode: ./gsr_rgbt_tools.sh collect --simulate"
    echo -e ""
    echo -e "${YELLOW}Problem:${RESET} Low performance or dropped frames"
    echo -e "${GREEN}Solutions:${RESET}"
    echo -e "  • Close unnecessary applications"
    echo -e "  • Use USB 3.0 ports for cameras"
    echo -e "  • Reduce camera resolution/frame rate"
    echo -e "  • Ensure adequate disk space"
    echo -e ""
    echo -e "${BOLD}GET MORE HELP:${RESET}"
    echo -e "  • Run system diagnostics: ./gsr_rgbt_tools.sh test"
    echo -e "  • Check documentation: docs/user/USER_GUIDE.md"
    echo -e "  • View logs in: outputs/logs/"
    echo -e ""
}

# Function to run data collection
run_collect() {
    echo -e "${BOLD}${BLUE}Starting data collection...${RESET}"

    # Check if virtual environment is activated
    check_venv || {
        echo -e "${YELLOW}!${RESET} Setting up environment first..."
        setup_environment || return 1
    }

    # Build command with options
    local cmd="python src/data_collection/main.py"

    if [ "$SIMULATE" = true ]; then
        echo -e "${YELLOW}Running in simulation mode${RESET}"
        # Set simulation mode environment variables or pass flags
    fi

    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config=$CONFIG_FILE"
    fi

    echo -e "\n${BOLD}Launching data collection GUI...${RESET}"
    $cmd

    return $?
}

# Function to run model training
run_train() {
    echo -e "${BOLD}${BLUE}Starting model training...${RESET}"

    # Check if virtual environment is activated
    check_venv || {
        echo -e "${YELLOW}!${RESET} Setting up environment first..."
        setup_environment || return 1
    }

    # Build command with options
    local cmd="python src/scripts/train_model.py"

    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config-path=$CONFIG_FILE"
    fi

    if [ -n "$MODEL_TYPE" ]; then
        cmd="$cmd --model-type=$MODEL_TYPE"
    fi

    if [ -n "$DATA_DIR" ]; then
        cmd="$cmd --data-dir=$DATA_DIR"
    fi

    if [ -n "$OUTPUT_DIR" ]; then
        cmd="$cmd --output-dir=$OUTPUT_DIR"
    fi

    if [ -n "$CV_FOLDS" ]; then
        cmd="$cmd --cv-folds=$CV_FOLDS"
    fi

    echo -e "\n${BOLD}Running training command: $cmd${RESET}"
    $cmd

    return $?
}

# Function to run model evaluation
run_evaluate() {
    echo -e "${BOLD}${BLUE}Starting model evaluation...${RESET}"

    # Check if virtual environment is activated
    check_venv || {
        echo -e "${YELLOW}!${RESET} Setting up environment first..."
        setup_environment || return 1
    }

    # Build command with options
    local cmd="python src/scripts/visualize_results.py"

    if [ -n "$MODEL_TYPE" ]; then
        cmd="$cmd --model=$MODEL_TYPE"
    fi

    if [ -n "$DATA_DIR" ]; then
        cmd="$cmd --data-dir=$DATA_DIR"
    fi

    if [ -n "$OUTPUT_DIR" ]; then
        cmd="$cmd --output-dir=$OUTPUT_DIR"
    fi

    if [ "$VISUALIZE" = true ]; then
        cmd="$cmd --plot-history --plot-predictions --model-comparison --annotate-graphs"
    fi

    echo -e "\n${BOLD}Running evaluation command: $cmd${RESET}"
    $cmd

    return $?
}

# Function to print final status and usage information
print_final_status() {
    echo -e "\nTo activate the virtual environment in the future, run:"
    echo -e "  ${BOLD}source .venv/bin/activate${RESET} (Linux/macOS)"
    echo -e "  ${BOLD}.venv\\Scripts\\activate${RESET} (Windows)"
    echo ""
    echo -e "To run the application:"
    echo -e "  ${BOLD}./gsr_rgbt_tools.sh collect${RESET}"
    echo -e "  or ${BOLD}./gsr_rgbt_tools.sh run --component=app${RESET}"
    echo -e "  or ${BOLD}python src/data_collection/main.py${RESET}"
    echo ""
    echo -e "For more detailed instructions, see the README.md file."
    echo ""

    return 0
}

# If no command is provided, show help
if [ -z "$COMMAND" ]; then
    COMMAND="help"
fi

# Main command handling
case "$COMMAND" in
    "setup")
        setup_environment
        print_final_status
        ;;
    "collect")
        run_collect
        ;;
    "train")
        run_train
        ;;
    "evaluate")
        run_evaluate
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
        show_help "$HELP_TOPIC"
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${RESET}"
        echo -e "Run './gsr_rgbt_tools.sh help' for usage information."
        exit 1
        ;;
esac

exit 0
