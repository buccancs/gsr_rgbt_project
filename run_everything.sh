#!/bin/bash

# run_everything.sh - Script to run all components of the GSR-RGBT Project
# This script runs all the main components of the project in sequence:
# 1. Setup (if needed)
# 2. System validation checks
# 3. Synchronization tests
# 4. Generate mock data (if needed)
# 5. Run the full ML pipeline

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Print header
echo -e "${BOLD}${BLUE}=========================================${RESET}"
echo -e "${BOLD}${BLUE}  GSR-RGBT Project - Run Everything     ${RESET}"
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

# Check if make is available
if ! command_exists make; then
    echo -e "${RED}✗${RESET} make command not found. Please install make and try again."
    exit 1
fi

# Check if Python is available
if command_exists python3; then
    PYTHON="python3"
    echo -e "${GREEN}✓${RESET} Python 3 is installed"
elif command_exists python; then
    PYTHON="python"
    # Check Python version
    PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
    if [ "$PYTHON_VERSION" -lt 3 ]; then
        echo -e "${RED}✗${RESET} Python 3 is required, but Python $PYTHON_VERSION is installed"
        echo -e "${YELLOW}Please install Python 3 and try again${RESET}"
        exit 1
    else
        echo -e "${GREEN}✓${RESET} Python 3 is installed"
    fi
else
    echo -e "${RED}✗${RESET} Python 3 is not installed"
    echo -e "${YELLOW}Please install Python 3 and try again${RESET}"
    exit 1
fi

# Step 1: Setup (if needed)
echo -e "\n${BOLD}${BLUE}Step 1: Checking setup${RESET}"
if ! check_venv; then
    echo -e "Running setup..."
    run_make_target "setup" "Setup" || exit 1
    source .venv/bin/activate
else
    echo -e "Setup already completed, skipping..."
fi

# Check for required components
echo -e "\n${BOLD}${BLUE}Step 1a: Checking for required components${RESET}"
ALL_COMPONENTS_INSTALLED=true

# Check for PySpin
echo -e "Checking for FLIR Spinnaker SDK..."
if ! $PYTHON -c "import PySpin" 2>/dev/null; then
    echo -e "${RED}✗${RESET} FLIR Spinnaker SDK (PySpin) is not installed"
    echo -e "${YELLOW}Running setup script to install required components...${RESET}"
    ./setup.sh

    # Check again after running setup
    if ! $PYTHON -c "import PySpin" 2>/dev/null; then
        echo -e "${RED}✗${RESET} FLIR Spinnaker SDK (PySpin) is still not installed"
        echo -e "${YELLOW}Please install it manually following the instructions in guide.md${RESET}"
        ALL_COMPONENTS_INSTALLED=false
    else
        echo -e "${GREEN}✓${RESET} FLIR Spinnaker SDK (PySpin) is now installed"
    fi
else
    echo -e "${GREEN}✓${RESET} FLIR Spinnaker SDK (PySpin) is installed"
fi

# Check for other critical dependencies
for package in pyshimmer numpy pandas cv2 PyQt5 tensorflow sklearn neurokit2; do
    echo -e "Checking for $package..."
    if ! $PYTHON -c "import $package" 2>/dev/null; then
        echo -e "${RED}✗${RESET} $package is not installed"
        echo -e "${YELLOW}Installing $package...${RESET}"
        pip install $package

        # Check again after installation
        if ! $PYTHON -c "import $package" 2>/dev/null; then
            echo -e "${RED}✗${RESET} $package installation failed"
            ALL_COMPONENTS_INSTALLED=false
        else
            echo -e "${GREEN}✓${RESET} $package is now installed"
        fi
    else
        echo -e "${GREEN}✓${RESET} $package is installed"
    fi
done

if [ "$ALL_COMPONENTS_INSTALLED" = false ]; then
    echo -e "\n${RED}WARNING: Some required components are missing.${RESET}"
    echo -e "${YELLOW}The application may not work correctly without these components.${RESET}"
    echo -e "${YELLOW}Do you want to continue anyway? (y/n)${RESET}"
    read -p "" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting...${RESET}"
        exit 1
    fi
    echo -e "${YELLOW}Continuing with missing components...${RESET}"
fi


# Step 2: System validation checks
echo -e "\n${BOLD}${BLUE}Step 2: Running system validation checks${RESET}"
run_make_target "test" "System validation checks" || {
    echo -e "${YELLOW}!${RESET} System validation checks failed, but continuing with other steps..."
}

# Step 3: Synchronization tests
echo -e "\n${BOLD}${BLUE}Step 3: Running synchronization tests${RESET}"
run_make_target "test_sync" "Synchronization tests" || {
    echo -e "${YELLOW}!${RESET} Synchronization tests failed, but continuing with other steps..."
}

# Step 4: Generate mock data (if needed)
echo -e "\n${BOLD}${BLUE}Step 4: Checking for existing data${RESET}"
if [ ! -d "data/recordings" ] || [ -z "$(ls -A data/recordings 2>/dev/null)" ]; then
    echo -e "No data found, generating mock data..."
    run_make_target "mock_data" "Generate mock data" || {
        echo -e "${RED}✗${RESET} Failed to generate mock data, cannot continue with ML pipeline"
        exit 1
    }
else
    echo -e "${GREEN}✓${RESET} Data already exists, skipping mock data generation"
fi

# Step 5: Run the full ML pipeline
echo -e "\n${BOLD}${BLUE}Step 5: Running the full ML pipeline${RESET}"
run_make_target "pipeline" "Full ML pipeline" || {
    echo -e "${YELLOW}!${RESET} ML pipeline failed or was incomplete"
}

# Ask if the user wants to run the data collection application
echo -e "\n${BOLD}${BLUE}Optional: Run the data collection application?${RESET}"
read -p "Do you want to run the data collection application? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_make_target "run_app" "Data collection application"
fi

echo -e "\n${BOLD}${GREEN}All steps completed!${RESET}"
echo -e "You have successfully run all components of the GSR-RGBT Project."
echo -e "Check the output above for any warnings or errors that may have occurred."
echo ""
