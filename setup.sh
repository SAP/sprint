#!/bin/bash
# Copyright (c) 2025 SAP SE or an SAP affiliate company and sprint contributors
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on any error

echo "🚀 SPRINT Framework Setup Script"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
USE_CONDA=false

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --conda         Use conda to manage Python environment (recommended)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Standard installation using system package manager"
    echo "  $0 --conda      # Use conda environment (recommended for research)"
    echo "  source $0       # Run with sourcing to keep environment active"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --conda)
                USE_CONDA=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Check operating system and architecture
detect_os() {
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    print_status "Detected OS: $OS"
    print_status "Detected Architecture: $ARCH"
}

# Install conda if not present
install_conda() {
    if command -v conda &> /dev/null; then
        print_success "Conda already installed: $(conda --version)"
        return 0
    fi
    
    print_status "Installing Miniconda..."
    local conda_installer
    local install_dir="$HOME/miniconda3"
    
     # Determine the correct installer based on OS and architecture
    if [[ "$OS" == "linux" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            conda_installer="Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
            conda_installer="Miniconda3-latest-Linux-aarch64.sh"
        else
            print_error "Unsupported Linux architecture: $ARCH"
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            conda_installer="Miniconda3-latest-MacOSX-arm64.sh"
        elif [[ "$ARCH" == "x86_64" ]]; then
            conda_installer="Miniconda3-latest-MacOSX-x86_64.sh"
        else
            print_error "Unsupported macOS architecture: $ARCH"
            exit 1
        fi
    fi
    
    # Download and install Miniconda
    local temp_installer="/tmp/miniconda.sh"
    print_status "Downloading $conda_installer..."
    
    if ! wget -q "https://repo.anaconda.com/miniconda/$conda_installer" -O "$temp_installer"; then
        print_error "Failed to download Miniconda installer"
        exit 1
    fi
    
    print_status "Installing Miniconda to $install_dir..."
    # Check for available shell and use the best one for running the installer
    if command -v bash &> /dev/null; then
        print_status "Using bash to run Miniconda installer..."
        bash "$temp_installer" -b -p "$install_dir"
    elif command -v zsh &> /dev/null; then
        print_status "bash not found, using zsh to run Miniconda installer..."
        zsh "$temp_installer" -b -p "$install_dir"
    elif command -v sh &> /dev/null; then
        print_warning "bash and zsh not found, using sh to run Miniconda installer..."
        sh "$temp_installer" -b -p "$install_dir"
    else
        print_error "No compatible shell (bash/zsh/sh) found. Cannot install Miniconda."
        rm "$temp_installer"
        exit 1
    fi
    
    rm "$temp_installer"
    
    # Initialize conda for current session
    export PATH="$install_dir/bin:$PATH"
    
    # Initialize conda for future sessions - now supports more shells
    if command -v bash &> /dev/null; then
        "$install_dir/bin/conda" init bash 2>/dev/null || true
    fi
    if command -v zsh &> /dev/null; then
        "$install_dir/bin/conda" init zsh 2>/dev/null || true
    fi
    
    print_success "Miniconda installed successfully"
}

# Setup conda environment
setup_conda_env() {
    print_status "Setting up conda environment 'sprint' with Python 3.9..."
    
    # Accept Anaconda Terms of Service for non-interactive installation
    print_status "Accepting Anaconda Terms of Service..."
    conda config --set channel_priority strict
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^sprint "; then
        print_warning "Removing existing 'sprint' conda environment..."
        conda env remove -n sprint -y
    fi
    
    # Create new environment with Python 3.9
    print_status "Creating conda environment 'sprint' with Python 3.9..."
    conda create -n sprint python=3.9 -y
    
    # Activate environment
    print_status "Activating conda environment 'sprint'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate sprint
    
    # Verify activation
    if [[ "$CONDA_DEFAULT_ENV" == "sprint" ]]; then
        print_success "Conda environment 'sprint' activated successfully"
        PYTHON_CMD="python"
        local python_version
        python_version=$(python --version 2>&1 | awk '{print $2}')
        print_status "Using Python $python_version from conda"
    else
        print_error "Failed to activate conda environment"
        exit 1
    fi
}

# Check for existing Python installations
check_existing_python() {
    local python_versions=("python3.9" "python3.8" "python3")
    
    for py_cmd in "${python_versions[@]}"; do
        if command -v "$py_cmd" >/dev/null 2>&1; then
            local version
            version=$("$py_cmd" --version 2>&1 | awk '{print $2}')
            if [[ "$version" == 3.8* ]] || [[ "$version" == 3.9* ]]; then
                print_success "Found compatible Python: $py_cmd ($version)"
                PYTHON_CMD="$py_cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Install Python 3.9
install_python() {
    if [[ "$USE_CONDA" == "true" ]]; then
        print_status "Setting up Python environment using conda..."
        install_conda
        setup_conda_env
        return
    fi
    
    # Check for existing compatible Python first
    if check_existing_python; then
        print_status "Using existing Python installation: $PYTHON_CMD"
        return
    fi
    
    print_status "Installing Python 3.9 via system package manager..."
    
    # check if sudo is available (e.g., not available in some docker containers)
    if [[ "$OS" == "linux" ]]; then
        if command -v sudo &> /dev/null; then
        # Check if we are root
            if [[ $EUID -eq 0 ]]; then
                SUDO_CMD=""
            else
                SUDO_CMD="sudo"
            fi
        else
            SUDO_CMD=""
        fi

        # Focus on apt for Linux
        if command -v apt-get &> /dev/null; then
            print_status "Installing Python 3.9 via apt..."
            ${SUDO_CMD} apt-get update
            ${SUDO_CMD} apt-get install -y software-properties-common
            ${SUDO_CMD} add-apt-repository -y ppa:deadsnakes/ppa
            ${SUDO_CMD} apt-get update
            ${SUDO_CMD} apt-get install -y python3.9 python3.9-venv python3.9-dev python3.9-distutils
            PYTHON_CMD="python3.9"
        else
            print_error "This script requires apt package manager (Ubuntu/Debian)"
            print_error "Please install Python 3.9 manually, use --conda flag, or ensure 'python3.9' command is available"
            exit 1
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is available
        if command -v brew &> /dev/null; then
            print_status "Installing Python 3.9 via Homebrew..."
            brew install python@3.9 || true
            # Check different possible locations
            if command -v python3.9 &> /dev/null; then
                PYTHON_CMD="python3.9"
            elif [[ -f "/opt/homebrew/bin/python3.9" ]]; then
                PYTHON_CMD="/opt/homebrew/bin/python3.9"
            elif [[ -f "/usr/local/bin/python3.9" ]]; then
                PYTHON_CMD="/usr/local/bin/python3.9"
            else
                print_error "Homebrew installation succeeded but python3.9 command not found"
                print_error "Try using --conda flag for a simpler setup"
                exit 1
            fi
        else
            print_error "Homebrew is required for macOS installation"
            print_error "Install Homebrew first or use --conda flag for easier setup:"
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            exit 1
        fi
    fi
    
    # Verify Python installation
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        print_error "Python installation failed - $PYTHON_CMD not found"
        exit 1
    fi
    
    PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION installed successfully"
}

# Set up SPRINT_PATH
setup_sprint_path() {
    print_status "Setting up SPRINT_PATH environment variable..."
    
    # Verify we're in the right directory
    if [[ ! -d "src/sprint_core" ]]; then
        print_error "This script must be run from the SPRINT repository root directory"
        exit 1
    fi
    
    SPRINT_PATH=$(pwd)
    export SPRINT_PATH="$SPRINT_PATH"
    print_success "SPRINT_PATH set to: $SPRINT_PATH"
}

# Create virtual environment (only for non-conda setups)
create_virtual_env() {
    if [[ "$USE_CONDA" == "true" ]]; then
        print_status "Using conda environment, skipping virtual environment creation"
        return
    fi
    
    print_status "Creating virtual environment 'sprint_env'..."
    
    if [[ -d "sprint_env" ]]; then
        print_warning "Removing existing virtual environment..."
        rm -rf sprint_env
    fi
    
    "$PYTHON_CMD" -m venv sprint_env
    print_success "Virtual environment 'sprint_env' created"
}

# Activate virtual environment
activate_virtual_env() {
    if [[ "$USE_CONDA" == "true" ]]; then
        print_status "Using conda environment (already activated)"
        return
    fi
    
    print_status "Activating virtual environment..."
    source sprint_env/bin/activate
    
    # Verify activation
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Upgrade pip
    python -m pip install --upgrade pip
}

# Install requirements
install_requirements() {
    print_status "Installing Python dependencies..."
    
    # Use the reviewer's suggested command to set SKLEARN variable only for this installation
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt
    
    print_success "Python dependencies installed successfully"
}

# Apply critical modifications
apply_modifications() {
    print_status "Applying critical modifications..."
    
    # Find private_transformers installation
    PRIVATE_TRANSFORMERS_PATH=$(python -c "import private_transformers; print(private_transformers.__file__)" 2>/dev/null | sed 's/__init__.py$//')
    
    if [[ -z "$PRIVATE_TRANSFORMERS_PATH" ]]; then
        print_error "Could not find private_transformers installation (Use manual setup instructions)"
        exit 1
    fi
    
    AUTOGRAD_FILE="$PRIVATE_TRANSFORMERS_PATH/autograd_grad_sample.py"
    
    if [[ -f "$AUTOGRAD_FILE" ]]; then
        print_status "Found autograd_grad_sample.py at: $AUTOGRAD_FILE"
        
        # Check if modification is already applied
        if grep -q "handles.append(layer.register_full_backward_hook(this_backward))" "$AUTOGRAD_FILE"; then
            print_success "Critical modification already applied"
        else
            print_status "Applying critical modification..."
            
            # Create backup
            cp "$AUTOGRAD_FILE" "$AUTOGRAD_FILE.backup"
            
            # Apply modification (replace register_backward_hook with register_full_backward_hook)
            sed -i.bak 's/handles\.append(layer\.register_backward_hook(this_backward))/handles.append(layer.register_full_backward_hook(this_backward))/g' "$AUTOGRAD_FILE"
            
            print_success "Critical modification applied to private_transformers"
            print_warning "Backup created at: $AUTOGRAD_FILE.backup"
        fi
    else
        print_error "Could not find autograd_grad_sample.py in private_transformers (Use manual setup instructions)"
        print_error "You may need to apply the modification manually"
        print_error "Location should be: <env_path>/lib/python3.9/site-packages/private_transformers/autograd_grad_sample.py"
        print_error "Change: replace 'register_backward_hook' with 'register_full_backward_hook'"
    fi
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p "$SPRINT_PATH/data/tokenized_datasets"
    mkdir -p "$SPRINT_PATH/data/models"
    mkdir -p "$SPRINT_PATH/data/finetuning"
    mkdir -p "$SPRINT_PATH/data/inference/accuracy"
    mkdir -p "$SPRINT_PATH/data/inference/runtime"
    
    print_success "Directory structure created"
}

# Print next steps
print_next_steps() {
    echo ""
    echo "🎉 SPRINT Framework Setup Complete!"
    echo "===================================="
    echo ""
    
    if [[ "$USE_CONDA" == "true" ]]; then
        echo "Conda environment 'sprint' is ready!"
        echo ""
        echo "To activate the environment in future sessions:"
        echo "  conda activate sprint"
        echo ""
        echo "To deactivate:"
        echo "  conda deactivate"
        echo ""
    else
        echo "Virtual environment 'sprint_env' is ready!"
        echo ""
        echo "To activate the environment:"
        echo "  source sprint_env/bin/activate"
        echo ""
    fi
    
    echo "Environment variables:"
    echo "  export SPRINT_PATH=$SPRINT_PATH"
    echo ""
    echo "Quick test:"
    echo "  cd src"
    echo "  python tokenize_dataset.py --dataset sst2 --model_type roberta"
    echo ""
    echo "Run experiments:"
    echo "  python run_dp_finetuning.py --config fine-tuning_example_cpu.yaml"
    echo "  python run_inference.py --config inference_example.yaml --crypten_config crypten_inference_config.yaml"
    echo ""
    print_success "Setup completed successfully!"
}

# Main execution
main() {
    parse_args "$@"
    detect_os
    install_python
    setup_sprint_path
    create_virtual_env
    activate_virtual_env
    install_requirements
    apply_modifications
    create_directories
    print_next_steps
}

# Run main function
main "$@"