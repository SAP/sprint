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

# Check operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Install Python 3.9
install_python() {
    print_status "Installing Python 3.9..."
    
    if [[ "$OS" == "linux" ]]; then
        # Focus on apt for Linux
        if command -v apt-get &> /dev/null; then
            print_status "Installing Python 3.9 via apt..."
            sudo apt-get update
            sudo apt-get install -y python3.9 python3.9-venv python3.9-dev python3.9-distutils
            PYTHON_CMD="python3.9"
        else
            print_error "This script requires apt package manager (Ubuntu/Debian)"
            print_error "Please install Python 3.9 manually and ensure 'python3.9' command is available"
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
                exit 1
            fi
        else
            print_error "Homebrew is required for macOS installation"
            print_error "Install Homebrew first:"
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            exit 1
        fi
    fi
    
    # Verify Python installation
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python installation failed - $PYTHON_CMD not found"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
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
}

# Create virtual environment
create_virtual_env() {
    print_status "Creating virtual environment 'sprint_env'..."
    
    if [[ -d "sprint_env" ]]; then
        print_warning "Virtual environment 'sprint_env' already exists"
    else
        $PYTHON_CMD -m venv sprint_env
        print_success "Virtual environment 'sprint_env' created"
    fi
}

# Activate virtual environment
activate_virtual_env() {
    print_status "Activating virtual environment..."
    source sprint_env/bin/activate
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" != "" ]]; then
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
    PRIVATE_TRANSFORMERS_PATH=$(python -c "import private_transformers; print(private_transformers.__file__)" 2>/dev/null | sed 's/__init__.py//')
    
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
            sed -i.bak 's/handles.append(layer.register_backward_hook(this_backward))/handles.append(layer.register_full_backward_hook(this_backward))/g' "$AUTOGRAD_FILE"
            
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
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source sprint_env/bin/activate"
    echo ""
    echo "2. Set SPRINT_PATH:"
    echo "   export SPRINT_PATH=$SPRINT_PATH"
    echo ""
    echo "3. Test data loading:"
    echo "   cd src"
    echo "   python tokenize_dataset.py --dataset sst2 --model_type roberta"
    echo ""
    echo "4. Run DP fine-tuning:"
    echo "   python run_dp_finetuning.py --config configs/fine-tuning_example_cuda.yaml"
    echo ""
    echo "5. Run inference:"
    echo "   python run_inference.py --config configs/inference_example.yaml"
    echo ""
    print_success "Setup completed successfully!"
}

# Main execution
main() {
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

# Check if script is run with bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi