#!/bin/bash
# Installation script for OWUI Adaptive Memory Plugin
# This script handles installation and verification in one step

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_header() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then 
   print_warning "Running as root is not recommended. Continue anyway? (y/N)"
   read -r response
   if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
       exit 1
   fi
fi

print_header "OWUI Adaptive Memory Plugin Installation"

# Step 1: Check Python version
print_status "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version meets requirements (>= $required_version)"
else
    print_error "Python $python_version is below minimum requirement ($required_version)"
    exit 1
fi

# Step 2: Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate || {
    print_warning "Failed to activate virtual environment, continuing with system Python"
}

# Step 3: Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip >/dev/null 2>&1

# Step 4: Install dependencies
print_status "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || {
        print_error "Failed to install dependencies"
        print_warning "Trying to install core dependencies individually..."
        
        # Try installing core dependencies one by one
        for package in pydantic numpy aiohttp pytz; do
            print_status "Installing $package..."
            pip install "$package" || print_warning "Failed to install $package"
        done
    }
    print_status "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Step 5: Run quick verification
print_header "Running Quick Verification"
python quick_verify.py || {
    print_error "Quick verification failed"
    print_warning "Attempting full verification with auto-fix..."
    
    # Try full verification with auto-fix
    python post_install_verification.py --auto-fix || {
        print_error "Installation verification failed"
        exit 1
    }
}

# Step 6: Check for OpenWebUI
print_header "Checking OpenWebUI Integration"
if command -v docker &> /dev/null; then
    if docker ps | grep -q openwebui; then
        print_status "OpenWebUI container detected"
        
        # Try to test integration
        print_status "Testing OpenWebUI integration..."
        python verify_openwebui_integration.py || {
            print_warning "OpenWebUI integration test failed"
            print_warning "This is normal if OpenWebUI is not yet configured"
        }
    else
        print_warning "OpenWebUI container not running"
        print_warning "Start OpenWebUI before uploading the plugin"
    fi
else
    print_warning "Docker not installed - skipping OpenWebUI check"
fi

# Step 7: Create necessary directories
print_status "Creating plugin directories..."
mkdir -p logs
mkdir -p memory-bank
mkdir -p tests/unit
mkdir -p tests/integration

# Step 8: Set permissions
print_status "Setting file permissions..."
chmod +x quick_verify.py
chmod +x post_install_verification.py
chmod +x verify_openwebui_integration.py
chmod 644 adaptive_memory_v4.0.py

# Final summary
print_header "Installation Complete!"

echo "Next steps:"
echo "1. Upload adaptive_memory_v4.0.py to OpenWebUI:"
echo "   - Go to Workspace → Functions"
echo "   - Click '+' to add new function"
echo "   - Upload the file and save"
echo ""
echo "2. Enable the filter for your models:"
echo "   - Go to Workspace → Models"
echo "   - Select models to use with memory"
echo "   - Enable the Adaptive Memory filter"
echo ""
echo "3. Configure the plugin (optional):"
echo "   - Click the settings icon on the filter"
echo "   - Adjust memory settings as needed"
echo ""
echo "4. Start chatting with persistent memory!"
echo ""
echo "For troubleshooting, run:"
echo "  python post_install_verification.py --save-report"
echo ""
print_status "Installation completed successfully!"

# Deactivate virtual environment if activated
deactivate 2>/dev/null || true