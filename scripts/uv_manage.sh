#!/bin/bash
# uv_manage.sh - A script to manage Python dependencies using uv

set -e

# Default values
REQUIREMENTS_FILE="requirements.txt"
COMMAND="install"

# Help message
show_help() {
    echo "Usage: ./uv_manage.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -i, --install              Install packages from requirements.txt (default)"
    echo "  -u, --uninstall            Uninstall packages from requirements.txt"
    echo "  -s, --sync                 Synchronize environment with requirements.txt"
    echo "  -r, --requirements FILE    Specify requirements file (default: requirements.txt)"
    echo ""
    echo "Examples:"
    echo "  ./uv_manage.sh -i          # Install packages from requirements.txt"
    echo "  ./uv_manage.sh -u          # Uninstall packages from requirements.txt"
    echo "  ./uv_manage.sh -s          # Sync environment with requirements.txt"
    echo "  ./uv_manage.sh -r dev-requirements.txt -i  # Install from custom requirements file"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--install)
            COMMAND="install"
            shift
            ;;
        -u|--uninstall)
            COMMAND="uninstall"
            shift
            ;;
        -s|--sync)
            COMMAND="sync"
            shift
            ;;
        -r|--requirements)
            REQUIREMENTS_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file '$REQUIREMENTS_FILE' not found."
    exit 1
fi

# Execute the command
case $COMMAND in
    install)
        echo "Installing packages from $REQUIREMENTS_FILE using uv..."
        uv pip install -r "$REQUIREMENTS_FILE"
        ;;
    uninstall)
        echo "Uninstalling packages from $REQUIREMENTS_FILE using uv..."
        uv pip uninstall -y -r "$REQUIREMENTS_FILE"
        ;;
    sync)
        echo "Synchronizing environment with $REQUIREMENTS_FILE using uv..."
        uv pip sync "$REQUIREMENTS_FILE"
        ;;
esac

echo "Done!"
