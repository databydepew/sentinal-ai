# Package Management with uv

This project uses [uv](https://github.com/astral-sh/uv) for Python package management instead of pip. uv is a much faster alternative to pip with improved dependency resolution.

## Using the uv_manage.sh Script

We've included a convenient script to help manage dependencies with uv:

```bash
# Install all dependencies from requirements.txt
./scripts/uv_manage.sh --install

# Uninstall all packages listed in requirements.txt
./scripts/uv_manage.sh --uninstall

# Synchronize your environment with requirements.txt
# This ensures your environment exactly matches the requirements
./scripts/uv_manage.sh --sync

# Use a different requirements file
./scripts/uv_manage.sh --requirements dev-requirements.txt --install
```

## Direct uv Commands

If you prefer to use uv directly:

```bash
# Install packages from requirements.txt
uv pip install -r requirements.txt

# Install a specific package
uv pip install package_name

# Uninstall packages
uv pip uninstall package_name

# Synchronize environment with requirements.txt
uv pip sync requirements.txt

# Generate a requirements.txt file from your environment
uv pip freeze > requirements.txt
```

## Benefits of uv over pip

- **Speed**: uv is significantly faster than pip for installing packages
- **Better dependency resolution**: Improved handling of complex dependency trees
- **Caching**: Better caching of wheel builds for faster reinstalls
- **Reproducibility**: More consistent environments across different systems

## Installing uv

If uv is not already installed, you can install it with:

```bash
# Using curl (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using pip
pip install uv
```
