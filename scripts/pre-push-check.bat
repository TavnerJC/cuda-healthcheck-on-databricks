@echo off
REM Pre-push quality checks (matches CI/CD)

echo 🔍 Running pre-push quality checks...
echo.

REM 1. Black formatting
echo 1️⃣ Checking Black formatting...
python -m black --check --line-length 100 cuda_healthcheck/ tests/
if errorlevel 1 (
    echo ❌ Black formatting failed
    echo    Fix with: python -m black --line-length 100 cuda_healthcheck/ tests/
    exit /b 1
)
echo ✅ Black formatting passed
echo.

REM 2. isort import sorting
echo 2️⃣ Checking import sorting...
python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/
if errorlevel 1 (
    echo ❌ Import sorting failed
    echo    Fix with: python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
    exit /b 1
)
echo ✅ Import sorting passed
echo.

REM 3. Flake8 linting
echo 3️⃣ Running Flake8...
python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
if errorlevel 1 (
    echo ❌ Flake8 failed
    exit /b 1
)
echo ✅ Flake8 passed
echo.

REM 4. Unit tests
echo 4️⃣ Running unit tests...
python -m pytest tests/ -v --tb=short -x
if errorlevel 1 (
    echo ❌ Tests failed
    exit /b 1
)
echo ✅ Tests passed
echo.

echo 🎉 All pre-push checks passed!
echo ✅ Safe to push to GitHub

