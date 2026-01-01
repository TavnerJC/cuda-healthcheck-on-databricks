@echo off
REM Auto-fix code quality issues

echo 🔧 Auto-fixing quality issues...
echo.

echo 1️⃣ Sorting imports with isort...
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
if errorlevel 1 (
    echo ❌ isort failed
    exit /b 1
)
echo ✅ Imports sorted
echo.

echo 2️⃣ Formatting code with Black...
python -m black --line-length 100 cuda_healthcheck/ tests/
if errorlevel 1 (
    echo ❌ Black failed
    exit /b 1
)
echo ✅ Code formatted
echo.

echo 🎉 Auto-fix complete!
echo 💡 Review changes with: git diff

