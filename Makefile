.PHONY: help install quality fix test test-fast pre-push qc clean

# Default target
help:
	@echo "🛠️  CUDA Healthcheck on Databricks - Development Commands"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make install       Install all dependencies and pre-commit hooks"
	@echo ""
	@echo "🔧 Quick Fixes:"
	@echo "  make fix           Auto-fix formatting and import issues"
	@echo "  make qc            Quick check: fix + verify quality"
	@echo ""
	@echo "🔍 Quality Checks:"
	@echo "  make quality       Run all quality checks (matches CI/CD)"
	@echo "  make pre-push      Full check before pushing (quality + tests)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test          Run all tests with coverage"
	@echo "  make test-fast     Run tests without coverage (faster)"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean         Remove generated files and caches"

# Install all dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest pytest-cov bandit radon pre-commit
	pre-commit install
	@echo "✅ Installation complete!"

# Auto-fix quality issues
fix:
	@echo "🔧 Auto-fixing quality issues..."
	@echo ""
	@echo "1️⃣ Sorting imports with isort..."
	python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
	@echo "✅ Imports sorted"
	@echo ""
	@echo "2️⃣ Formatting code with Black..."
	python -m black --line-length 100 cuda_healthcheck/ tests/
	@echo "✅ Code formatted"
	@echo ""
	@echo "🎉 Auto-fix complete! Review changes with: git diff"

# Run quality checks (matches CI/CD exactly)
quality:
	@echo "🔍 Running quality checks (matches CI/CD)..."
	@echo ""
	@echo "1️⃣ Checking Black formatting..."
	python -m black --check --line-length 100 cuda_healthcheck/ tests/
	@echo "✅ Black passed"
	@echo ""
	@echo "2️⃣ Checking import sorting..."
	python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/
	@echo "✅ isort passed"
	@echo ""
	@echo "3️⃣ Running Flake8..."
	python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "✅ Flake8 passed"
	@echo ""
	@echo "4️⃣ Running Flake8 (full check)..."
	python -m flake8 cuda_healthcheck/ tests/ --count --max-complexity=10 --max-line-length=100 --statistics --exit-zero
	@echo ""
	@echo "5️⃣ Running MyPy (type checking)..."
	python -m mypy cuda_healthcheck/ --ignore-missing-imports --no-strict-optional || true
	@echo "✅ MyPy completed"
	@echo ""
	@echo "🎉 All quality checks passed!"

# Run tests with coverage
test:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=cuda_healthcheck --cov-report=term-missing --cov-report=html
	@echo "✅ Tests complete! Coverage report: htmlcov/index.html"

# Run tests without coverage (faster)
test-fast:
	@echo "🧪 Running tests (fast mode)..."
	python -m pytest tests/ -v --tb=short
	@echo "✅ Tests complete!"

# Full pre-push check
pre-push: quality test-fast
	@echo ""
	@echo "🎉 All checks passed!"
	@echo "✅ Safe to push to GitHub"
	@echo ""
	@echo "Next steps:"
	@echo "  git add ."
	@echo "  git commit -m 'your message'"
	@echo "  git push origin main"

# Quick check: fix + verify
qc: fix quality
	@echo ""
	@echo "✅ Quality fixed and verified!"
	@echo "💡 Review changes: git diff"
	@echo "💡 Ready to commit!"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov bandit-report.json
	@echo "✅ Clean complete!"

