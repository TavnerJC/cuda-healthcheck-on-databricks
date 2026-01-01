# Code Quality Pre-Flight Checklist

## 🎯 Analysis: GitHub Actions Failures

### What Went Wrong

| Run | Check Failed | Root Cause | Fix Time |
|-----|-------------|------------|----------|
| **#69** | Black formatting | Missing Black auto-format | 5 min |
| **#70** | isort import sorting | Missing isort in pre-commit | 5 min |

**Total wasted CI/CD time:** ~30 minutes (2 runs × 15 min average)

---

## 🔍 Root Cause Analysis

### Issue 1: Incomplete Pre-Commit Config

**Current `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        args: [--line-length=100]
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

**❌ Missing:** `isort` hook

### Issue 2: Pre-Commit Not Always Run

Pre-commit hooks rely on:
1. Developer installing pre-commit hooks locally
2. Developer remembering to run them before push

**Problem:** Easy to forget or skip

---

## ✅ Recommended Solutions

### Solution 1: Enhanced Pre-Commit Config (IMMEDIATE)

Add **isort** to catch import sorting issues:

```yaml
# Add to .pre-commit-config.yaml
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
```

### Solution 2: Pre-Push Quality Script (RECOMMENDED)

Create a **local quality check script** that runs EXACTLY what GitHub Actions runs:

**File:** `scripts/pre-push-check.sh` (or `.bat` for Windows)

```bash
#!/bin/bash
set -e

echo "🔍 Running pre-push quality checks..."
echo ""

# 1. Black formatting
echo "1️⃣ Checking Black formatting..."
python -m black --check --line-length 100 cuda_healthcheck/ tests/
if [ $? -eq 0 ]; then
    echo "✅ Black formatting passed"
else
    echo "❌ Black formatting failed"
    echo "   Fix with: python -m black --line-length 100 cuda_healthcheck/ tests/"
    exit 1
fi
echo ""

# 2. isort import sorting
echo "2️⃣ Checking import sorting..."
python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/
if [ $? -eq 0 ]; then
    echo "✅ Import sorting passed"
else
    echo "❌ Import sorting failed"
    echo "   Fix with: python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/"
    exit 1
fi
echo ""

# 3. Flake8 linting
echo "3️⃣ Running Flake8..."
python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
if [ $? -eq 0 ]; then
    echo "✅ Flake8 passed"
else
    echo "❌ Flake8 failed"
    exit 1
fi
echo ""

# 4. MyPy type checking (non-blocking)
echo "4️⃣ Running MyPy type checking..."
python -m mypy cuda_healthcheck/ --ignore-missing-imports --no-strict-optional || true
echo "✅ MyPy completed (warnings only)"
echo ""

# 5. Unit tests
echo "5️⃣ Running unit tests..."
python -m pytest tests/ -v --tb=short -x
if [ $? -eq 0 ]; then
    echo "✅ Tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
echo ""

echo "🎉 All pre-push checks passed!"
echo "✅ Safe to push to GitHub"
```

### Solution 3: Auto-Fix Script (DEVELOPER FRIENDLY)

Create a script that **automatically fixes** common issues:

**File:** `scripts/fix-quality.sh`

```bash
#!/bin/bash

echo "🔧 Auto-fixing code quality issues..."
echo ""

# 1. Sort imports
echo "1️⃣ Sorting imports with isort..."
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
echo "✅ Imports sorted"
echo ""

# 2. Format code
echo "2️⃣ Formatting code with Black..."
python -m black --line-length 100 cuda_healthcheck/ tests/
echo "✅ Code formatted"
echo ""

# 3. Check remaining issues
echo "3️⃣ Checking for remaining issues..."
python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
echo ""

echo "🎉 Auto-fix complete!"
echo "💡 Review changes with: git diff"
```

### Solution 4: Makefile for Easy Commands

**File:** `Makefile`

```makefile
.PHONY: quality fix test install

# Install development dependencies
install:
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest pytest-cov
	pre-commit install

# Auto-fix quality issues
fix:
	@echo "🔧 Auto-fixing quality issues..."
	python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
	python -m black --line-length 100 cuda_healthcheck/ tests/
	@echo "✅ Done! Review with: git diff"

# Check code quality (matches CI/CD)
quality:
	@echo "🔍 Running quality checks..."
	python -m black --check --line-length 100 cuda_healthcheck/ tests/
	python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/
	python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "✅ Quality checks passed!"

# Run tests
test:
	python -m pytest tests/ -v --cov=cuda_healthcheck --cov-report=term-missing

# Full pre-push check (quality + tests)
pre-push: quality test
	@echo "🎉 All checks passed! Safe to push."

# Quick quality fix and check
qc: fix quality
	@echo "✅ Quality fixed and verified!"
```

**Usage:**
```bash
make fix        # Auto-fix formatting issues
make quality    # Check quality (matches CI/CD)
make test       # Run tests
make pre-push   # Full check before pushing
make qc         # Quick: fix + check
```

---

## 📋 Step-by-Step Implementation

### Step 1: Update Pre-Commit Config

```bash
# Update .pre-commit-config.yaml to include isort
```

### Step 2: Create Quality Scripts

```bash
# For Linux/Mac
mkdir -p scripts
touch scripts/fix-quality.sh
chmod +x scripts/fix-quality.sh

touch scripts/pre-push-check.sh
chmod +x scripts/pre-push-check.sh

# For Windows
mkdir scripts
# Create .bat versions
```

### Step 3: Create Makefile

```bash
# Add Makefile to project root
```

### Step 4: Update Developer Workflow

Add to `CONTRIBUTING.md`:

```markdown
## Before Pushing to GitHub

### Quick Fix (Recommended)
```bash
make qc  # Auto-fix and verify quality
```

### Manual Workflow
```bash
# 1. Auto-fix issues
make fix

# 2. Review changes
git diff

# 3. Run full checks
make pre-push

# 4. Commit and push
git add .
git commit -m "your message"
git push
```

### If You Forget
The CI/CD will catch issues, but this wastes time (~15 min per run).
Always run `make qc` before pushing!
```

---

## 🎯 Expected Impact

### Before (Current State)

| Step | Time | Success Rate |
|------|------|--------------|
| Write code | 30 min | - |
| Push to GitHub | 1 min | - |
| **CI/CD fails** | **15 min** | **❌ 50%** |
| Fix formatting | 5 min | - |
| Push again | 1 min | - |
| **CI/CD fails again** | **15 min** | **❌ 25%** |
| Fix imports | 5 min | - |
| Push third time | 1 min | - |
| CI/CD passes | 15 min | ✅ 100% |
| **Total** | **~73 min** | **3 pushes** |

### After (With New Tools)

| Step | Time | Success Rate |
|------|------|--------------|
| Write code | 30 min | - |
| **Run `make qc`** | **2 min** | **✅ Catches issues** |
| Fix locally (if needed) | 2 min | - |
| **Run `make qc` again** | **2 min** | **✅ Pass** |
| Push to GitHub | 1 min | - |
| CI/CD passes | 15 min | ✅ 100% |
| **Total** | **~52 min** | **1 push** |

**Time Saved:** ~21 minutes per feature  
**CI/CD Runs Saved:** 2 fewer failed runs  
**Developer Confidence:** ⬆️⬆️⬆️

---

## 🚀 Quick Start Commands

### For This Project (Immediate Use)

```bash
# 1. Fix current issues
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
python -m black --line-length 100 cuda_healthcheck/ tests/

# 2. Verify
python -m black --check --line-length 100 cuda_healthcheck/ tests/
python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/
python -m flake8 cuda_healthcheck/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Test
python -m pytest tests/ -v

# 4. Push confidently
git add .
git commit -m "your message"
git push origin main
```

---

## 📊 Commonalities Identified

### GitHub Actions Checks (from `code-quality.yml`)

| Check | Tool | Command | In Pre-Commit? |
|-------|------|---------|----------------|
| **Code formatting** | Black | `black --check --line-length 100` | ✅ YES |
| **Import sorting** | isort | `isort --check-only --profile black` | ❌ **MISSING** |
| **Linting** | Flake8 | `flake8 --max-line-length 100` | ✅ YES |
| **Type checking** | MyPy | `mypy --ignore-missing-imports` | ✅ YES |
| **Security** | Bandit | `bandit -r cuda_healthcheck/` | ❌ Optional |
| **Complexity** | Radon | `radon cc cuda_healthcheck/` | ❌ Optional |

**Critical Gap:** isort not in pre-commit config

---

## 🎓 Best Practices Going Forward

### 1. Always Run Local Checks Before Push

```bash
# One command to rule them all
make pre-push
```

### 2. Use Auto-Fix First

```bash
# Let the tools fix what they can
make fix
```

### 3. Install Git Hooks

```bash
# One-time setup
pre-commit install
```

### 4. CI/CD Should Be Final Validation, Not First Check

**Philosophy:**
- Local checks = Fast feedback (2 min)
- CI/CD = Confirmation (15 min)

Don't use CI/CD as your primary quality check!

---

## 📝 Recommended Next Steps

### Immediate (5 minutes)
1. ✅ Update `.pre-commit-config.yaml` to add isort
2. ✅ Create `Makefile` with quality commands
3. ✅ Run `make fix` before next push

### Short-term (1 hour)
1. Create `scripts/pre-push-check.sh`
2. Create `scripts/fix-quality.sh`
3. Update `CONTRIBUTING.md` with new workflow
4. Test the new workflow on a feature branch

### Long-term (Ongoing)
1. Make `make pre-push` part of your muscle memory
2. Consider adding Git hook that runs checks automatically
3. Add quality metrics dashboard to README
4. Train team members on new workflow

---

## 🎉 Summary

**Problem:** CI/CD failures due to formatting and import sorting  
**Root Cause:** isort missing from pre-commit, no local quality check habit  
**Solution:** Enhanced pre-commit + Makefile + quality scripts  
**Impact:** ~20 min saved per feature, fewer failed CI/CD runs  

**Key Insight:** The CI/CD is already checking isort (line 42-44 of code-quality.yml), but our pre-commit config wasn't! This created a gap where local checks passed but CI/CD failed.

---

**Want me to implement these solutions now?** I can:
1. Update `.pre-commit-config.yaml` to add isort
2. Create the `Makefile`
3. Create the quality scripts
4. Update documentation

