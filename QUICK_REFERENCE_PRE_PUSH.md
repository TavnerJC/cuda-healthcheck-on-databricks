# Quick Reference: Pre-Push Quality Checks

## 🎯 The Golden Rule

**ALWAYS run quality checks locally BEFORE pushing to GitHub!**

## ⚠️ CRITICAL: Line Length Configuration

Our codebase uses **100-character line length** (not Black's default of 88):

```bash
# ✅ CORRECT (matches CI/CD)
python -m black --line-length 100 cuda_healthcheck/ tests/
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/

# ❌ WRONG (uses Black's default of 88, will fail CI/CD)
python -m black cuda_healthcheck/ tests/
python -m isort cuda_healthcheck/ tests/
```

**Always include `--line-length 100`** or use the provided scripts which have it configured.

---

## ⚡ Quick Commands

### Windows PowerShell
```powershell
# Quick fix + check (recommended)
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/; python -m black --line-length 100 cuda_healthcheck/ tests/; python -m pytest tests/ -v --tb=short

# Or use the batch scripts
.\scripts\fix-quality.bat
.\scripts\pre-push-check.bat
```

### Linux/Mac (with Makefile)
```bash
# Quick fix + check (recommended)
make qc

# Or full pre-push check
make pre-push

# Or just fix issues
make fix

# Or just check quality
make quality
```

---

## 📋 Pre-Push Checklist

Before every `git push`, run:

```bash
# 1. Fix formatting issues
python -m isort --profile black --line-length 100 cuda_healthcheck/ tests/
python -m black --line-length 100 cuda_healthcheck/ tests/

# 2. Verify quality
python -m black --check --line-length 100 cuda_healthcheck/ tests/
python -m isort --check-only --profile black --line-length 100 cuda_healthcheck/ tests/

# 3. Run tests
python -m pytest tests/ -v --tb=short
```

**Or just use:** `make qc` (if you have Makefile support)

---

## 🔧 Tools We Use

| Tool | Purpose | Command |
|------|---------|---------|
| **isort** | Sort imports | `python -m isort --profile black --line-length 100` |
| **Black** | Code formatting | `python -m black --line-length 100` |
| **Flake8** | Linting | `python -m flake8 --max-line-length 100` |
| **MyPy** | Type checking | `python -m mypy --ignore-missing-imports` |
| **Pytest** | Unit tests | `python -m pytest tests/ -v` |

---

## 🚫 Common Mistakes

### Mistake 1: Forgetting isort
```bash
❌ Only running Black
✅ Run isort FIRST, then Black
```

### Mistake 2: Pushing without local checks
```bash
❌ git add . && git commit -m "fix" && git push
✅ make qc && git add . && git commit -m "fix" && git push
```

### Mistake 3: Not reviewing auto-fixes
```bash
❌ make fix && git add . && git commit
✅ make fix && git diff && git add . && git commit
```

---

## 🎓 Workflow Examples

### Example 1: New Feature
```bash
# 1. Write code
vim cuda_healthcheck/new_feature.py

# 2. Quick fix
make fix

# 3. Review changes
git diff

# 4. Full check
make pre-push

# 5. Commit and push
git add .
git commit -m "feat: add new feature"
git push origin main
```

### Example 2: Bug Fix
```bash
# 1. Fix bug
vim cuda_healthcheck/detector.py

# 2. Quick check
make qc

# 3. If issues found, they're auto-fixed
git diff

# 4. Commit
git add .
git commit -m "fix: resolve detector bug"
git push origin main
```

### Example 3: CI/CD Failed (Recovery)
```bash
# GitHub Actions failed with formatting issue

# 1. Pull latest
git pull origin main

# 2. Auto-fix
make fix

# 3. Verify
make quality

# 4. Push fix
git add .
git commit -m "style: fix formatting issues"
git push origin main
```

---

## 📊 Time Comparison

### Without Local Checks
```
Write code (30 min)
Push to GitHub (1 min)
Wait for CI/CD (15 min)  ❌ FAIL
Fix formatting (5 min)
Push again (1 min)
Wait for CI/CD (15 min)  ❌ FAIL
Fix imports (5 min)
Push third time (1 min)
Wait for CI/CD (15 min)  ✅ PASS
━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: ~73 minutes, 3 pushes
```

### With Local Checks
```
Write code (30 min)
Run `make qc` (2 min)
Fix issues (2 min)
Run `make qc` again (2 min)  ✅ PASS
Push to GitHub (1 min)
Wait for CI/CD (15 min)  ✅ PASS
━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: ~52 minutes, 1 push
SAVED: 21 minutes! 🎉
```

---

## 🆘 Troubleshooting

### "isort: command not found"
```bash
pip install isort
```

### "black: command not found"
```bash
pip install black
```

### "make: command not found" (Windows)
```bash
# Use the .bat scripts instead
.\scripts\fix-quality.bat
.\scripts\pre-push-check.bat
```

### "Tests fail locally but pass in CI"
```bash
# Make sure you have the latest dependencies
pip install -r requirements.txt --upgrade
```

---

## 💡 Pro Tips

1. **Alias for quick checks:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   alias qc="make qc"
   alias pre-push="make pre-push"
   ```

2. **Git hook for automatic checks:**
   ```bash
   pre-commit install
   # Now checks run automatically on every commit
   ```

3. **IDE integration:**
   - VS Code: Install Black, isort, Flake8 extensions
   - PyCharm: Enable Black formatter in settings
   - Set format-on-save for automatic fixes

4. **CI/CD as final gate, not first check:**
   - Use local checks for fast feedback (2 min)
   - Use CI/CD for final validation (15 min)
   - Never rely on CI/CD to catch formatting issues

---

## 📚 Further Reading

- **Pre-commit hooks:** https://pre-commit.com/
- **Black formatter:** https://black.readthedocs.io/
- **isort:** https://pycqa.github.io/isort/
- **Flake8:** https://flake8.pycqa.org/

---

## 🎯 Remember

**2 minutes of local checks = 30 minutes of CI/CD time saved!**

Always run `make qc` before pushing! 🚀

