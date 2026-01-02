# Compatibility Matrix Testing Workflow

## Overview

Automated testing workflow that validates all 9 combinations of Databricks Runtime versions and PyTorch CUDA variants to ensure the CUDA Healthcheck Tool correctly detects compatibility issues.

---

## Test Matrix

### Combinations Tested

| Runtime | Driver | CUDA | cu120 | cu121 | cu124 |
|---------|--------|------|-------|-------|-------|
| 14.3    | 535    | 12.0 | ✅    | ✅    | ❌    |
| 15.1    | 550    | 12.4 | ✅    | ✅    | ✅    |
| 15.2    | 550    | 12.4 | ✅    | ✅    | ✅    |

**Total:** 9 combinations (8 compatible + 1 known incompatibility)

---

## Workflow Features

### 1. **Matrix Strategy**
Tests all 9 combinations in parallel using GitHub Actions matrix strategy:
- 3 Databricks runtimes: 14.3, 15.1, 15.2
- 3 PyTorch CUDA variants: cu120, cu121, cu124

### 2. **Mock Environment**
Each test:
- Simulates the Databricks runtime environment
- Creates mock `pip freeze` output
- Sets appropriate environment variables
- Runs the healthcheck tool

### 3. **Expected Behavior Validation**
- **8 combinations** should PASS (compatible)
- **1 combination** should FAIL (Runtime 14.3 + cu124 - known incompatibility)
- Test validates that tool correctly identifies each scenario

### 4. **Automated Reporting**
- Generates comprehensive compatibility matrix report
- Shows test results in GitHub Actions summary
- Comments on PRs with detailed results
- Stores artifacts for 90 days

### 5. **Known Incompatibility Detection**
Validates that the tool correctly flags **Runtime 14.3 + cu124**:
- **Root cause:** Driver 535 < required 550
- **Severity:** BLOCKER
- **Fix options:** Downgrade PyTorch OR upgrade runtime

---

## Workflow Jobs

### Job 1: `compatibility-matrix`
Runs 9 parallel test jobs (one for each combination)

**Steps:**
1. Checkout code
2. Set up Python 3.11
3. Install CUDA Healthcheck Tool
4. Create mock pip freeze output
5. Generate test script
6. Run compatibility validation
7. Upload test artifacts

**Output Artifacts:**
- `test-results-{runtime}-{variant}/` for each combination
  - `mock_pip_freeze_*.txt` - Mock pip output
  - `test_compatibility_*.py` - Test script
  - `output_*.log` - Test execution log
  - `result_*.txt` - PASS or FAIL

### Job 2: `generate-report`
Collects results and generates comprehensive report

**Steps:**
1. Download all test artifacts
2. Generate compatibility matrix markdown
3. Include detailed logs for each combination
4. Upload report artifact
5. Comment on PR (if applicable)
6. Add to workflow summary

**Output:**
- `compatibility-report/` artifact
  - `compatibility_matrix.md` - Full report

### Job 3: `test-summary`
Provides high-level summary

**Steps:**
1. Check matrix results
2. Add summary to workflow output

---

## Test Script Logic

For each combination, the test script:

```python
def test_compatibility():
    # 1. Parse mock pip freeze output
    packages = parse_cuda_packages(pip_output)
    torch_cuda_branch = packages.get("torch_cuda_branch")
    
    # 2. Validate compatibility
    result = validate_torch_branch_compatibility(
        runtime_version=float(runtime),
        torch_cuda_branch=torch_cuda_branch
    )
    
    # 3. Check expected vs actual behavior
    expected_pass = (expected_result == "pass")
    actual_pass = result["is_compatible"]
    
    # 4. Return success if behavior matches expectation
    return 0 if (expected_pass == actual_pass) else 1
```

---

## Expected Results

### Compatible Combinations (8)

#### Runtime 14.3 (Driver 535, CUDA 12.0)
- ✅ **cu120**: Compatible - Driver 535 supports CUDA 12.0
- ✅ **cu121**: Compatible - Driver 535 supports CUDA 12.1
- ❌ **cu124**: **INCOMPATIBLE** - Driver 535 < 550 (required by cu124)

#### Runtime 15.1 (Driver 550, CUDA 12.4)
- ✅ **cu120**: Compatible - Backward compatible
- ✅ **cu121**: Compatible - Backward compatible
- ✅ **cu124**: Compatible - Driver 550 supports CUDA 12.4

#### Runtime 15.2 (Driver 550, CUDA 12.4)
- ✅ **cu120**: Compatible - Backward compatible
- ✅ **cu121**: Compatible - Backward compatible
- ✅ **cu124**: Compatible - Driver 550 supports CUDA 12.4

### Known Incompatibility (1)

**Runtime 14.3 + torch cu124:**
- **Status:** ❌ INCOMPATIBLE (as expected)
- **Detection:** Tool should return `is_compatible: False` and `severity: BLOCKER`
- **Root Cause:** Driver 535 (immutable on Runtime 14.3) < 550 (required by PyTorch cu124)
- **Fix Options:**
  1. Downgrade to cu120: `pip install torch --index-url https://download.pytorch.org/whl/cu120`
  2. Upgrade runtime to 15.1+

---

## Mock Data Example

### cu124 Mock Pip Freeze
```
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
numpy==1.26.4
pandas==2.0.0
```

### cu120 Mock Pip Freeze
```
torch==2.4.1+cu120
nvidia-cublas-cu12==12.1.3.1
nvidia-nvjitlink-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
numpy==1.26.4
pandas==2.0.0
```

---

## Workflow Triggers

The workflow runs automatically on:
- ✅ **Push to main or develop** - Validates changes don't break compatibility detection
- ✅ **Pull requests** - Reports compatibility matrix in PR comments
- ✅ **Manual trigger** - Can be run on-demand via `workflow_dispatch`

---

## PR Comment Example

When the workflow runs on a PR, it posts a comment like:

```markdown
# 🧪 Compatibility Matrix Test Results

## Test Configuration
**Workflow Run:** 123  
**Commit:** abc123def  
**Branch:** feature/new-detection

---

## Compatibility Matrix

| Runtime | Driver | CUDA | cu120 | cu121 | cu124 |
|---------|--------|------|-------|-------|-------|
| 14.3    | 535    | 12.0 | ✅    | ✅    | ❌    |
| 15.1    | 550    | 12.4 | ✅    | ✅    | ✅    |
| 15.2    | 550    | 12.4 | ✅    | ✅    | ✅    |

**Legend:**
- ✅ Compatible (as expected)
- ❌ Incompatible (as expected) or Test failed unexpectedly
- ⚠️ Test did not run or result unavailable

---

[Full detailed results in workflow run...]
```

---

## Viewing Results

### GitHub Actions UI
1. Go to **Actions** tab
2. Click on **Compatibility Matrix Testing** workflow
3. View the latest run
4. Check the summary for the compatibility matrix
5. Download artifacts for detailed logs

### Artifacts
- **compatibility-report** (90 days retention)
  - Contains full markdown report
- **test-results-{runtime}-{variant}** (30 days retention)
  - Contains logs and results for each combination

### PR Comments
- Automatically posted on pull requests
- Shows compatibility matrix at a glance
- Links to full workflow run

---

## Interpreting Results

### All Green (✅)
- Tool correctly detects all compatible combinations
- Tool correctly flags Runtime 14.3 + cu124 as incompatible
- **Status:** PASS

### Any Red (❌) Unexpected
- Tool failed to detect an incompatibility
- Tool incorrectly flagged a compatible combination
- **Status:** FAIL - Investigation required

### Runtime 14.3 + cu124 Specific
- **Expected:** ❌ (BLOCKER)
- **Validation:** Tool should return `is_compatible: False`
- **Validation:** Tool should provide 2 fix options

---

## Integration with CI/CD

This workflow complements the existing test suite:
- **Unit tests** (40 tests) - Test individual functions
- **Compatibility matrix** (9 tests) - Test real-world scenarios
- **Code quality** - Lint, format, type check

**Combined Coverage:**
- ✅ Unit-level validation
- ✅ Integration-level validation
- ✅ Compatibility-level validation

---

## Maintenance

### Adding New Runtimes
To add Runtime 16.0:

```yaml
matrix:
  runtime: ["14.3", "15.1", "15.2", "16.0"]  # Add here
  # ...
  include:
    # Add expected behavior for all 3 combinations
    - runtime: "16.0"
      torch_variant: "cu120"
      expected_result: "pass"
      driver: "560"
      cuda: "12.6"
    # ... (add cu121, cu124)
```

### Adding New CUDA Variants
To add cu130:

```yaml
matrix:
  torch_variant: ["cu120", "cu121", "cu124", "cu130"]  # Add here
  # Update expected behavior matrix
```

---

## Troubleshooting

### Test Fails Unexpectedly
1. Check the detailed log in artifacts
2. Verify mock pip freeze output is correct
3. Confirm compatibility detection logic
4. Check if runtime mappings are up to date

### Workflow Doesn't Run
1. Check workflow file syntax (YAML)
2. Verify branch triggers
3. Check GitHub Actions permissions

### PR Comment Not Posted
1. Verify `permissions: pull-requests: write`
2. Check GITHUB_TOKEN has correct permissions
3. Look for errors in generate-report job

---

## Benefits

✅ **Automated Validation** - No manual testing of all 9 combinations  
✅ **Regression Detection** - Catches breaking changes immediately  
✅ **Documentation** - Compatibility matrix is always up-to-date  
✅ **CI/CD Integration** - Runs on every push and PR  
✅ **Comprehensive Reporting** - Detailed logs for troubleshooting  
✅ **PR Transparency** - Results visible directly in pull requests  

---

## Files

- **Workflow:** `.github/workflows/compatibility-matrix.yml` (417 lines)
- **Documentation:** `docs/COMPATIBILITY_MATRIX_TESTING.md` (this file)

---

## Version

- **Workflow Version:** 1.0
- **CUDA Healthcheck Tool:** 0.5.0+
- **GitHub Actions:** ubuntu-latest, Python 3.11

---

## See Also

- [Main Test Suite](../tests/test_cuda_version_mismatch_detection.py) - 40 unit tests
- [Code Quality Workflow](../.github/workflows/code-quality.yml) - Linting, formatting
- [Databricks Runtime Detection](./DATABRICKS_RUNTIME_DETECTION.md) - Runtime detection docs
- [Driver Version Mapping](./DRIVER_VERSION_MAPPING.md) - Driver compatibility docs

