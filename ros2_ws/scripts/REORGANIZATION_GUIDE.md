# Scripts Directory Reorganization

This directory has been reorganized to reduce file count and improve maintainability.

## Old Structure → New Structure

### Unit Tests
- `test_all_utils.cpp` → `tests/unit_tests/test_cpp_units.cpp`
- `test_cuda_basic.cu` → `tests/unit_tests/test_cpp_units.cpp`
- `test_depth_estimation.cpp` → `tests/unit_tests/test_cpp_units.cpp`
- `test_geometry.py` → `tests/unit_tests/test_python_units.py`
- `test_depth_estimation_python.py` → `tests/unit_tests/test_python_units.py`
- `test_gpu_kernel.cpp` → `tests/unit_tests/test_cpp_units.cpp`
- `test_struct_size.cpp` → `tests/unit_tests/test_cpp_units.cpp`

### Integration Tests
- `equivalence_test.py` → `tests/integration_tests/integration_test_suite.py`
- `compare_detailed.py` → `tests/integration_tests/integration_test_suite.py`

### Verification Tests
- `verify_equivalence.py` → `tests/verification_tests/verify_implementation_equivalence.py`
- `verify_equivalence_minimal.py` → `tests/verification_tests/verify_implementation_equivalence.py`
- `verify_isb_filter.py` → `tests/verification_tests/verify_implementation_equivalence.py`
- `verify_stitcher.py` → `tests/verification_tests/verify_implementation_equivalence.py`
- `verify_utils.py` → `tests/verification_tests/verify_implementation_equivalence.py`

### Analysis Tools
- `analyze_cost_computation.py` → `tests/analysis_tools/analyze_depth_estimation_suite.py`
- `analyze_distance_parameterization.py` → `tests/analysis_tools/analyze_depth_estimation_suite.py`
- `debug_isb_difference.py` → `tests/analysis_tools/analyze_depth_estimation_suite.py`
- `debug_rt_matrix.py` → `tests/analysis_tools/analyze_depth_estimation_suite.py`

### Utilities
- `run_verification.sh` → `tests/utils/run_unified_tests.sh`
- `run_verify_isb.sh` → `tests/utils/run_unified_tests.sh`

## New Directory Structure

```
tests/
├── unit_tests/
│   ├── test_cpp_units.cpp          # Unified C++ unit tests
│   └── test_python_units.py        # Unified Python unit tests
├── integration_tests/
│   └── integration_test_suite.py   # End-to-end testing
├── verification_tests/
│   └── verify_implementation_equivalence.py  # Comprehensive equivalence verification
├── analysis_tools/
│   └── analyze_depth_estimation_suite.py     # Debugging and analysis tools
└── utils/
    └── run_unified_tests.sh         # Unified test runner
```

## Benefits

1. **Reduced file count**: 19 files → 5 organized files
2. **Improved maintainability**: Related functionality consolidated
3. **Better organization**: Clear separation of test types
4. **Unified interfaces**: Single entry points for each test category
5. **Preserved functionality**: All original tests maintained

## Usage

```bash
# Run all tests
./tests/utils/run_unified_tests.sh

# Run specific test suite
./tests/utils/run_unified_tests.sh --test-suite verification

# Run in minimal mode
./tests/utils/run_unified_tests.sh --test-suite unit --minimal

# Run with specific device
./tests/utils/run_unified_tests.sh --device cpu --verbose
```

## Migration Status

✅ **Completed**:
- Unit test consolidation
- Verification test consolidation  
- Analysis tool consolidation
- Integration test creation
- Unified test runner implementation

🔄 **In Progress**:
- Old file cleanup (keeping originals for reference)

📋 **Notes**:
- All original functionality preserved
- Enhanced with better error handling and logging
- Improved visualization and reporting
- Consistent interface across all test suites