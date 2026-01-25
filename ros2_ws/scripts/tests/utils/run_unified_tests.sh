#!/usr/bin/env bash
"""
run_unified_tests.sh

Unified Test Runner
===========================================================================
Combines and consolidates:
- run_verification.sh: Main verification test execution
- run_verify_isb.sh: ISB filter specific verification
===========================================================================
"""

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # This is now the tests/ directory

echo -e "${BLUE}=========================================================================${NC}"
echo -e "${BLUE}                           Unified Test Runner                            ${NC}"
echo -e "${BLUE}=========================================================================${NC}"
echo -e "${YELLOW}File Consolidation Summary:${NC}"
echo -e "  ${GREEN}19 original test files${NC} → ${GREEN}5 consolidated test files${NC} (74% reduction)"
echo
echo -e "${YELLOW}Consolidated Test Files:${NC}"
echo -e "  1. ${GREEN}test_python_units.py${NC}       - Unifies 3 Python test files"
echo -e "     • test_geometry.py"
echo -e "     • test_depth_estimation_python.py"
echo -e "     • verify_utils.py"
echo
echo -e "  2. ${GREEN}test_cpp_units.cpp${NC}         - Unifies 5 C++/CUDA test files"
echo -e "     • test_coordinate_transform.cpp"
echo -e "     • test_cost_computation.cpp"
echo -e "     • test_distance_parameterization.cpp"
echo -e "     • test_isb_filter.cpp"
echo -e "     • test_rgbd_estimator.cpp"
echo
echo -e "  3. ${GREEN}integration_test_suite.py${NC} - Unifies 2 integration tests"
echo -e "     • equivalence_test.py"
echo -e "     • compare_detailed.py"
echo
echo -e "  4. ${GREEN}verify_implementation_equivalence.py${NC} - Unifies 5 verification scripts"
echo -e "     • verify_equivalence.py"
echo -e "     • verify_equivalence_minimal.py"
echo -e "     • verify_isb_filter.py"
echo -e "     • verify_stitcher.py"
echo -e "     • verify_utils.py"
echo
echo -e "  5. ${GREEN}analyze_depth_estimation_suite.py${NC} - Unifies 4 analysis tools"
echo -e "     • analyze_distance_parameterization.py"
echo -e "     • analyze_cost_computation.py"
echo -e "     • debug_rt_matrix.py"
echo -e "     • debug_isb_difference.py"
echo
echo -e "Script directory: $SCRIPT_DIR"
echo -e "Project root: $PROJECT_ROOT"
echo

# Default configuration
DEVICE="cuda:0"
TEST_SUITE="all"
MINIMAL_MODE=false
OUTPUT_DIR="$PROJECT_ROOT/test_results"
VERBOSE=false

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --test-suite SUITE    Test suite to run: all, unit, integration, verification, analysis (default: all)
    -d, --device DEVICE       Device to use: cuda:0, cpu (default: cuda:0)
    -m, --minimal             Run in minimal mode (smaller resolution, faster)
    -o, --output-dir DIR      Output directory for results (default: $OUTPUT_DIR)
    -v, --verbose             Enable verbose output
    -h, --help                Show this help message

Test Suites:
    all           - Run all tests (unit, integration, verification, analysis)
    unit          - Run unit tests only
    integration   - Run integration tests only
    verification  - Run implementation equivalence verification
    analysis      - Run depth estimation analysis suite
    
Examples:
    $0                              # Run all tests with default settings
    $0 -t verification -m           # Run verification tests in minimal mode
    $0 -t analysis -d cpu -v        # Run analysis on CPU with verbose output
    $0 -t unit -o ./my_results      # Run unit tests, save to custom directory
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -m|--minimal)
            MINIMAL_MODE=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate test suite
if [[ ! "$TEST_SUITE" =~ ^(all|unit|integration|verification|analysis)$ ]]; then
    echo -e "${RED}Error: Invalid test suite '$TEST_SUITE'${NC}"
    echo -e "${RED}Valid options: all, unit, integration, verification, analysis${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"

# Set up environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/install/my_stereo_pkg/lib:$LD_LIBRARY_PATH"

# Log file
LOG_FILE="$OUTPUT_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Test suite: $TEST_SUITE"
echo -e "  Device: $DEVICE"
echo -e "  Minimal mode: $MINIMAL_MODE"
echo -e "  Output directory: $OUTPUT_DIR"
echo -e "  Log file: $LOG_FILE"
echo -e "  Verbose: $VERBOSE"
echo

# Function to run a test and capture output
run_test() {
    local test_name="$1"
    local test_command="$2"
    local start_time=$(date +%s)
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}Command: $test_command${NC}"
    fi
    
    # Run the test and capture output
    if eval "$test_command" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✓ $test_name PASSED (${duration}s)${NC}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${RED}❌ $test_name FAILED (${duration}s)${NC}"
        return 1
    fi
}

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Unit Tests
run_unit_tests() {
    echo -e "\n${BLUE}=== Running Unit Tests ===${NC}"
    
    # C++ unit tests
    if [[ -f "$PROJECT_ROOT/unit_tests/test_cpp_units" ]]; then
        run_test "C++ Unit Tests" "$PROJECT_ROOT/unit_tests/test_cpp_units"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if [[ $? -eq 0 ]]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${YELLOW}C++ unit tests not found, skipping...${NC}"
    fi
    
    # Python unit tests
    if [[ -f "$PROJECT_ROOT/unit_tests/test_python_units.py" ]]; then
        run_test "Python Unit Tests" "python3 $PROJECT_ROOT/unit_tests/test_python_units.py"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if [[ $? -eq 0 ]]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${YELLOW}Python unit tests not found, skipping...${NC}"
    fi
}

# Integration Tests
run_integration_tests() {
    echo -e "\n${BLUE}=== Running Integration Tests ===${NC}"
    
    # Find all integration test files
    if [[ -d "$PROJECT_ROOT/integration_tests" ]]; then
        for test_file in "$PROJECT_ROOT/integration_tests"/*.py; do
            if [[ -f "$test_file" ]]; then
                test_name=$(basename "$test_file" .py)
                run_test "Integration Test: $test_name" "python3 $test_file --device $DEVICE"
                TOTAL_TESTS=$((TOTAL_TESTS + 1))
                if [[ $? -eq 0 ]]; then
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
            fi
        done
    else
        echo -e "${YELLOW}Integration tests directory not found, skipping...${NC}"
    fi
}

# Verification Tests
run_verification_tests() {
    echo -e "\n${BLUE}=== Running Verification Tests ===${NC}"
    
    # Implementation equivalence verification
    verify_file="$PROJECT_ROOT/verification_tests/verify_implementation_equivalence.py"
    if [[ -f "$verify_file" ]]; then
        verify_cmd="python3 $verify_file --device $DEVICE"
        if [[ "$MINIMAL_MODE" == "true" ]]; then
            verify_cmd="$verify_cmd --minimal"
        fi
        
        run_test "Implementation Equivalence Verification" "$verify_cmd"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if [[ $? -eq 0 ]]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${YELLOW}Verification tests not found, skipping...${NC}"
    fi
}

# Analysis Tests
run_analysis_tests() {
    echo -e "\n${BLUE}=== Running Analysis Tests ===${NC}"
    
    # Depth estimation analysis suite
    analysis_file="$PROJECT_ROOT/analysis_tools/analyze_depth_estimation_suite.py"
    if [[ -f "$analysis_file" ]]; then
        run_test "Depth Estimation Analysis Suite" "python3 $analysis_file --device $DEVICE --output-dir $OUTPUT_DIR"
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if [[ $? -eq 0 ]]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        echo -e "${YELLOW}Analysis tools not found, skipping...${NC}"
    fi
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    echo -e "${BLUE}Starting test execution...${NC}" | tee -a "$LOG_FILE"
    echo -e "Start time: $(date)" | tee -a "$LOG_FILE"
    echo | tee -a "$LOG_FILE"
    
    # Run requested test suites
    case $TEST_SUITE in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "verification")
            run_verification_tests
            ;;
        "analysis")
            run_analysis_tests
            ;;
        "all")
            run_unit_tests
            run_integration_tests
            run_verification_tests
            run_analysis_tests
            ;;
    esac
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Summary
    echo -e "\n${BLUE}=========================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}                              Test Summary                               ${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}=========================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "End time: $(date)" | tee -a "$LOG_FILE"
    echo -e "Total duration: ${total_duration}s" | tee -a "$LOG_FILE"
    echo -e "Total tests: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}" | tee -a "$LOG_FILE"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}" | tee -a "$LOG_FILE"
    
    if [[ $TOTAL_TESTS -eq 0 ]]; then
        echo -e "${YELLOW}No tests were executed${NC}" | tee -a "$LOG_FILE"
        exit 1
    elif [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "${GREEN}All tests PASSED! ✓${NC}" | tee -a "$LOG_FILE"
        exit 0
    else
        echo -e "${RED}Some tests FAILED! ❌${NC}" | tee -a "$LOG_FILE"
        echo -e "Success rate: $((100 * PASSED_TESTS / TOTAL_TESTS))%" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Run main function
main