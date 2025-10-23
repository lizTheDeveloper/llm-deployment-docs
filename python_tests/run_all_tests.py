#!/usr/bin/env python3
"""
Run all lab tests in sequence
"""

import subprocess
import sys
import time

LABS = [
    ("Lab 1: Keras Quick Refresher", "lab1_keras_refresher.py"),
    ("Lab 2: GradientTape Refresher", "lab2_gradient_tape.py"),
    ("Lab 3: Hello LLM/Unsloth", "lab3_hello_unsloth.py"),
    ("Lab 4: Knowledge Distillation", "lab4_distillation_simple.py"),
    ("Lab 5: Model Pruning", "lab5_pruning_simple.py"),
    ("Lab 6: Model Quantization", "lab6_quantization_simple.py"),
    ("Lab 7: FastAPI Deployment", "lab7_fastapi_deployment.py"),
    ("Lab 8: FastAPI Tool Calling", "lab8_fastapi_tool_calling.py"),
]

def run_lab(name, script):
    """Run a single lab script"""
    print("\n" + "=" * 70)
    print(f"Running: {name}")
    print("=" * 70)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python", script],
            cwd="/Users/annhoward/src/Mastering_LLM_Deployment/python_tests",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {name} PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"✗ {name} FAILED ({elapsed:.1f}s)")
            print("\nSTDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars
            print("\nSTDERR:")
            print(result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {name} TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"✗ {name} ERROR: {e}")
        return False

def main():
    print("=" * 70)
    print("LLM DEPLOYMENT LABS - TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {len(LABS)} lab tests...\n")
    
    results = {}
    start_time = time.time()
    
    for name, script in LABS:
        success = run_lab(name, script)
        results[name] = success
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\n{passed}/{len(results)} tests passed")
    print(f"Total time: {total_time:.1f}s\n")
    
    if failed > 0:
        print(f"⚠️  {failed} test(s) failed")
        return 1
    else:
        print("🎉 All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

