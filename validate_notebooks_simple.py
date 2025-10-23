#!/usr/bin/env python3
"""
Simple Notebook Validator - Checks for common issues without false positives.
"""

import json
import re
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate a single notebook for common issues."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    errors = []
    warnings = []
    cells = nb['cells']
    
    # Track important variables
    model_defined_at = None
    teacher_model_defined_at = None
    student_model_defined_at = None
    tokenizer_defined_at = None
    unsloth_imported_at = None
    first_import_at = None
    
    for idx, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        if not source.strip():
            continue
        
        # Check for model definitions
        if re.search(r'\bmodel\s*[,=].*FastLanguageModel\.from_pretrained', source):
            if model_defined_at is None:
                model_defined_at = idx
        
        if re.search(r'\bteacher_model\s*[,=].*FastLanguageModel\.from_pretrained', source):
            teacher_model_defined_at = idx
        
        if re.search(r'\bstudent_model\s*[,=].*FastLanguageModel\.from_pretrained', source):
            student_model_defined_at = idx
        
        if re.search(r'\btokenizer\s*=.*from_pretrained', source):
            tokenizer_defined_at = idx
        
        # Check for model usage before definition
        if re.search(r'\bmodel\.(eval|train|generate|forward|parameters|to)\(', source):
            if model_defined_at is not None and idx < model_defined_at:
                errors.append(f"Cell {idx}: 'model' used before definition in Cell {model_defined_at}")
        
        if re.search(r'\bteacher_model\.(eval|train|generate|forward|parameters|to)\(', source):
            if teacher_model_defined_at is not None and idx < teacher_model_defined_at:
                errors.append(f"Cell {idx}: 'teacher_model' used before definition in Cell {teacher_model_defined_at}")
        
        if re.search(r'\bstudent_model\.(eval|train|generate|forward|parameters|to)\(', source):
            if student_model_defined_at is not None and idx < student_model_defined_at:
                errors.append(f"Cell {idx}: 'student_model' used before definition in Cell {student_model_defined_at}")
        
        if re.search(r'\btokenizer\.(encode|decode|from_pretrained)\(', source):
            if tokenizer_defined_at is not None and idx < tokenizer_defined_at:
                errors.append(f"Cell {idx}: 'tokenizer' used before definition in Cell {tokenizer_defined_at}")
        
        # Check Unsloth import order
        if 'from unsloth import' in source or 'import unsloth' in source:
            unsloth_imported_at = idx
        
        if 'import ' in source and first_import_at is None and 'unsloth' not in source:
            first_import_at = idx
    
    # Validate Unsloth import order
    if unsloth_imported_at is not None and first_import_at is not None:
        if unsloth_imported_at > first_import_at:
            warnings.append(
                f"Unsloth imported at Cell {unsloth_imported_at} but other imports at Cell {first_import_at}. "
                f"Unsloth should be imported first to avoid initialization conflicts."
            )
    
    return errors, warnings

def main():
    print("=" * 70)
    print("📓 SIMPLE NOTEBOOK VALIDATOR")
    print("=" * 70)
    
    solution_notebooks = sorted(Path("solution_notebooks").glob("*.ipynb"))
    lab_notebooks = sorted(Path("lab_notebooks").glob("*.ipynb"))
    
    all_notebooks = list(solution_notebooks) + list(lab_notebooks)
    
    if not all_notebooks:
        print("❌ No notebooks found!")
        return 1
    
    print(f"\nValidating {len(all_notebooks)} notebooks...\n")
    
    total_errors = 0
    total_warnings = 0
    failed_notebooks = []
    
    for notebook_path in all_notebooks:
        errors, warnings = validate_notebook(str(notebook_path))
        
        if errors or warnings:
            print(f"🔍 {notebook_path.name}")
            print("-" * 70)
            
            if errors:
                print(f"  ❌ ERRORS ({len(errors)}):")
                for error in errors:
                    print(f"     • {error}")
                total_errors += len(errors)
                failed_notebooks.append(notebook_path.name)
            
            if warnings:
                print(f"  ⚠️  WARNINGS ({len(warnings)}):")
                for warning in warnings:
                    print(f"     • {warning}")
                total_warnings += len(warnings)
            
            print()
    
    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total notebooks: {len(all_notebooks)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    
    if failed_notebooks:
        print(f"\n❌ Notebooks with errors:")
        for name in failed_notebooks:
            print(f"  • {name}")
        return 1
    elif total_warnings > 0:
        print(f"\n⚠️  All notebooks passed but have {total_warnings} warnings")
        return 0
    else:
        print(f"\n✅ All notebooks passed validation!")
        return 0

if __name__ == "__main__":
    sys.exit(main())



