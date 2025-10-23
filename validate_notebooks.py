#!/usr/bin/env python3
"""
Notebook Static Analysis Tool
Validates that notebooks have correct variable definitions, import order, and cell dependencies.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class NotebookValidator:
    def __init__(self, notebook_path: str):
        self.notebook_path = notebook_path
        with open(notebook_path, 'r') as f:
            self.notebook = json.load(f)
        self.cells = self.notebook['cells']
        self.errors = []
        self.warnings = []
        
    def extract_variables(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Extract defined and used variables from code."""
        defined = set()
        used = set()
        
        # Find variable assignments (simple heuristic)
        # Matches: var = ..., var, var2 = ..., etc.
        assignment_pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s*='
        for match in re.finditer(assignment_pattern, code, re.MULTILINE):
            vars_str = match.group(1)
            for var in vars_str.split(','):
                var = var.strip()
                if var and not var.startswith('_'):
                    defined.add(var)
        
        # Find function definitions
        func_pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        for match in re.finditer(func_pattern, code, re.MULTILINE):
            defined.add(match.group(1))
        
        # Find class definitions
        class_pattern = r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'
        for match in re.finditer(class_pattern, code, re.MULTILINE):
            defined.add(match.group(1))
        
        # Find variable usage (simple heuristic - look for identifiers)
        # This is imperfect but catches most cases
        usage_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        for match in re.finditer(usage_pattern, code):
            var = match.group(1)
            # Exclude Python keywords and common builtins
            if var not in {'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 
                          'import', 'from', 'as', 'with', 'try', 'except', 'finally',
                          'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
                          'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
                          'set', 'tuple', 'open', 'enumerate', 'zip', 'sum', 'min', 'max'}:
                used.add(var)
        
        return defined, used
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements."""
        imports = []
        import_pattern = r'^\s*(?:from\s+[\w.]+\s+)?import\s+[\w.,\s*()]+(?:\s+as\s+\w+)?'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            imports.append(match.group(0).strip())
        return imports
    
    def validate_variable_order(self):
        """Check that variables are defined before use."""
        defined_vars = set()
        
        for idx, cell in enumerate(self.cells):
            if cell['cell_type'] != 'code':
                continue
            
            source = ''.join(cell['source'])
            if not source.strip():
                continue
            
            cell_defined, cell_used = self.extract_variables(source)
            
            # Check if any used variables haven't been defined yet
            undefined = cell_used - defined_vars - cell_defined
            
            # Filter out likely false positives
            undefined = {v for v in undefined if not (
                v.startswith('_') or  # Private vars
                v.isupper() or  # Constants
                len(v) <= 2  # Short vars like 'f', 'ax', etc.
            )}
            
            if undefined:
                self.errors.append(
                    f"Cell {idx}: Variables used before definition: {', '.join(sorted(undefined))}"
                )
            
            # Add defined variables to the set
            defined_vars.update(cell_defined)
    
    def validate_import_order(self):
        """Check that imports come before usage."""
        imports_seen = set()
        code_seen = False
        
        for idx, cell in enumerate(self.cells):
            if cell['cell_type'] != 'code':
                continue
            
            source = ''.join(cell['source'])
            if not source.strip():
                continue
            
            cell_imports = self.extract_imports(source)
            has_non_import = bool(re.search(r'^\s*[^#\n]', source, re.MULTILINE)) and not cell_imports
            
            if cell_imports:
                if code_seen and idx > 5:  # Allow imports in first few cells
                    self.warnings.append(
                        f"Cell {idx}: Import statements after code execution (consider moving to top)"
                    )
                imports_seen.update(cell_imports)
            
            if has_non_import and not cell_imports:
                code_seen = True
    
    def validate_model_definitions(self):
        """Check for common model definition patterns."""
        model_defined = None
        model_used = []
        
        for idx, cell in enumerate(self.cells):
            if cell['cell_type'] != 'code':
                continue
            
            source = ''.join(cell['source'])
            
            # Check if model is defined
            if re.search(r'\bmodel\s*[,=]', source) and \
               re.search(r'FastLanguageModel\.from_pretrained|AutoModel', source):
                if model_defined is None:
                    model_defined = idx
            
            # Check if model is used
            if re.search(r'\bmodel\.|\bmodel\(', source):
                model_used.append(idx)
        
        if model_defined is not None:
            for use_idx in model_used:
                if use_idx < model_defined:
                    self.errors.append(
                        f"Cell {use_idx}: 'model' used before definition in Cell {model_defined}"
                    )
    
    def validate_unsloth_import_order(self):
        """Check that Unsloth is imported first."""
        first_import_cell = None
        unsloth_import_cell = None
        
        for idx, cell in enumerate(self.cells):
            if cell['cell_type'] != 'code':
                continue
            
            source = ''.join(cell['source'])
            
            if 'from unsloth import' in source or 'import unsloth' in source:
                unsloth_import_cell = idx
            
            if 'import ' in source and first_import_cell is None:
                first_import_cell = idx
        
        if unsloth_import_cell is not None and first_import_cell is not None:
            if unsloth_import_cell > first_import_cell:
                self.warnings.append(
                    f"Cell {unsloth_import_cell}: Unsloth should be imported first "
                    f"(before Cell {first_import_cell})"
                )
    
    def validate_cell_numbering(self):
        """Check that cell numbers are sequential."""
        cell_numbers = []
        
        for idx, cell in enumerate(self.cells):
            if cell['cell_type'] != 'code':
                continue
            
            source = ''.join(cell['source'])
            
            # Look for cell numbers like "# 1️⃣" or "# Step 1:"
            match = re.search(r'#\s*([0-9])[️⃣]', source)
            if match:
                cell_numbers.append((idx, int(match.group(1))))
        
        # Check if numbers are sequential
        if cell_numbers:
            expected = 1
            for idx, num in cell_numbers:
                if num != expected and num != expected - 1:  # Allow some flexibility
                    self.warnings.append(
                        f"Cell {idx}: Cell number {num} out of sequence (expected ~{expected})"
                    )
                expected = num + 1
    
    def validate(self) -> bool:
        """Run all validations."""
        print(f"\n🔍 Validating: {Path(self.notebook_path).name}")
        print("=" * 70)
        
        self.validate_variable_order()
        self.validate_import_order()
        self.validate_model_definitions()
        self.validate_unsloth_import_order()
        self.validate_cell_numbering()
        
        # Print results
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
        
        return len(self.errors) == 0


def main():
    """Validate all notebooks in solution_notebooks and lab_notebooks."""
    
    print("=" * 70)
    print("📓 NOTEBOOK STATIC ANALYSIS TOOL")
    print("=" * 70)
    
    # Find all notebooks
    solution_notebooks = sorted(Path("solution_notebooks").glob("*.ipynb"))
    lab_notebooks = sorted(Path("lab_notebooks").glob("*.ipynb"))
    
    all_notebooks = list(solution_notebooks) + list(lab_notebooks)
    
    if not all_notebooks:
        print("❌ No notebooks found!")
        return 1
    
    print(f"\nFound {len(all_notebooks)} notebooks to validate")
    
    total_errors = 0
    total_warnings = 0
    failed_notebooks = []
    
    for notebook_path in all_notebooks:
        validator = NotebookValidator(str(notebook_path))
        passed = validator.validate()
        
        total_errors += len(validator.errors)
        total_warnings += len(validator.warnings)
        
        if not passed:
            failed_notebooks.append(notebook_path.name)
    
    # Summary
    print("\n" + "=" * 70)
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
    else:
        print(f"\n✅ All notebooks passed validation!")
        return 0


if __name__ == "__main__":
    sys.exit(main())



