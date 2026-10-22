import os
import re
from pathlib import Path

# This script fixes remaining STE violations in docstrings more aggressively
# It focuses on common patterns: helping verbs as main verbs, passive voice, gerunds, long sentences

# Patterns to identify STE violations
PASSIVE_PATTERNS = [
    (r'(\w+) is (\w+ed)\b', r'\2'),  # "X is Y-ed" → "X"
    (r'(\w+) was (\w+ed)\b', r'\2'),  # "X was Y-ed" → "X"
    (r'(\w+) are (\w+ed)\b', r'\2'),
    (r'(\w+) were (\w+ed)\b', r'\2'),
]

# Fixing helper function to process each file
def fix_ste_violations_in_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    original = content

    # More targeted fixes for common STE violations
    # Fix "this is" / "it is" / "he is" / "she is" / "they are" constructions
    content = re.sub(r'\bThis is the\b', 'This is the', content)  # This is actually fine in context

    # Fix specific known problematic patterns in docstrings
    replacements = [
        # Helping verbs as main verbs
        (r'is the (best|worst|fastest|slowest)', r'is the \1'),  # These are ok in context
        # Fix passive voice in docstrings (docstring text, not code)
        (r'This is the primary teaching function for the core concept',
         'This tutorial teaches the core concept'),
        (r'This is the same as a sync function',
         'This behaves the same as a sync function'),
        (r'This is a coroutine',
         'This is a coroutine'),  # Actually fine
        (r'A staticmethod is a coroutine',
         'A staticmethod defines a coroutine'),
        (r'A classmethod is a coroutine',
         'A classmethod defines a coroutine'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed complex STE issues in: {filepath}")

# Process all Python files
py_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            py_files.append(str(Path(root) / file))

print(f"Processing {len(py_files)} Python files")

for filepath in py_files:
    fix_ste_violations_in_file(filepath)

print("Done fixing complex STE violations")
