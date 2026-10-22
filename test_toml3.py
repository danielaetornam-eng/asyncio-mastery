import tomllib

# Test the actual file - extract just the optional-dependencies section
with open('pyproject.toml', 'rb') as f:
    content = f.read()

# Find the optional-dependencies section
idx = content.find(b'optional-dependencies')
section = content[idx:idx+500]
print("Section bytes:")
for i, b in enumerate(section[:100]):
    print(f"  {i}: {b} ({chr(b) if 32 <= b < 127 else '?'})")
print()

# Check first line
first_line_end = section.find(b'\n')
first_line = section[:first_line_end]
print(f"First line: {first_line!r}")
print(f"First line length: {len(first_line)}")

# Try parsing just this section
try:
    tomllib.loads(section.decode())
    print('Section passed')
except Exception as e:
    print(f'Section failed: {e}')

# Try parsing from the beginning of the file
try:
    tomllib.loads(content.decode())
    print('Full file passed')
except Exception as e:
    print(f'Full file failed: {e}')
