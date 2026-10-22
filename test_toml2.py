import tomllib

# Test the actual file - extract just the optional-dependencies section
with open('pyproject.toml', 'rb') as f:
    content = f.read()

# Find the optional-dependencies section
idx = content.find(b'optional-dependencies')
section = content[idx:idx+500]
print("Section bytes:")
print(repr(section))
print()

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
    # Find the exact position
    import traceback
    traceback.print_exc()
