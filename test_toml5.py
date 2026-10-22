import tomllib

# Test regular table format (not inline)
test_toml = b'''optional-dependencies.dev = ["pytest>=7.4.0"]
optional-dependencies.docs = ["sphinx>=7.2.0"]'''
try:
    tomllib.loads(test_toml.decode())
    print('Regular table format passed')
except Exception as e:
    print(f'Regular table format failed: {e}')

# Test with array of tables
test_toml2 = b'''[optional-dependencies]
dev = ["pytest>=7.4.0"]
docs = ["sphinx>=7.2.0"]'''
try:
    tomllib.loads(test_toml2.decode())
    print('Array of tables format passed')
except Exception as e:
    print(f'Array of tables format failed: {e}')

# Test the actual file
with open('pyproject.toml', 'rb') as f:
    content = f.read()
try:
    tomllib.loads(content.decode())
    print('Full file passed')
except Exception as e:
    print(f'Full file failed: {e}')
