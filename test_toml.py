import tomllib

# Test with dev key
test_toml = b'optional-dependencies = { "dev" = ["pytest"] }'
try:
    tomllib.loads(test_toml.decode())
    print('Test with "dev" key passed')
except Exception as e:
    print(f'Test with "dev" key failed: {e}')

# Test with different key
test_toml2 = b'optional-dependencies = { "development" = ["pytest"] }'
try:
    tomllib.loads(test_toml2.decode())
    print('Test with "development" key passed')
except Exception as e:
    print(f'Test with "development" key failed: {e}')

# Test with docs key
test_toml3 = b'optional-dependencies = { "docs" = ["sphinx"] }'
try:
    tomllib.loads(test_toml3.decode())
    print('Test with "docs" key passed')
except Exception as e:
    print(f'Test with "docs" key failed: {e}')

# Test with both
test_toml4 = b'optional-dependencies = { "dev" = ["pytest"], "docs" = ["sphinx"] }'
try:
    tomllib.loads(test_toml4.decode())
    print('Test with both keys passed')
except Exception as e:
    print(f'Test with both keys failed: {e}')

# Test the actual file
with open('pyproject.toml', 'rb') as f:
    content = f.read()
try:
    tomllib.loads(content.decode())
    print('Full file passed')
except Exception as e:
    print(f'Full file failed: {e}')
