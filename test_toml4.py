import tomllib

# Test inline table on single line
test_toml = b'optional-dependencies = { "dev" = ["pytest"], "docs" = ["sphinx"] }'
try:
    tomllib.loads(test_toml.decode())
    print('Single line inline table passed')
except Exception as e:
    print(f'Single line inline table failed: {e}')

# Test multi-line inline table
test_toml2 = b'''optional-dependencies = {
    "dev" = ["pytest"],
    "docs" = ["sphinx"],
}'''
try:
    tomllib.loads(test_toml2.decode())
    print('Multi-line inline table passed')
except Exception as e:
    print(f'Multi-line inline table failed: {e}')

# Test with version strings
test_toml3 = b'''optional-dependencies = {
    "dev" = ["pytest>=7.4.0"],
    "docs" = ["sphinx>=7.2.0"],
}'''
try:
    tomllib.loads(test_toml3.decode())
    print('Multi-line with versions passed')
except Exception as e:
    print(f'Multi-line with versions failed: {e}')

# Test the actual file
with open('pyproject.toml', 'rb') as f:
    content = f.read()
try:
    tomllib.loads(content.decode())
    print('Full file passed')
except Exception as e:
    print(f'Full file failed: {e}')
