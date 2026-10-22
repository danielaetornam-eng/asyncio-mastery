with open('pyproject.toml', 'rb') as f:
    content = f.read()
idx = content.find(b'per-file-ignores')
section = content[idx:idx+100]
for i, b in enumerate(section):
    print(f'  {i}: {b} ({chr(b) if 32 <= b < 127 else "?"})')
