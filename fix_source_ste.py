import os
import re
from pathlib import Path

# Common STE violations and their replacements
contractions = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "needn't": "need not",
    "mightn't": "might not",
    "shan't": "shall not",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "how's": "how is",
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "let's": "let us",
}

unapproved_terms = {
    "comprehensive": "complete",
    "extensive": "broad",
    "utilize": "use",
    "facilitate": "help",
    "enable": "allow",
    "ensure": "make sure",
    "leverage": "use",
    "optimize": "improve",
    "streamline": "simplify",
    "robust": "strong",
    "seamless": "smooth",
    "intuitive": "easy",
    "user-friendly": "easy to use",
    "state-of-the-art": "modern",
    "cutting-edge": "modern",
    "best-in-class": "best",
    "enterprise-grade": "production",
    "production-ready": "production",
    "god-tier": "advanced",
    "ultimate": "complete",
    "heavily": "fully",
    "highly": "very",
    "significantly": "much",
    "substantially": "much",
    "considerably": "much",
    "dramatically": "much",
    "exponentially": "rapidly",
    "paradigm": "model",
    "synergy": "cooperation",
    "holistic": "complete",
    "granular": "detailed",
    "actionable": "usable",
    "deliverable": "result",
    "stakeholder": "party",
    "touchpoint": "contact",
    "bandwidth": "capacity",
    "low-hanging fruit": "easy task",
    "move the needle": "make progress",
    "deep dive": "detailed review",
    "circle back": "return",
    "reach out": "contact",
    "ping": "contact",
    "sync up": "meet",
    "huddle": "meet",
    "standup": "meeting",
    "stand-up": "meeting",
}

# Find all .py files
py_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            py_files.append(str(Path(root) / file))

print(f"Found {len(py_files)} Python files")

for filepath in py_files:
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix contractions in docstrings (but not in code blocks)
    lines = content.split('\n')
    in_docstring = False
    in_code = False

    for i, line in enumerate(lines):
        # Track docstring state
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            in_docstring = not in_docstring
            continue

        if in_docstring:
            for contraction, replacement in contractions.items():
                # Use word boundary to avoid false matches
                pattern = r'\b' + re.escape(contraction) + r'\b'
                lines[i] = re.sub(pattern, replacement, lines[i], flags=re.IGNORECASE)

            for term, replacement in unapproved_terms.items():
                pattern = r'\b' + re.escape(term) + r'\b'
                lines[i] = re.sub(pattern, replacement, lines[i], flags=re.IGNORECASE)

    content = '\n'.join(lines)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")

print("Done fixing STE violations in source files")
