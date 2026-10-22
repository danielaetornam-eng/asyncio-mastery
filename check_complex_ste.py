import os
import re
from pathlib import Path


# Function to check if a sentence violates STE rules
def check_sentence(sentence):
    issues = []

    # Rule 1: Max 20 words per sentence
    word_count = len(sentence.split())
    if word_count > 20:
        issues.append(f"LONG ({word_count} words)")

    # Rule 2: Max 2 clauses per sentence (rough check for conjunctions)
    conjunctions = ["and", "but", "or", "nor", "for", "yet", "so"]
    clause_count = 1
    for conj in conjunctions:
        clause_count += len(re.findall(r'\b' + conj + r'\b', sentence, re.IGNORECASE))
    if clause_count > 2:
        issues.append(f"MANY CLAUSES ({clause_count})")

    # Rule 3: Check for passive voice (is, was, are, were, been, been being + past participle)
    passive_patterns = [
        r'\bis\s+\w+ed\b',
        r'\bare\s+\w+ed\b',
        r'\bwas\s+\w+ed\b',
        r'\bwere\s+\w+ed\b',
        r'\bhas been\s+\w+ed\b',
        r'\bhave been\s+\w+ed\b',
        r'\bwill be\s+\w+ed\b',
    ]
    for pattern in passive_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            issues.append("PASSIVE VOICE")
            break

    # Rule 4: Check for gerunds as nouns (ending in -ing)
    gerund_nouns = r'\b\w+ing\b'
    if re.search(gerund_nouns, sentence):
        # Check if it's used as a noun (common in passive/giving context)
        if re.search(r'\b(when|while|during|after|before)\s+\w+ing', sentence, re.IGNORECASE):
            issues.append("GERUND AS NOUN")

    # Rule 5: Check for helping verbs as main verbs (is, has, had, will, can, etc.)
    helping_verb_patterns = [
        r'\bis\s+(a|an|the|this|that|these|those)\s+\w+\b',
        r'\bhas\s+(a|an|the|this|that|these|those)\s+\w+\b',
        r'\bhad\s+(a|an|the|this|that|these|those)\s+\w+\b',
        r'\bmust\s+(a|an|the|this|that|these|those)\s+\w+\b',
        r'\bcan\s+(a|an|the|this|that|these|those)\s+\w+\b',
    ]
    for pattern in helping_verb_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            issues.append("HELPING VERB AS MAIN")
            break

    # Rule 6: Check for long noun clusters (3+ words in a row without verbs)
    words = re.findall(r'\b\w+\b', sentence)
    noun_cluster = 0
    for word in words:
        if word[0].isupper() or word in ['the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from']:
            noun_cluster += 1
            if noun_cluster >= 3:
                issues.append("LONG NOUN CLUSTER")
                break
        else:
            noun_cluster = 0

    return issues

# Function to fix a sentence
def fix_sentence(sentence):
    # Simple fixes for demonstration
    # In real implementation, this would be much more sophisticated

    # Fix long sentences (keep first 20 words, add "..." to indicate truncation)
    if len(sentence.split()) > 20:
        words = sentence.split()
        sentence = ' '.join(words[:17]) + "..."

    # Fix multiple clauses - split on conjunctions
    conjunctions = [" and ", " but ", " or ", " nor ", " for ", " yet ", " so "]
    for conj in conjunctions:
        if sentence.count(conj) > 1:
            # Simple split - in real implementation would need context
            parts = re.split(r'(?<!\w)' + re.escape(conj) + r'(?!\w)', sentence, maxsplit=1)
            if len(parts) > 1:
                sentence = parts[0].strip() + '. ' + parts[1].strip() + '.'

    # Fix passive voice - simple replacements
    passive_patterns = [
        (r'is \w+ed', 'becomes'),
        (r'was \w+ed', 'became'),
        (r'are \w+ed', 'become'),
        (r'were \w+ed', 'became'),
        (r'has been \w+ed', 'is completed'),
        (r'will be \w+ed', 'will be completed'),
    ]
    for pattern, replacement in passive_patterns:
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

    return sentence

# Find all .py files and check docstrings
py_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            py_files.append(str(Path(root) / file))

print(f"Checking {len(py_files)} Python files for complex STE violations")

# Files to examine
files_to_check = [
    'src/asyncio_mastery/basics/coroutines.py',
    'src/asyncio_mastery/basics/tasks.py',
    'src/asyncio_mastery/basics/event_loop.py',
    'src/asyncio_mastery/basics/futures.py',
    'src/asyncio_mastery/patterns/circuit_breaker.py',
    'src/asyncio_mastery/patterns/rate_limiter.py',
    'src/asyncio_mastery/patterns/worker_pool.py',
    'src/asyncio_mastery/patterns/connection_pool.py',
    'src/asyncio_mastery/patterns/batching.py',
]

for filepath in files_to_check:
    if Path(filepath).exists():
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        in_docstring = False
        current_docstring = []
        docstring_start = 0

        for i, line in enumerate(lines):
            # Track docstring state
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                if not in_docstring:
                    # Start of docstring
                    in_docstring = True
                    current_docstring = [line]
                    docstring_start = i
                else:
                    # End of docstring
                    current_docstring.append(line)
                    in_docstring = False

                    # Analyze the docstring
                    docstring_text = '\n'.join(current_docstring)

                    # Extract sentences
                    sentences = re.split(r'[.!?]+', docstring_text)
                    for sentence in sentences:
                        if sentence.strip():
                            issues = check_sentence(sentence)
                            if issues:
                                print(f"\n{filepath}:{docstring_start+1}")
                                print(f"Sentence: {sentence[:100]}...")
                                print(f"Issues: {', '.join(issues)}")

                    current_docstring = []
            elif in_docstring:
                current_docstring.append(line)

print("\nComplex STE violation check complete")
