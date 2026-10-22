import re
from pathlib import Path

# Read all markdown files
files_to_check = [
    'README.md',
    'docs/getting_started.md',
    'docs/concepts.md',
    'docs/patterns.md'
]

# STE validation checks
violations = []

# 1. Check for contractions
contraction_pattern = re.compile(r"\b(can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|needn't|daren't|mightn't|shan't|it's|that's|what's|who's|there's|here's|where's|how's|I'm|you're|he's|she's|we're|they're|I've|you've|we've|they've|I'd|you'd|he'd|she'd|we'd|they'd|I'll|you'll|he'll|she'll|we'll|they'll|let's)\b", re.IGNORECASE)

# 2. Check for passive voice indicators
passive_indicators = [
    'is being', 'are being', 'was being', 'were being',
    'has been', 'have been', 'had been',
    'will be', 'would be', 'should be', 'could be', 'must be',
    'is done', 'are done', 'was done', 'were done',
    'is made', 'are made', 'was made', 'were made',
    'is used', 'are used', 'was used', 'were used',
    'is called', 'are called', 'was called', 'were called',
    'is known', 'are known', 'was known', 'were known',
    'is designed', 'are designed',
    'is intended', 'are intended',
    'is expected', 'are expected',
    'is required', 'are required',
    'is supposed to', 'are supposed to',
]

# 3. Check for unapproved terms
unapproved_terms = {
    'comprehensive': 'complete',
    'extensive': 'broad',
    'utilize': 'use',
    'utilization': 'use',
    'facilitate': 'help',
    'enable': 'allow',
    'ensure': 'make sure',
    'leverage': 'use',
    'optimize': 'improve',
    'streamline': 'simplify',
    'robust': 'strong',
    'seamless': 'smooth',
    'intuitive': 'easy',
    'user-friendly': 'easy to use',
    'state-of-the-art': 'modern',
    'cutting-edge': 'modern',
    'best-in-class': 'best',
    'enterprise-grade': 'production',
    'production-ready': 'production',
    'god-tier': 'advanced',
    'ultimate': 'complete',
    'heavily': 'fully',
    'highly': 'very',
    'significantly': 'much',
    'substantially': 'much',
    'considerably': 'much',
    'dramatically': 'much',
    'exponentially': 'rapidly',
    'paradigm': 'model',
    'synergy': 'cooperation',
    'holistic': 'complete',
    'granular': 'detailed',
    'actionable': 'usable',
    'deliverable': 'result',
    'stakeholder': 'party',
    'touchpoint': 'contact',
    'bandwidth': 'capacity',
    'low-hanging fruit': 'easy task',
    'move the needle': 'make progress',
    'deep dive': 'detailed review',
    'circle back': 'return',
    'reach out': 'contact',
    'ping': 'contact',
    'sync up': 'meet',
    'huddle': 'meet',
    'standup': 'meeting',
    'stand-up': 'meeting',
}

for filepath in files_to_check:
    if Path(filepath).exists():
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        in_code_block = False
        for i, line in enumerate(lines, 1):
            # Track code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or line.strip().startswith('`'):
                continue

            # Check contractions
            for match in contraction_pattern.finditer(line):
                violations.append(f'{filepath}:{i}: Contraction found: {match.group()}')

            # Check passive voice
            for indicator in passive_indicators:
                if indicator in line.lower():
                    violations.append(f'{filepath}:{i}: Possible passive voice: {indicator}')

            # Check unapproved terms
            for term, replacement in unapproved_terms.items():
                if term in line.lower():
                    violations.append(f'{filepath}:{i}: Unapproved term "{term}" (use "{replacement}")')

            # Check sentence length (rough estimate)
            sentences = re.split(r'[.!?]+', line)
            for sent in sentences:
                words = len(sent.split())
                if words > 20:
                    violations.append(f'{filepath}:{i}: Long sentence ({words} words): {sent[:80]}...')

print(f'Total violations found: {len(violations)}')
for v in violations[:100]:
    print(v)
if len(violations) > 100:
    print(f'... and {len(violations) - 100} more')
