import re
from pathlib import Path

# Check all markdown files
files_to_check = [
    'README.md',
    'README_STE.md',
    'STE_COMPLIANCE_REPORT.md',
    'docs/getting_started.md',
    'docs/getting_started_STE.md',
    'docs/getting_started_STE_compliance_report.md',
    'docs/concepts.md',
    'docs/patterns.md'
]

violations = []

contraction_pattern = re.compile(r"\b(can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|needn't|daren't|mightn't|shan't|it's|that's|what's|who's|there's|here's|where's|how's|I'm|you're|he's|she's|we're|they're|I've|you've|we've|they've|I'd|you'd|he'd|she'd|we'd|they'd|I'll|you'll|he'll|she'll|we'll|they'll|let's)\b", re.IGNORECASE)

unapproved_terms = {
    'comprehensive': 'complete',
    'extensive': 'broad',
    'utilize': 'use',
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
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or line.strip().startswith('`'):
                continue

            for match in contraction_pattern.finditer(line):
                violations.append(f'{filepath}:{i}: Contraction: {match.group()}')

            for term in unapproved_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', line, re.IGNORECASE):
                    violations.append(f'{filepath}:{i}: Unapproved term: {term}')

print(f'Total violations: {len(violations)}')
for v in violations:
    print(v)
