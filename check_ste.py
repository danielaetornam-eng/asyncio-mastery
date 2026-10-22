import re

with open('docs/concepts.md') as f:
    content = f.read()

pattern = re.compile(r'\b(ping|comprehensive|extensive|production-ready|god-tier)\b', re.IGNORECASE)
matches = pattern.findall(content)
print('Unapproved terms:', matches)

contraction_pattern = re.compile(r"\b(can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|needn't|daren't|mightn't|shan't|it's|that's|what's|who's|there's|here's|where's|how's|I'm|you're|he's|she's|we're|they're|I've|you've|we've|they've|I'd|you'd|he'd|she'd|we'd|they'd|I'll|you'll|he'll|she'll|we'll|they'll|let's)\b", re.IGNORECASE)
contraction_matches = contraction_pattern.findall(content)
print('Contractions:', len(contraction_matches))
if contraction_matches:
    print('First few:', contraction_matches[:5])
