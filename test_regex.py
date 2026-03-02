
import re

def strip_markdown(text):
    """
    Universally strips markdown code blocks (```json, ```xml, ```, etc) using regex.
    """
    # Remove ``` followed by optional language identifier (or nothing)
    # This covers ```json, ```xml, ```toml, ```, etc.
    text = re.sub(r'```\w*', '', text)
    return text.strip()

test_cases = [
    "```json\n{\"foo\": \"bar\"}\n```",
    "```xml\n<root>foo</root>\n```",
    "Here is code:\n```toml\nkey = \"val\"\n```",
    "```\nPlain block\n```",
    "No markdown here."
]

print("--- Testing strip_markdown ---")
for i, case in enumerate(test_cases):
    cleaned = strip_markdown(case)
    print(f"Case {i+1}:")
    print(f"Original: {case!r}")
    print(f"Cleaned : {cleaned!r}")
    if "```" in cleaned:
        print("❌ FAILED: Markdown still present!")
    else:
        print("✅ PASSED")
    print("-" * 20)
