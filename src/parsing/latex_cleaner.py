"""
arXiv RAG v1 - LaTeX Cleaner

Clean up LaTeX artifacts from parsed text.
Handles unconverted commands, special characters, and formatting.
"""

import re
from typing import Optional


# Common LaTeX commands to strip (keeping content inside braces)
STRIP_COMMANDS_KEEP_CONTENT = [
    r"\\textbf",
    r"\\textit",
    r"\\emph",
    r"\\texttt",
    r"\\textrm",
    r"\\textsf",
    r"\\textsc",
    r"\\underline",
    r"\\textcolor\{[^}]*\}",  # \textcolor{red}{text}
    r"\\colorbox\{[^}]*\}",
    r"\\mbox",
    r"\\hbox",
    r"\\vbox",
    r"\\fbox",
    r"\\makebox(?:\[[^\]]*\])?",
    r"\\parbox(?:\[[^\]]*\])?\{[^}]*\}",
    # Positioning commands (often in titles)
    r"\\raisebox\{[^}]*\}(?:\[[^\]]*\])?",
    r"\\scalebox\{[^}]*\}",
    r"\\resizebox\{[^}]*\}\{[^}]*\}",
    r"\\rotatebox\{[^}]*\}",
    r"\\phantom",
    r"\\hphantom",
    r"\\vphantom",
]

# Commands to remove entirely (including content)
REMOVE_COMMANDS_WITH_CONTENT = [
    r"\\cite\{[^}]*\}",
    r"\\citep\{[^}]*\}",
    r"\\citet\{[^}]*\}",
    r"\\citeauthor\{[^}]*\}",
    r"\\citeyear\{[^}]*\}",
    r"\\ref\{[^}]*\}",
    r"\\eqref\{[^}]*\}",
    r"\\pageref\{[^}]*\}",
    r"\\autoref\{[^}]*\}",
    r"\\Cref\{[^}]*\}",
    r"\\cref\{[^}]*\}",
    r"\\label\{[^}]*\}",
    r"\\footnote\{[^}]*\}",
    r"\\thanks\{[^}]*\}",
    r"\\marginpar\{[^}]*\}",
    # Graphics commands (often logos in titles)
    r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\input\{[^}]*\}",
    r"\\include\{[^}]*\}",
]

# Commands to remove (no content)
REMOVE_COMMANDS_NO_CONTENT = [
    r"\\noindent",
    r"\\indent",
    r"\\par\b",
    r"\\newline",
    r"\\linebreak",
    r"\\pagebreak",
    r"\\newpage",
    r"\\clearpage",
    r"\\hfill",
    r"\\vfill",
    r"\\hspace\{[^}]*\}",
    r"\\vspace\{[^}]*\}",
    r"\\smallskip",
    r"\\medskip",
    r"\\bigskip",
    r"\\centering",
    r"\\raggedright",
    r"\\raggedleft",
]

# Special character replacements
SPECIAL_CHARS = {
    r"\\%": "%",
    r"\\&": "&",
    r"\\#": "#",
    r"\\_": "_",
    r"\\$": "$",
    r"\\{": "{",
    r"\\}": "}",
    r"\\textbackslash": "\\",
    r"\\textasciitilde": "~",
    r"\\textasciicircum": "^",
    r"\\ldots": "...",
    r"\\dots": "...",
    r"\\cdots": "...",
    r"---": "—",  # em dash
    r"--": "–",   # en dash
    r"``": '"',
    r"''": '"',
    r"`": "'",
    r"~": " ",    # non-breaking space
}

# Quote replacements
QUOTE_PATTERNS = [
    (r"``([^']+)''", r'"\1"'),
    (r"`([^']+)'", r"'\1'"),
]


def clean_latex_text(text: str) -> str:
    """
    Clean LaTeX artifacts from text.

    Args:
        text: Text with potential LaTeX commands

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    result = text

    # 1. Remove commands with content (citations, refs, etc.)
    for pattern in REMOVE_COMMANDS_WITH_CONTENT:
        result = re.sub(pattern, "", result)

    # 2. Remove commands without content
    for pattern in REMOVE_COMMANDS_NO_CONTENT:
        result = re.sub(pattern, "", result)

    # 3. Strip formatting commands but keep content
    for cmd in STRIP_COMMANDS_KEEP_CONTENT:
        # Match \command{content} and extract content
        pattern = cmd + r"\{([^}]*)\}"
        result = re.sub(pattern, r"\1", result)

    # 4. Handle nested braces - multiple passes for deeply nested
    for _ in range(3):  # Up to 3 levels of nesting
        # Match \command{content} and keep just content
        prev = result
        result = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", result)
        if prev == result:
            break

    # 5. Remove remaining backslash commands without braces
    result = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?=\s|$|[^a-zA-Z{])", "", result)

    # 6. Special character replacements
    for latex, char in SPECIAL_CHARS.items():
        result = result.replace(latex, char)

    # 7. Quote normalization
    for pattern, replacement in QUOTE_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # 8. Clean up math mode markers (but preserve content)
    # Inline math: $...$ or \(...\)
    result = re.sub(r"\$([^$]+)\$", r"\1", result)
    result = re.sub(r"\\\(([^)]+)\\\)", r"\1", result)

    # 9. Clean up orphaned braces
    result = re.sub(r"\{\s*\}", "", result)  # Empty braces
    result = result.replace("{", "").replace("}", "")  # Remaining braces

    # 10. Clean up whitespace
    result = re.sub(r"[ \t]+", " ", result)  # Multiple spaces to single
    result = re.sub(r"\n{3,}", "\n\n", result)  # Max 2 newlines
    result = re.sub(r"^\s+", "", result, flags=re.MULTILINE)  # Leading whitespace

    return result.strip()


def clean_section_title(title: str) -> str:
    """
    Clean a section title.

    Args:
        title: Raw section title

    Returns:
        Cleaned title
    """
    if not title:
        return ""

    result = clean_latex_text(title)

    # Remove section numbering
    result = re.sub(r"^[\d.]+\s*", "", result)

    # Remove trailing punctuation
    result = re.sub(r"[.:;,]+$", "", result)

    return result.strip()


def clean_paper_title(title: str) -> str:
    """
    Clean a paper title (more aggressive than section title).

    Args:
        title: Raw paper title from LaTeX

    Returns:
        Cleaned title
    """
    if not title:
        return ""

    result = title

    # 1. Remove LaTeX comments (% to end of line, but not \%)
    result = re.sub(r"(?<!\\)%[^\n]*\n?", "", result)

    # 2. Remove newlines and normalize whitespace
    result = re.sub(r"\s+", " ", result)

    # 3. Apply standard LaTeX cleaning
    result = clean_latex_text(result)

    # 4. Remove leading/trailing special characters
    result = re.sub(r"^[:\-–—\s]+", "", result)
    result = re.sub(r"[:\-–—\s]+$", "", result)

    # 5. Fix common artifacts
    result = result.replace("\\", "").replace("  ", " ")

    return result.strip()


def clean_equation_latex(latex: str) -> str:
    """
    Clean equation LaTeX for display.
    Preserves math commands but removes environment wrappers.

    Args:
        latex: Raw equation LaTeX

    Returns:
        Cleaned equation LaTeX
    """
    if not latex:
        return ""

    result = latex

    # Remove equation environments
    result = re.sub(r"\\begin\{(equation|align|gather|multline)\*?\}", "", result)
    result = re.sub(r"\\end\{(equation|align|gather|multline)\*?\}", "", result)

    # Remove labels
    result = re.sub(r"\\label\{[^}]*\}", "", result)

    # Clean whitespace
    result = re.sub(r"\s+", " ", result)

    return result.strip()


def extract_text_content(latex: str) -> str:
    """
    Extract plain text content from LaTeX, removing all commands.
    More aggressive than clean_latex_text - for embedding purposes.

    Args:
        latex: LaTeX content

    Returns:
        Plain text content
    """
    if not latex:
        return ""

    result = latex

    # Remove all environments
    result = re.sub(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", "", result, flags=re.DOTALL)

    # Remove all remaining commands
    result = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", result)

    # Apply standard cleaning
    result = clean_latex_text(result)

    # Remove any remaining backslashes
    result = result.replace("\\", "")

    # Clean up braces
    result = result.replace("{", "").replace("}", "")

    return result.strip()


def is_math_heavy(text: str, threshold: float = 0.3) -> bool:
    """
    Check if text is heavily mathematical.

    Args:
        text: Text to check
        threshold: Ratio threshold (0.3 = 30% math)

    Returns:
        True if text is math-heavy
    """
    if not text:
        return False

    # Count math delimiters
    math_markers = len(re.findall(r"\$|\\\(|\\\)|\\\[|\\\]|\\begin\{", text))
    words = len(text.split())

    if words == 0:
        return True

    return (math_markers / words) > threshold
