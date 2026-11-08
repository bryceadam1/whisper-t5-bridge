"""FLAN 2021 templates: load them, recover the input field from a rendered prompt, and re-render.

The templates come from google-research/FLAN (Apache-2.0) at a pinned commit. The two template files
are pure dict literals, so they are parsed with `ast.literal_eval` and never imported or executed.
"""

from __future__ import annotations

import ast
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FLAN_COMMIT = "e9e4ec6e2701182c7a91af176f705310da541277"
TEMPLATE_FILES = ("templates.py", "flan_templates_branched.py")

# DPI task_name prefix -> (FLAN template key, input field, task family). Only tasks whose source
# datasets carry permissive licences (see README) and whose prompt has a single speakable input.
# fix_punct and true_case (CC0) also qualify but are left out: their web-scraped text carries too much
# adult-site spam for keyword filtering to clean.
TASKS = {
    "glue/sst2": ("sst2", "sentence", "sentiment"),
    "trec": ("trec", "text", "question_type"),
    "trivia_qa": ("trivia_qa", "question", "closed_book_qa"),
    "natural_questions": ("natural_questions", "question", "closed_book_qa"),
    "gem/common_gen": ("common_gen", "concepts", "concept_to_sentence"),
    "ai2_arc": ("arc", "question", "multiple_choice"),
    "openbookqa": ("openbookqa", "question", "multiple_choice"),
    "lambada": ("lambada", "sentence", "continuation"),
}

OPTIONS = "options_"


def task_spec(task_name: str):
    for prefix, spec in TASKS.items():
        if task_name.startswith(prefix):
            return spec
    return None


def _download(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TEMPLATE_FILES:
        path = cache_dir / f"{FLAN_COMMIT[:12]}_{name}"
        if not path.exists():
            url = f"https://raw.githubusercontent.com/google-research/FLAN/{FLAN_COMMIT}/flan/v2/{name}"
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def _patterns_literal(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "PATTERNS":
            return ast.literal_eval(node.value)
    raise ValueError(f"no PATTERNS literal in {path}")


def _strip_options(template: str) -> str:
    # Mirrors how FLAN builds its no-options variants (templates.py, PATTERNS_NO_OPTIONS).
    return template.replace("\n\n{options_}", "").replace("\n{options_}", "").replace("{options_}", "")


@dataclass(frozen=True)
class Template:
    key: str
    text: str
    input_field: str
    has_options: bool
    regex: re.Pattern

    def render(self, value: str, options: str | None) -> tuple[str, str]:
        """Return (full prompt, instruction) where the instruction is the prompt minus the input."""
        filled = self.text.replace("{" + OPTIONS + "}", options or "")
        before, _, after = filled.partition("{" + self.input_field + "}")
        return (before + value + after).strip(), tidy(before + after)


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _compile(template: str, input_field: str) -> re.Pattern:
    parts = re.split(r"\{(\w+)\}", template.strip())
    regex = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            regex.append(re.escape(part))
        else:
            regex.append(f"(?P<{part}>.+?)")
    return re.compile("".join(regex), re.DOTALL)


def load_templates(cache_dir: Path) -> dict[str, list[Template]]:
    """Forward-direction templates per FLAN key: the input field appears in the prompt, and the only
    other placeholder allowed is the options list."""
    literals = [_patterns_literal(p) for p in _download(cache_dir)]
    out: dict[str, list[Template]] = {}
    for key, input_field, _ in TASKS.values():
        seen = set()
        for patterns in literals:
            for prompt, _target in patterns.get(key, []):
                for text in (prompt, _strip_options(prompt)):
                    fields = set(re.findall(r"\{(\w+)\}", text))
                    if input_field not in fields or fields - {input_field, OPTIONS} or text in seen:
                        continue
                    seen.add(text)
                    out.setdefault(key, []).append(
                        Template(key, text, input_field, OPTIONS in fields, _compile(text, input_field))
                    )
        # Most specific first, so a bare "{question}" template cannot swallow another template's text.
        out[key].sort(key=lambda t: -len(re.sub(r"\{\w+\}", "", t.text)))
    return out


def match(prompt: str, templates: list[Template]) -> tuple[str, str | None] | None:
    """Recover (input, options) from a rendered prompt, or None if no template matches exactly."""
    prompt = prompt.strip()
    for t in templates:
        m = t.regex.fullmatch(prompt)
        if m:
            groups = m.groupdict()
            return groups[t.input_field].strip(), (groups.get(OPTIONS) or "").strip() or None
    return None


def sample_renderings(templates: list[Template], value: str, options: str | None, k: int, rng: random.Random):
    """Up to k distinct (template text, prompt, instruction) renderings. Templates that need an options
    list are only used when the source row had one."""
    usable = [t for t in templates if options or not t.has_options]
    picks = rng.sample(usable, min(k, len(usable)))
    return [(t.text, *t.render(value, options)) for t in picks]
