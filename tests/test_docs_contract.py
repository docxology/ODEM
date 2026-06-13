from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from tests.fixtures.cli import REPO_ROOT, run_cli


def test_markdown_local_links_resolve():
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
    link_pattern = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
    missing: list[str] = []

    for doc in docs:
        for target in link_pattern.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            resolved = (doc.parent / path_part).resolve()
            if not resolved.exists():
                missing.append(f"{doc.relative_to(REPO_ROOT)} -> {target}")

    assert missing == []


def test_root_readme_and_agents_are_present_and_linked():
    readme = REPO_ROOT / "README.md"
    agents = REPO_ROOT / "AGENTS.md"

    assert readme.exists()
    assert agents.exists()
    assert "AGENTS.md" in readme.read_text(encoding="utf-8")
    assert "pytest -q" in agents.read_text(encoding="utf-8")
    assert "python main.py validate --config parameters.yaml" in agents.read_text(encoding="utf-8")


def test_docs_name_shared_refactor_modules():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
        if path.exists()
    )

    for module_name in (
        "odem.arrays",
        "odem.slicing",
        "odem.summary_schema",
        "odem.run_outputs",
        "odem.model_registry",
        "odem.numerics",
        "odem.plot_style",
        "odem.plot_data",
        "odem.visual_checks",
        "odem.release_evidence",
    ):
        assert module_name in combined


def test_docs_index_links_every_docs_markdown_file():
    index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    missing = []

    for doc in sorted((REPO_ROOT / "docs").glob("*.md")):
        if doc.name == "index.md":
            continue
        if f"]({doc.name})" not in index:
            missing.append(doc.name)

    assert missing == []


def test_maintainer_docs_cover_release_and_validation_contracts():
    required_docs = {
        "maintainer-guide.md": ("odem.config", "odem.experiment", "safe extension"),
        "validation-contract.md": ("manifest status", "SHA-256", "static PDF"),
        "release-evidence.md": ("3024", "/tmp/odem-full", "local archive"),
        "ci-and-benchmarks.md": ("Python 3.11", "benchmark_visuals.py", "run_release_sweep.py"),
        "test-strategy.md": ("contract tests", "fixtures", "slow gates"),
    }

    for filename, phrases in required_docs.items():
        text = (REPO_ROOT / "docs" / filename).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text


def test_docs_omit_removed_entrypoint_wording():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
        if path.exists()
    )

    removed_terms = (
        "leg" + "acy",
        "sh" + "im",
        "place" + "holder",
        "/".join(("functions", "report_bundle", "report_bundle.py")),
        "/".join(("functions", "plotting", "plot.py")),
        "to_" + "leg" + "acy",
        "from_" + "leg" + "acy",
    )

    assert [term for term in removed_terms if term in combined.lower()] == []


def test_documented_cli_subcommands_are_in_help():
    completed = run_cli("-m", "odem.cli", "--help", check=True)
    documented = set()
    command_pattern = re.compile(r"python main\.py\s+([a-z][a-z-]*)")
    for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]:
        documented.update(command_pattern.findall(doc.read_text(encoding="utf-8")))

    assert documented
    assert documented <= set(completed.stdout.split())


def test_documented_cli_options_are_live_and_use_preferred_spellings():
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
    text_by_doc = {doc: doc.read_text(encoding="utf-8") for doc in docs if doc.exists()}
    assert all("--quiet-progress" not in text for text in text_by_doc.values())

    command_pattern = re.compile(r"python main\.py\s+([a-z][a-z-]*)\s+([^\n`$]*)")
    options_by_command: dict[str, set[str]] = {}
    for text in text_by_doc.values():
        for command, rest in command_pattern.findall(text):
            options_by_command.setdefault(command, set()).update(re.findall(r"--[a-z][a-z-]*", rest))

    for command, options in options_by_command.items():
        help_text = run_cli("-m", "odem.cli", command, "--help", check=True).stdout
        missing = sorted(option for option in options if option not in help_text)
        assert missing == []


def test_api_reference_python_snippets_execute_in_isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (REPO_ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)

    assert snippets
    for snippet in snippets:
        namespace = {
            "Path": Path,
            "np": np,
            "repo_root": REPO_ROOT,
            "tmp_path": tmp_path,
        }
        exec(compile(snippet, "docs/api-reference.md", "exec"), namespace)
