"""Guards on the documentation that carries normative weight.

`NUMERICS.md` is the approved contract: clauses and precedents are cited by
identifier from source comments, test docstrings, commit messages and the C++
preservation notes. A citation that no longer resolves is not cosmetic. It means
a reader following the reference to check an accuracy claim lands nowhere, and
the usual response to that is to assume the claim is fine.

These tests are cheap and structural. They do not judge the content of any
clause; they check that the document's own cross-references, and the identifiers
the code cites, still point at something.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
NUMERICS = ROOT / "NUMERICS.md"
COEFFICIENT_EVIDENCE = ROOT / "docs" / "evidence" / "coefficient_immutability.md"
CONTRACT_DOCUMENTS = (NUMERICS, COEFFICIENT_EVIDENCE)
HISTORY = ROOT / "docs" / "history"


def _numerics_text() -> str:
    return NUMERICS.read_text(encoding="utf-8")


def _anchors(text: str) -> set[str]:
    """Every fragment a link in this document could legitimately target.

    Two kinds are recognised. GitHub derives an implicit anchor from each
    heading by lowercasing, stripping punctuation and replacing spaces with
    hyphens. Explicit ``<a id="...">`` anchors are also honoured, and are what
    this document uses for the clause and precedent identifiers: an implicit
    anchor changes whenever the heading text is edited, so a citation of "C-2"
    would break the first time someone rewords the title of C-2.
    """
    found = {
        "#" + re.sub(r"[^\w\s-]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    found |= {"#" + a for a in re.findall(r'<a id="([^"]+)"></a>', text)}
    return found


@pytest.mark.parametrize(
    "document", CONTRACT_DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT))
)
def test_numerics_internal_links_resolve(document):
    """Contract and evidence links resolve, including across the file boundary.

    Eleven of the thirteen internal links in this document were broken when
    this test was written, all in the same way: the link was spelled from the
    clause title while the heading also carried a trailing status marker such
    as "— `APPROVED`", which GitHub folds into the generated anchor. The cure
    was explicit anchors; this test is what stops it recurring.
    """
    text = document.read_text(encoding="utf-8")
    links = sorted(set(re.findall(r"\]\(([^)]+)\)", text)))
    assert links, "no links found; the extraction pattern is wrong"

    broken = []
    for link in links:
        if re.match(r"[a-z]+:", link):
            continue
        filename, _, fragment = link.partition("#")
        target = (document.parent / filename).resolve() if filename else document
        if not target.is_file() or (
            fragment
            and "#" + fragment not in _anchors(target.read_text(encoding="utf-8"))
        ):
            broken.append(link)
    assert not broken, (
        f"{document.relative_to(ROOT)} contains links that do not resolve:\n  "
        + "\n  ".join(broken)
    )


def test_coefficient_history_is_linked_from_its_clause():
    """Moving the history must not make its witnesses undiscoverable."""
    clause = _numerics_text().split("### C-15.7", 1)[1].split("## C-16", 1)[0]
    target = COEFFICIENT_EVIDENCE.relative_to(ROOT).as_posix()
    links = re.findall(r"\]\(([^)]+)\)", clause)
    assert any(link.partition("#")[0] == target for link in links), (
        "C-15.7 no longer links to its failure history and campaign evidence"
    )


#: Identifiers cited from source, tests or the C++ preservation notes. Listing
#: them explicitly, rather than scraping every "C-n" shaped string, keeps the
#: test from passing merely because a citation was deleted along with its
#: target.
CITED_IDENTIFIERS = [
    "C-1",
    "C-2",
    "C-3",
    "C-3.2",
    "C-3.4",
    "C-4",
    "C-5.1",
    "C-5.3",
    "C-5.4",
    "C-6.1",
    "C-6.2",
    "C-7",
    "C-8.1",
    "C-9",
    "C-10.2",
    "C-11.3",
    "C-13",
    "C-14.1",
    "C-14.2",
    "C-14.4",
    "C-15",
    "C-15.1",
    "C-15.2",
    "C-15.3",
    "C-15.4",
    "C-15.6",
    "C-15.7",
    "C-16",
    "C-16.1",
    "C-16.2",
    "C-16.3",
    "C-16.4",
    "C-16.5",
    "C-16.6",
    "C-16.8",
    "C-16.9",
    "C-17",
    "C-17.1",
    "C-17.3",
    "R-1",
    "R-2",
    "R-4",
    "R-5",
    "R-6",
    "R-9",
    "R-10",
    "R-11",
]


@pytest.mark.parametrize("identifier", CITED_IDENTIFIERS)
def test_cited_clause_is_defined(identifier):
    """Each cited clause or precedent has a definition, not just mentions.

    "Defined" means the identifier introduces a section: a Markdown heading, or
    the bold lead-in that the earlier precedents R-1 to R-8 use. A clause that
    appears only inside the prose of some other clause has been deleted and its
    citations left dangling.
    """
    text = _numerics_text()
    escaped = re.escape(identifier)
    defined = re.search(
        rf"^(#+\s*{escaped}[\s—-]|\*\*{escaped}\s*—)", text, re.MULTILINE
    )
    assert defined, (
        f"{identifier} is cited from the codebase but NUMERICS.md does not "
        f"define it as a clause or precedent."
    )


def _first_table_block(section: str) -> list[tuple[str, str]]:
    """Return ``(first-column, whole-row)`` for the first Markdown table.

    A "table" is a maximal run of consecutive lines beginning with ``|``. The
    header row and the ``|---|`` separator are dropped. Taking the block whole,
    rather than collecting rows that look familiar, is what makes an
    unrecognised row detectable instead of invisible.
    """
    block: list[str] = []
    started = False
    for line in section.splitlines():
        if line.startswith("|"):
            started = True
            block.append(line)
        elif started:
            break
    body = [ln for ln in block if not set(ln) <= set("|- :")][1:]
    return [(ln.split("|")[1].strip(), ln) for ln in body]


def test_certified_families_agree_with_the_code():
    """C-6.1 requires the contract table and CERTIFIED_STAGE_TYPES to agree.

    The clause states that adding an entry to either without the other is a
    false certification. It is the one place where a documentation edit can, by
    itself, make the package advertise an accuracy claim nobody validated, so
    the agreement is checked rather than trusted.

    Every row of the table is examined, not only the families this test knows
    about. An earlier version compared a fixed list of four families and
    therefore did not notice when "Linear multistep, ``r > 1``" was marked
    certified -- the exact edit the clause forbids, on a family that is refused
    outright. A table row with no corresponding stage type cannot be certified
    by definition, because there is no route for it to certify.
    """
    from adjungo.core.method import StageType
    from adjungo.optimization.interface import CERTIFIED_STAGE_TYPES

    text = _numerics_text()
    table = text[text.index("### C-6.1"): text.index("### C-6.2")]

    #: Table row label -> the stage type that implements it. A family absent
    #: from this map has no implementing route at all.
    ROUTES = {
        "Explicit Runge–Kutta": StageType.EXPLICIT,
        "DIRK": StageType.DIRK,
        "SDIRK": StageType.SDIRK,
        "Fully implicit (dense `A`)": StageType.IMPLICIT,
        "Linear multistep, `r > 1`": None,
        "IMEX / additive splitting": None,
    }

    rows = [
        ln for ln in table.splitlines()
        if ln.startswith("| ") and "|---" not in ln and not ln.startswith("| Family")
        and not ln.startswith("| Claim")
    ]
    labels = [ln.split("|")[1].strip() for ln in rows]

    # The family table is the FIRST contiguous table in the section; the later
    # tables in C-6.1 are evidence tables keyed by claim. The block is taken
    # whole rather than by matching known labels: an earlier version stopped at
    # the first unrecognised label, so appending a fabricated family to the end
    # of the table simply truncated the parse and the row was never examined.
    family_rows = _first_table_block(table)
    family_labels = {label for label, _ in family_rows}

    unknown = family_labels - set(ROUTES)
    assert not unknown, (
        f"C-6.1 lists families this test does not know how to verify: "
        f"{sorted(unknown)}. A family cannot be certified in the contract "
        "until its implementing stage type is recorded here."
    )
    absent = set(ROUTES) - family_labels
    assert not absent, (
        f"C-6.1 no longer lists {sorted(absent)}. Removing a family from the "
        "table silently drops its refusal or its certification."
    )
    # The family table is the first one in the section; later tables in C-6.1
    # are evidence tables keyed by claim, so stop at the first unknown label.
    family_rows = []
    for label, row in zip(labels, rows):
        if label not in ROUTES:
            break
        family_rows.append((label, row))

    assert len(family_rows) == len(ROUTES), (
        f"C-6.1 family table has {len(family_rows)} recognised rows, expected "
        f"{len(ROUTES)}. Rows found: {[lbl for lbl, _ in family_rows]}. "
        "A new family was added to the contract without being added here."
    )
    for label, row in family_rows:
        certified_in_contract = "**certified**" in row
        stage_type = ROUTES[label]

        if stage_type is None:
            assert not certified_in_contract, (
                f"C-6.1 marks {label!r} as certified, but no stage-solver "
                "route implements it. A family with no implementation cannot "
                "have had its gradient and Hessian validated."
            )
            continue

        certified_in_code = stage_type in CERTIFIED_STAGE_TYPES
        assert certified_in_contract == certified_in_code, (
            f"{label}: NUMERICS.md C-6.1 says "
            f"{'certified' if certified_in_contract else 'not certified'} but "
            f"CERTIFIED_STAGE_TYPES says "
            f"{'certified' if certified_in_code else 'not certified'}"
        )

    # The converse direction: nothing may be certified in code without a row.
    routed = {st for st in ROUTES.values() if st is not None}
    unaccounted = CERTIFIED_STAGE_TYPES - routed
    assert not unaccounted, (
        f"CERTIFIED_STAGE_TYPES certifies {unaccounted}, which has no row in "
        "the C-6.1 table."
    )


def test_the_readme_status_table_agrees_with_the_code():
    """The README's own certified table is checked, not only its prose.

    ``tests/test_examples.py`` already forbids the README from advertising a
    *refused* family, and executes its quick start. Both of those guards
    point the same way: they catch the README claiming more than the code
    delivers. Nothing compared the README's status table with
    ``CERTIFIED_STAGE_TYPES``, so the opposite drift was invisible -- and it
    had happened, in the section this table sits above, which listed
    factorisation reuse as unimplemented long after C-15 delivered and
    counted it.

    ``docs/architecture.md`` is checked in both directions. This gives the
    README, the document a reader meets first, the same treatment.
    """
    from adjungo.core.method import StageType
    from adjungo.optimization.interface import CERTIFIED_STAGE_TYPES

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## What is supported"):]

    #: README row label -> implementing stage type, or ``None`` where the
    #: family has no route at all. The labels carry their method names, so a
    #: reworded row surfaces here as unknown rather than being skipped.
    ROUTES = {
        "Explicit Runge–Kutta (`explicit_euler`, `heun`, `rk4`)": (
            StageType.EXPLICIT
        ),
        "DIRK (`implicit_trapezoid` / Crank–Nicolson)": StageType.DIRK,
        "SDIRK (`implicit_midpoint`, `sdirk2`, `sdirk3`)": StageType.SDIRK,
        "Fully implicit, dense `A` (`gauss2`)": StageType.IMPLICIT,
        "BDF (`bdf2`, `bdf3`)": None,
        "Adams (`adams_bashforth2`, `adams_moulton2`)": None,
        "IMEX / additive splitting": None,
    }

    rows = _first_table_block(section)
    labels = {label for label, _ in rows}

    unknown = labels - set(ROUTES)
    assert not unknown, (
        f"README lists method families this test cannot verify: "
        f"{sorted(unknown)}. Record the implementing stage type here, or "
        "None if the family has no route."
    )
    missing = set(ROUTES) - labels
    assert not missing, (
        f"README no longer lists {sorted(missing)}. Dropping a row removes "
        "a certification or a refusal without removing the capability."
    )

    for label, row in rows:
        certified_in_readme = "**certified**" in row
        stage_type = ROUTES[label]

        if stage_type is None:
            assert not certified_in_readme, (
                f"README marks {label!r} certified, but no stage-solver route "
                "implements it."
            )
            continue

        certified_in_code = stage_type in CERTIFIED_STAGE_TYPES
        assert certified_in_readme == certified_in_code, (
            f"{label}: README says "
            f"{'certified' if certified_in_readme else 'not certified'} but "
            f"CERTIFIED_STAGE_TYPES says "
            f"{'certified' if certified_in_code else 'not certified'}"
        )

    # Under-claiming is the direction the other README guards miss.
    routed = {ROUTES[label] for label, row in rows if "**certified**" in row}
    unadvertised = CERTIFIED_STAGE_TYPES - routed
    assert not unadvertised, (
        f"the code certifies {unadvertised}, which the README does not "
        "present as certified. Delivered capability must not exceed what the "
        "README tells a reader is available."
    )


# ---------------------------------------------------------------------------
# The archived reports
# ---------------------------------------------------------------------------


def _archived_reports() -> list[Path]:
    return sorted(p for p in HISTORY.glob("*.md") if p.name != "README.md")


def test_archived_reports_are_marked_superseded():
    """Every archived report says so in its first line.

    These files contain obsolete tolerances, defunct file paths and test counts
    from trees that no longer exist. Read without the banner they look like
    documentation, and several of them contradict the current contract.
    """
    reports = _archived_reports()
    assert reports, "docs/history/ holds no archived reports"

    for report in reports:
        first = report.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("> **SUPERSEDED"), (
            f"{report.name} does not open with the superseded banner"
        )


def test_every_archived_report_is_accounted_for():
    """The archive index names every file it is supposed to account for.

    ``docs/history/README.md`` is the record of where each report's durable
    content went, and is the precondition for ever deleting them. An archived
    file missing from that table has been quietly dropped from the accounting.
    """
    index = (HISTORY / "README.md").read_text(encoding="utf-8")

    present = {p.name for p in _archived_reports()}

    missing = sorted(n for n in present if f"`{n}`" not in index)
    assert not missing, (
        "docs/history/README.md does not account for: " + ", ".join(missing)
    )

    # The converse. A row naming a file that is gone asserts that the file's
    # content was accounted for and is still available to check, which is then
    # false. That is worse than no row: the index is the stated precondition
    # for deleting these files, so a stale row can authorise a second deletion.
    cited = {m.group(1) for m in re.finditer(r"^\| `([A-Za-z0-9_]+\.md)`", index, re.MULTILINE)}
    vanished = sorted(cited - present)
    assert not vanished, (
        "docs/history/README.md accounts for files that are not there: "
        + ", ".join(vanished)
        + ". Remove the row, or restore the file."
    )


def test_no_ad_hoc_reports_remain_in_the_repository_root():
    """Only the four maintained documents live at the top level.

    The archived files accumulated there one session at a time, each
    reasonable on its own, until the root held more obsolete reports than
    current documentation. This test is the ratchet.
    """
    allowed = {"README.md", "NUMERICS.md", "AGENTS.md", "CHANGELOG.md"}
    present = {p.name for p in ROOT.glob("*.md")}

    unexpected = sorted(present - allowed)
    assert not unexpected, (
        "unexpected Markdown files in the repository root: "
        + ", ".join(unexpected)
        + ". Session reports and working notes belong in docs/history/ with a "
        "superseded banner and an entry in docs/history/README.md."
    )


# --------------------------------------------------------------------------
# Collection parity
#
# `testpaths` is what makes bare `pytest` safe. CI checks this too, but the
# CI check was vacuous from the day it was written, so the property is held
# here as well, where it can be run and falsified locally.
# --------------------------------------------------------------------------


def _collected_node_ids(*args: str) -> list[str]:
    """Node IDs from a collection run, with the quiet level pinned.

    ``addopts`` is replaced rather than extended. ``pyproject.toml`` sets
    ``-q``, so passing ``-q`` here would be the *second* one, and at that
    level pytest stops listing node IDs and prints per-file counts instead.
    ``--strict-markers`` is restored explicitly because collection is
    exactly when it applies.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *args,
            "--collect-only",
            "-q",
            "-o",
            "addopts=--strict-markers",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,  # the assertion below reports the output, not CalledProcessError
    )
    assert completed.returncode == 0, (
        f"collection failed for {args or ('<bare>',)}:\n{completed.stdout}"
        f"\n{completed.stderr}"
    )
    return sorted(
        line.strip() for line in completed.stdout.splitlines() if "::" in line
    )


def test_bare_collection_and_scoped_collection_agree():
    """``pytest`` and ``pytest tests/`` must collect the same node IDs.

    Root-level ``test_*.py`` files once made these two commands disagree,
    and one of them raised at import and broke bare collection outright.
    ``testpaths = ["tests"]`` in ``pyproject.toml`` is what keeps them
    equal, and this test is what notices if that stops being true.

    The node IDs are compared, not a count and not a summary line. The
    equivalent CI step compared ``tail -1`` of a doubled-``-q`` run, which
    is an empty line, so it compared ``""`` with ``""`` and passed while
    bare collection saw 833 tests and a deliberately narrowed scoped
    collection saw 819. A guard that cannot fail is not a guard; see the
    same ``-q`` doubling hazard recorded in ``AGENTS.md`` under R-11.
    """
    bare = _collected_node_ids()
    scoped = _collected_node_ids("tests/")

    assert bare, "bare collection produced no node IDs; the comparison would be vacuous"

    only_bare = sorted(set(bare) - set(scoped))
    only_scoped = sorted(set(scoped) - set(bare))
    assert bare == scoped, (
        "bare `pytest` and `pytest tests/` collect different tests.\n"
        f"  only bare ({len(only_bare)}): {only_bare[:5]}\n"
        f"  only scoped ({len(only_scoped)}): {only_scoped[:5]}"
    )


# --------------------------------------------------------------------------
# Language level (C-11.4)
#
# The declared floor is a claim about which interpreters can import this
# package. A single development interpreter cannot observe a violation: the
# one used here runs every construct in the tree. CI is what checks it, so
# the matrix must name the floor and nothing may be pinned below it.
# --------------------------------------------------------------------------


def _version(text: str) -> tuple[int, int]:
    """``3.13``, ``"3.13"``, ``py313`` and ``>=3.13`` as a comparable pair."""
    digits = re.search(r"(\d+)\.(\d+)", text.replace("py3", "3."))
    assert digits, f"no version found in {text!r}"
    return int(digits.group(1)), int(digits.group(2))


def test_the_ci_matrix_exercises_the_declared_python_floor():
    """C-11.4: every interpreter CI names is one the package admits.

    ``requires-python`` said ``>=3.10`` while the suite imported
    ``typing.Self`` (3.11) and referenced ``copy.replace`` (3.13) at module
    scope. Both fail at collection rather than skipping. The matrix named
    3.10 and 3.12 and would have caught either, but had not run; see R-12.

    Two properties are held here. The floor must be exercised, so that the
    claim is measured rather than asserted, and no job may pin below it,
    because such a job installs a package whose own metadata refuses that
    interpreter.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = _version(config["project"]["requires-python"])

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    listed = re.search(r"python-version:\s*\[([^\]]*)\]", workflow)
    assert listed, "no python-version matrix found in ci.yml"
    matrix = [_version(entry) for entry in listed.group(1).split(",")]

    assert floor in matrix, (
        f"ci.yml never runs the declared floor {floor[0]}.{floor[1]}; "
        f"the matrix is {matrix}. The floor is then unmeasured."
    )

    pinned = [_version(v) for v in re.findall(r'python-version:\s*"([\d.]+)"', workflow)]
    below = [v for v in pinned + matrix if v < floor]
    assert not below, (
        f"ci.yml pins {below}, below the declared floor {floor[0]}.{floor[1]}. "
        "Those jobs cannot install this package."
    )


def test_ci_installs_every_extra_the_suite_can_skip_on():
    """A skip that CI also takes is a test that never runs anywhere.

    ``tests/test_examples.py`` skips the rocket example when sympy is
    absent, so that a contributor who installed only ``[dev]`` gets a skip
    rather than a collection error. That courtesy becomes a hole the moment
    CI takes the same skip: the example, its closed-form anchor and its
    independent-reference comparisons would all report green while never
    executing. Every optional-dependency group must therefore appear in
    every CI install line.

    This is the same failure mode as R-12, one layer down: there the check
    existed and had not run; here the check would run and decline.
    """
    extras = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    installs = re.findall(r"pip install -e [\"']\.\[([^\]]*)\]", workflow)
    assert installs, "no editable install with extras found in ci.yml"

    for requested in installs:
        named = {name.strip() for name in requested.split(",")}
        missing = extras - named
        assert not missing, (
            f"ci.yml installs .[{requested}] but pyproject declares {sorted(extras)}; "
            f"{sorted(missing)} would be absent, so every test guarded on it "
            "would silently skip in CI."
        )


def test_the_tool_targets_are_the_declared_floor():
    """C-11.4: mypy and ruff check the language level the package promises.

    A checker aimed above the floor cannot see syntax the floor rejects, and
    one aimed below it reports failures for a version nobody supports. Both
    settings had drifted: ``py310`` remained in the ruff and black targets,
    and mypy checked 3.12, long after the code required more.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = _version(config["project"]["requires-python"])

    assert _version(config["tool"]["mypy"]["python_version"]) == floor
    assert _version(config["tool"]["ruff"]["target-version"]) == floor
    for target in config["tool"]["black"]["target-version"]:
        assert _version(target) == floor


# --------------------------------------------------------------------------
# docs/architecture.md
#
# The document this replaced was an ASCII diagram of an intended design. It
# listed `solvers/imex.py`, automatic differentiation of user callbacks, and
# genericity over the scalar type, none of which were ever written; it omitted
# `validation/`, which is the repository's primary oracle. Nothing detected the
# drift because nothing could: no part of the document was checkable.
#
# These tests make the structural claims checkable. They deliberately do not
# judge prose.
# --------------------------------------------------------------------------

ARCHITECTURE = ROOT / "docs" / "architecture.md"


def _module_map() -> set[str]:
    """Paths claimed by the fenced module map, as ``package/module.py``."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    block = re.search(r"```\nadjungo/\n(.*?)```", text, re.DOTALL)
    assert block, "docs/architecture.md has no fenced module map beginning 'adjungo/'"

    claimed: set[str] = set()
    package = ""
    for line in block.group(1).splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        token = line.strip().split()[0]
        if token.endswith("/"):
            # A nested package keeps its parent prefix; a top-level one resets.
            package = token if indent <= 2 else package + token
        elif token.endswith(".py"):
            claimed.add(package + token)
    return claimed


def test_architecture_module_map_matches_the_package():
    """Every module the map names exists, and every module is named.

    Both directions matter, and for different reasons. A named module that does
    not exist is the advertised-capability defect: `solvers/imex.py` appeared in
    the design document and in the architecture diagram, and a reader had no way
    to tell that no such file had ever been written. A module that exists but is
    unnamed is the opposite failure -- `validation/reference.py`, the primary
    oracle of C-14.1, was absent from the diagram entirely.
    """
    package_root = ROOT / "adjungo"
    actual = {
        str(p.relative_to(package_root))
        for p in package_root.rglob("*.py")
        if p.name != "__init__.py"
    }
    claimed = _module_map()

    phantom = sorted(claimed - actual)
    assert not phantom, (
        "docs/architecture.md describes modules that do not exist: "
        + ", ".join(phantom)
        + ". Describing unwritten code as part of the architecture is the "
        "defect class NUMERICS.md C-1 exists to prevent."
    )

    undocumented = sorted(actual - claimed)
    assert not undocumented, (
        "docs/architecture.md omits: "
        + ", ".join(undocumented)
        + ". A reimplementer reading the map would not know these exist."
    )


def test_architecture_symbol_map_resolves():
    """Every code symbol in the glm_opt.tex-to-code table is a real attribute.

    The symbol map is the bridge a reimplementer crosses between the derivation
    and the code, and it is the only place recording that `V` is overloaded:
    `glm_opt.tex` uses it both for the GLM propagation matrix, which is
    `method.V`, and for an unrelated bilinear form that has no code symbol at
    all. A stale attribute name here sends the reader to the wrong operator.
    """
    import adjungo
    from adjungo.stepping.adjoint import AdjointTrajectory
    from adjungo.stepping.sensitivity import (
        AdjointSensitivityTrajectory,
        SensitivityTrajectory,
    )
    from adjungo.stepping.trajectory import Trajectory

    owners = {
        "method": adjungo.GLMethod,
        "problem": adjungo.Problem,
        "trajectory": Trajectory,
        "adjoint": AdjointTrajectory,
        "sensitivity": SensitivityTrajectory,
        "adj_sensitivity": AdjointSensitivityTrajectory,
    }

    text = ARCHITECTURE.read_text(encoding="utf-8")
    table = text.split("## Symbol map")[1].split("\n---")[0]

    checked = 0
    unresolved = []
    for row in table.splitlines():
        if not row.startswith("|"):
            continue
        cells = row.split("|")
        if len(cells) < 3:
            continue
        for ref in re.findall(r"`([a-z_]+)\.([A-Za-z_]+)", cells[2]):
            owner, attribute = ref
            if owner not in owners:
                continue
            checked += 1
            target = owners[owner]
            fields = set(getattr(target, "__annotations__", {}))
            if not hasattr(target, attribute) and attribute not in fields:
                unresolved.append(f"{owner}.{attribute}")

    assert checked >= 10, (
        f"only {checked} symbol-map entries were checkable; the table's shape "
        "has changed and this test is no longer examining it"
    )
    assert not unresolved, (
        "docs/architecture.md maps glm_opt.tex symbols onto attributes that do "
        "not exist: " + ", ".join(sorted(set(unresolved)))
    )


def test_architecture_does_not_claim_uncertified_families():
    """The 'not built' table and C-6.1 must not contradict each other.

    The previous architecture document described additive/IMEX splitting and
    partitioned methods as part of the design, in the same register as the
    parts that worked. Both are refused by `GLMOptimizer`.
    """
    from adjungo.optimization.interface import CERTIFIED_STAGE_TYPES

    text = ARCHITECTURE.read_text(encoding="utf-8")
    not_built = text.split("## What is not built")[1].split("\n---")[0].lower()

    for family, marker in (("imex", "IMEX"), ("multistep", "multistep")):
        assert family in not_built, (
            f"docs/architecture.md no longer records {marker} as unbuilt. "
            "It is still refused by GLMOptimizer."
        )

    certified = {stage_type.name.lower() for stage_type in CERTIFIED_STAGE_TYPES}
    for name in certified:
        assert f"| {name} |" not in not_built.replace("`", ""), (
            f"docs/architecture.md lists {name} as not built, but "
            "CERTIFIED_STAGE_TYPES certifies it."
        )


# ---------------------------------------------------------------------------
# The LaTeX design documents
# ---------------------------------------------------------------------------

#: Rules stated in ``docs/linalg_requirements.tex`` that later certification
#: work refuted, each paired with the clause that corrects it. The document is
#: the design source for the eventual C++ library, so a reimplementer who
#: reads it without the correction reintroduces a defect that has already been
#: found and cured once.
REFUTED_DESIGN_RULES = ["C-15.1", "C-15.2", "C-16.1", "C-16"]


def _linalg_requirements() -> str:
    """The document with runs of whitespace collapsed.

    LaTeX source is hard-wrapped, so a phrase this guard looks for is very
    likely to straddle a line break.
    """
    text = (ROOT / "docs" / "linalg_requirements.tex").read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_linalg_requirements_declares_itself_a_design_document():
    """The document must not read as a description of delivered code.

    It specifies sparse and matrix-free backends, Krylov solves, IMEX
    splitting, automatic differentiation, and multistep paths. None of those
    is built, and one of them is explicitly refused.
    """
    text = _linalg_requirements()
    assert "Status: design document" in text
    assert "not a description of the delivered" in text


@pytest.mark.parametrize("clause", REFUTED_DESIGN_RULES)
def test_linalg_requirements_records_its_refuted_rules(clause):
    """Each refuted rule names the clause that corrects it."""
    assert clause in _linalg_requirements(), (
        f"docs/linalg_requirements.tex no longer cites {clause}. Its status "
        f"preamble is what stops a C++ port re-deriving factorization reuse "
        f"from the tableau, or reading state linearity as zero curvature."
    )


def test_linalg_requirements_does_not_derive_reuse_from_the_tableau():
    """The refuted predicate must not return under its old name.

    ``has_reusable_factorization() { return stage_type == SDIRK; }`` reads as
    a settled fact. Equal diagonal coefficients do not make the assembled
    stage matrices equal, because ``I - h*a_ii*F_i`` also depends on ``F_i``.
    The name now says the tableau only permits an attempt.
    """
    text = _linalg_requirements()
    assert "has_reusable_factorization" not in text
    assert "may_attempt_factorization_reuse" in text


def test_sources_citing_the_design_document_carry_the_caveat():
    """A citation of the design document must not present it as settled.

    Two dispatch modules name this document as the decision tree they
    implement. Both implement rules the document states incorrectly, so an
    unqualified citation points a reader at a refuted source.
    """
    citing = [
        path
        for path in (ROOT / "adjungo").rglob("*.py")
        if "linalg_requirements" in path.read_text(encoding="utf-8")
    ]
    assert citing, "no source cites linalg_requirements.tex; update this guard"

    uncaveated = [
        path.relative_to(ROOT)
        for path in citing
        if "refuted" not in path.read_text(encoding="utf-8")
    ]
    assert not uncaveated, (
        "these modules cite docs/linalg_requirements.tex without recording "
        "that parts of it were refuted:\n  "
        + "\n  ".join(str(p) for p in uncaveated)
    )


#: Derivative tensors that must never appear under a ``\sum_j a_{ji}`` in
#: ``docs/runge_kutta_opt.tex``. Differentiating the stage residual gives
#: ``df_j/dz_i = delta_ji F_i``, so the surviving Jacobian carries the free
#: index ``i`` and factors out of the sum. Writing index ``j`` inside the sum
#: is precedent R-5, the defect that made every multi-stage method in this
#: repository return an inexact gradient.
SUMMATION_INDEXED_TENSORS = [
    r"\(F_k\^j\)",
    r"\(G_k\^j\)",
    r"F_\{zz\}\^\{k,j\}",
    r"F_\{zu\}\^\{k,j\}",
    r"F_\{uu\}\^\{k,j\}",
]


def _runge_kutta_opt() -> str:
    return (ROOT / "docs" / "runge_kutta_opt.tex").read_text(encoding="utf-8")


def _runge_kutta_opt_body() -> list[tuple[int, str]]:
    """The document with its correction notice removed.

    The notice has to quote the defective equation in order to explain it, so
    scanning the whole file would flag the explanation as the defect.
    """
    lines = _runge_kutta_opt().splitlines()
    start = next(
        (n for n, line in enumerate(lines) if line.startswith(r"\section{")),
        0,
    )
    assert start, "no numbered section found; the notice boundary moved"
    return [(n + 1, line) for n, line in enumerate(lines) if n >= start]


def test_runge_kutta_opt_records_its_stage_index_correction():
    """The document once stated the adjoint with the R-5 defect in it.

    It is the specification a C++ port would follow, so the correction has to
    be visible in it, not only in the commit that made it.
    """
    text = re.sub(r"\s+", " ", _runge_kutta_opt())
    assert "Correction notice: the stage index in the adjoint" in text
    assert "precedent R-5" in text


@pytest.mark.parametrize("tensor", SUMMATION_INDEXED_TENSORS)
def test_runge_kutta_opt_never_indexes_a_tensor_by_the_summation_index(tensor):
    """No ``a_{ji}`` may be followed by a ``j``-indexed derivative tensor.

    The aggregate definitions legitimately weight ``mu_k^j`` by ``a_{ji}``;
    what may not appear is a Jacobian or curvature operator carrying ``j``
    inside that sum.
    """
    offenders = [
        (number, line.strip())
        for number, line in _runge_kutta_opt_body()
        if re.search(r"a_\{ji\}", line) and re.search(tensor, line)
    ]
    assert not offenders, (
        f"docs/runge_kutta_opt.tex indexes {tensor} by the summation index "
        f"inside a sum weighted by a_{{ji}}. The Jacobian must carry the free "
        f"stage index and factor out of the sum (precedent R-5):\n  "
        + "\n  ".join(f"line {n}: {t}" for n, t in offenders)
    )


def test_the_adjoint_block_matrix_is_the_forward_transpose():
    """``M_k`` must carry its Jacobian on the row, not the column.

    This is the same statement as the test above, checked on the assembled
    block matrix where the original document contradicted its own claim that
    ``M_k = A_k^T``.
    """
    rng = np.random.default_rng(0)
    n, d, h = 3, 2, 0.17
    a = rng.normal(size=(d, d))
    F = [rng.normal(size=(n, n)) for _ in range(d)]
    eye = np.eye(n)

    def assemble(block):
        out = np.zeros((d * n, d * n))
        for i in range(d):
            for j in range(d):
                out[i * n : (i + 1) * n, j * n : (j + 1) * n] = block(i, j)
        return out

    forward = assemble(lambda i, j: (eye if i == j else 0.0) - h * a[i, j] * F[j])
    row_indexed = assemble(
        lambda i, j: (eye if i == j else 0.0) - h * a[j, i] * F[i].T
    )
    column_indexed = assemble(
        lambda i, j: (eye if i == j else 0.0) - h * a[j, i] * F[j].T
    )

    assert np.allclose(forward.T, row_indexed, rtol=0.0, atol=1e-14)
    assert not np.allclose(forward.T, column_indexed, atol=1e-8), (
        "the two forms agree, so this fixture cannot discriminate; it needs "
        "distinct per-stage Jacobians and a non-symmetric tableau"
    )


def _injection_tables(text: str) -> list[tuple[int, list[str]]]:
    """Each injection table in a document, as (line number, defect cells)."""
    tables: list[tuple[int, list[str]]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("| Injected defect |"):
            continue
        rows: list[str] = []
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            rows.append(row.split("|")[1].strip())
        tables.append((index + 1, rows))
    return tables


def test_each_injection_campaign_is_reported_once():
    """A campaign belongs to the clause whose evidence it is.

    These tables are regenerated by script, and a script that locates one by
    searching for its header can splice a fresh campaign into the table
    above it. C-6's certification evidence was once overwritten in exactly
    that way with C-15.7's coefficient-immutability rows, leaving the clause
    introducing a 312-test baseline and then listing ninety defects injected
    into ``__setstate__``. Two clauses reporting the same injected defect is
    the signature.
    """
    tables = [
        (document, line, rows)
        for document in CONTRACT_DOCUMENTS
        for line, rows in _injection_tables(document.read_text(encoding="utf-8"))
    ]
    assert len(tables) >= 4

    seen: dict[str, tuple[Path, int]] = {}
    for document, line, rows in tables:
        location = (document.relative_to(ROOT), line)
        assert rows, f"empty injection table at {location[0]}:{line}"
        for defect in rows:
            first = seen.setdefault(defect, location)
            assert first == location, (
                f"{location[0]}:{line} repeats the defect first reported at "
                f"{first[0]}:{first[1]}: {defect!r}"
            )


def test_a_reported_campaign_size_matches_its_table():
    """``n of n detected`` is a count of the rows that follow it.

    Only a complete claim is checked. A table whose lead states nothing, or
    states a partial ``n of m``, is skipped, so four of the five campaigns
    recorded across the contract and its evidence are currently unguarded.
    Recognition of partial claims is deferred. Tables without a stated
    campaign size provide no total to compare; the final assertion requires
    at least one recognized complete claim.
    """
    checked = 0
    for document in CONTRACT_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        lines = text.splitlines()
        for line, rows in _injection_tables(text):
            lead = "\n".join(lines[max(0, line - 8) : line - 1])
            stated = re.search(r"(\d+) of \1\b", lead)
            if stated is None:
                continue
            checked += 1
            assert len(rows) == int(stated.group(1)), (
                f"{document.relative_to(ROOT)}:{line} claims "
                f"{stated.group(1)} defects and lists {len(rows)}"
            )
    assert checked, "no injection table states its size"
