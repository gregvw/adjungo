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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NUMERICS = ROOT / "NUMERICS.md"
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


def test_numerics_internal_links_resolve():
    """Every ``[...](#...)`` in NUMERICS.md points at a real anchor.

    Eleven of the thirteen internal links in this document were broken when
    this test was written, all in the same way: the link was spelled from the
    clause title while the heading also carried a trailing status marker such
    as "— `APPROVED`", which GitHub folds into the generated anchor. The cure
    was explicit anchors; this test is what stops it recurring.
    """
    text = _numerics_text()
    anchors = _anchors(text)
    links = sorted(set(re.findall(r"\]\((#[^)]+)\)", text)))

    assert links, "no internal links found; the extraction pattern is wrong"

    broken = [link for link in links if link not in anchors]
    assert not broken, (
        "NUMERICS.md contains links to anchors that do not exist:\n  "
        + "\n  ".join(broken)
    )


#: Identifiers cited from source, tests or the C++ preservation notes. Listing
#: them explicitly, rather than scraping every "C-n" shaped string, keeps the
#: test from passing merely because a citation was deleted along with its
#: target.
CITED_IDENTIFIERS = [
    "C-2",
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
    "C-13",
    "C-14.1",
    "C-14.2",
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
