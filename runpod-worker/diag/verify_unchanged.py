"""Mechanical proof that the diagnostics patch leaves the separation algorithm untouched.

Usage: python verify_unchanged.py BASE_handler.py PATCHED_handler.py

How it works (stdlib only):
  1. Both files are parsed with `ast`.
  2. From the PATCHED tree we remove ONLY nodes that match an explicit allow-list of
     diagnostic patterns (listed below). Anything else that differs is NOT removed.
  3. The remaining PATCHED tree must be AST-identical (ast.dump, node by node, in order)
     to the BASE tree. So every constant, every function body and every statement of the
     original algorithm is byte-for-byte equivalent at AST level.
  4. Extra guarantees about the removed statements:
       * no remaining (algorithm) code references any `_diag_*` name or `diag_fields`;
       * diagnostic calls used as statements only take pure arguments (names, constants,
         bool/int/_label_count calls) so they cannot change algorithm state;
       * `diag_fields` is only ever consumed by the `**diag_fields` entry of the response.
"""
import ast
import sys

BASE_PATH, PATCHED_PATH = sys.argv[1], sys.argv[2]
base = ast.parse(open(BASE_PATH, encoding="utf-8").read(), BASE_PATH)
patched = ast.parse(open(PATCHED_PATH, encoding="utf-8").read(), PATCHED_PATH)

ADDED_IMPORTS = {"hashlib", "importlib.metadata"}
ADDED_TOP_ASSIGNS = {"_diag_extra", "DIAG_PATCH_VERSION", "DIAG_BASE_COMMIT", "ROWS_CAP"}
PURE_CALLS = {"bool", "int", "len", "type", "_label_count"}
removed = {"top_level": [], "statements": []}
failures = []


def fail(message):
    failures.append(message)


def is_name(node, ident=None, prefix=None):
    return isinstance(node, ast.Name) and (
        (ident is not None and node.id == ident) or (prefix is not None and node.id.startswith(prefix))
    )


def pure_expr(node):
    """Argument expressions of a diagnostic call must be side-effect free."""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.Lambda)):
            return False
        if isinstance(sub, ast.Call):
            func = sub.func
            if not (isinstance(func, ast.Name) and func.id in PURE_CALLS):
                return False
        if isinstance(sub, (ast.Attribute,)) and not isinstance(sub.ctx, ast.Load):
            return False
    return True


def diagnostic_statement(stmt, in_function):
    """Return a label if `stmt` is one of the allowed diagnostic statements, else None."""
    # _diag_xxx(...) as a statement
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if is_name(call.func, prefix="_diag_"):
            args = list(call.args) + [kw.value for kw in call.keywords]
            if all(pure_expr(arg) or isinstance(arg, (ast.Dict, ast.Constant)) for arg in args):
                # dict arguments (requested/effective bounds) may only contain plain names/constants
                for arg in args:
                    if isinstance(arg, ast.Dict) and not all(
                        isinstance(v, (ast.Name, ast.Constant)) for v in arg.values
                    ):
                        fail(f"{in_function}: non-trivial dict argument in diagnostic call")
                return "diag_call"
            fail(f"{in_function}: diagnostic call with non-pure argument (line {stmt.lineno})")
        # _diag_extra.clear()
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "clear"
            and is_name(func.value, ident="_diag_extra")
        ):
            return "diag_extra_clear"
    # _diag_extra["x"] = <expr>
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Subscript) and is_name(target.value, ident="_diag_extra"):
            if not pure_expr(stmt.value):
                fail(f"{in_function}: non-pure value stored into _diag_extra (line {stmt.lineno})")
            return "diag_extra_store"
        # diag["speaker_count_step_s"] = round(step, 5)   (pre-existing hook dict, new key only)
        if (
            in_function == "_diagnostic_hook"
            and isinstance(target, ast.Subscript)
            and is_name(target.value, ident="diag")
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "speaker_count_step_s"
        ):
            return "hook_new_key"
        # diag_fields = {} / diag_fields = _diag_output_fields(...)
        if is_name(target, ident="diag_fields") and in_function == "handler":
            value = stmt.value
            if isinstance(value, ast.Dict) and not value.keys:
                return "diag_fields_init"
            if isinstance(value, ast.Call) and is_name(value.func, ident="_diag_output_fields"):
                return "diag_fields_build"
    # if diarization and not result["segments"]: diag_fields = {...}
    if isinstance(stmt, ast.If) and in_function == "handler":
        body_ok = (
            len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Assign)
            and len(stmt.body[0].targets) == 1
            and is_name(stmt.body[0].targets[0], ident="diag_fields")
            and not stmt.orelse
        )
        test = ast.dump(stmt.test)
        if body_ok and "Not" in test and "'diarization'" in test and "'segments'" in test:
            return "diag_fields_skipped_branch"
    return None


class Strip(ast.NodeTransformer):
    def __init__(self):
        self.function = "<module>"

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = previous
        return node

    def _filter(self, statements):
        kept = []
        for stmt in statements:
            label = diagnostic_statement(stmt, self.function)
            if label:
                removed["statements"].append((self.function, label))
                continue
            kept.append(stmt)
        return kept or [ast.Pass()]

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                setattr(node, field, self._filter(value))
        for handler in getattr(node, "handlers", []) or []:
            handler.body = self._filter(handler.body)
        return super().generic_visit(node)

    def visit_Dict(self, node):
        # drop the single `**diag_fields` entry of the response dict
        keys, values = [], []
        for key, value in zip(node.keys, node.values):
            if key is None and is_name(value, ident="diag_fields"):
                removed["statements"].append((self.function, "response_merge"))
                continue
            keys.append(key)
            values.append(self.visit(value) if value is not None else value)
        node.keys, node.values = keys, values
        return node


# ---- 1) top-level additions -------------------------------------------------------------
base_names = {
    n.name for n in base.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
}
kept_top = []
for node in patched.body:
    if isinstance(node, ast.Import) and {a.name for a in node.names} <= ADDED_IMPORTS:
        removed["top_level"].append(("import", ",".join(a.name for a in node.names)))
        continue
    if isinstance(node, ast.FunctionDef) and node.name not in base_names and (
        node.name.startswith("_diag_")
        or node.name in {
            "_as_number", "_effective_settings", "_library_versions", "_count_words",
            "_binding_stats", "_capped_rows", "_build_diarization_debug",
        }
    ):
        removed["top_level"].append(("function", node.name))
        continue
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ADDED_TOP_ASSIGNS
    ):
        removed["top_level"].append(("assign", node.targets[0].id))
        continue
    kept_top.append(node)

# an added helper must never shadow a base definition
for kind, name in removed["top_level"]:
    if kind == "function" and name in base_names:
        fail(f"added function shadows a base function: {name}")

# ---- 2) strip diagnostic statements inside the remaining functions -------------------------
stripper = Strip()
stripped = [stripper.visit(node) for node in kept_top]

# ---- 3) AST equality with the base ---------------------------------------------------------
if len(stripped) != len(base.body):
    fail(f"top-level node count differs: base={len(base.body)} patched(stripped)={len(stripped)}")
for index, (b, p) in enumerate(zip(base.body, stripped)):
    if ast.dump(b) != ast.dump(p):
        label = getattr(b, "name", None) or type(b).__name__
        fail(f"top-level node #{index} ({label}) differs from base after removing diagnostics")

# ---- 4) algorithm code must not read any diagnostic state ----------------------------------
# Names that already exist in the BASE (e.g. _diag_runs / _diag_current, the pre-existing hook
# log) are not "new diagnostic state"; only names introduced by the patch are forbidden.
base_defined = set(base_names)
for node in base.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                base_defined.add(target.id)
new_names = {name for kind, name in removed["top_level"] if kind in ("function", "assign")}
new_names.add("diag_fields")
for node in stripped:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in new_names and sub.id not in base_defined:
            fail(f"algorithm code still references diagnostic name: {sub.id}")

# `diag_fields` consumers in the ORIGINAL patched handler: only the response merge
handler = next(n for n in patched.body if isinstance(n, ast.FunctionDef) and n.name == "handler")
loads = [
    sub for sub in ast.walk(handler)
    if isinstance(sub, ast.Name) and sub.id == "diag_fields" and isinstance(sub.ctx, ast.Load)
]
merges = [
    value for dict_node in ast.walk(handler) if isinstance(dict_node, ast.Dict)
    for key, value in zip(dict_node.keys, dict_node.values)
    if key is None and is_name(value, ident="diag_fields")
]
if len(loads) != len(merges):
    fail(f"diag_fields is read outside the response merge ({len(loads)} loads, {len(merges)} merges)")

# ---- report ----------------------------------------------------------------------------------
print(f"base top-level nodes     : {len(base.body)}")
print(f"patched top-level nodes  : {len(patched.body)}")
print(f"added top-level items    : {len(removed['top_level'])}")
for kind, name in removed["top_level"]:
    print(f"   + {kind}: {name}")
counts = {}
for function, label in removed["statements"]:
    counts[(function, label)] = counts.get((function, label), 0) + 1
print(f"added statements inside existing functions: {len(removed['statements'])}")
for (function, label), count in sorted(counts.items()):
    print(f"   + {function}: {label} x{count}")

if failures:
    print("\nUNCHANGED_CHECK_FAILED")
    for message in failures:
        print(" -", message)
    sys.exit(1)
print("\nUNCHANGED_CHECK_OK: after removing the listed diagnostic nodes, every original node is AST-identical.")
