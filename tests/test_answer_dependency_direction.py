"""Keep answer orchestration above independently importable evidence policies."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "backend" / "agent" / "nodes"
ORCHESTRATORS = {
    "answer",
}


def answer_dependencies():
    modules = {path.stem: path for path in NODES.glob("answer*.py")}
    edges = {name: set() for name in modules}
    for name, path in modules.items():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    targets.append(node.module or "")
                elif (node.module or "").startswith("backend.agent.nodes."):
                    targets.append(node.module.rsplit(".", 1)[-1])
                elif node.module == "backend.agent.nodes":
                    targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                targets.extend(
                    alias.name.rsplit(".", 1)[-1]
                    for alias in node.names
                    if alias.name.startswith("backend.agent.nodes.")
                )
            edges[name].update(target for target in targets if target in modules)
    return edges


class AnswerDependencyDirectionTests(unittest.TestCase):
    def test_terminal_answer_does_not_restore_removed_lexical_fta_rewriter(self):
        tree = ast.parse((NODES / "answer.py").read_text(encoding="utf-8"))
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("_ensure_fta_structure_only_completeness", calls)

    def test_all_answer_modules_have_an_acyclic_import_graph(self):
        graph = answer_dependencies()
        complete = set()

        def visit(name, stack):
            self.assertNotIn(name, stack, " -> ".join([*stack, name]))
            if name in complete:
                return
            for target in graph[name]:
                visit(target, [*stack, name])
            complete.add(name)

        for name in graph:
            visit(name, [])

    def test_lower_layers_cannot_import_orchestration(self):
        for name, dependencies in answer_dependencies().items():
            with self.subTest(module=name):
                if name not in ORCHESTRATORS:
                    self.assertFalse(dependencies & ORCHESTRATORS)
                elif name != "answer":
                    self.assertNotIn("answer", dependencies)

    def test_each_domain_loads_without_importing_the_answer_entrypoint(self):
        for name in ("answer_evidence_context", "answer_source_fields", "answer_structure_views"):
            with self.subTest(module=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import importlib, sys; "
                        f"importlib.import_module('backend.agent.nodes.{name}'); "
                        "assert 'backend.agent.nodes.answer' not in sys.modules",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
