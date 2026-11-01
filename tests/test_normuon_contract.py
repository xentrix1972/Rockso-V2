"""CPU-only contract checks for the corrected Rockso V2 NorMuon coefficient.

Run from the repository root: python -B tests/test_normuon_contract.py
Only selected AST nodes are compiled. The trainer, torch and Triton are never
imported, and shape metadata never allocates a matrix. Kernel numerics, BF16
rounding and training quality are deliberately outside this test's scope.
"""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = ROOT / "rockso_hybrid/vendor/normuon_optimizer_v3.py"


def load_function(path, name, namespace, owner=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner) if owner else tree
    node = copy.deepcopy(next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class Scalar:
    """One test value plus shape metadata; enough tensor methods for this contract."""

    def __init__(self, value, shape=(6832, 2048), dtype="float32"):
        self.value, self.shape, self.dtype = float(value), shape, dtype
        self.grad = None

    @staticmethod
    def value_of(other):
        return other.value if isinstance(other, Scalar) else other

    def __mul__(self, other):
        return Scalar(self.value * self.value_of(other), self.shape)

    __rmul__ = __mul__

    def __sub__(self, other):
        return Scalar(self.value - self.value_of(other), self.shape)

    def __ge__(self, other):
        return Scalar(self.value >= self.value_of(other), self.shape)

    def size(self, dim):
        return self.shape[dim]

    def lerp_(self, other, weight):
        self.value += weight * (other.value - self.value)
        return self

    def bfloat16(self):
        return self

    def to(self, dtype):
        return self

    def detach(self):
        return self

    def float(self):
        return Scalar(self.value, self.shape)

    def clone(self):
        return Scalar(self.value, self.shape, self.dtype)

    def copy_(self, other):
        self.value = other.value
        return self

    def addcmul_(self, left, right, value):
        self.value += value * left.value * right.value
        return self

    def add_(self, other, alpha):
        self.value += alpha * other.value
        return self


class NorMuonContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.decay = load_function(OPTIMIZER, "cautious_wd_and_update", {})

        def update(route):
            def apply(parameter, direction, effective_lr, effective_wd):
                self.calls.append((route, effective_lr, effective_wd))
                self.decay(parameter, direction, effective_lr, effective_wd)
            return apply

        self.namespace = {
            "math": math,
            "torch": SimpleNamespace(float32="float32", bfloat16="bfloat16"),
            "polar_express": lambda grad, num_iters: grad,
            "normuon_variance_reduction": lambda direction, buffer, beta2, red_dim: direction,
            "cautious_wd_and_update": update("plain"),
            "cautious_wd_and_update_sr": update("sr"),
        }
        self.step = load_function(OPTIMIZER, "step", self.namespace, owner="NorMuon")

    def execute_step(self, shape=(6832, 2048), lr=0.011, lr_mul=None, wd_mul=None, mode="plain", missing_grad=False):
        parameter = Scalar(2.0, shape, "bfloat16" if mode == "master" else "float32")
        parameter.grad = None if missing_grad else Scalar(0.5, shape)
        if lr_mul is not None:
            parameter.lr_mul = lr_mul
        if wd_mul is not None:
            parameter.wd_mul = wd_mul
        group = dict(params=[parameter], lr=lr, weight_decay=0.025, momentum=0.95,
                     beta2=0.95, num_polar_iters=5,
                     fp32_master_weights=mode == "master", stochastic_rounding=mode == "sr")
        optimizer = SimpleNamespace(param_groups=[group], state={parameter: {
            "momentum_buffer": Scalar(0.0, shape), "second_momentum_buffer": Scalar(0.0, shape)}})
        self.step(optimizer)
        self.assertEqual(group["lr"], lr, "A parameter shape must never rewrite the base rate")
        return parameter

    def test_shape_independent_rate_and_coupled_decay(self):
        for shape in ((2048, 2048), (512, 2048), (6832, 2048), (2048, 6832)):
            for lr in (0.011, 0.0055, 0.007, 0.0035, 0.0):
                with self.subTest(shape=shape, lr=lr):
                    self.execute_step(shape=shape, lr=lr)
                    _, actual_lr, actual_wd = self.calls[-1]
                    self.assertAlmostEqual(actual_lr, lr, places=14)
                    self.assertAlmostEqual(actual_wd, lr * 0.025, places=14)

    def test_explicit_multipliers_and_all_update_routes(self):
        for mode in ("plain", "master", "sr"):
            for multiplier in (0.25, 1.0, 2.0):
                with self.subTest(mode=mode, multiplier=multiplier):
                    self.execute_step(lr_mul=multiplier, wd_mul=0.5, mode=mode)
                    route, actual_lr, actual_wd = self.calls[-1]
                    self.assertEqual(route, "sr" if mode == "sr" else "plain")
                    self.assertAlmostEqual(actual_lr, 0.011 * multiplier, places=14)
                    self.assertAlmostEqual(actual_wd, actual_lr * 0.025 * 0.5, places=14)

    def test_missing_gradient_skips_update(self):
        self.assertEqual(self.execute_step(missing_grad=True).value, 2.0)
        self.assertEqual(self.calls, [])

    def test_zero_rate_changes_neither_weight_nor_decay(self):
        self.assertEqual(self.execute_step(lr=0.0).value, 2.0)
        self.assertEqual(self.calls[-1][1:], (0.0, 0.0))

    def test_cautious_decay_alignment_including_zero(self):
        for weight, direction in ((2, 0.5), (2, -0.5), (-2, -0.5), (-2, 0.5), (2, 0)):
            with self.subTest(weight=weight, direction=direction):
                parameter = Scalar(weight)
                lr, wd = 0.011, 0.011 * 0.025
                self.decay(parameter, Scalar(direction), lr, wd)
                expected = weight - (wd * weight if weight * direction >= 0 else 0) - lr * direction
                self.assertAlmostEqual(parameter.value, expected, places=14)

    def test_stochastic_rounding_preserves_coupled_pre_rounding_formula(self):
        tree = ast.parse(OPTIMIZER.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cautious_wd_and_update_sr")
        nodes = [n for n in function.body if isinstance(n, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id in ("pf", "mask", "new") for target in n.targets)]
        self.assertEqual(len(nodes), 3)
        for weight, direction in ((2, 0.5), (2, -0.5), (-2, -0.5), (2, 0)):
            namespace = dict(p=Scalar(weight), v=Scalar(direction), eff_lr=0.011, eff_wd=0.011 * 0.025)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OPTIMIZER), "exec"), namespace)
            expected = weight - (0.011 * 0.025 * weight if weight * direction >= 0 else 0) - 0.011 * direction
            self.assertAlmostEqual(namespace["new"].value, expected, places=14)

    def test_schedule_preserves_configured_base_rates(self):
        scale = load_function(ROOT / "rockso_hybrid/runtime.py", "scale_lr", {"math": math}, owner="HybridOptimizer")
        for muon_lr, adam_lr in ((0.011, 0.0012), (0.007, 0.0005)):
            for factor in (0.0, 0.5, 1.0):
                with self.subTest(muon_lr=muon_lr, adam_lr=adam_lr, factor=factor):
                    optimizer = SimpleNamespace(base_lrs=(muon_lr, adam_lr),
                        muon=SimpleNamespace(param_groups=[{"lr": 999}]),
                        adam=SimpleNamespace(param_groups=[{"lr_multiplier": m} for m in (1, 4, 64)]))
                    scale(optimizer, factor)
                    self.assertEqual(optimizer.base_lrs, (muon_lr, adam_lr))
                    self.assertAlmostEqual(optimizer.muon.param_groups[0]["lr"], muon_lr * factor)
                    for group in optimizer.adam.param_groups:
                        self.assertAlmostEqual(group["lr"], adam_lr * group["lr_multiplier"] * factor)

    def test_future_launch_telemetry_reports_actual_rate(self):
        path = ROOT / "telemetry.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                 any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and
                     target.slice.value == "lr/normuon_gate_up_effective" for target in n.targets)]
        self.assertEqual(len(nodes), 1)
        for rate in (0.011, 0.0055, 0.007, 0.0035, 0.0):
            namespace = {"row": {"learning_rates": {"normuon": rate}}, "metrics": {}, "math": math}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
            self.assertEqual(namespace["metrics"]["lr/normuon_gate_up_effective"], rate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
