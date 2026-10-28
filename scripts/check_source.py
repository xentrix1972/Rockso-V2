#!/usr/bin/env python3
"""Check the explicit source allowlist without importing the training stack."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = (ROOT / 'TRACKED_FILES.txt').read_text().splitlines()
    assert names and len(names) == len(set(names)), 'Empty or duplicate allowlist'
    python_count = json_count = 0
    for name in names:
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT), 'Path escaped repository'
        assert path.is_file() and not path.is_symlink(), name
        data = path.read_bytes()
        assert len(data) < 250_000, f'Unexpected large artifact: {name}'
        text = data.decode('utf-8')
        assert '\x00' not in text, f'Binary content: {name}'
        if path.suffix == '.py':
            tree = ast.parse(text, filename=name)
            compile(tree, name, 'exec')
            python_count += 1
        elif path.suffix == '.json':
            json.loads(text)
            json_count += 1
    config = json.loads((ROOT / 'configs/rockso_v2_32.json').read_text())
    assert (config['n_layer'], config['n_embd'], config['mlp_hidden']) == (32, 2048, 6832)
    launcher = (ROOT / 'scripts/launch_rsi_32.sh').read_text()
    for flag in ('--muon-lr 0.011', '--adam-lr 0.0012', '--weight-decay 0.025',
                 '--normuon-stochastic-rounding', '--fp8-mlp-pad-to 6912'):
        assert flag in launcher, f'Recipe mismatch: {flag}'
    print(f'PASS: {len(names)} allowed text files; {python_count} Python and {json_count} JSON parsed.')
    print('No ML modules imported; no model, checkpoint, data or GPU accessed.')


if __name__ == '__main__':
    main()
