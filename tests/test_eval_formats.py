from pathlib import Path
import importlib.util

r=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,r/(name+'.py'))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# These two modules import only Python's standard library.
p=load('code_prompt_formats')
o=load('code_output_policy')
mb=p.mbpp_task_prompt('Add two numbers.', ['assert add(2, 3) == 5'])
assert 'assert add(2, 3) == 5' in mb
ap=p.approach_first_task_prompt(mb)
assert p.APPROACH_FIRST_INSTRUCTION in ap and 'assert add(2, 3) == 5' in ap
hp=p.humaneval_task_prompt('def add(a, b):\n    """Return the sum of a and b."""\n','add')
assert 'def add(a, b):' in hp and 'Return the sum of a and b.' in hp
raw='Brief explanation.\n```python\ndef add(a, b):\n    return a + b\n```\nAfterword.'
code,kind=o.extract_code(raw)
assert code=='def add(a, b):\n    return a + b\n'
assert kind=='single_python_fence_with_surrounding_prose'
assert o.closing_fence_end(raw)==raw.index('\nAfterword.')
assert o.extract_code('```python\nprint(1)')[0] is None
assert o.extract_code('print(1)')==('print(1)','plain_python')
print('STDLIB_PROMPT_AND_OUTPUT_POLICY_CHECKS_PASS: 8 assertions; no generated code executed')
