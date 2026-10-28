"""V2 coding prompt presentation; public task text only, no solution editing."""
import ast
import inspect


APPROACH_FIRST_VERSION = 'task_approach_first_3sentences_v1'
APPROACH_FIRST_INSTRUCTION = (
    'First explain your approach briefly in plain text (at most three sentences), '
    'including any important edge cases. Then return the complete implementation, '
    'including any needed imports, in one Python code block.'
)


def approach_first_task_prompt(prompt):
    """Apply the exact instruction tested on the 37-40k equal-weight soup."""
    endings = (
        'Return the complete implementation in one Python code block.',
        'Return the complete implementation, including any needed imports, in one Python code block.',
    )
    matches = [ending for ending in endings if prompt.endswith(ending)]
    if len(matches) != 1:
        raise ValueError('Expected a native task prompt with its original response instruction')
    return prompt[:-len(matches[0])] + APPROACH_FIRST_INSTRUCTION


def mbpp_task_prompt(text, public_tests):
    """Present only the target's published description and public test_list.

    No canonical code, challenge tests, or inferred signature is accepted here.
    This is an explicitly zero-shot instruction protocol.
    """
    return (f'In Python, solve the following task:\n\n{text}\n\n'
            'Use the function names shown in these examples; your implementation must pass them:\n\n'
            f'```python\n' + '\n'.join(public_tests) + '\n```\n\n'
            'Return the complete implementation, including any needed imports, in one Python code block.')


def humaneval_task_prompt(prompt, entry_point):
    """Move the public docstring outside the code scaffold, verbatim in meaning.

    HumanEval's function name, signature, imports, examples and description stay
    present. No hidden tests or canonical solution are accepted by this API.
    """
    tree = ast.parse(prompt)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == entry_point)
    if (not isinstance(node.body[-1], ast.Expr)
            or not isinstance(node.body[-1].value, ast.Constant)
            or not isinstance(node.body[-1].value.value, str)
            or any(not isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body[:-1])):
        raise ValueError('Expected a public scaffold containing imports and its description')
    doc = node.body[-1]
    if doc.lineno <= node.lineno:
        raise ValueError('Expected the docstring below the signature')
    lines = prompt.splitlines(keepends=True)
    scaffold = ''.join(lines[:doc.lineno - 1]).rstrip()
    trailing = ''.join(lines[node.end_lineno:]).strip()
    if trailing:
        raise ValueError('Additional source after the target function requires explicit handling')
    description = inspect.cleandoc(doc.value.value)
    return (f'In Python, implement `{entry_point}` according to this description and examples:\n\n'
            f'{description}\n\n'
            f'Use this exact function name and signature, with the supplied imports:\n\n'
            f'```python\n{scaffold}\n```\n\n'
            'Return the complete implementation in one Python code block.')
