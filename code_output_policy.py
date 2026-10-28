"""Explicit code-output extraction and stopping for native V2 code benchmarks.

Formatting only: never alter code, invent imports, select by tests, or repair it.
"""
import re
POLICY_VERSION='single_complete_python_fence_with_surrounding_prose_v1'
FENCE=re.compile(r'(?m)^[ \t]*```([^\r\n`]*)\r?$')

def extract_code(text):
    if '```' not in text:
        return text, 'plain_python'
    fences=list(FENCE.finditer(text))
    if len(fences)!=2 or text.count('```')!=2:
        return None, 'ambiguous_or_incomplete_fences'
    first,last=fences
    if first.group(1).strip().lower() not in ('python','py','') or last.group(1).strip():
        return None, 'unsupported_fence'
    start=first.end()
    if start>=len(text) or text[start]!='\n':
        return None, 'incomplete_fence'
    return text[start+1:last.start()], 'single_python_fence_with_surrounding_prose'

def closing_fence_end(text):
    """Return the first complete Python block's end, before trailing explanation."""
    fences=list(FENCE.finditer(text))
    if len(fences)<2:
        return None
    first,last=fences[:2]
    if first.group(1).strip().lower() not in ('python','py','') or last.group(1).strip():
        return None
    # Match the same unambiguous two-fence rule used by extraction on the prefix.
    if extract_code(text[:last.end()])[0] is None:
        return None
    return last.end()

def install_scorer(module):
    original=module.assemble
    def assemble(task,row,generation,mode='benchmark',reference=False):
        if mode!='instruct' or reference:
            return original(task,row,generation,mode,reference)
        candidate,kind=extract_code(generation)
        if candidate is None:
            return None, '', 'format_failure'
        code,tests,extraction=original(task,row,candidate,mode,reference)
        return code,tests,kind if code is not None else extraction
    module.assemble=assemble
    return original
