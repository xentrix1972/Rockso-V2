"""Production V2 loader for the existing native benchmark harness."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import rockso_chat_v5_prefixlm_145m as chat

_PREFIXLM_PREFILL = True
_ROPE_IMPL = "auto"
_MODEL_INFO = None


def load_model(checkpoint_path, tokenizer_path, device="cuda", dtype="bf16"):
    global _MODEL_INFO
    model, _ = chat.load_checkpoint(
        checkpoint_path, device=device, dtype=dtype,
        prefixlm_prefill=_PREFIXLM_PREFILL, rope_impl=_ROPE_IMPL,
        tokenizer_path=tokenizer_path,
    )
    _MODEL_INFO = model.inference_provenance
    return model, model.checkpoint_step, model.checkpoint_tokens_seen
