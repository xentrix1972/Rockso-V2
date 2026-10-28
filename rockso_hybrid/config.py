from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    schema: str = "rockso_hybrid_v2_1"
    vocab_size: int = 65536
    n_layer: int = 32
    n_embd: int = 2048
    n_head: int = 32
    n_kv_head: int = 8
    # Actual ReLU-squared hidden width, not Rockso's historical 1.5*d_ff argument.
    mlp_hidden: int = 10240
    mlp_activation: str = "relu2"
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    value_residual: bool = True
    attention_gate: bool = True
    attention_gate_channels: int = 128
    xsa: bool = True
    value_embedding: bool = True
    ve_start: int = 1
    ve_every: int = 3
    ve_gate_channels: int = 12
    mudd_layers: tuple = (16, 31)
    mudd_rank: int = 32
    # Global/full Rockso RoPE is the initial controlled comparison.
    local_window: int = 0
    global_every: int = 4
    global_nope: bool = False
    rope_fraction: float = 1.0
    rope_base: float = 10000.0
    rope_math: str = "rockso"
    attention_scale: float | None = None
    sigmoid_logit_cap: bool = False
    smoothing: bool = False
    residual_multipliers: bool = False

    def __post_init__(self):
        if self.schema != "rockso_hybrid_v2_1":
            raise ValueError("Unsupported architecture schema")
        if self.mlp_activation not in ("relu2", "swiglu"):
            raise ValueError("mlp_activation must be relu2 or swiglu")
        for name in ("vocab_size", "n_layer", "n_embd", "n_head", "n_kv_head",
                     "mlp_hidden", "max_seq_len", "global_every", "ve_every", "mudd_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("Invalid head dimensions/GQA grouping")
        if not 0 < self.rope_fraction <= 1 or self.rope_dim % 2 or self.rope_dim < 2:
            raise ValueError("RoPE dimension must be positive and even")
        if self.rope_base <= 0 or self.local_window < 0:
            raise ValueError("Invalid RoPE base or local window")
        if self.rope_math not in ("rockso", "float32"):
            raise ValueError("rope_math must be rockso or float32")
        if self.global_nope and not self.local_window:
            raise ValueError("global_nope requires an explicit local/global schedule")
        if self.attention_scale is not None and self.attention_scale <= 0:
            raise ValueError("Attention scale must be positive")
        if self.smoothing:
            raise ValueError("Rockso V2 does not use gradient smoothing")
        if self.attention_gate_channels > self.n_embd or self.ve_gate_channels > self.n_embd:
            raise ValueError("Gate input width exceeds residual width")
        if min(self.attention_gate_channels, self.ve_gate_channels) < 1:
            raise ValueError("Gate input widths must be positive")
        if not 0 <= self.ve_start < self.n_layer:
            raise ValueError("Invalid value embedding start layer")
        if len(set(self.mudd_layers)) != len(self.mudd_layers):
            raise ValueError("Duplicate MUDD layer")
        if any(i < 2 or i >= self.n_layer for i in self.mudd_layers):
            raise ValueError("MUDD sites must have three distinct available input depths")

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def rope_dim(self):
        return int(self.head_dim * self.rope_fraction)

    def local(self, layer):
        return self.local_window > 0 and (layer + 1) % self.global_every != 0

    def ve_layer(self, layer):
        return self.value_embedding and layer >= self.ve_start and (layer - self.ve_start) % self.ve_every == 0

    def taps(self, layer):
        return (0, layer // 2, layer)

    def to_dict(self):
        d = asdict(self)
        d["mudd_layers"] = list(self.mudd_layers)
        return d

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown architecture options: {sorted(unknown)}")
        data = dict(data)
        if "mudd_layers" in data:
            data["mudd_layers"] = tuple(data["mudd_layers"])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
