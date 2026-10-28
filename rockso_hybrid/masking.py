"""Rockso packing contract: 1=prefix/meta/pad, 0=response target."""
import torch


def segment_ids(span_ends, length):
    ends = span_ends.long().clamp(0, length)
    valid = ends > 0
    boundary = torch.zeros(ends.shape[0], length + 1, dtype=torch.int32, device=ends.device)
    boundary.scatter_add_(1, ends * valid, valid.int())
    boundary[:, 0] = 0
    return boundary.cumsum(1)[:, :length].contiguous()


def prefix_positions(segments, metadata):
    response = (metadata == 0).int().cumsum(1)
    previous = torch.cat((torch.zeros_like(response[:, :1]), response[:, :-1]), 1)
    start = torch.ones_like(segments, dtype=torch.bool)
    start[:, 1:] = segments[:, 1:] != segments[:, :-1]
    base = torch.where(start, previous, torch.full_like(previous, -1)).cummax(1).values
    return (response - base) == 0


def mask_modifier(segments, prefix, mode, window=0):
    length = segments.shape[1]

    def allowed(b, h, q, k):
        # BlockMask construction can visit the padded edge of a tile.
        qi, ki = q.clamp(max=length - 1), k.clamp(max=length - 1)
        same = segments[b, qi] == segments[b, ki]
        causal = k <= q
        if window:
            causal = causal & (k >= q - window)
        visible = causal | prefix[b, ki] if mode == "prefixlm" else causal
        if mode == "causal":
            same = torch.ones_like(same)
        return same & visible & (q < length) & (k < length)

    return allowed


def build_masks(span_ends, metadata, mode="prefixlm", backend="sdpa", local_window=0):
    if mode not in ("prefixlm", "doc_causal", "causal"):
        raise ValueError(f"Unknown attention mode: {mode}")
    if backend not in ("sdpa", "flex"):
        raise ValueError(f"Unknown attention backend: {backend}")
    b, t = metadata.shape
    seg = segment_ids(span_ends, t)
    pref = prefix_positions(seg, metadata)
    masks = {}
    for window in dict.fromkeys((0, local_window)):
        fn = mask_modifier(seg, pref, mode, window)
        if backend == "flex":
            if metadata.device.type != "cuda":
                raise ValueError("Use SDPA for CPU reference tests")
            from torch.nn.attention.flex_attention import create_block_mask
            masks[window] = create_block_mask(fn, B=b, H=None, Q_LEN=t, KV_LEN=t,
                                              device=str(metadata.device), _compile=True)
        else:
            bi = torch.arange(b, device=metadata.device)[:, None, None]
            q = torch.arange(t, device=metadata.device)[None, :, None]
            k = torch.arange(t, device=metadata.device)[None, None, :]
            masks[window] = fn(bi, 0, q, k).unsqueeze(1)
    return masks


def response_labels(input_ids, metadata, span_ends, pad_id=None):
    labels = input_ids.clone()
    labels[metadata == 1] = -100
    length = input_ids.shape[1]
    valid = (span_ends > 0) & (span_ends < length)
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None].expand_as(span_ends)
    labels[rows[valid], span_ends[valid].long()] = -100
    if pad_id is not None:
        labels[input_ids == pad_id] = -100
    return labels
