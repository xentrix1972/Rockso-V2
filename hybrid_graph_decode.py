"""Optional single-row CUDA-graph decode using the native cached block operations.

Uses the existing cached block operations. Fixed bucket lengths add masked KV
slots, so numerical/token parity must be measured before adopting this path.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from hybrid_chat_adapter import _cached_block

@dataclass(frozen=True)
class GraphCache:
    owner: int
    generation: int
    length: int
    prefix_length: int

class GraphDecodeModel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if not getattr(reference,'is_rockso_hybrid',False):
            raise TypeError('CUDA graph decode supports native hybrid models only')
        if reference.native.embed.weight.device.type != 'cuda':
            raise ValueError('CUDA graph decode requires a CUDA model')
        self.reference=reference.eval()
        self.is_rockso_hybrid=True
        self.checkpoint_step=reference.checkpoint_step
        self.checkpoint_tokens_seen=reference.checkpoint_tokens_seen
        self.inference_provenance=dict(reference.inference_provenance)
        self.block_size=reference.block_size
        self.rope_impl=reference.rope_impl
        self.prefixlm_prefill=reference.prefixlm_prefill
        self.generation=0
        self.length=0
        self.graphs={}
        c=reference.config;device=reference.native.embed.weight.device
        dtype=reference.native.embed.weight.dtype
        heads=c.n_kv_head if reference.native.native_gqa else c.n_head
        self.storage=tuple((torch.zeros((1,heads,self.block_size,c.head_dim),device=device,dtype=dtype),
                            torch.zeros((1,heads,self.block_size,c.head_dim),device=device,dtype=dtype))
                           for _ in reference.native.blocks)
        self.token=torch.zeros((1,1),device=device,dtype=torch.long)
        self.position=torch.ones((1,1),device=device,dtype=torch.long)
        self.prefix=torch.zeros((1,1),device=device,dtype=torch.long)
        self.rows=torch.zeros(1,device=device,dtype=torch.long)
        self.buckets=sorted(set([x for x in [128,256,512,1024,self.block_size] if x<=self.block_size]))

    @torch.inference_mode()
    def _step(self,capacity):
        model=self.reference;native=model.native;c=model.config
        keys=torch.arange(capacity,device=self.token.device)[None,:]
        masks={}
        for window in dict.fromkeys((0,c.local_window)):
            allowed=keys<=self.position
            if window:
                allowed=allowed&((keys>=self.position-window)|(keys<self.prefix))
            masks[window]=allowed[:,None,None,:]
        write=(self.rows,self.position.reshape(-1))
        with torch.autocast(device_type='cuda',enabled=False):
            x=native.embed(self.token)
            ve=native.value_embed(self.token) if native.value_embed is not None else None
            history={};first_v=None
            for i,block in enumerate(native.blocks):
                if i in model._retained:history[i]=x
                taps=tuple(history[j] if j!=i else x for j in c.taps(i)) if block.mudd is not None else ()
                window=c.local_window if c.local(i) else 0
                storage=tuple(t[:,:,:capacity] for t in self.storage[i])
                x,raw_v,_=_cached_block(block,x,self.position,masks[window],first_v,
                    ve if block.ve_gate is not None else None,taps,storage,native.native_gqa,static_write=write)
                if i==0 and c.value_residual:first_v=raw_v
            logits=F.linear(native.norm(x),native.lm_head.weight).float()
            if c.sigmoid_logit_cap:logits=23*torch.sigmoid((logits+5)/7.5)
        return logits

    @torch.inference_mode()
    def capture(self):
        # Capture once before real prefill; warmup mutations cannot affect any
        # saved prompt. Refill the cache from the reference prefill afterwards.
        for capacity in self.buckets:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self._step(capacity)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=self._step(capacity)
            self.graphs[capacity]=(graph,output)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def forward(self,input_ids,position_ids=None,kv_caches=None,prefill_causal=None):
        if input_ids.shape[0]!=1:
            raise ValueError('CUDA graph decode is single-row only')
        if not self.graphs:
            raise RuntimeError('Call capture() before generation')
        if kv_caches is None:
            if not 0 < input_ids.shape[1] <= self.block_size:
                raise ValueError('Prompt must fit the model context')
            logits,cache=self.reference(input_ids,position_ids=position_ids,prefill_causal=prefill_causal)
            for dst,src in zip(self.storage,cache.layers):
                for a,b in zip(dst,src):
                    a.zero_();a[:,:,:cache.length].copy_(b)
            self.generation+=1
            self.length=cache.length
            self.prefix.fill_(cache.prefix_length)
            return logits,GraphCache(id(self),self.generation,cache.length,cache.prefix_length)
        if (not isinstance(kv_caches,GraphCache) or kv_caches.owner!=id(self)
                or kv_caches.generation!=self.generation or kv_caches.length!=self.length):
            raise ValueError('Cache was superseded by another prefill; interleaved rows are unsupported')
        if input_ids.shape!=(1,1) or kv_caches.length>=self.block_size:
            raise ValueError('Expected one new token inside context')
        self.token.copy_(input_ids)
        if position_ids is None:self.position.fill_(kv_caches.length)
        else:self.position.copy_(position_ids)
        capacity=next(n for n in self.buckets if n>=kv_caches.length+1)
        graph,logits=self.graphs[capacity];graph.replay()
        self.length=kv_caches.length+1
        return logits,GraphCache(id(self),self.generation,self.length,kv_caches.prefix_length)


@torch.inference_mode()
def enable_cuda_graph_decode(reference, prompt_ids, check_tokens=32):
    """Capture and compare real checkpoint logits before enabling graph replay.

    This wrapper owns one mutable sequence cache. It is for sequential generation
    only: likelihood scoring and branched/batched caches keep the reference model.
    No model weights or sampling, condition, stopping or scoring rules change.
    """
    if isinstance(reference, GraphDecodeModel):
        return reference
    started=time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    candidate=GraphDecodeModel(reference)
    device=reference.native.embed.weight.device
    ids=torch.tensor([list(prompt_ids)],device=device,dtype=torch.long)
    if ids.shape[1] < 1 or ids.shape[1] >= reference.block_size:
        raise ValueError('Parity prompt must leave room for decode')
    candidate.capture()
    expected,ref_cache=reference(ids)
    actual,graph_cache=candidate(ids)
    if not torch.equal(expected,actual):
        raise AssertionError('CUDA graph prefill differs from the reference')
    checked=0
    for _ in range(min(check_tokens,reference.block_size-ids.shape[1])):
        token=expected[:,-1].argmax(-1,keepdim=True)
        expected,ref_cache=reference(token,kv_caches=ref_cache)
        actual,graph_cache=candidate(token,kv_caches=graph_cache)
        if not torch.equal(expected,actual):
            error=float((expected-actual).abs().max())
            raise AssertionError(f'CUDA graph logits differ on decode {checked+1}: max_abs={error}')
        checked+=1
    receipt={
        'backend':'single_row_cuda_graph', 'buckets':candidate.buckets,
        'startup_decode_steps_bitwise_equal':checked,
        'scope':'Current checkpoint startup probe; not proof for every possible input',
        'capture_and_check_seconds':time.clock_gettime(time.CLOCK_MONOTONIC_RAW)-started,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'weights_changed':False, 'sampling_changed':False, 'stopping_changed':False,
    }
    candidate.inference_provenance['cuda_graph_decode']=receipt
    print(f"[decode] CUDA graph replay enabled; {checked} startup steps bitwise equal; "
          f"setup {receipt['capture_and_check_seconds']:.2f}s",flush=True)
    return candidate.eval()
