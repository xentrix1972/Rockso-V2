"""CPU contract tests for the separate export loader; no production assets required."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import infer_export as inference
from rockso_hybrid import ModelConfig, RocksoHybrid


class ExportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        config = ModelConfig(vocab_size=32, n_layer=4, n_embd=32, n_head=4, n_kv_head=2,
            mlp_hidden=64, mlp_activation='swiglu', max_seq_len=32, mudd_layers=(2, 3),
            mudd_rank=4, attention_gate_channels=8, rope_fraction=0.5, rope_math='float32',
            value_residual=False, residual_multipliers=True)
        model = RocksoHybrid(config).to(dtype=torch.bfloat16).preserve_precision()
        cls.state = {'schema':'rockso_hybrid_export_1', 'attention_mode':'prefixlm',
            'config':config.to_dict(), 'model':model.state_dict(),
            'source_sha256':{p:inference.sha256(inference.ROOT/p) for p in inference.SOURCE_FILES}}

    def test_strict_load_preserves_values_precision_and_aliases(self):
        model, aliases = inference.bind_model(self.state)
        self.assertIs(model.embed.weight, model.lm_head.weight)
        self.assertIs(model.blocks[2].mudd.reduce.weight, model.blocks[3].mudd.reduce.weight)
        self.assertEqual(len(aliases), 2)
        for name, actual in model.state_dict().items():
            self.assertEqual(actual.dtype, self.state['model'][name].dtype)
            self.assertTrue(torch.equal(actual, self.state['model'][name]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_refuses_wrong_schema_mode_source_missing_tensor_and_alias_disagreement(self):
        for key, value in [('schema','rockso_hybrid_checkpoint_1'), ('attention_mode','causal'),
                           ('source_sha256',{})]:
            state = dict(self.state, **{key:value})
            with self.assertRaises(ValueError): inference.bind_model(state)
        state = dict(self.state, model=dict(self.state['model']))
        state['model'].pop('norm.weight')
        with self.assertRaises(RuntimeError): inference.bind_model(state)
        state = dict(self.state, model=dict(self.state['model']))
        state['model']['lm_head.weight'] = state['model']['lm_head.weight'] + 1
        with self.assertRaisesRegex(ValueError, 'shared parameters'): inference.bind_model(state)

    def test_prefix_stays_bidirectional_and_answer_stays_causal(self):
        mask = inference.generation_masks(torch.ones((1, 7), dtype=torch.long), 4)[0][0, 0]
        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[:4, 4:].any())
        self.assertTrue(mask[4:, :4].all())
        self.assertTrue(mask[4, 4])
        self.assertFalse(mask[4, 5])
        self.assertFalse(mask[5, 6])

    def test_condition_order_and_native_envelope(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens): return text
        tok = Tokenizer()
        self.assertEqual(inference.prompt_ids(tok, ' hi ', 'direct'),
            '<|im_start|><|object_ref_start|>hi<|im_end|>')
        self.assertEqual(inference.prompt_ids(tok, 'hi', 'synth,cot', ' sys '),
            '<|im_start|><|quad_end|><|object_ref_end|>sys\n\nhi<|im_end|>')
        with self.assertRaises(ValueError): inference.prompt_ids(tok, 'hi', 'synth,direct')

    def test_answer_stop_and_context_limit(self):
        class Model:
            config = type('Config', (), {'max_seq_len':8, 'local_window':0})()
            def __call__(self, sequence, masks):
                output = torch.zeros((1, sequence.shape[1], 32))
                output[:, -1, 7] = 1
                return output
        class Tokenizer:
            def decode(self, ids, skip_special_tokens): return str(ids)
        result = inference.generate(Model(), Tokenizer(), [2,4,3], 3)
        self.assertEqual(result['generated_ids'], [7])
        self.assertEqual(result['stop_reason'], 'answer_end')
        self.assertEqual(result['text'], '[]')
        with self.assertRaises(ValueError): inference.generate(Model(), Tokenizer(), [2,4,3], 6)


if __name__ == '__main__':
    unittest.main()
