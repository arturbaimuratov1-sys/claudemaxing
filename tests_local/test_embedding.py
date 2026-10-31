"""fp16 storage of the tied embedding/head weight: lossless, or not done at all."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from solution.embedding import shrink_tied_embedding  # noqa: E402


def tiny(tied=True):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      tie_word_embeddings=tied)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).half().cuda()  # loaded in fp16 ...
    model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.float()
    if tied:
        model.tie_weights()
    for p in model.parameters():  # ... upcast and frozen, as prepare_model_for_kbit_training does
        p.requires_grad_(False)
    return model


def test_lossless_and_same_outputs():
    model = tiny()
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert head.weight is emb.weight and emb.weight.dtype == torch.float32
    ref = emb.weight.detach().clone()
    ids = torch.randint(0, 1000, (2, 17), device="cuda")
    out_ref = emb(ids)
    calls = []
    emb.register_forward_hook(lambda m, i, o: calls.append(o.dtype))  # peft-style hook still runs
    assert shrink_tied_embedding(model) is False  # outside fp16 autocast: not yet
    with torch.autocast("cuda", dtype=torch.float16):
        assert shrink_tied_embedding(model) is True
    assert emb.weight.dtype == torch.float16 and head.weight is emb.weight
    assert torch.equal(emb.weight.float(), ref)
    out = emb(ids)
    assert out.dtype == torch.float32 and torch.equal(out, out_ref)
    assert calls == [torch.float32]
    assert shrink_tied_embedding(model) is True  # idempotent
    with torch.autocast("cuda", dtype=torch.float16):
        assert torch.equal(emb(ids), out_ref)
    import copy

    assert torch.equal(copy.deepcopy(emb)(ids), out_ref)


def test_not_exact_left_alone():
    model = tiny()
    w = model.get_input_embeddings().weight
    w.data[0, 0] = 1.0 + 2 ** -20  # not an fp16 number
    with torch.autocast("cuda", dtype=torch.float16):
        assert shrink_tied_embedding(model) is False
    assert w.data[0, 0].item() == 1.0 + 2 ** -20
    assert w.dtype == torch.float32


def test_untied_or_trainable_left_alone():
    with torch.autocast("cuda", dtype=torch.float16):
        model = tiny(tied=False)
        assert shrink_tied_embedding(model) is False
        model = tiny()
        model.get_input_embeddings().weight.requires_grad_(True)
        assert shrink_tied_embedding(model) is False
