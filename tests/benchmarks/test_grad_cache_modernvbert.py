from benchmarks.grad_cache_modernvbert import _canonical_text_name


def test_canonical_text_name_accepts_text_only_and_wrapped_models() -> None:
    assert (
        _canonical_text_name("layers.0.attn.Wqkv.weight")
        == "model.text_model.layers.0.attn.Wqkv.weight"
    )
    assert (
        _canonical_text_name("text_model.layers.0.attn.Wqkv.weight")
        == "model.text_model.layers.0.attn.Wqkv.weight"
    )
    assert (
        _canonical_text_name("_orig_mod.layers.0.attn.Wqkv.weight")
        == "model.text_model.layers.0.attn.Wqkv.weight"
    )
