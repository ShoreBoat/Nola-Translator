"""Regression test for decoding Qwen3-ASR generated token batches.

The model returns a 2-D ``[batch, tokens]`` tensor.  Qwen's processor ``decode``
forwards to the tokenizer's single-sequence decoder, while the official Qwen3-ASR
inference path uses ``batch_decode`` for this tensor.  Passing the whole batch to
``decode`` can corrupt byte-level CJK token decoding and surface U+FFFD replacement
characters in captions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nola_translator_engine.recognition.qwen_runtime import QwenRuntime


class _Batch(dict):
    def to(self, _device, _dtype):
        return self


class _BatchOnlyProcessor:
    def __init__(self) -> None:
        self.batch = _Batch({"input_ids": torch.zeros((1, 4), dtype=torch.long)})
        self.batch_decode_options: dict[str, object] | None = None

    def apply_chat_template(self, _messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "PROMPT|"

    def __call__(self, *, text, audio, return_tensors, padding):
        assert text == ["PROMPT|"]
        assert len(audio) == 1
        assert return_tensors == "pt"
        assert padding is True
        return self.batch

    def decode(self, *_args, **_kwargs):
        raise AssertionError("generated ids are a batch; single-sequence decode() must not be used")

    def batch_decode(self, ids, **kwargs):
        assert tuple(ids.shape) == (1, 2)
        self.batch_decode_options = kwargs
        return ["language Chinese<asr_text>中文正常"]

    def parse_output(self, text: str) -> dict[str, str | None]:
        prefix, body = text.split("<asr_text>", 1)
        language = prefix.removeprefix("language ").strip() or None
        return {"language": language, "transcription": body.strip()}


class _Model:
    device = "cpu"

    def generate(self, **_kwargs):
        # Four prompt tokens followed by two generated tokens.
        return torch.tensor([[1, 2, 3, 4, 9, 8]])


def test_transcribe_batch_decodes_generated_ids_like_official_qwen(monkeypatch) -> None:
    processor = _BatchOnlyProcessor()
    pipeline = SimpleNamespace(processor=processor, model=_Model())
    monkeypatch.setattr(QwenRuntime, "_load_pipeline", lambda _self, _quant: pipeline)

    runtime = QwenRuntime(Path("models/qwen"), device="cpu")
    text, language = runtime.transcribe(np.zeros(1600, dtype=np.float32))

    assert (text, language) == ("中文正常", "zh")
    assert processor.batch_decode_options == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
