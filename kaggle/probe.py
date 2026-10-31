"""Print the library code the pad-free path has to match, and the padding of the real batches."""
import inspect
import json

import transformers.models.qwen2.modeling_qwen2 as m

for f in (m.Qwen2Model.forward, m.Qwen2DecoderLayer.forward, m.Qwen2Attention.forward,
          m.Qwen2RotaryEmbedding.forward, m.Qwen2MLP.forward):
    print(inspect.getsource(f))

import trl.trainer.sft_trainer as s  # noqa: E402

src = inspect.getsource(s)
i = src.find("class DataCollatorForLanguageModeling")
print(src[i:i + 6000])

# padding: Qwen tokenizer over the 256 training rows in alpaca format, batches of 4 in file order
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
rows = [json.loads(line) for line in open("kit/data/train.jsonl", encoding="utf-8")]
print("rows", len(rows), "keys", list(rows[0]))
lens = []
for r in rows:
    text = (r.get("instruction", "") + "\n" + r.get("input", "") + "\n" + r.get("output", ""))
    lens.append(min(512, len(tok(text)["input_ids"]) + 30))
real = sum(lens)
padded = sum(4 * max(lens[k:k + 4]) for k in range(0, len(lens), 4))
print("approx real tokens", real, "padded", padded, "ratio", round(padded / real, 3),
      "max batch real", max(sum(lens[k:k + 4]) for k in range(0, len(lens), 4)))
