import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModel

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_LEN = 256

class Qwen3EmbedWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        self.m.config.use_cache = False
        self.eval()

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state  # [B,T,H]

        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        b = torch.arange(hs.size(0), device=hs.device)
        last = hs[b, lengths]       # [B,H]

        emb = F.normalize(last, p=2, dim=1)  # [B,H]
        return emb

model = Qwen3EmbedWrapper().eval()

dummy_ids  = torch.zeros((1, MAX_LEN), dtype=torch.long)
dummy_mask = torch.ones((1, MAX_LEN), dtype=torch.long)

traced = torch.jit.trace(model, (dummy_ids, dummy_mask), strict=False)
traced = torch.jit.freeze(traced)

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, MAX_LEN), dtype=int),
        ct.TensorType(name="attention_mask", shape=(1, MAX_LEN), dtype=int),
    ],
)

mlmodel.save("Qwen3Embedding06B.mlpackage")
print("saved Qwen3Embedding06B.mlpackage")