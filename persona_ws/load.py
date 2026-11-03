"""Load a trained arm from its checkpoint dir (train_sft.py output), or a reference model."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import attach
from .sidecar import SideCar, SideCarConfig
from .workspace import Workspace, WorkspaceConfig


def load_arm(spec, device="cuda", dtype=torch.bfloat16):
    """spec: a checkpoint dir, or 'hf:<repo>' for a plain HF model (base / AI2 SFT reference).
    Returns (model, tok, handle). handle is None for plain, LoRA and full fine-tune models."""
    if spec.startswith("hf:"):
        repo = spec[3:]
        tok_name = repo if repo.endswith("-SFT") or repo.endswith("-DPO") else f"{repo}-SFT"
        tok = AutoTokenizer.from_pretrained(tok_name)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device).eval()
        return model, tok, None
    cfg = json.load(open(os.path.join(spec, "config.json")))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"] or f"{cfg['model']}-SFT")
    m = cfg["module"]
    if m["type"] == "full":  # no-module baseline: the checkpoint is the whole model
        return AutoModelForCausalLM.from_pretrained(os.path.join(spec, "model"), dtype=dtype).to(device).eval(), tok, None
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=dtype).to(device)
    c = model.config
    handle = None
    if m["type"] == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec)
    elif m["type"] == "sidecar":
        handle = attach(model, SideCar(SideCarConfig(c.hidden_size, c.num_hidden_layers, d_side=m["d_side"])))
    else:
        wcfg = WorkspaceConfig(c.hidden_size, c.num_hidden_layers, n_slots=m["n_slots"], d_ws=m["d_ws"],
                               read_layers=tuple(m["read_layers"]), write_layers=tuple(m["write_layers"]),
                               hard=m["hard"], hard_rank=m["hard_rank"])
        # P is a saved buffer; a placeholder of the right shape is overwritten by load()
        P = {i: torch.zeros(c.hidden_size, m["hard_rank"]) for i in wcfg.write_layers} if wcfg.hard else None
        handle = attach(model, Workspace(wcfg, P))
    if handle is not None:
        handle.load(os.path.join(spec, "persona.pt"))
    model.eval()
    return model, tok, handle
