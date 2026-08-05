"""REAUDIT ROUND2 probe F — prior-PASS regression spot checks + residual-gap evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path("/mnt/c/projects/tts-self-d1c")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

out: list[tuple[str, str, str]] = []


def rec(tag, ok, msg):
    out.append((tag, "PASS" if ok else "FAIL", msg))
    print(f"F{tag} {'PASS' if ok else 'FAIL'} {msg}", flush=True)


def info(tag, msg):
    out.append((tag, "INFO", msg))
    print(f"F{tag} INFO {msg}", flush=True)


def load_script(name):
    p = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from minispeech.data.collate_codec_seq import collate_codec_seq  # noqa: E402
from minispeech.data.d1b_oracles import (  # noqa: E402
    eval_contract_content_sha256,
    load_eval_contract,
    verify_recipe_source_manifest,
)
from minispeech.data.discrete_oracles import load_formal_subset  # noqa: E402
from minispeech.models.discrete_speech_lm import (  # noqa: E402
    DiscreteSpeechLM,
    DiscreteSpeechLMConfig,
)


def semantic_model():
    cfg = DiscreteSpeechLMConfig.tiny()
    cfg.termination_mode = "semantic_eos_v1"
    cfg.eos_id = cfg.codebook_size
    cfg.codec_vocab_size = cfg.codebook_size
    cfg.semantic_vocab_size = cfg.codebook_size + 1
    return DiscreteSpeechLM(cfg)


# ---- F1 residual head receives live gradients (prior blocker 1)
torch.manual_seed(0)
m = semantic_model()
b = collate_codec_seq(
    [
        {"utt_id": "s", "input_ids": torch.tensor([3, 4]), "codes": torch.randint(0, m.codebook_size, (2, m.n_codebooks)), "speaker_id": 0},
        {"utt_id": "l", "input_ids": torch.tensor([5, 6, 7]), "codes": torch.randint(0, m.codebook_size, (4, m.n_codebooks)), "speaker_id": 1},
    ],
    pad_token_id=0,
    speech_eos_id=m.eos_id,
    termination_mode="semantic_eos_v1",
)
o = m(
    b["input_ids"],
    b["codes"],
    text_mask=b["text_mask"],
    frame_mask=b["codec_frame_mask"],
    semantic_targets=b["semantic_targets"],
    semantic_mask=b["semantic_mask"],
    speaker_id=b["speaker_id"],
    cb0_weight=0.0,
    eos_loss_weight=0.0,
    residual_weights=torch.ones(m.n_codebooks - 1),
)
o.loss["total"].backward()
res = [p for p in m.heads.residuals.parameters() if p.requires_grad]
nog = [i for i, p in enumerate(res) if p.grad is None or float(p.grad.abs().sum()) == 0.0]
rec("1a", res and not nog, f"residual params with live grads (cb0_w=0, eos_w=0): {len(res) - len(nog)}/{len(res)}")
rec("1b", bool(o.loss["total"].requires_grad), "loss['total'].requires_grad=True")

# ---- F2 immediate EOS -> empty codes through the PUBLIC generate()
m2 = semantic_model()


class AlwaysEOS(torch.nn.Module):
    def forward(self, h):  # noqa: ANN001
        lg = h.new_full((*h.shape[:-1], m2.semantic_vocab_size), -1.0e9)
        lg[..., m2.eos_id] = 1.0e9
        return lg


m2.heads.semantic = AlwaysEOS()
g = m2.generate(torch.tensor([[3, 4, 5]]), max_frames=4, min_frames=0, max_cb0_repeat=100, temperature=0.0, top_k=0)
codes = g.codes if hasattr(g, "codes") else g
lengths = list(g.lengths.tolist()) if hasattr(g, "lengths") else None
rec("2a", tuple(codes.shape)[:2] == (1, 0), f"generate() immediate EOS -> codes.shape={tuple(codes.shape)} lengths={lengths}")
rec("2b", lengths == [0], f"lengths==[0] (no phantom frame) lengths={lengths}")
info("2c", f"rows[0]={json.dumps(g.rows[0], ensure_ascii=False, default=str) if hasattr(g, 'rows') and g.rows else 'n/a'}")

# ---- F3 oracle select8 composition + formal load + file_sha256
contract = load_eval_contract(ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json")
for key in ("dev32", "select8", "gate24", "full32"):
    s = load_formal_subset(key, contract=contract)
    rec(f"3-{key}", bool(s.get("file_sha256")) and s.get("formal_eligible") is True, f"formal load {key}: n={len(s['utt_ids'])} split={s.get('split')} formal_eligible={s.get('formal_eligible')} file_sha256={str(s['file_sha256'])[:16]}")
sel = load_formal_subset("select8", contract=contract)
g24 = load_formal_subset("gate24", contract=contract)
f32 = load_formal_subset("full32", contract=contract)
rus = [u for u in sel["utt_ids"] if "ruslan" in str(u).lower()]
cv = [u for u in sel["utt_ids"] if "ruslan" not in str(u).lower()]
rec("3-comp", len(rus) == 2 and len(cv) == 6, f"select8 composition = {len(rus)} RUSLAN + {len(cv)} CV")
rec(
    "3-part",
    set(sel["utt_ids"]) | set(g24["utt_ids"]) == set(f32["utt_ids"]) and not (set(sel["utt_ids"]) & set(g24["utt_ids"])),
    f"select8 | gate24 == full32 and disjoint (|s|={len(sel['utt_ids'])} |g|={len(g24['utt_ids'])} |f|={len(f32['utt_ids'])})",
)

# ---- F4 selector _sha256_file on a multi-chunk blob
m76 = load_script("76_select_discrete_checkpoint.py")
tmp = Path(tempfile.mkdtemp(prefix="reaudit2reg_"))
blob = tmp / "blob.bin"
data = os.urandom(3 * 1024 * 1024 + 17)
blob.write_bytes(data)
rec("4a", m76._sha256_file(blob) == hashlib.sha256(data).hexdigest(), "_sha256_file matches hashlib on a 3 MiB+ blob (chunk loop iterates)")

# ---- F5 contract self-check + no legacy stop fields
raw = json.loads((ROOT / "configs/eval_contracts/d1c_semantic_eos_v2_0_0.json").read_text(encoding="utf-8"))
rec("5a", eval_contract_content_sha256(raw) == str(raw.get("sha256")), "contract self-check holds")
from minispeech.infer.discrete_config import contract_infer_from_eval_contract  # noqa: E402

ci = contract_infer_from_eval_contract(contract)
legacy = [k for k in ("stop_threshold", "stop_consecutive", "frames_per_text_token", "stop_overrun_bias", "stop_earliest_frac") if k in ci]
rec("5b", not legacy, f"contract infer keys={sorted(ci)} legacy_present={legacy}")

# ---- F6 frozen source manifest integrity AFTER all probes
man = raw.get("recipe_source_manifest") or {}
errs = verify_recipe_source_manifest(man, root=ROOT)
rec("6a", not errs, f"post-probe recipe_source_manifest mismatches={errs} (n_locked={len(man)})")

# ---- F7 residual gap evidence: which §11 fields does the real emitter actually populate?
base = Path("/tmp")
cands = sorted(base.glob("reaudit2sel_*/base/discrete_eval_report_v2.json"))
if cands:
    rep = json.loads(cands[-1].read_text(encoding="utf-8"))
    info("7a", f"emitted report gates keys={sorted(rep.get('gates') or {})}")
    info("7b", f"emitted report summaries keys={sorted(rep.get('summaries') or {})}")
    missing = [k for k in ("text_usage",) if k not in (rep.get("gates") or {})]
    info("7c", f"§11 terms never populated by the emitter -> always vacuous: gates.text_usage missing={bool(missing)}, summaries.phrase_loop_repetition missing={'phrase_loop_repetition' not in (rep.get('summaries') or {})}, gates/summaries.runaway_or_repetition missing={'runaway_or_repetition' not in (rep.get('gates') or {}) and 'runaway_or_repetition' not in (rep.get('summaries') or {})}")
else:
    info("7a", "no emitted baseline report found for gates/summaries introspection")

print("\n=== SUMMARY F ===")
nf = 0
for tag, v, msg in out:
    if v == "FAIL":
        nf += 1
        print(f"FAIL F{tag}: {msg}")
print(f"F_total={len(out)} F_fail={nf}")
