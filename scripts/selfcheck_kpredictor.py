"""Self-check for the monotone survival-head KPredictor (no retraining).

Validates the train/infer alignment fix:
  (a) keep-prob p_i is monotone non-increasing for every chunk/batch element
  (b) a single low intermediate hazard does NOT zero the tail (vs old cummin)
  (c) the structure can produce a WIDE k range (not capped at 8)
Plus: helper correctness, ckpt arch compatibility, loss finiteness + grad flow.
"""
import sys, os, json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from model.modeling import (
    KPredictor, monotone_keep_logprob, predictor_loss,
    oracle_to_per_slot_labels,
)

torch.manual_seed(0)
DEV = "cuda:7" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEV}")

MAX_SLOTS = 34
NUM_LAYERS = 32
D_KV = 1024   # Llama-3 8B GQA KV dim: num_kv_heads(8) * head_dim(128)


def make_predictor():
    p = KPredictor(num_layers=NUM_LAYERS, d_kv=D_KV, max_slots=MAX_SLOTS).to(DEV).float()
    return p


def keep_probs_per_chunk(pred, anchor_kv, mem_sizes, eps):
    """Return list of [B, m_c] keep-probs p_i from the live forward path."""
    logits = pred(anchor_kv, mem_sizes, eps)
    return [monotone_keep_logprob(lg)[0].exp() for lg in logits]


def synth_anchor(B, total_M):
    return torch.randn(B, NUM_LAYERS, 2, total_M, D_KV, device=DEV)


# ----------------------------------------------------------------------------
print("\n=== helper: monotone_keep_logprob matches manual cumprod ===")
a = torch.tensor([[3.0, 2.0, -1.0, 0.5, -4.0]], device=DEV)
log_p, log1m_p = monotone_keep_logprob(a)
p = log_p.exp()
p_manual = torch.cumprod(torch.sigmoid(a), dim=-1)
print("p        :", p.cpu().numpy().round(5))
print("p_manual :", p_manual.cpu().numpy().round(5))
assert torch.allclose(p, p_manual, atol=1e-5), "cumprod mismatch"
# log(1-p) consistency
assert torch.allclose(log1m_p.exp(), 1 - p, atol=1e-5), "log1mexp mismatch"
# extreme inputs stay finite
a_ext = torch.tensor([[40.0, 40.0, -40.0, -40.0]], device=DEV)
lp, l1m = monotone_keep_logprob(a_ext)
assert torch.isfinite(lp).all() and torch.isfinite(l1m).all(), "non-finite on extremes"
print("extreme-input log_p / log1m_p finite: OK")

# ----------------------------------------------------------------------------
print("\n=== (a) monotonicity of p_i across all chunks/batch ===")
pred = make_predictor()
pred.eval()
B, mem_sizes = 4, [34, 30, 20]
with torch.no_grad():
    pk = keep_probs_per_chunk(pred, synth_anchor(B, sum(mem_sizes)),
                              mem_sizes, torch.full((B,), 0.1, device=DEV))
max_increase = 0.0
for c, p_c in enumerate(pk):
    diff = p_c[:, 1:] - p_c[:, :-1]          # should be <= 0 everywhere
    max_increase = max(max_increase, float(diff.max().cpu()))
assert max_increase <= 1e-6, f"non-monotone! max slot-to-slot increase = {max_increase}"
print(f"all chunks monotone non-increasing (max increase {max_increase:.2e}) : OK")

# ----------------------------------------------------------------------------
print("\n=== init prior: bias=4 keeps k near max_slots at init ===")
with torch.no_grad():
    _, ks_b0 = pred.predict_k(synth_anchor(1, sum(mem_sizes)), mem_sizes,
                              torch.tensor(0.1, device=DEV))
print(f"initial per-chunk k (random init): {ks_b0}  vs chunk sizes {mem_sizes}")
# at init every slot should be kept -> k_c == m_c (capped at each chunk size)
assert all(k >= m - 1 for k, m in zip(ks_b0, mem_sizes)), \
    "init k should keep ~all slots per chunk"

# ----------------------------------------------------------------------------
print("\n=== (b) one noisy intermediate slot no longer flattens the whole tail ===")
THRESH = 0.5
# Per-slot outputs a jagged (old-style independent) head might emit: confident
# keep (h~0.97) everywhere, with ONE noisy low slot at index 5 (h~0.30).
base = torch.full((1, MAX_SLOTS), 3.48, device=DEV)   # sigmoid(3.48)~0.97
base[0, 5] = -0.85                                     # sigmoid(-0.85)~0.30 (dip)
tail = slice(6, None)
# OLD scheme it replaces: per-slot sigmoid -> cummin (min) projection.
#   min is winner-take-all: the lone dip becomes a hard floor and the ENTIRE
#   tail is clamped to it (all 0.30), erasing every downstream distinction.
p_old = torch.cummin(torch.sigmoid(base), dim=-1).values[0]
old_tail_spread = float((p_old[tail].max() - p_old[tail].min()).cpu())
# NEW survival: cumulative product. The dip is one bounded multiplicative
#   factor; the tail keeps DISTINCT, strictly-decreasing values.
p_new = monotone_keep_logprob(base)[0].exp()[0]
new_tail_min_gap = float((p_new[tail][:-1] - p_new[tail][1:]).min().cpu())
print(f"old cummin tail spread (~0 => flattened to the dip floor): {old_tail_spread:.2e}")
print(f"new survival tail strictly decreasing (min slot gap > 0): {new_tail_min_gap:.2e}")
assert old_tail_spread < 1e-6, "old cummin should flatten the tail to one value"
assert new_tail_min_gap > 0, "new tail should keep distinct monotone values"
print("one noisy slot no longer flattens/erases the tail: OK")

# ----------------------------------------------------------------------------
print("\n=== (c) k range is wide / not capped at 8 ===")
# Sweep a global hazard offset; show predict_k spans the full [1, max_slots].
ks_seen = set()
with torch.no_grad():
    for off in torch.linspace(-6, 6, 25):
        a = torch.full((1, MAX_SLOTS), float(off), device=DEV)
        p = monotone_keep_logprob(a)[0].exp()
        k = int((p > THRESH).sum().clamp(min=1).cpu())
        ks_seen.add(k)
ks_sorted = sorted(ks_seen)
print(f"k values reachable by sweeping hazard offset: {ks_sorted}")
assert max(ks_sorted) == MAX_SLOTS, "must be able to reach max_slots=34"
assert any(k > 8 for k in ks_sorted), "must exceed the old cap of 8"
assert min(ks_sorted) == 1, "must be able to reach k=1"
print("k spans 1..34 (incl. values >8 and =34): OK")

# Also: distribution over random inputs with random eps (representational range)
pred2 = make_predictor()
pred2.eval()
rng_ks = []
with torch.no_grad():
    for _ in range(40):
        B = 1
        ms = [34, 34, 34]
        eps = torch.tensor(float(torch.empty(1).uniform_(0.01, 0.2)), device=DEV)
        # jitter the head bias so we exercise the output range, not just init
        with torch.no_grad():
            pred2.head[-1].bias.uniform_(-4, 6)
        _, kb0 = pred2.predict_k(synth_anchor(B, sum(ms)), ms, eps)
        rng_ks.extend(kb0)
import collections
print(f"k histogram over random heads: {dict(sorted(collections.Counter(rng_ks).items()))}")

# ----------------------------------------------------------------------------
print("\n=== ckpt arch compatibility (load k_predictor.* from trained adapter) ===")
CKPT = "experiment/pact_v5_mrqa/output/instruction_adapter.pt"
ckpt_path = os.path.join(PROJECT_ROOT, CKPT)
if os.path.exists(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    kp_sd = {k[len("k_predictor."):]: v for k, v in sd.items()
             if k.startswith("k_predictor.")}
    print(f"found {len(kp_sd)} k_predictor.* tensors in adapter")
    if kp_sd:
        fresh = make_predictor()
        missing, unexpected = fresh.load_state_dict(kp_sd, strict=False)
        print(f"load_state_dict -> missing={list(missing)[:3]}... "
              f"unexpected={list(unexpected)[:3]}...")
        assert not unexpected, f"architecture drift: unexpected keys {unexpected}"
        print("trained k_predictor weights load into new structure: OK "
              "(arch unchanged; only bias-init + downstream interpretation changed)")
else:
    print(f"[skip] no adapter at {ckpt_path}")

# ----------------------------------------------------------------------------
print("\n=== loss finiteness + gradient flow ===")
pred3 = make_predictor()
pred3.train()
ms = [34, 30]
anchor = synth_anchor(2, sum(ms))
eps = torch.full((2,), 0.1, device=DEV)
logits = pred3(anchor, ms, eps)
oracle = [torch.tensor([8, 34], device=DEV), torch.tensor([4, 16], device=DEV)]
loss = predictor_loss(logits, ms, oracle, pos_weight=3.0)
loss.backward()
gnorm = sum(p.grad.norm().item() for p in pred3.parameters() if p.grad is not None)
print(f"loss = {float(loss):.4f}  finite={torch.isfinite(loss).item()}  "
      f"total grad norm = {gnorm:.4f}")
assert torch.isfinite(loss) and gnorm > 0, "loss/grad broken"
print("loss finite + gradients flow (incl. pos_weight): OK")

print("\nALL SELF-CHECKS PASSED")
