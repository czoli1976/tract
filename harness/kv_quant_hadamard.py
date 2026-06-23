import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from transformers import GPT2Model, GPT2Tokenizer
torch.manual_seed(0)
tok = GPT2Tokenizer.from_pretrained("gpt2"); model = GPT2Model.from_pretrained("gpt2").eval()
ne, nh = model.config.n_embd, model.config.n_head
D = ne // nh  # head dim (64 for gpt2)
text = ("Key-value cache quantization stores the attention cache in low precision to save "
        "memory while keeping every token. Keys have outlier channels, values do not. ") * 6
ids = tok(text, return_tensors="pt").input_ids[:, :160]; S = ids.shape[1]
cap = {}
def hook(L):
    def f(m, i, o):
        q, k, v = o.split(ne, dim=2)
        h = lambda x: x.view(1, S, nh, D)[0].permute(1, 0, 2).contiguous().numpy()
        cap[L] = (h(q), h(k), h(v))
    return f
for L in (2, 6, 10): model.h[L].attn.c_attn.register_forward_hook(hook(L))
with torch.no_grad(): model(ids)

def hadamard(n):                              # normalized Sylvester Hadamard: Hᵀ=H, H@H=I
    H = np.ones((1, 1));
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)
Hm = hadamard(D)

def qd(x, bits, per):                          # x [S,D]; per='channel' (col scale) or 'token' (row scale)
    levels = (1 << bits) - 1; ax = 0 if per == 'channel' else 1
    lo = x.min(ax, keepdims=True); hi = x.max(ax, keepdims=True)
    sc = np.where(hi > lo, (hi - lo) / levels, 1.0)
    return lo + np.clip(np.round((x - lo) / sc), 0, levels) * sc

def attn_last(q, k, v):                         # [H,S,D], last query only
    H_, Sq, Dd = q.shape; i = Sq - 1; outs = []
    for h in range(H_):
        sc = (k[h] @ q[h, i]) / np.sqrt(Dd); w = np.exp(sc - sc.max()); w /= w.sum(); outs.append(w @ v[h])
    return np.stack(outs)

def dev(q, k, v, kf, vf, rot=False):
    f = attn_last(q, k, v)
    if rot:                                     # rotate K (scores invariant) and V, then un-rotate V out
        kf = np.stack([kf[h] @ Hm for h in range(nh)])
        qf = np.stack([q[h] @ Hm for h in range(nh)])
        vf = np.stack([vf[h] @ Hm for h in range(nh)])
        g = attn_last(qf, kf, vf); g = np.stack([g[h] @ Hm for h in range(nh)])   # H symmetric ⇒ un-rotate
    else:
        g = attn_last(q, kf, vf)
    return np.linalg.norm(g - f) / np.linalg.norm(f)

def per_channel_K(k, bits): return np.stack([qd(k[h], bits, 'channel') for h in range(nh)])
def per_token_K(k, bits):   return np.stack([qd(k[h], bits, 'token')   for h in range(nh)])
def per_token_V(v, bits):   return np.stack([qd(v[h], bits, 'token')   for h in range(nh)])

print(f"GPT-2  S={S}, heads={nh}, D={D}  — attention rel-deviation vs full f32 (lower=better)\n")
print("                              int4 Keys                         int2 Keys")
print(" layer | perCh  perTok  perTok+Hada  perCh+Hada | perCh  perTok  perTok+Hada")
for L in (2, 6, 10):
    q, k, v = cap[L]
    def D4(kf, vf, rot=False): return dev(q, k, v, kf, vf, rot)
    # int4
    a = D4(per_channel_K(k, 4), per_token_V(v, 4))                 # KIVI per-channel
    b = D4(per_token_K(k, 4),   per_token_V(v, 4))                 # per-token raw
    # for the rotated variants we rotate the RAW k/v inside dev(rot=True), quantizing the rotated tensor:
    def rotq(bits, kper):
        kr = np.stack([(k[h] @ Hm) for h in range(nh)]); vr = np.stack([(v[h] @ Hm) for h in range(nh)])
        kq = np.stack([qd(kr[h], bits, kper) for h in range(nh)]); vq = np.stack([qd(vr[h], bits, 'token') for h in range(nh)])
        f = attn_last(q, k, v)
        qf = np.stack([q[h] @ Hm for h in range(nh)])
        g = attn_last(qf, kq, vq); g = np.stack([g[h] @ Hm for h in range(nh)])
        return np.linalg.norm(g - f) / np.linalg.norm(f)
    c = rotq(4, 'token')                                          # per-token + Hadamard
    d = rotq(4, 'channel')                                        # per-channel + Hadamard (should ~match a)
    # int2
    e = D4(per_channel_K(k, 2), per_token_V(v, 2))
    f2 = D4(per_token_K(k, 2),  per_token_V(v, 2))
    g2 = rotq(2, 'token')
    print(f" {L:>5} | {a:.4f}  {b:.4f}   {c:.4f}     {d:.4f}  | {e:.4f}  {f2:.4f}   {g2:.4f}")
print("\nQuestion the A/B answers: does a fixed Hadamard let the staleness-free per-token-K layout")
print("match per-channel KIVI on REAL outliers? Compare 'perTok+Hada' vs 'perCh'.")
