"""D21 step 4: short recovery fine-tune of a rank-r compressed D13 base interface (projected gradient descent).

Runs scripts/register_d13.py UNCHANGED (same base training data, objective KL+CE+distillation, held-out val-doc
selection) from a compressed init (scripts/d21_compress.py --save_specs svd<r>...), but after every optimizer step
every A / M / B matrix is projected back to rank r (truncated SVD), so every saved checkpoint is exactly rank r
(params as in the svd<r> row). All register_d13.py flags pass through; D21 flags:
    --d21_rank r          projection rank (required)
    --d21_groups AMB      which matrix groups are projected (default AMB, as in svd<r>)

    python scripts/d21_recover.py --d21_rank 128 --init outputs/D21_ckpt/svd128.pt --encoder Qwen/Qwen3.5-4B \
        --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --steps 400 --eval_every 100 --mem_safe 1 --out outputs/D21_rec128
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

STATE = {"rank": 0, "groups": "AMB", "params": None}


@torch.no_grad()
def project(p: torch.Tensor, r: int):
    if r * (p.shape[0] + p.shape[1]) >= p.numel():
        return
    # p is (rank r) + one small optimizer step: a randomized range finder with oversampling recovers its top-r part
    U, S, V = torch.svd_lowrank(p.float(), q=min(r + 32, min(p.shape)), niter=3)
    p.copy_((U[:, :r] * S[:r]) @ V[:, :r].t())


class _ProjAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        out = super().step(closure)
        if STATE["params"] is None:                     # matrices of the interface, found by shape among the groups
            ps = []
            for g in self.param_groups:
                for p in g["params"]:
                    if p.dim() == 2:
                        ps.append(p)
            STATE["params"] = ps
        for p in STATE["params"]:
            grp = STATE["kind"].get(id(p))
            if grp in STATE["groups"]:
                project(p, STATE["rank"])
        return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag in ("--d21_rank", "--d21_groups"):
        if flag in argv:
            k = argv.index(flag)
            STATE[flag[6:]] = int(argv[k + 1]) if flag == "--d21_rank" else argv[k + 1]
            del argv[k: k + 2]
    assert STATE["rank"] > 0, "--d21_rank required"
    import fit_d13
    import register_d13
    _build = fit_d13.build

    def build(a, init=None, mlp_hidden=0):
        out = _build(a, init=init, mlp_hidden=mlp_hidden)
        itf = out[3]
        kind = {}
        for li in itf.layers:
            kind[id(itf.A[str(li)].weight)] = "A"
            if itf.mlp_hidden:
                kind[id(itf.M[str(li)][0].weight)] = "M"
                kind[id(itf.M[str(li)][2].weight)] = "M"
            if not itf.identity_B:
                kind[id(itf.B[str(li)])] = "B"
        STATE["kind"] = kind
        for p in list(itf.A.parameters()) + list(itf.M.parameters()) + list(itf.B.parameters()):   # init is rank r
            if p.dim() == 2 and kind.get(id(p)) in STATE["groups"]:
                project(p.data, STATE["rank"])
        return out

    fit_d13.build = build
    torch.optim.AdamW = _ProjAdamW
    print(f"[D21 recover] projected fine-tune: rank {STATE['rank']} groups {STATE['groups']}", flush=True)
    return register_d13.main(argv)


if __name__ == "__main__":
    sys.exit(main())
