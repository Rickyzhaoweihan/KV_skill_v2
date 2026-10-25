"""D12b §5.23: verified conflict headline (27B kv vs 27B text, paired) on ALL / CLEAN / FLAGGED items of ConflictQA-CF and
FaithEval-CF, flagged keys from outputs/D12b_max8k_flagged.json (scripts/d12b_leakcheck.py). CPU, per-item jsonl only.
    python scripts/d12b_conflict_clean.py
"""
import json, math
import os
R=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs') + '/'
F=json.load(open(R+'D12b_max8k_flagged.json'))['sets']
def L(p): return {json.loads(x)['key']:json.loads(x) for x in open(R+p+'/eval_test.ckpt.jsonl')}
def ms(xs):
    n=len(xs); m=sum(xs)/n; return m, math.sqrt(sum((x-m)**2 for x in xs)/max(n-1,1))/math.sqrt(n)
for s,flag in (('cqacf','cqacf'),('fecf','fecf')):
    bad=set(F[flag]['flagged_keys'])
    T0=L(f'D12p_9b_deep_v2_{s}'); T1=L(f'D12p_9b_deep_v2s1_{s}'); Trc=L(f'D12p_9b_deep_v2_rc_{s}')
    runs=[('deep_v2 seed 0',T0,T0),('deep_v2 seed 0 recompute',Trc,Trc),('deep_v2 seed 1',T1,T1),
          ('lever-full',L(f'D12p_lvfull_{s}'),T0),('lv4',L(f'D12b_lv4_{s}'),T0)]
    for sub in ('all','clean','flagged'):
        for name,E,T in runs:
            ks=[k for k in E if k in T and ((sub=='all') or (sub=='clean' and k not in bad) or (sub=='flagged' and k in bad))]
            if not ks: continue
            kv=sum(E[k]['kv']['score'] for k in ks)/len(ks); tx=sum(T[k]['text']['score'] for k in ks)/len(ks)
            wr=sum(E[k]['wrong_doc']['score'] for k in ks)/len(ks)
            m,se=ms([E[k]['kv']['score']-T[k]['text']['score'] for k in ks])
            src='own' if T is E else 'seed0 text'
            print(f"| {s} | {sub} | {name} | {len(ks)} | {kv:.3f} | {tx:.3f} ({src}) | {wr:.3f} | {m:+.3f} ± {se:.3f} | {(m/se if se else float('nan')):.1f} |")
