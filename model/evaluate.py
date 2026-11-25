"""Frozen policy comparison using full-ledger reconstruction and refitted models."""
import argparse,json
from dataclasses import replace
import pandas as pd
import refined as r

assert str(r.z['inventory_method']) == 'full_ledger_through_2026-08-31'
p=argparse.ArgumentParser();p.add_argument('--window',choices=['short','continuous','all'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--original',action='store_true',help='Also evaluate the earlier no-recovery MIP');args=p.parse_args()
base_cfg=r.cfg0
pilot=json.loads((r.P/'pilot_candidate.json').read_text())
out=r.R/'full_ledger_results.csv'
rows=pd.read_csv(out).to_dict('records') if args.resume and out.exists() else []
for window,lo,hi in [('short',r.jtest,r.END),('continuous',r.jt,r.END)]:
    if args.window not in (window,'all'): continue
    r.cfg0=base_cfg
    b,ref=r.replay(lo,hi,'test_'+window+'_LSC',use_mip=False)
    for policy,overrides in [('LSC',None),('recovery_MIP',pilot['overrides'])]+([('original_MIP',{})] if args.original else []):
        if any(x['window']==window and x['policy']==policy for x in rows): continue
        if overrides is None: a=b
        else:
            r.cfg0=replace(base_cfg,**overrides)
            a,_=r.replay(lo,hi,'test_'+window+'_'+policy,mode='learned',ref=ref)
        a.update(window=window,policy=policy,inventory_reduction_pct=100*(1-a['value']/b['value']),service_change_pp=a['field']-b['field'],production_reduction_pct=100*(1-a['production']/b['production']),external_change=a['external']-b['external'])
        rows.append(a)
        pd.DataFrame(rows).to_csv(out,index=False)
    del ref
