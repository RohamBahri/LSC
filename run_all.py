"""Reproduce the full-ledger recovery-enabled MIP results."""
import argparse,os,sys,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent; model=root/'model'
p=argparse.ArgumentParser();p.add_argument('--rebuild',action='store_true');p.add_argument('--data-dir',type=Path);p.add_argument('--window',choices=['short','continuous','all'],default='all');p.add_argument('--original',action='store_true');args=p.parse_args()
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
if args.rebuild:
    if args.data_dir is None:p.error('--rebuild requires --data-dir with datasets.zip and Inventory.zip')
    env['LSC_DATA_DIR']=str(args.data_dir.resolve())
    for f in ('prepare.py','forecast.py','censored.py','learn.py','calibrate.py'):
        subprocess.run([sys.executable,str(model/f)],cwd=model,env=env,check=True)
if not (model/'data.npz').exists():
    p.error('No prepared data. Use --rebuild --data-dir /path/to/archives first.')
cmd=[sys.executable,str(model/'evaluate.py'),'--window',args.window]
if args.original:cmd.append('--original')
subprocess.run(cmd,cwd=model,env=env,check=True)
