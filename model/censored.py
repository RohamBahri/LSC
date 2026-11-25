"""Pooled 21-day censored Poisson likelihood with a recorded-count anchor.
Censor flags are conservative proxies, not verified observations of lost sales.
"""
import numpy as np, pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from scipy.stats import poisson
P=Path(__file__).parent; z=np.load(P/'data.npz'); old=np.load(P/'pred.npz'); Y=z['sales'].copy(); pre=z['pre']; out=z['output']; moves=z['moves']; res=z['resid']; val=z['val']; Y[:,:,0]+=np.maximum(-res[:,:,0],0); T,K,L=Y.shape
j24,jt,jm,jtest=[int(z[k]) for k in ['j24','jt','jm','jtest']]
def cs(a):return np.concatenate([np.zeros((1,*a.shape[1:]),np.float32),np.cumsum(a,axis=0,dtype=np.float32)])
cy=cs(Y); ci=cs(moves.sum(2)); co=cs(out); ce=cs(pre>0); cb=cs(pre<=Y)
def feat(t):
 x=[np.repeat(np.arange(K),L),np.tile(np.arange(L),K),np.repeat(val,L)]
 for w in [7,21,56,112,365]:
  a=cy[t]-cy[t-w];x.extend([a.ravel(),np.repeat(a.sum(1),L),(ci[t]-ci[t-w]).ravel(),np.repeat((co[t]-co[t-w]).sum(1),L),(ce[t]-ce[t-w]).ravel()/w])
 x.append(pre[t].ravel());return np.column_stack(x).astype('float32')
def lab(t):
 y=(cy[t+21]-cy[t]).ravel(); c=(cb[t+21]-cb[t]).ravel()>0; return y,c
# signed encoding for censor flag: -(count+1). Anchor weight 0.25 regularizes weakly identified cells.
def loss(y,eta):
 c=y<0; k=np.where(c,-y-1,y); mu=np.exp(np.clip(eta,-12,7)); g=mu-k; h=mu.copy(); ix=c&(k>0)
 if ix.any():
  m=mu[ix]; q=k[ix]; logsf=poisson.logsf(q-1,m); logpm=poisson.logpmf(q-1,m); r=np.where(np.isfinite(logsf),np.exp(np.clip(np.log(m)+logpm-logsf,-40,20)),np.maximum(q-m,0)); hc=np.maximum(r*(m+r-q),1e-5);g[ix]=.25*g[ix]-.75*r;h[ix]=.25*h[ix]+.75*hc
 ix=c&(k==0);g[ix]*=.25;h[ix]*=.25
 return g,np.maximum(h,1e-5)
if __name__=='__main__':
 result={};rows=[]
 for period,cut,end in [('tune',jt,jm),('test',jtest,T)]:
  tr=np.arange(j24+90,cut-20,7); ts=np.arange(cut,end-20,7); X=np.concatenate([feat(t) for t in tr]); y,c=map(np.concatenate,zip(*[lab(t) for t in tr])); model=LGBMRegressor(objective=loss,n_estimators=90,num_leaves=31,learning_rate=.065,min_child_samples=100,n_jobs=4,verbosity=-1);model.fit(X,np.where(c,-y-1,y),categorical_feature=[0,1]); del X
  pp=np.zeros((end-cut,K,L),np.float32)
  for t in range(cut,end,3):
   raw=np.exp(np.clip(model.predict(feat(t),raw_score=True),-12,5)).reshape(K,L);base=old[period+'_ops'][t-cut];raw=np.minimum(raw,2*base+.25); pp[t-cut:min(t+3,end)-cut]=raw
  result[period+'_censor']=pp
  obs,cc=map(np.concatenate,zip(*[lab(t) for t in ts]));
  for name,pr in [('ops',old[period+'_ops']),('censor',pp)]:
   b=np.concatenate([pr[t-cut].ravel() for t in ts]);
   for group,mask in [('all',np.ones(len(obs),bool)),('proxy_uncensored',~cc)]:
    o=obs[mask];p=b[mask];dv=2*np.mean(np.where(o>0,o*np.log(np.maximum(o,1e-10)/np.maximum(p,1e-10)),0)-o+p);rows.append(dict(period=period,model=name,group=group,n=len(o),deviance=dv,observed=o.sum(),predicted=p.sum()))
  print(rows[-4:],flush=True)
 np.savez_compressed(P/'censor_pred.npz',**result);pd.DataFrame(rows).to_csv(P/'censor_metrics.csv',index=False)
