"""Calibrate residual from action model against demand with Poisson log-offset on tuning data.
Only current positive shipment proposals; no action correction on nonproposed cells.
"""
import warnings;warnings.filterwarnings('ignore',message='X does not have valid feature names')
import learn as a
import pandas as pd,numpy as np,joblib
from scipy.optimize import minimize
P=a.P; models=joblib.load(P/'models.joblib');clf,qty=models['action']['shipment'];dm=models['demand'];rows=[];coeff=[];preds={}
def inputs(t,H):
 x=a.feat(t); mask=(a.D[t].ravel()>0)&(a.loc>0); base=dm[H,False].predict(x).reshape(a.K,a.L); direct=dm[H,True].predict(np.column_stack([x,a.D[t].ravel(),a.O[t].ravel()]).astype('float32')).reshape(a.K,a.L); A=a.D[t].ravel()[mask]; E=clf.predict_proba(x[mask])[:,1]*qty.predict(x[mask]) if mask.any() else np.zeros(0);y=(a.cy[t+1+H]-a.cy[t+1]).ravel()[mask] if t+1+H<=a.T else np.zeros(len(A));Z=np.column_stack([np.ones(len(A)),np.log1p(A)-np.log1p(E),np.log1p(E)]);return mask,base,direct,Z,y
for H in [7,21]:
 X=[];y=[];off=[]
 for t in range(a.jt,a.jm-H):
  mask,b,d,z,yy=inputs(t,H);X.append(z);y.append(yy);off.append(b.ravel()[mask]);
 X=np.concatenate(X);y=np.concatenate(y);off=np.maximum(np.concatenate(off),1e-8)
 for typ,ix in [('flag',[0]),('residual',[0,1,2])]:
  x=X[:,ix]
  def obj(beta):
   eta=x@beta; lam=off*np.exp(np.clip(eta,-6,6));return np.sum(lam-y*eta)+.5*np.sum(beta[1:]**2)
  beta=minimize(obj,np.zeros(len(ix)),method='BFGS').x;coeff.append(dict(horizon=H,kind=typ,intercept=beta[0],residual=beta[1] if len(beta)>1 else 0,routine=beta[2] if len(beta)>2 else 0))
  if typ=='flag':flag=beta
  else:resid=beta
 for period,lo,hi in [('tune',a.jt,a.jm),('test',a.jtest,a.END)]:
  arr=[]; agg={k:[[],[]] for k in ['history','direct','flag','residual']}
  for t in range(lo,hi):
   mask,b,d,z,y=inputs(t,H);f=b.copy();r=b.copy();f.ravel()[mask]*=np.exp(np.clip(z[:,0]*flag[0],-6,6));r.ravel()[mask]*=np.exp(np.clip(z@resid,-6,6));arr.append(r)
   if t<hi-H:
    for name,p in [('history',b),('direct',d),('flag',f),('residual',r)]:agg[name][0].append(y);agg[name][1].append(p.ravel()[mask])
  for name,(ys,ps) in agg.items():
   yy=np.concatenate(ys);p=np.maximum(np.concatenate(ps),1e-8);rows.append(dict(period=period,H=H,kind=name,n=len(yy),actual=yy.sum(),pred=p.sum(),deviance=a.mean_poisson_deviance(yy,p),mae=np.abs(yy-p).mean()))
  if H==21:preds[period+'_residual']=np.array(arr,dtype=np.float32)
 print('H',H,coeff[-2:],rows[-4:],flush=True)
pd.DataFrame(coeff).to_csv(P/'residual_coefficients.csv',index=False);pd.DataFrame(rows).to_csv(P/'residual_metrics.csv',index=False);np.savez_compressed(P/'residual_pred.npz',**preds)
