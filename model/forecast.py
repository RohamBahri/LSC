import numpy as np,pandas as pd
from lightgbm import LGBMRegressor,LGBMClassifier
from pathlib import Path
P=Path(__file__).parent; z=np.load(P/'data.npz'); Y=z['sales'].copy();pre=z['pre']; output=z['output']; moves=z['moves']; resid=z['resid'];val=z['val'];j24,j25,jt,jm,jtest=[int(z[k]) for k in ['j24','j25','jt','jm','jtest']];T,K,L=Y.shape;Y[:,:,0]+=np.maximum(-resid[:,:,0],0); stock=pre>0
cs=lambda a:np.concatenate([np.zeros((1,*a.shape[1:]),np.float32),np.cumsum(a,axis=0,dtype=np.float32)])
cy=cs(Y); ci=cs(moves.sum(2));co=cs(output); ce=cs(stock); csy=cs(Y*stock)
# 21-day direct forecast. Exposure-adjusted target imputation is deliberately sensitivity, not identified demand.
def feat(t,ops):
 x=[np.repeat(np.arange(K),L),np.tile(np.arange(L),K),np.repeat(val,L)]
 for w in [7,21,56,112,365]:
  a=cy[t]-cy[t-w];x.extend([a.ravel(),np.repeat(a.sum(1),L)])
  if ops:x.extend([(ci[t]-ci[t-w]).ravel(),np.repeat((co[t]-co[t-w]).sum(1),L),(ce[t]-ce[t-w]).ravel()/w])
 if ops:x.append(pre[t].ravel())
 return np.column_stack(x).astype('float32')
def target(t):return (cy[t+21]-cy[t]).ravel()
def labels(t,corr):
 y=target(t)
 if not corr:return y
 rate=((csy[t]-csy[t-365])+60*(cy[t]-cy[t-365])/365)/(ce[t]-ce[t-365]+60)
 add=np.minimum(rate*(21-(ce[t+21]-ce[t])),3);add[:,0]=0
 return y+0.5*add.ravel()
preds={};metrics=[]
for period,cut,end in [('tune',jt,jm),('test',jtest,T)]:
 tr=np.arange(j24+90,cut-20,7); ts=np.arange(cut,end-20,7); evalt=np.arange(cut,end,3); obs=np.concatenate([target(t) for t in ts]);
 for name,ops,corr in [('ml',False,False),('ops',True,False),('correct',True,True)]:
  X=np.concatenate([feat(t,ops) for t in tr]);yy=np.concatenate([labels(t,corr) for t in tr]); model=LGBMRegressor(objective='poisson',n_estimators=130,num_leaves=31,learning_rate=.07,min_child_samples=100,n_jobs=4,verbosity=-1);model.fit(X,yy,categorical_feature=[0,1]); b=model.predict(np.concatenate([feat(t,ops) for t in ts]));dev=2*np.mean(np.where(obs>0,obs*np.log(np.maximum(obs,1e-12)/np.maximum(b,1e-8)),0)-obs+b); metrics.append([period,name,dev,np.abs(obs-b).mean(),obs.sum(),b.sum()]);print(metrics[-1],flush=True)
  pred=np.zeros((end-cut,K,L),np.float32)
  for t in evalt:pred[t-cut:min(t+3,end)-cut]=model.predict(feat(t,ops)).reshape(K,L)
  preds[period+'_'+name]=pred
  if name=='ops':
   tail=LGBMClassifier(n_estimators=100,num_leaves=23,learning_rate=.07,min_child_samples=100,n_jobs=4,verbosity=-1);tail.fit(X,yy>=5,categorical_feature=[0,1]);tp=np.zeros_like(pred)
   for t in evalt:tp[t-cut:min(t+3,end)-cut]=tail.predict_proba(feat(t,ops))[:,1].reshape(K,L)
   preds[period+'_tail']=tp; print('tail probability',tp.mean(),'observed',np.mean(obs>=5),flush=True)
  del X,yy
np.savez_compressed(P/'pred.npz',**preds);pd.DataFrame(metrics,columns=['period','model','deviance','mae','observed','pred']).to_csv(P/'forecast.csv',index=False)
