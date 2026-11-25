import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.metrics import average_precision_score,roc_auc_score,mean_poisson_deviance
import joblib
P=Path(__file__).parent; z=np.load(P/'data.npz'); sales=z['sales']; net=z['net']; pre=z['pre']; moves=z['moves']; output=z['output']; res=z['resid'];v=z['val']; dates=z['dates']; T,K,L=sales.shape; j24,j25,jt,jm,jtest=[int(z[k]) for k in ['j24','j25','jt','jm','jtest']]; END=T-14
Y=sales.copy();Y[:,:,0]+=np.maximum(-res[:,:,0],0)
cs=lambda a:np.concatenate([np.zeros((1,*a.shape[1:]),np.float32),np.cumsum(a,axis=0,dtype=np.float32)])
cy=cs(Y); D=moves.sum(2); O=moves.sum(3); ci=cs(D); co=cs(output); ce=cs(pre>0); cd=cs(O); sku=np.repeat(np.arange(K),L); loc=np.tile(np.arange(L),K)
def feat(t,state=None):
 s=np.maximum(pre[t]-sales[t],0) if state is None else state
 x=[sku,loc,np.repeat(v,L),np.full(K*L,t%7),s.ravel(),np.repeat(s.sum(1),L),np.repeat(s[:,0],L)]
 for w in [7,21,56,112,365]:
  a=cy[t+1]-cy[t+1-w];x.extend([a.ravel(),np.repeat(a.sum(1),L),(ci[t]-ci[t-w]).ravel(),(cd[t]-cd[t-w]).ravel(),np.repeat((co[t]-co[t-w]).sum(1),L),(ce[t]-ce[t-w]).ravel()/w])
 return np.column_stack(x).astype('float32')
def fit(kind,X,y):
 args=dict(n_estimators=70,num_leaves=23,learning_rate=.09,min_child_samples=100,n_jobs=4,verbosity=-1,random_state=7)
 m=LGBMClassifier(**args) if kind=='class' else LGBMRegressor(objective='poisson',**args);m.fit(X,y,categorical_feature=[0,1]);return m
route=D.sum(1)>0; histprod=np.roll(output[:,:,0],-14,axis=0);histprod[-14:]=0
def action_rows(tt,kind):
 xx=[];yy=[];rr=[]
 for t in tt:
  x=feat(t).reshape(K,L,-1)
  if kind=='shipment':
   for j in np.flatnonzero(route[t,1:])+1:xx.append(x[:,j]);yy.append(D[t,:,j]);rr.append(np.full(K,t*L+j))
  else:xx.append(x[:,0]);yy.append(histprod[t]);rr.append(np.full(K,t*L))
 return np.concatenate(xx),np.concatenate(yy),np.concatenate(rr)
def metrics(y,p,q,r,kind,period):
 hit=total=unit=topcnt=0
 for a in np.unique(r):
  ix=np.flatnonzero(r==a);n=(y[ix]>0).sum()
  if n:
   sel=ix[np.argsort(-p[ix],kind='stable')[:n]];hit+=(y[sel]>0).sum();total+=n;unit+=y[sel].sum();topcnt+=y[ix].sum()
 return dict(period=period,action=kind,n=len(y),positive=int((y>0).sum()),prevalence=float((y>0).mean()),AP=average_precision_score(y>0,p),AUC=roc_auc_score(y>0,p),topK_recall=hit/max(total,1),topK_unit_capture=unit/max(topcnt,1),count_mae=np.abs(y-q).mean(),pred_units=q.sum(),actual_units=y.sum())
if __name__=='__main__':
 rows=[];models={};tr=np.arange(j24+90,jt-14,2)
 for kind in ['shipment','production']:
  X,y,r=action_rows(tr,kind);print('training',kind,X.shape,flush=True);clf=fit('class',X,y>0);qty=fit('count',X[y>0],y[y>0]);models[kind]=(clf,qty)
  for period,tt in [('tune',np.arange(jt,jm-14)),('test',np.arange(jtest,END))]:
   X,y,r=action_rows(tt,kind);p=clf.predict_proba(X)[:,1];q=p*qty.predict(X);rows.append(metrics(y,p,q,r,kind+'_ML',period));score=X[:,7 if kind=='shipment' else 8]/(1+X[:,4 if kind=='shipment' else 5]);rows.append(metrics(y,score,np.zeros_like(score),r,kind+'_recent_sales_rank',period));rows[-1]['count_mae']=np.nan;rows[-1]['pred_units']=np.nan;print(rows[-2],flush=True)
  pd.DataFrame(rows).to_csv(P/'action_metrics.csv',index=False)
 fr=[];preds={};demandmodels={}
 def frow(t,proposal):
  x=feat(t)
  return np.column_stack([x,D[t].ravel(),O[t].ravel()]).astype('float32') if proposal else x
 for H in [7,21]:
  for proposal in [False,True]:
   tr=np.arange(j24+90,jt-H-1,7);X=np.concatenate([frow(t,proposal) for t in tr]);y=np.concatenate([(cy[t+1+H]-cy[t+1]).ravel() for t in tr]);m=fit('count',X,y);demandmodels[H,proposal]=m;del X,y
   for period,cut,end in [('tune',jt,jm),('test',jtest,END)]:
    tt=np.arange(cut,end-H,H);X=np.concatenate([frow(t,proposal) for t in tt]);y=np.concatenate([(cy[t+1+H]-cy[t+1]).ravel() for t in tt]);p=m.predict(X);field=X[:,1]>0;rec=np.concatenate([D[t].ravel()>0 for t in tt]);
    for group,mask in [('all_field',field),('received_today',field&rec),('not_received',field&~rec)]:
     yy=y[mask];pp=np.maximum(p[mask],1e-8);fr.append(dict(period=period,horizon=H,proposal=proposal,group=group,n=len(yy),actual=yy.sum(),pred=pp.sum(),deviance=mean_poisson_deviance(yy,pp),mae=np.abs(yy-pp).mean()))
    if H==21:preds[period+('_proposal' if proposal else '_hist')]=np.array([m.predict(frow(t,proposal)).reshape(K,L) for t in range(cut,end)],dtype=np.float32)
    print(fr[-3],flush=True)
   pd.DataFrame(fr).to_csv(P/'forecast_proposal_metrics.csv',index=False)
 np.savez_compressed(P/'proposal_pred.npz',**preds);joblib.dump(dict(action=models,demand=demandmodels),P/'models.joblib')
 lanes=[]
 for o in range(L):
  for j in range(L):
   if o==j:continue
   a=moves[j25:END,:,o,j].sum(1);n=a>0
   if n.any():lanes.append(dict(origin=o,destination=j,days=int(n.sum()),units=int(a.sum()),median=float(np.median(a[n])),single=int((a==1).sum()),small_2=int(((a>0)&(a<=2)).sum()),per_30d=n.mean()*30,dominant_weekday=pd.to_datetime(dates[j25:END][n]).day_name().value_counts().index[0]))
 pd.DataFrame(lanes).to_csv(P/'lanes.csv',index=False)
