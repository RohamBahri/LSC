"""Unified daily replay: joint MIP, responsive proposals, dynamic learning features."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import warnings;warnings.filterwarnings('ignore',message='X does not have valid feature names')
import numpy as np,pandas as pd,joblib,json
from pathlib import Path
from dataclasses import replace
import mip as m
P=Path(__file__).resolve().parent;R=P.parent/'refinement_results';R.mkdir(exist_ok=True)
z=np.load(P/'data.npz');K,L=m.K,m.L;T=m.T;dates=z['dates'];codes=z['codes'];v=m.v
jt,jm,jtest=[int(z[k]) for k in ('jt','jm','jtest')];END=T-14
models=joblib.load(P/'models.joblib');cfg0=m.Config(**json.loads((P/'selected.json').read_text()));beta=pd.read_csv(P/'residual_coefficients.csv').query("horizon==21 and kind=='residual'").iloc[0]
sku=np.repeat(np.arange(K),L);loc=np.tile(np.arange(L),K)
class History:
 def __init__(self,lo):
  y=m.sales[:lo].copy();y[:,:,0]+=np.maximum(-m.res[:lo,:,0],0)
  self.h=[]
  for a in [y,m.moves[:lo].sum(2),m.moves[:lo].sum(3),m.out[:lo],m.pre[:lo]>0]:
   c=np.zeros((T+1,K,L),np.float32);np.cumsum(a,axis=0,dtype=np.float32,out=c[1:lo+1]);self.h.append(c)
 def before(self,t,y,opening,completion):
  for i,a in [(0,y),(4,opening>0),(3,completion)]:self.h[i][t+1]=self.h[i][t]+a
 def after(self,t,ship):
  for i,a in [(1,ship.sum(1)),(2,ship.sum(2))]:self.h[i][t+1]=self.h[i][t]+a
 def feat(self,t,inv):
  cy,ci,cd,co,ce=self.h
  x=[sku,loc,np.repeat(v,L),np.full(K*L,t%7),inv.ravel(),np.repeat(inv.sum(1),L),np.repeat(inv[:,0],L)]
  for w in (7,21,56,112,365):
   a=cy[t+1]-cy[t+1-w];x.extend([a.ravel(),np.repeat(a.sum(1),L),(ci[t]-ci[t-w]).ravel(),(cd[t]-cd[t-w]).ravel(),np.repeat((co[t]-co[t-w]).sum(1),L),(ce[t]-ce[t-w]).ravel()/w])
  return np.column_stack(x).astype('float32')
def expect(kind,x):
 c,q=models['action'][kind];return c.predict_proba(x,num_threads=2)[:,1]*q.predict(x,num_threads=2)
def baseline_ship(inv,s):
 x=np.zeros_like(s);tmp=inv.copy()
 for o in range(L):
  for j in range(L):
   if o!=j:x[:,o,j]=np.minimum(tmp[:,o],s[:,o,j]);tmp[:,o]-=x[:,o,j]
 return x
def proposal(mode,s0,p0,pos,x,ref):
 if mode=='fixed':return s0.copy(),p0.copy()
 sh=s0.copy().astype(int);pr=p0.copy().astype(int);refpos,refx=ref
 xx=x.reshape(K,L,-1);rr=refx.reshape(K,L,-1)
 for j in range(1,L):
  if not s0[:,0,j].any():continue
  old=s0[:,0,j]; diff=refpos[:,j]-pos[:,j]
  delta=diff if mode=='target' else expect('shipment',xx[:,j])-expect('shipment',rr[:,j])
  delta=np.where(diff>0,np.maximum(delta,0),np.where(diff<0,np.minimum(delta,0),0))
  new=np.clip(np.rint(old+delta),0,old+4).astype(int);sh[:,0,j]=np.where(refpos[:,j]+old>0,new,old)
 d=expect('production',xx[:,0])-expect('production',rr[:,0]);diff=refpos.sum(1)-pos.sum(1)
 d=np.where(diff>0,np.maximum(d,0),np.where(diff<0,np.minimum(d,0),0))
 pr[:,0]=np.where(p0[:,0]>0,np.clip(np.rint(p0[:,0]+d),0,p0[:,0]+3),0).astype(int)
 cap=max(cfg0.cap,int(p0.sum()));ex=max(0,int(pr.sum())-cap)
 for k in np.argsort(-np.maximum(pr[:,0]-p0[:,0],0)):
  if not ex:break
  a=min(ex,max(0,int(pr[k,0]-p0[k,0])));pr[k,0]-=a;ex-=a
 assert pr.sum()<=cap
 return sh,pr

def replay(lo,hi,name,mode='fixed',use_mip=True,signal=True,trim=-1.,ref=None,level=0,net_sales=False,theta=None,lateral=True):
 n=hi-lo;lead=14;cfg=replace(cfg0,planlower=(trim<0),shipcut=max(0,trim),lateral=lateral,extra_factory=(theta is None));inv=m.pre[lo].copy().astype(int)
 pipe=np.zeros((n+lead+30,K,L),np.int32);wip=np.zeros_like(pipe);wip[:lead]=m.out[lo:lo+lead];h=History(lo);rows=[];solves=[];actions=[];states=[];extra=m.hidden(lo,hi,level)
 for i,t in enumerate(range(lo,hi)):
  before=inv.sum()+pipe[i:].sum()+wip[i:].sum();completion=wip[i].copy();inv+=pipe[i]+completion;pipe[i]=0;wip[i]=0
  opening=inv.copy();sd=m.sales[t] if not net_sales else np.maximum(m.net[t],0);positive=np.maximum(m.res[t],0)+np.maximum(-m.net[t],0);inv+=positive;ext=np.maximum(-m.res[t],0);served=np.minimum(inv,ext);inv-=served;rr=np.minimum(inv,sd);inv-=rr;hh=np.minimum(inv,extra[i]);inv-=hh
  observed=rr+hh;observed[:,0]+=served[:,0];h.before(t,observed,opening,completion)
  x=h.feat(t,inv);pos=inv+pipe[i+1:].sum(0)+wip[i+1:].sum(0)
  if ref is None:states.append((pos.copy(),x.copy()))
  s0=m.moves[t].copy();p0=m.out[t+lead].copy();s,p=proposal(mode,s0,p0,pos,x,ref[i] if ref is not None else None)
  if use_mip or theta is not None:
   mu=models['demand'][21,False].predict(x,num_threads=2).reshape(K,L)
   if signal:
    A=s0.sum(1).ravel();mask=(A>0)&(loc>0);xx=ref[i][1] if ref is not None else x;E=expect('shipment',xx[mask]) if mask.any() else np.array([])
    mu.ravel()[mask]*=np.exp(np.clip(beta.intercept+beta.residual*(np.log1p(A[mask])-np.log1p(E))+beta.routine*np.log1p(E),-6,6))
   if theta is not None:
    from scipy.stats import nbinom
    mean=np.maximum(mu*14/21,1e-9); pnb=1.25/(1.25+mean)
    for j in range(1,L):
     for k in np.flatnonzero(s[:,0,j]):
      if (m.extcum[t,k]-m.extcum[t-365,k]).sum()>0:continue
      q=int(s[k,0,j]); pp=int(pos[k,j]); threshold=theta*v[k]/1000
      keep=0
      for unit in range(1,q+1):
       if nbinom.sf(pp+unit-1,1.25,pnb[k,j])>=threshold:keep=unit
       else:break
      s[k,0,j]=keep
  if use_mip:
   ship,prod,st=m.optimize(inv,pipe[i+1:],wip[i+1:],s,p,mu,t,replace(cfg,cap=max(cfg.cap,int(p0.sum()))));assert st['status']==0 and st['gap']<=.005001,st;solves.append(dict(date=dates[t],**st))
  else:ship=baseline_ship(inv,s);prod=p
  assert (ship.sum(2)<=inv).all() and prod.sum()<=max(cfg.cap,int(p0.sum()))
  inv-=ship.sum(2)
  for o in range(L):
   for j in range(L):
    if o!=j:pipe[i+(3 if o==0 or j==0 else 1+cfg.prep),:,j]+=ship[:,o,j]
  wip[i+lead]+=prod;h.after(t,ship)
  now=inv.sum()+pipe[i+1:].sum()+wip[i+1:].sum();assert now-before-positive.sum()+served.sum()+rr.sum()+hh.sum()-prod.sum()==0
  fg=((inv.sum(1)+pipe[i+1:].sum((0,2)))*v).sum();work=(wip[i+1:].sum((0,2))*v).sum();q=ship.sum(0); bases=(inv[:,1:]*v[:,None]).sum(); edm=(inv[:,0]*v).sum(); transit=(pipe[i+1:].sum((0,2))*v).sum()
  rows.append(dict(date=dates[t],field=int((rr+hh)[:,1:].sum()),demand=int((sd+extra[i])[:,1:].sum()),recorded=int(rr[:,1:].sum()),hidden=int(hh[:,1:].sum()),external=int(served.sum()+rr[:,0].sum()),external_demand=int(ext.sum()+sd[:,0].sum()),finished=fg,wip=work,value=fg+work,bases=bases,edm=edm,transit=transit,production=int(prod.sum()),lateral=int(q[1:,1:].sum()),factory=int(q[0].sum()),routes=int((q>0).sum()),small=int(((q>0)&(q<cfg.minimum)).sum()),ship_response=int(np.abs(s-s0).sum()),prod_response=int(np.abs(p-p0).sum()),ship_trim=int(np.maximum(s-ship,0).sum())))
  if name.startswith('test_'):
   for k,o,j in zip(*np.nonzero(ship+s+s0)):actions.append((dates[t],codes[k],'ship',o,j,int(s0[k,o,j]),int(s[k,o,j]),int(ship[k,o,j])))
   for k,j in zip(*np.nonzero(prod+p+p0)):actions.append((dates[t],codes[k],'production',j,j,int(p0[k,j]),int(p[k,j]),int(prod[k,j])))
 d=pd.DataFrame(rows);d.to_csv(R/(name+'_daily.csv'),index=False);pd.DataFrame(solves).to_csv(R/(name+'_solves.csv'),index=False)
 if actions:pd.DataFrame(actions,columns=['date','sku','type','origin','destination','historical','responsive_proposal','optimized']).to_csv(R/(name+'_actions.csv'),index=False)
 r=dict(name=name,mode=mode,signal=signal,trim=trim,level=level,net_sales=net_sales,theta=theta,lateral_enabled=lateral,days=n,field=100*d.field.sum()/d.demand.sum(),field_units=d.field.sum(),demand=d.demand.sum(),external=d.external.sum(),external_demand=d.external_demand.sum(),value=d.value.mean(),bases=d.bases.mean(),edm=d.edm.mean(),transit=d.transit.mean(),finished=d.finished.mean(),wip=d.wip.mean(),production=d.production.sum(),lateral=d.lateral.sum(),factory=d.factory.sum(),routes=d.routes.sum(),small=d.small.sum(),ship_response=d.ship_response.sum(),prod_response=d.prod_response.sum(),ship_trim=d.ship_trim.sum(),solves=len(solves),max_gap=max([s['gap'] for s in solves],default=0),max_seconds=max([s['seconds'] for s in solves],default=0));print(r,flush=True);return r,states

def main():
 rows=[]
 def go(lo,hi,name,ref=None,**kw):
  r,rr=replay(lo,hi,name,ref=ref,**kw);rows.append(r);pd.DataFrame(rows).to_csv(R/'summary.csv',index=False);return r,rr
 for period,lo,hi in [('tune',jt,jm),('test',jtest,END)]:
  b,ref=go(lo,hi,period+'_LSC',use_mip=False)
  go(lo,hi,period+'_learned',ref,mode='learned')
  go(lo,hi,period+'_no_extra_lateral',ref,mode='learned',lateral=False)
  if period=='tune':
   for th in [1e-5,.0003,.003]:go(lo,hi,period+'_threshold_'+str(th),ref,mode='learned',theta=th,lateral=False)
   d=pd.DataFrame(rows);d=d[d.theta.notna()];good=d[(d.field>=b['field']-.2)&(d.external>=b['external'])&(d.routes<=b['routes'])];chosen=float((good if len(good) else d).sort_values('value').iloc[0].theta);print('THRESHOLD SELECTED',chosen,'eligible',len(good),flush=True)
  else:
   go(lo,hi,period+'_threshold_'+str(chosen),ref,mode='learned',theta=chosen,lateral=False)
   go(lo,hi,period+'_threshold_only',ref,mode='learned',theta=chosen,lateral=False,use_mip=False)
  del ref
 b,ref=go(jt,END,'extended_LSC',use_mip=False)
 go(jt,END,'extended_learned',ref,mode='learned')
 go(jt,END,'extended_no_extra_lateral',ref,mode='learned',lateral=False)
if __name__=='__main__':main()
