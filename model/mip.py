"""Full domestic-portfolio rolling integer decision model.
Two inventory valuation checkpoints; first-day movements and production are executed.
Terminal factory allocations are provisional, not executed. See README for caveats.
"""
import os,time,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from policy import *
dates=pd.date_range("2021-01-01",periods=T).astype(str); codes=np.load(P/"data.npz")["codes"].astype(str)
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from collections import defaultdict
from dataclasses import dataclass,asdict
@dataclass
class Config:
 name:str='mip'; forecast:str='ops'; lead:int=14; prep:int=1; minimum:int=2; cap:int=45
 cutcost:float=1.5; prodcost:float=2.; movecost:float=.45; fixedcost:float=2.; fieldweight:float=6.; extweight:float=3.
 protect:bool=True; newroutes:bool=False; lateral:bool=True; addprod:bool=True; limit:float=3.; planlower:bool=True; guard:int=-1; holding:float=.15; shipcut:float=6.; extra_factory:bool=True
class Model:
 def __init__(self): self.c=[];self.lo=[];self.hi=[];self.integer=[];self.ri=[];self.ci=[];self.a=[];self.lb=[];self.ub=[]
 def var(self,c=0,lo=0,hi=np.inf,integer=0):
  i=len(self.c);self.c.append(c);self.lo.append(lo);self.hi.append(hi);self.integer.append(integer);return i
 def row(self,terms,lo=-np.inf,hi=np.inf):
  r=len(self.lb);self.lb.append(lo);self.ub.append(hi)
  for k,v in terms.items():
   if v:self.ri.append(r);self.ci.append(k);self.a.append(v)
 def solve(self,limit):
  A=coo_matrix((self.a,(self.ri,self.ci)),shape=(len(self.lb),len(self.c))).tocsc();s=time.perf_counter(); r=milp(np.array(self.c),integrality=np.array(self.integer),bounds=Bounds(self.lo,self.hi),constraints=LinearConstraint(A,self.lb,self.ub),options={'time_limit':limit,'mip_rel_gap':.005});elapsed=time.perf_counter()-s
  if r.x is None:return None,dict(status=r.status,seconds=elapsed,nvar=len(self.c),nint=sum(self.integer),ncon=len(self.lb),gap=np.nan)
  x=r.x.copy(); ii=np.flatnonzero(self.integer);x[ii]=np.rint(x[ii]);ax=A@x;viol=max(np.max(np.maximum(np.array(self.lb)-ax,0),initial=0),np.max(np.maximum(ax-np.array(self.ub),0),initial=0),np.max(np.maximum(np.array(self.lo)-x,0),initial=0),np.max(np.maximum(x-np.array(self.hi),0),initial=0)); assert viol<1e-4,viol
  return x,dict(status=r.status,seconds=elapsed,nvar=len(self.c),nint=sum(self.integer),ncon=len(self.lb),gap=getattr(r,'mip_gap',np.nan),obj=r.fun,violation=viol)
# Protect historically external-facing products, based only on past dates.
extcum=np.concatenate([np.zeros((1,K,L)),np.cumsum(np.maximum(-res,0),axis=0)])
dispcum=np.concatenate([np.zeros((1,K)),np.cumsum(moves[:,:,0,:].sum(2),axis=0)])
price=v/np.median(v); price=np.clip(price,.25,4)
try: cp=np.load(P/'censor_pred.npz')
except FileNotFoundError:cp=None

def optimize(inv,pipe,wip,proposed,planned,mu,t,cfg):
 """pipe/wip offset 0 = tomorrow. All inputs at decision after current-day sales."""
 H=cfg.lead+7; E=5; m=Model(); xvars={}; pvars={}; yvars={}; outgoing=defaultdict(dict); effect=[defaultdict(dict),defaultdict(dict)]; routevars={}; protected=(extcum[t]-extcum[t-365]).sum(1)>0
 # clip baseline dispatches to available stock in a deterministic order, same as historical replay
 base=np.zeros_like(proposed); tmp=inv.copy()
 for o in range(L):
  for j in range(L):
   if o!=j:base[:,o,j]=np.minimum(tmp[:,o],proposed[:,o,j]);tmp[:,o]-=base[:,o,j]
 # demand mean adjustment for length and extra buffer, NB tails with a fitted-model mean.
 me=np.maximum(mu*E/21,1e-8); ml=np.maximum(mu*H/21,1e-8); q=np.minimum(nbinom.ppf(.94,1.25,1.25/(1.25+ml)),35).astype(int)
 pe=inv+pipe[:E].sum(0)+wip[:E].sum(0); pl=inv+pipe[:H].sum(0)+wip[:H].sum(0)
 # Donor guard: optional extra shipments must leave some forecast coverage, and protect external products.
 histbuf=np.ceil(np.maximum((dispcum[t]-dispcum[t-56])/56*cfg.lead,(dispcum[t]-dispcum[t-365])/365*cfg.lead)).astype(int)
 # Only the proposed, feasible legacy quantity is protected in the route. Optional shipment size is limited by forecast deficits.
 extra_surplus=np.maximum(tmp-np.minimum(q, np.maximum(1,np.ceil(ml)).astype(int)),0); extra_surplus[:,0]=np.maximum(tmp[:,0]-np.maximum(histbuf,q[:,0]),0)
 if cfg.protect:extra_surplus[protected,:]=0
 for o in range(L):
  for j in range(L):
   if o==j:continue
   legacy=base[:,o,j]; has=bool(legacy.sum());
   if not has and not cfg.newroutes:continue
   if not has and not cfg.lateral and o!=0:continue
   if j==0:extra=np.zeros(K,int)
   else:extra=np.minimum(extra_surplus[:,o],np.maximum(q[:,j]-pl[:,j],0));extra=np.where(mu[:,j]>.12,extra,0)
   if not cfg.lateral and o!=0:extra[:]=0
   if not cfg.extra_factory and o==0:extra[:]=0
   # restrict additional load to 50% of legacy volume + 1, or 12 for new route
   extra_lim=max(1,int(.5*legacy.sum())) if has else 12
   ids={}; extra_ids={}
   for k in np.flatnonzero(legacy+extra):
    b=int(legacy[k]);a=int(extra[k]); delay=3 if o==0 or j==0 else 1+cfg.prep
    # mandatory feasible baseline movements: their quantities carry private information. Optional extras optimized simultaneously.
    ix=m.var(c=cfg.movecost*price[k],lo=b if (cfg.planlower or o!=0 or (cfg.protect and protected[k])) else 0,hi=b+a,integer=1);xvars[k,o,j]=ix;ids[ix]=1;outgoing[k,o][ix]=1
    for h,Hh in enumerate([E,H]):
     effect[h][k,o][ix]=-1
     if delay<=Hh:effect[h][k,j][ix]=1
    if not cfg.planlower and b:
     d=m.var(c=cfg.shipcut*price[k],hi=b);m.row({d:1,ix:1},lo=b)
   if not ids:continue
   if has:
    m.row(ids,hi=int(legacy.sum())+extra_lim)
    # Existing planned shipments grandfathered, including single-unit emergency moves.
    if cfg.planlower:routevars[o,j]=None
    else:
     zz=m.var(c=cfg.fixedcost,hi=1,integer=1);routevars[o,j]=zz;m.row({**ids,zz:-sum(m.hi[i] for i in ids)},hi=0);m.row({**ids,zz:-min(cfg.minimum,int(legacy.sum()))},lo=0)
   else:
    zz=m.var(c=cfg.fixedcost,hi=1,integer=1);routevars[o,j]=zz;m.row({**ids,zz:-min(extra_lim,sum(m.hi[i] for i in ids))},hi=0);m.row({**ids,zz:-cfg.minimum},lo=0)
 for (k,o),terms in outgoing.items():m.row(terms,hi=int(base[k,o].sum()+extra_surplus[k,o]))
 # Optional additional production, integer, with common capacity. Non-EDM planned output kept fixed by replay.
 post0=tmp+pipe[:H].sum(0)+wip[:H].sum(0)+base.sum(1)
 for k in range(K):
  b=int(planned[k,0]); need=max(0,int(q[k].sum()-pl[k].sum())); extra=min(3,need) if cfg.addprod and mu[k].sum()>.25 else 0
  if not b and not extra:continue
  lower=b if cfg.protect and protected[k] else 0
  if cfg.guard>=0:
   # Conservative floor from planner-informed stock coverage: a reduction requires actual excess factory stock after committed dispatches.
   post=post0[k]
   # remove incoming historical dispatches that were double-counted at their origin in tmp; only add destination quantities once
   future_need=np.maximum(q[k,1:]-post[1:],0).sum()
   reserve=max(histbuf[k]+q[k,0],future_need+q[k,0])
   reduction=max(0,int(post[0]-reserve-cfg.guard))
   lower=max(lower,b-reduction)
  ix=m.var(c=(cfg.prodcost+cfg.holding)*price[k],lo=lower,hi=b+extra,integer=1);pvars[k]=ix;effect[1][k,0][ix]=1
  # deviation costs: reduction from proposed order and increase beyond proposal treated separately
  if b:
   d=m.var(c=cfg.cutcost*price[k],hi=b);m.row({d:1,ix:1},lo=b)
  if extra:
   d=m.var(c=.7*price[k],hi=extra);m.row({d:1,ix:-1},lo=-b)
 if pvars:m.row({ix:1 for ix in pvars.values()},hi=max(cfg.cap,int(planned.sum()))-int(planned[:,1:].sum()))
 # Provisional late allocation of factory stock/production to bases. Separate route activations and minimum loads.
 for j in range(1,L):
  yy={}
  for k in np.flatnonzero((q[:,j]>pl[:,j]) & ((pl[:,0]>0)|np.isin(np.arange(K),list(pvars)))):
   if cfg.protect and protected[k]:continue
   ub=int(min(q[k,j]-pl[k,j],pl[k,0]+(m.hi[pvars[k]] if k in pvars else 0)))
   if ub<=0:continue
   iy=m.var(c=(cfg.movecost+.18)*price[k],hi=ub,integer=1);yvars[k,j]=iy;yy[iy]=1;effect[1][k,0][iy]=-1;effect[1][k,j][iy]=1
  if yy:
   zz=m.var(c=cfg.fixedcost,hi=1,integer=1);m.row({**yy,zz:-sum(m.hi[i] for i in yy)},hi=0);m.row({**yy,zz:-cfg.minimum},lo=0)
 # Terminal allocations must depart by H-3; factory stock that arrives later cannot fund them.
 for k in set(k for k,j in yvars):
  terms={iy:1 for (kk,j),iy in yvars.items() if kk==k}
  for (kk,o,j),ix in xvars.items():
   if kk!=k:continue
   if o==0:terms[ix]=1
   elif j==0:terms[ix]=-1
  if k in pvars:terms[pvars[k]]=-1
  m.row(terms,hi=int(inv[k,0]+pipe[:H-3,k,0].sum()+wip[:H-3,k,0].sum()))
 # Expected fulfilled units at 5 and lead+7 days. Increasing unit positions receive decreasing survival weights.
 # These checkpoints overlap: they approximate service value, not a stochastic multi-period recourse model.
 for h,(mean,pos,weight) in enumerate([(me,pe,cfg.fieldweight),(ml,pl,cfg.fieldweight)]):
  tails=nbinom.sf(np.arange(35)[None,None,:],1.25,1.25/(1.25+mean[:,:,None]))
  for k,j in set(effect[h])|set(zip(*np.where(mean>.035))):
   term={ix:-sgn for ix,sgn in effect[h][k,j].items()}; const=int(pos[k,j]);
   # terminal factory net balance must not be negative, even when marginal values vanish
   if term:m.row(term,hi=const)
   # fixed stock not affected by any decision is a constant and can be omitted from objective.
   if not term:continue
   for pr in tails[k,j]:
    valw=weight*pr*(cfg.extweight if j==0 or protected[k] else 1.)
    if valw<.015:break
    si=m.var(c=-valw,hi=1);term[si]=1
   m.row(term,hi=const)
 x,stat=m.solve(cfg.limit)
 if x is None:
  # Safe fallback to feasible historical dispatches and proposed production; logged separately.
  return base,planned.copy(),stat
 ship=np.zeros_like(proposed); prod=planned.copy();prod[:,0]=0
 for (k,o,j),ix in xvars.items():ship[k,o,j]=round(x[ix])
 for k,ix in pvars.items():prod[k,0]=round(x[ix])
 assert (ship.sum(2)<=inv).all(); assert prod.sum()<=max(cfg.cap,int(planned.sum()))
 return ship,prod,stat

def replay(a,b,period,cfg,level=0,seed=713,log=False):
 n=b-a;extra=hidden(a,b,level,seed); inv=pre[a].copy().astype(int);pipe=np.zeros((n+cfg.lead+30,K,L),int);wip=np.zeros_like(pipe);wip[:cfg.lead]=out[a:a+cfg.lead]; stats=[]; daily=[]; actions=[];tot=np.zeros(12); start=time.perf_counter()
 for i,t in enumerate(range(a,b)):
  before=inv.sum()+pipe[i:].sum()+wip[i:].sum();inv+=pipe[i]+wip[i];pipe[i]=0;wip[i]=0; positive=np.maximum(res[t],0)+np.maximum(-net[t],0);inv+=positive;ext=np.maximum(-res[t],0);served=np.minimum(inv,ext);inv-=served;rr=np.minimum(inv,sales[t]);inv-=rr;hh=np.minimum(inv,extra[i]);inv-=hh
  planned=out[t+cfg.lead].copy() if t+cfg.lead<T else np.zeros((K,L),int)
  # identical proxy plan and opening WIP for baseline and MIP; final lead days unknown beyond extract, set to zero for both.
  if cfg.name=='LSC':
   ship=np.zeros_like(moves[t]);tmp=inv.copy()
   for o in range(L):
    for j in range(L):
     if o!=j:ship[:,o,j]=np.minimum(tmp[:,o],moves[t,:,o,j]);tmp[:,o]-=ship[:,o,j]
   prod=planned
  else:
   if cfg.forecast=='censor':
    if cp is None:raise RuntimeError('censor_pred.npz missing')
    mu=.7*cp[period+'_censor'][i]+.3*pred[period+'_ops'][i]
   elif cfg.forecast=='history':mu=(cs[t]-cs[t-365])*21/365
   else:mu=pred[period+'_'+cfg.forecast][i].copy()
   ship,prod,st=optimize(inv,pipe[i+1:],wip[i+1:],moves[t],planned,mu,t,cfg);stats.append(dict(date=str(dates[t]),**st))
  inv-=ship.sum(2); assert (inv>=0).all();
  for o in range(L):
   for j in range(L):
    if o!=j:pipe[i+(3 if o==0 or j==0 else 1+cfg.prep),:,j]+=ship[:,o,j]
  wip[i+cfg.lead]+=prod
  now=inv.sum()+pipe[i+1:].sum()+wip[i+1:].sum();err=now-before-positive.sum()+served.sum()+rr.sum()+hh.sum()-prod.sum(); assert err==0,err
  fg=(inv.sum(1)+pipe[i+1:].sum((0,2)))*v; work=wip[i+1:].sum((0,2))*v; qty=ship.sum(0); f=rr[:,1:].sum()+hh[:,1:].sum();ef=served.sum()+rr[:,0].sum();et=ext.sum()+sales[t,:,0].sum();dem=sales[t,:,1:].sum()+extra[i,:,1:].sum();record=rr[:,1:].sum(); hfill=hh[:,1:].sum()
  daily.append(dict(date=str(dates[t]),field=f,demand=dem,record=record,hiddenfill=hfill,external=ef,external_demand=et,value=fg.sum()+work.sum(),finished=fg.sum(),production=prod.sum(),lateral=qty[1:,1:].sum(),factory=qty[0].sum(),routes=(qty>0).sum(),small=((qty>0)&(qty<cfg.minimum)).sum(),cut=np.maximum(planned-prod,0).sum(),added=np.maximum(prod-planned,0).sum()))
  if log:
   for k,o,j in zip(*np.nonzero(ship+moves[t])):actions.append((str(dates[t]),'ship',str(codes[k]),o,j,int(ship[k,o,j]),int(moves[t,k,o,j])))
   for k,j in zip(*np.nonzero(prod+planned)):actions.append((str(dates[t]),'make',str(codes[k]),j,j,int(prod[k,j]),int(planned[k,j])))
 d=pd.DataFrame(daily);s=pd.DataFrame(stats);tag=f'{period}_{cfg.name}_{level}_{seed}';d.to_csv(P/(tag+'_daily.csv'),index=False)
 if len(s):s.to_csv(P/(tag+'_solves.csv'),index=False)
 if log:pd.DataFrame(actions,columns=['date','action','sku','origin_index','dest_index','quantity','proposal']).to_csv(P/(tag+'_actions.csv'),index=False)
 result=dict(period=period,hidden=level,seed=seed,**asdict(cfg),fill=100*d.field.sum()/d.demand.sum(),recordfill=100*d.record.sum()/sales[a:b,:,1:].sum(),extfill=100*d.external.sum()/d.external_demand.sum(),totalfill=100*(d.external.sum()+d.field.sum())/(d.external_demand.sum()+d.demand.sum()),value=d.value.mean(),finished=d.finished.mean(),made=d.production.sum(),lat=d.lateral.sum(),fac=d.factory.sum(),shipdays=d.routes.sum(),small=d.small.sum(),cut=d.cut.sum(),added=d.added.sum(),fieldserved=d.field.sum(),extserved=d.external.sum(),fielddemand=d.demand.sum(),extdemand=d.external_demand.sum(),solve_count=len(s),fallback=int((s.status!=0).sum()) if len(s) else 0,avgsolve=s.seconds.mean() if len(s) else 0,maxsolve=s.seconds.max() if len(s) else 0,maxvars=s.nvar.max() if len(s) else 0,maxint=s.nint.max() if len(s) else 0,wall=time.perf_counter()-start)
 print({k:result[k] for k in ['period','name','hidden','fill','extfill','value','made','lat','shipdays','cut','added','avgsolve','fallback','wall']},flush=True);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['tune','test','smoke'],default='tune'); args=ap.parse_args(); rows=[]
 if args.phase=='smoke':
  replay(jt,jt+5,'tune',Config(name='smoke'),log=True);raise SystemExit
 if args.phase=='tune':
  configs=[Config(name='LSC'),Config(name='safe',cutcost=2.0,prodcost=2.2,addprod=False),Config(name='balanced',cutcost=1.4,prodcost=2.,addprod=False),Config(name='flex',cutcost=1.4,prodcost=2.,newroutes=True,addprod=True),Config(name='aggressive',cutcost=.7,prodcost=2.,protect=False,addprod=True,newroutes=True)]
  for c in configs:
   rows.append(replay(jt,jm,'tune',c));pd.DataFrame(rows).to_csv(P/'mip_tune.csv',index=False)
 elif args.phase=='test':
  tune=pd.read_csv(P/'mip_tune.csv');base=tune.iloc[0]; g=tune.iloc[1:].copy();g['score']=g.value/base.value+3*np.maximum(0,(base.fill-g.fill)/100)+3*np.maximum(0,(base.extfill-g.extfill)/100)+.03*np.maximum(0,(g.shipdays-base.shipdays)/base.shipdays); win=g.sort_values('score').iloc[0]; kw={k:win[k] for k in Config.__dataclass_fields__};c=Config(**kw);print('selected',c,flush=True)
  for lvl in [0,.1]:
   for cc in [Config(name='LSC'),c,Config(**{**kw,'name':c.name+'_censor','forecast':'censor'})] if cp is not None else [Config(name='LSC'),c]:
    rows.append(replay(jtest,T,'test',cc,level=lvl,log=(lvl==0)));pd.DataFrame(rows).to_csv(P/'mip_test.csv',index=False)
