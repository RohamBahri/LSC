import numpy as np,pandas as pd,itertools
from scipy.stats import nbinom
from pathlib import Path
P=Path(__file__).parent; z=np.load(P/'data.npz');sales=z['sales'];net=z['net'];pre=z['pre'];out=z['output'];moves=z['moves'];res=z['resid'];v=z['val'];T,K,L=sales.shape;jt,jm,jtest=[int(z[k]) for k in ['jt','jm','jtest']];pred=np.load(P/'pred.npz'); Y=sales.copy();Y[:,:,0]+=np.maximum(-res[:,:,0],0);cs=np.concatenate([np.zeros((1,K,L)),np.cumsum(Y,axis=0)]);exp=np.concatenate([np.zeros((1,K,L)),np.cumsum(pre>0,axis=0)]);
# hidden requests, weighted by trailing demand at same cell and elsewhere, only historical zero opening, same draws across policies
rng=np.random.default_rng(713)
def hidden(a,b,level,seed=713):
 rr=np.random.default_rng(seed); n=round(sales[a:b,:,1:].sum()*level);
 if n==0:return np.zeros((b-a,K,L),int)
 w=np.zeros((b-a,K,L-1))
 for i,t in enumerate(range(a,b)):
  h=cs[t]-cs[t-365];w[i]=(pre[t,:,1:]==0)*(h[:,1:]+.10*h[:,1:].sum(1)[:,None]/(L-1))*(h[:,1:].sum(1)[:,None]>0)
 h=np.zeros((b-a,K,L),int)
 if n:
  indices=rr.choice(w.size,n,replace=True,p=w.ravel()/w.sum()); i,k,l=np.unravel_index(indices,w.shape);np.add.at(h,(i,k,l+1),1)
 return h
# target distribution: calibrated mean forecast + independent large-order indicator to impose monotonic upper tails if desired.
def targets(a,b,period,model,q,lead,tailboost):
 ans=[]; means=[]
 for i,t in enumerate(range(a,b)):
  hist=(cs[t]-cs[t-365])/365*21
  if model=='history': mu=hist
  elif model=='blend':mu=.65*pred[period+'_ops'][i]+.35*hist
  elif model=='blendcorrect':mu=.65*pred[period+'_correct'][i]+.35*hist
  else:mu=pred[period+'_'+model][i].copy()
  H=lead+3+3; m=np.maximum(mu*H/21,1e-7); r=1.25; p=r/(r+m); tar=nbinom.ppf(q,r,p).astype(int)
  if tailboost and model!='history':
   tp=pred[period+'_tail'][i];tar=np.maximum(tar,np.where(tp>1-q,5,0))
  # warehouse external reserve forecast over full lead; no speculative creation for no-demand product
  tar=np.minimum(tar,30);ans.append(tar);means.append(m)
 return np.array(ans),np.array(means)
# shipment minimum per origin-destination-day, measured in total PCP units. Independent production capacity 30/day assumed, adjustable.
# Route selection greedy using common state: transfer only if recipient needs and donor has extra; then production orders net all covered requirements.
def run(a,b,period,model='history',q=.90,lead=14,minimum=2,prep=1,lateral=True,tailboost=False,level=0,baseline=False,cap=30,seed=713):
 extra=hidden(a,b,level,seed); n=b-a; dem=sales[a:b]+extra; inv=pre[a].copy().astype(int); pipe=np.zeros((n+lead+12,K,L),int); wip=np.zeros((n+lead+12,K),int); # no preperiod pipelines known, same opening except factory WIP not known
 targets0,means=targets(a,b,period,model,q,lead,tailboost) if not baseline else (None,None)
 fieldfill=recordfill=hiddenfill=extfill=exttotal=made=lat=fac=shipcount=small=lost=0; stocks=[]; finished=[]; endfield=[]; conservation=0
 for i,t in enumerate(range(a,b)):
  before=inv.sum()+pipe[i:].sum()+wip[i:].sum(); inv+=pipe[i];pipe[i]=0; inv[:,0]+=wip[i];wip[i]=0
  # exogenous net receipts/returns and external withdrawals: preserve explicit obligations and track them separately
  positive=np.maximum(res[t],0)+np.maximum(-net[t],0); inv+=positive; ext=np.maximum(-res[t],0); served_ext=np.minimum(inv,ext); inv-=served_ext;extfill+=served_ext.sum();exttotal+=ext.sum()
  if baseline:
   inv+=out[t];made+=out[t].sum(); new=out[t].sum()
  else:new=0
  # immediate service, recorded before added requests; same convention for all policies
  rr=np.minimum(inv,sales[t]);inv-=rr;hh=np.minimum(inv,extra[i]);inv-=hh; recordfill+=rr[:,1:].sum();hiddenfill+=hh[:,1:].sum();fieldfill+=(rr[:,1:]+hh[:,1:]).sum();extfill+=rr[:,0].sum();exttotal+=sales[t,:,0].sum()
  # distribution decisions after demand, common transit times. baseline historical dispatches are realized action requests, not forecasts.
  def dispatch(i0,j,qty):
   nonlocal lat,fac,shipcount,small
   qty=np.minimum(inv[:,i0],qty); total=qty.sum()
   if total:
    inv[:,i0]-=qty;delay=3 if i0==0 or j==0 else 1+prep;pipe[i+delay,:,j]+=qty;shipcount+=1;small+=int(total<minimum)
    if i0==0:fac+=total
    elif j!=0:lat+=total
   return qty
  if baseline:
   for o in range(L):
    for j in range(L):
     if o!=j and moves[t,:,o,j].any():dispatch(o,j,moves[t,:,o,j])
  elif i%3==0:
   tar=targets0[i]; mu=means[i]; incoming=pipe[i+1:].sum(0); pos=inv+incoming; deficit=np.maximum(tar-pos,0); deficit[:,0]=0
   # route-level consolidation across products; only move if target recipient deficit, donor above target + one reserve
   if lateral:
    for _ in range(L*2):
     best=None;bestscore=0
     for o in range(1,L):
      surplus=np.minimum(inv[:,o],np.maximum(pos[:,o]-tar[:,o]-1,0));
      if not surplus.any():continue
      for j in range(1,L):
       if o==j:continue
       qty=np.minimum(surplus,deficit[:,j]); size=qty.sum()
       if size<minimum:continue
       # prioritize recipients with near-term demand and economically meaningful avoided manufacturing
       score=np.sum(qty*(1+mu[:,j]/(1+tar[:,j])))-.4*size
       if score>bestscore:bestscore=score;best=(o,j,qty)
     if best is None:break
     o,j,qty=best; qty=dispatch(o,j,qty);pos[:,o]-=qty;pos[:,j]+=qty;deficit[:,j]-=qty
   # allocate factory stock over all bases, protect forecast external commitments at factory
   for _ in range(L-1):
    avail=np.maximum(inv[:,0]-tar[:,0],0);best=None;bestscore=0
    for j in range(1,L):
     qty=np.minimum(avail,deficit[:,j]);size=qty.sum()
     if size<minimum:continue
     score=np.sum(qty*(1+mu[:,j]/(1+tar[:,j])))
     if score>bestscore:bestscore=score;best=(j,qty)
    if best is None:break
    j,qty=best;qty=dispatch(0,j,qty);pos[:,j]+=qty;deficit[:,j]-=qty
   # production replaces only remaining net deficits after reassignment, subtract unallocated available factory and WIP
   factorypos=inv[:,0]+incoming[:,0]+wip[i+1:].sum(0);need=np.maximum(deficit.sum(1)+tar[:,0]-factorypos,0)
   # prioritize expected demand per unit of desired stock; total shared 3-day manufacturing capacity
   priority=mu.sum(1)/(1+pos.sum(1)); order=np.argsort(-priority); qty=np.zeros(K,int);rem=cap*3
   for k in order:
    if not rem:break
    qk=min(need[k],rem);qty[k]=qk;rem-=qk
   wip[i+lead]+=qty;made+=qty.sum();new=qty.sum()
  now=inv.sum()+pipe[i+1:].sum()+wip[i+1:].sum(); conservation=max(conservation,abs(now-before-positive.sum()+served_ext.sum()+rr.sum()+hh.sum()-new))
  f=(inv.sum(1)+pipe[i+1:].sum((0,2)));finished.append((f*v).sum());stocks.append(((f+wip[i+1:].sum(0))*v).sum());endfield.append((inv[:,1:]*v[:,None]).sum())
 return dict(period=period,model='LSC_replay' if baseline else model,q=q,lead=lead,minship=minimum,prep=prep,lateral=lateral,tail=tailboost,hidden=level,seed=seed,cap=cap,fill=100*fieldfill/dem[:,:,1:].sum(),recordfill=100*recordfill/sales[a:b,:,1:].sum(),hiddenfill=100*hiddenfill/max(1,extra.sum()),extfill=100*extfill/max(1,exttotal),value=np.mean(stocks),finished=np.mean(finished),fieldvalue=np.mean(endfield),made=made,lateral_units=lat,factory_units=fac,shipdays=shipcount,small_shipdays=small,conservation=conservation,demand=dem[:,:,1:].sum(),external=exttotal)
if __name__=='__main__':
 rows=[]
 for model,q,minimum in itertools.product(['history','ops','correct','blendcorrect'],[.90,.95,.98],[2]):
  r=run(jt,jm,'tune',model,q,minimum=minimum,tailboost=model!='history');rows.append(r);pd.DataFrame(rows).to_csv(P/'tuning_partial.csv',index=False);print(model,q,minimum,r['fill'],r['value'],flush=True)
 rows.append(run(jt,jm,'tune',baseline=True));df=pd.DataFrame(rows);df.to_csv(P/'tuning.csv',index=False);print(df.sort_values('fill',ascending=False).head(12).to_string(index=False),flush=True)
 # choose per model nearest benchmark inventory in tuning (not test), for each minship. also retain service frontier by evaluating all q for test only diagnostic
 base=df[df.model=='LSC_replay'].iloc[0];chosen=[]
 for (m,mi),g in df[df.model!='LSC_replay'].groupby(['model','minship']):
  ix=(g.value-base.value).abs().idxmin();chosen.append(df.loc[ix])
 rows=[]
 for lvl in [0,.10,.20]:
  rows.append(run(jtest,T,'test',level=lvl,baseline=True))
  for c in chosen:
   rows.append(run(jtest,T,'test',c.model,c.q,minimum=int(c.minship),tailboost=bool(c['tail']),level=lvl));pd.DataFrame(rows).to_csv(P/'test_partial.csv',index=False)
 df2=pd.DataFrame(rows);df2.to_csv(P/'test.csv',index=False); print(df2[['model','q','minship','hidden','fill','extfill','value','made','lateral_units','shipdays','conservation']].to_string(index=False),flush=True)
 # stress test selected blended correction at different lead/minship and no redistribution; same q tuned
 c=[c for c in chosen if c.model=='blendcorrect' and c.minship==2][0];rows=[]
 for lead,mi,prep,lat in [(7,2,1,True),(14,1,0,True),(14,2,1,True),(14,4,1,True),(14,6,1,True),(21,2,1,True),(14,2,3,True),(14,2,1,False)]:
  rows.append(run(jtest,T,'test',c.model,c.q,lead=lead,minimum=mi,prep=prep,lateral=lat,tailboost=True,level=.10))
 pd.DataFrame(rows).to_csv(P/'stress.csv',index=False)
