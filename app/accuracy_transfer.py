"""Effective-map accuracy inputs, causal mixed-age references and reading features."""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .accuracy_context import PROFILE_KEYS,DELTA_KEYS,song
from .accuracy_models import predict_model
from .ml import FEATURES,BEST_SOURCES,SIMILARITY_SCALES,vector
from .recency import played_date,sample_weights

READING=('reading_density','overlap_fraction','overlap_peak','rhythm_entropy',
         'rhythm_switch_rate','offgrid_fraction','tempo_change_rate')
HIDDEN=('hd_memory_window','hd_overlap','hd_density','hd_rhythm','hd_rhythm_switch','hd_offgrid','hd_tempo_changes')
SCHEMA='effective-map-accuracy-v3'
ALL_FEATURES=(*FEATURES,*READING,*HIDDEN,*PROFILE_KEYS,*(f'relative_{k}' for k in DELTA_KEYS))


def effective_analysis(a):
    return {**a,**{k:a.get('mechanical_'+k,a[k]) for k in ('stars','aim','speed')}}


def effective_vector(a):
    a=effective_analysis(a);v=vector(a)
    # Timing and geometry modifiers are already represented by effective values.
    for k in ('hidden','hard_rock','double_time','half_time','easy','flashlight','hidden_low_ar',
              'hidden_ar','hidden_ar_squared','hidden_rhythm'):
        v[FEATURES.index(k)]=0.
    return v


def reading_vector(a):
    r=[float(a.get(k,0)) for k in READING]
    hd=float('HD' in a.get('mods','NM'))
    hidden=[hd*float(a.get('preempt_ms',600))/1000,
            hd*r[1],hd*r[0],hd*r[3],hd*r[4],hd*r[5],hd*r[6]]
    return [*r,*hidden]


def visibility_class(mods):
    return ('HD' in mods,'FL' in mods,'CL' in mods)


class TransferContext:
    def __init__(self,records,families=None):
        self.rows=[p for p in records if p.get('analysis') is not None and played_date(p) is not None
                   and p.get('passed_known',1) and p.get('complete',1) and p['passed']
                   and 'NF' not in p['mods'] and 'FL' not in p['mods']]
        self.maps=[effective_analysis(p['analysis']) for p in self.rows]
        self.x=np.asarray([effective_vector(a) for a in self.maps],dtype=float).reshape(-1,len(FEATURES))
        self.times=np.asarray([played_date(p).timestamp() for p in self.rows])
        self.curated=np.asarray([p['source'] in BEST_SOURCES for p in self.rows],dtype=bool)
        self.families=families
        self.groups=np.asarray([families[p['map_id']] if families else '|'.join(song(a))
                                for p,a in zip(self.rows,self.maps)])

    def input(self,target,now=None,allowed=None):
        if not self.rows:return None
        now=now or datetime.now(timezone.utc);a=effective_analysis(target)
        v=np.asarray(effective_vector(a));t=now.timestamp()
        group=self.families.get(a['id']) if self.families else '|'.join(song(a))
        eligible=self.times<t
        eligible &= np.asarray([p['map_id']!=a['id'] and p['analysis'].get('checksum')!=a.get('checksum') for p in self.rows])
        if group is not None and str(group):eligible &= self.groups!=group
        if allowed is not None:eligible &= np.isin(self.groups,list(allowed))
        ids=np.flatnonzero(eligible)
        if len(ids)<3:return None
        distances=np.square((self.x-v)/np.asarray(SIMILARITY_SCALES)).sum(axis=1)
        # Reading evidence differs by visibility/scoring; HR/DT/HT/EZ are transferable.
        classes=[visibility_class(p['mods']) for p in self.rows]
        # The mechanical player's profile is shared with NM; Hidden is a
        # separate conditional reading head, not a different pool of abilities.
        target_class=visibility_class(a['mods'].replace('HD','') or 'NM')
        distances+=np.asarray([3. if c!=target_class else 0. for c in classes])
        ordinary=ids[~self.curated[ids]];curated=ids[self.curated[ids]]
        recent=ordinary[np.argsort(-self.times[ordinary],kind='stable')[:100]]
        similar=ordinary[np.argsort(distances[ordinary],kind='stable')[:60]]
        selected=sorted(set(recent.tolist()+similar.tolist()))
        selected+=sorted(curated.tolist(),key=lambda i:hashlib.sha256(self.rows[i]['id'].encode()).digest())[:30]
        h=[self.rows[i] for i in selected]
        if len(h)<3:return None
        w=np.asarray(sample_weights(h,now=now));ordinary_mask=~self.curated[selected]
        acc=np.asarray([p['accuracy'] for p in h]);x=self.x[selected]
        baseline=ordinary_mask if ordinary_mask.any() else ~ordinary_mask
        mean=float(np.average(acc[baseline],weights=w[baseline]))
        curated_mean=float(np.average(acc[~ordinary_mask],weights=w[~ordinary_mask])) if (~ordinary_mask).any() else mean
        d=distances[selected].copy()
        if ordinary_mask.sum()>=5:d[~ordinary_mask]+=16
        near=np.argsort(d,kind='stable')[:20]
        nw=w[near]*np.exp(-.5*np.minimum(d[near],60))*np.where(ordinary_mask[near],1,.25)
        adjusted=np.clip(acc[near]-(a['stars']-x[near,0])*.025,0,1)
        neighbor=float(np.average(adjusted,weights=nw))
        levels=[]
        for key,q in zip(DELTA_KEYS,(.7,.7,.7,.9,.9,.9,.7,.9)):
            values=np.asarray([self.maps[i].get(key,0) for i in selected]);order=np.argsort(values,kind='stable')
            levels.append(float(values[order][min(len(h)-1,np.searchsorted(w[order].cumsum(),q*w.sum()))]))
        recent_mask=(t-self.times[selected]<30*86400)&ordinary_mask
        delta=float(np.average(acc[recent_mask],weights=w[recent_mask])-mean) if recent_mask.any() else 0.
        exact=np.asarray([classes[i]==target_class for i in selected])
        profile=[mean,neighbor,curated_mean,math.log1p(w[ordinary_mask].sum()),math.log1p(w[~ordinary_mask].sum()),
                 math.log1p(w[exact].sum()),float(nw.sum()**2/np.square(nw).sum()),*levels,delta]
        relative=np.asarray([a.get(k,0) for k in DELTA_KEYS])-levels
        result=np.concatenate([v,reading_vector(a),profile,relative])
        if not np.isfinite(result).all():return None
        return {'x':result,'mean':mean,'neighbor':neighbor,'profile_stars':levels[0],
                'history_ids':[p['id'] for p in h],'history_dates':[p['played_at'] for p in h],
                'matched_history':int(exact.sum()),'recent_count':len(recent),'similar_count':len(similar)}


def artifact_path(database,user):
    database=Path(database).resolve()
    key=hashlib.sha256(f'{database}:{user.strip().casefold()}'.encode()).hexdigest()[:24]
    return database.parent/'models'/f'{key}-accuracy-v3.json'


def reading_head_vector(x):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x.reshape(1,-1)
    z=x[:,len(FEATURES)+len(READING):len(FEATURES)+len(READING)+len(HIDDEN)].copy()
    # Physical units and shrinkage avoid standardising a handful of HD scores
    # into enormous binary coefficients. No fixed penalty is added for HD.
    z[:,0] **= 2
    z[:,2]=np.log1p(z[:,2])
    z[:,6]=np.log1p(z[:,6])
    return z


def predict_effective(model,x,base,components=False):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x.reshape(1,-1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite accuracy input')
    values=x.copy()
    start=len(FEATURES)+len(READING)
    values[:,start:start+len(HIDDEN)]=0
    mechanical=predict_model(model['base'],values,base)
    correction=reading_head_vector(x)@np.asarray(model['reading_head']['coef'])
    result=np.clip(mechanical+correction,0,1)
    return (result,mechanical,correction) if components else result


@lru_cache(maxsize=8)
def _index(database,user,revision,day):
    from .storage import Store
    store=Store(database)
    with store.batch():rows=store.plays_with_analysis(user)
    return TransferContext(rows)


@lru_cache(maxsize=8)
def _read(path,mtime,size):return json.loads(Path(path).read_text(encoding='utf-8'))


def predict_transfer(store,user,target):
    path=artifact_path(store.path,user)
    try:
        stat=path.stat();m=_read(str(path),stat.st_mtime_ns,stat.st_size)
        if m.get('schema')!=SCHEMA or m.get('features')!=list(ALL_FEATURES) or not m.get('accepted'):return {'status':'candidate'}
        if m['user']!=user.strip().casefold() or Path(m['deployment_database']).resolve()!=store.path.resolve():return {'status':'wrong_player_or_database'}
        if 'FL' in target['mods'] or 'NF' in target['mods']:return {'status':'unsupported_conditions'}
        now=datetime.now(timezone.utc)
        ctx=_index(str(store.path.resolve()),user.strip().casefold(),store.cache_token(user),now.date().isoformat())
        a=ctx.input(target,now)
        if a is None:return {'status':'insufficient_history'}
        if not m['stars_range'][0]<=effective_analysis(target)['stars']<=m['stars_range'][1]:return {'status':'out_of_range'}
        values,mechanical,reading=predict_effective(m['model'],a['x'],a['neighbor'],components=True)
        estimate=float(values[0])
        direct=target['mods'] in m['validated_mods']
        metrics=m['report']['test']['by_mod'].get(target['mods'],m['report']['test'])
        return {'status':'trained' if direct else 'transferred','accuracy':estimate,
                'error_q90':m['accuracy_error_q90'],'algorithm':m['model']['algorithm'],
                'trained_at':m['trained_at'],'metrics':metrics,'validated_mods':m['validated_mods'],
                'validation_metrics':m['report']['test']['by_mod']['NM'],
                'mechanical_transfer':not direct,'reading_features':{k:target.get(k,0) for k in READING},
                'base_accuracy':float(mechanical[0]),'reading_adjustment_pp':float(reading[0]*100),
                'reading_samples':m['model']['reading_head']['samples'],
                'history_maps':len(a['history_ids']),'similar_history':a['similar_count'],
                'protocol':m['report']['protocol'],'prospective_validation':False}
    except (OSError,ValueError,KeyError,TypeError,IndexError):return {'status':'not_trained'}
