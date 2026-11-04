"""Causal completed-pass references for accuracy regression, shared by fit and inference."""
from datetime import datetime,timezone
import hashlib
import math

import numpy as np

from .ml import FEATURES,SIMILARITY_SCALES,BEST_SOURCES,vector
from .recency import played_date,sample_weights

PROFILE_KEYS=('mean_accuracy','neighbor_accuracy','curated_accuracy','ordinary_log_count','curated_log_count',
              'exact_mod_log_count','neighbor_effective_count','stars_level','aim_level','speed_level','ar_level',
              'stream_level','burst_level','rhythm_level','peak_level','recent_accuracy_delta')
DELTA_KEYS=('stars','aim','speed','ar','stream_bpm','burst_bpm','rhythm_variation','peak_load')
CONTEXT_FEATURES=(*FEATURES,*PROFILE_KEYS,*(f'relative_{k}' for k in DELTA_KEYS),'curated_target')
SCHEMA='completed-accuracy-context-v2'


def song(a):
    return tuple(''.join(c for c in a.get(k,'').casefold() if c.isalnum()) for k in ('artist','title'))


def prior_history(records,target,now=None,families=None,allowed=None):
    now=now or datetime.now(timezone.utc)
    target_family=families.get(target['id']) if families else None
    target_song=song(target)
    result=[]
    for p in records:
        a=p.get('analysis');date=played_date(p)
        if a is None or date is None or date>=now or 'NF' in p['mods']:continue
        if not p.get('complete',1) or not p.get('passed_known',1) or not p['passed']:continue
        if p['map_id']==target['id'] or a.get('checksum')==target.get('checksum'):continue
        if families:
            if families[p['map_id']]==target_family or (allowed is not None and families[p['map_id']] not in allowed):continue
        elif all(target_song) and song(a)==target_song:continue
        result.append(p)
    ordinary=sorted((p for p in result if p['source'] not in BEST_SOURCES),key=lambda p:(played_date(p),p['id']),reverse=True)[:100]
    curated=sorted((p for p in result if p['source'] in BEST_SOURCES),key=lambda p:hashlib.sha256(p['id'].encode()).digest())[:30]
    return [*ordinary,*curated]


def accuracy_input(records,target,now=None,families=None,allowed=None,curated_target=False):
    now=now or datetime.now(timezone.utc)
    h=prior_history(records,target,now,families,allowed)
    if len(h)<3:return None
    weights=np.asarray(sample_weights(h,now=now),dtype=float)
    ordinary=np.asarray([p['source'] not in BEST_SOURCES for p in h],dtype=bool)
    curated=~ordinary
    exact=np.asarray([p['mods']==target['mods'] for p in h],dtype=bool)
    acc=np.asarray([p['accuracy'] for p in h],dtype=float)
    mask=ordinary if ordinary.any() else curated
    mean=float(np.average(acc[mask],weights=weights[mask]))
    curated_mean=float(np.average(acc[curated],weights=weights[curated])) if curated.any() else mean
    x=np.asarray([vector(p['analysis']) for p in h],dtype=float)
    v=np.asarray(vector(target),dtype=float)
    distances=np.square((x-v)/np.asarray(SIMILARITY_SCALES)).sum(axis=1)
    if ordinary.sum()>=5:
        distances[~ordinary]+=16
    distances[~exact]+=3
    ids=np.argsort(distances,kind='stable')[:20]
    neighbor_weights=weights[ids]*np.exp(-.5*np.minimum(distances[ids],60))
    neighbor_weights*=np.where(ordinary[ids],1,.25)
    adjusted=np.clip(acc[ids]-(float(target['stars'])-x[ids,0])*.025,0,1)
    nearest=float(np.average(adjusted,weights=neighbor_weights))
    levels=[]
    for key,q in zip(DELTA_KEYS,(.7,.7,.7,.9,.9,.9,.7,.9)):
        values=np.asarray([p['analysis'].get(key,0) for p in h],dtype=float)
        order=np.argsort(values,kind='stable');w=weights[order]
        levels.append(float(values[order][min(len(h)-1,np.searchsorted(w.cumsum(),q*w.sum()))]))
    recent=np.asarray([(now-played_date(p)).total_seconds()<30*86400 for p in h])&ordinary
    delta=float(np.average(acc[recent],weights=weights[recent])-mean) if recent.any() else 0.
    effective=float(neighbor_weights.sum()**2/np.square(neighbor_weights).sum())
    profile=[mean,nearest,curated_mean,math.log1p(weights[ordinary].sum()),math.log1p(weights[curated].sum()),
             math.log1p(weights[exact].sum()),effective,*levels,delta]
    relative=np.asarray([target.get(k,0) for k in DELTA_KEYS])-np.asarray(levels)
    result=np.concatenate([v,profile,relative,[float(curated_target)]])
    if not np.isfinite(result).all():return None
    return {'x':result,'mean':mean,'neighbor':nearest,'history_ids':[p['id'] for p in h],
            'history_dates':[p['played_at'] for p in h],'profile_stars':levels[0]}
