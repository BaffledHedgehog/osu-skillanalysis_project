"""Experimental references matched by skill-relative load at their own date.

An old easy chart is not an easy example if it was demanding for the player
at that time. Historical ability coordinates are built only from earlier data.
"""
from datetime import datetime,timezone
import math

import numpy as np

from .accuracy_transfer import TransferContext,ALL_FEATURES,effective_analysis,visibility_class
from .accuracy_context import song
from .accuracy_capability import quantile
from .recency import played_date,sample_weights


FEATURES=('progress_neighbor_median','progress_neighbor_q85','progress_best_median',
          'progress_best_q85','progress_ordinary_log_weight','progress_best_log_weight',
          'progress_effective_count','progress_reference_age_days_log',
          'progress_match_distance','progress_log_star_ratio','progress_log_aim_ratio',
          'progress_log_speed_ratio')
LEVEL_KEYS=('stars_level','aim_level','speed_level')


def relative_coordinates(a,x):
    a=effective_analysis(a)
    levels=np.asarray([x[ALL_FEATURES.index(k)] for k in LEVEL_KEYS])
    values=np.asarray([a[k] for k in ('stars','aim','speed')])
    return np.log(np.maximum(values,.01)/np.maximum(levels,.01))


class ProgressContext(TransferContext):
    feature_names=FEATURES

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._historical_cache={}

    def historical_coordinates(self,allowed):
        key=None if allowed is None else frozenset(allowed)
        if key not in self._historical_cache:
            result=np.full((len(self.rows),3),np.nan)
            for i,p in enumerate(self.rows):
                # Base context only: no recursive references or future outcomes.
                prior=super().input(p['analysis'],played_date(p),allowed)
                if prior is not None:result[i]=relative_coordinates(p['analysis'],prior['x'])
            self._historical_cache[key]=result
        return self._historical_cache[key]

    def input(self,target,now=None,allowed=None):
        current=super().input(target,now,allowed)
        if current is None:return None
        now=now or datetime.now(timezone.utc);a=effective_analysis(target)
        coordinates=self.historical_coordinates(allowed)
        point=relative_coordinates(a,current['x'])
        group=self.families.get(a['id']) if self.families else '|'.join(song(a))
        eligible=(self.times<now.timestamp())&np.isfinite(coordinates).all(axis=1)
        eligible &= np.asarray([p['map_id']!=a['id'] and p['analysis'].get('checksum')!=a.get('checksum') for p in self.rows])
        if group is not None and str(group):eligible &= self.groups!=group
        if allowed is not None:eligible &= np.isin(self.groups,list(allowed))
        target_class=visibility_class(a['mods'].replace('HD','') or 'NM')
        eligible &= np.asarray([visibility_class(p['mods'].replace('HD','') or 'NM')==target_class for p in self.rows])
        ids=np.flatnonzero(eligible)
        if not len(ids):
            current['capability']=[current['neighbor']]*4+[0,0,0,0,0,*point.tolist()]
            current['progress_neighbor']=current['neighbor']
            return current
        relative=np.square((coordinates[ids]-point)/np.asarray([.25,.25,.25])).sum(axis=1)
        # Timing/reading conditions remain comparable even when motor demands
        # are matched across different stages of the player's progress.
        for key,scale in [('ar',1.5),('od',1.5),('rhythm_entropy',1.5),('slider_fraction',.4)]:
            relative+=np.square((np.asarray([self.maps[i].get(key,0) for i in ids])-a.get(key,0))/scale)
        h=[self.rows[i] for i in ids];weights=np.asarray(sample_weights(h,now=now))
        acc=np.asarray([p['accuracy'] for p in h]);curated=self.curated[ids]
        def stats(mask):
            ix=np.flatnonzero(mask);ix=ix[np.argsort(relative[ix],kind='stable')[:30]]
            if not len(ix):return [current['neighbor'],current['neighbor'],0.],ix,np.array([])
            w=weights[ix]*np.exp(-.5*np.minimum(relative[ix],60))
            return [quantile(acc[ix],w,.5),quantile(acc[ix],w,.85),math.log1p(float(w.sum()))],ix,w
        ordinary,ix,w=stats(~curated);best,_,_=stats(curated)
        if len(ix):
            effective=float(w.sum()**2/np.square(w).sum())
            age=float(np.average((now.timestamp()-self.times[ids[ix]])/86400,weights=w))
            distance=float(np.average(relative[ix],weights=w))
        else:effective=age=distance=0.
        current['progress_neighbor']=ordinary[0] if len(ix) else best[0]
        current['capability']=[ordinary[0],ordinary[1],best[0],best[1],ordinary[2],best[2],
                               effective,math.log1p(age),distance,*point.tolist()]
        return current
