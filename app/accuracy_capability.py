"""Experimental causal accuracy references at comparable motor load and OD.

Timing references are approximate features, not measured UR or forced accuracy.
They assume mostly 100 judgements among non-miss errors, and are consumed only
by the offline experiment until its acceptance checks succeed.
"""
import math
from statistics import NormalDist

import numpy as np

from .accuracy_transfer import TransferContext,effective_analysis,visibility_class
from .accuracy_context import song
from .recency import sample_weights


FEATURES = ('ordinary_near_median','ordinary_near_q85','best_near_median','best_near_q85',
            'ordinary_near_log_weight','best_near_log_weight','precision_at_od_median',
            'precision_at_od_q85','precision_log_weight','precision_available',
            'comfortable_acc_q85','comfortable_log_weight','aim_load_ratio','speed_load_ratio')


def timing_accuracy_at_od(accuracy,source_od,target_od,object_count):
    """Approximate timing-only transfer between effective OD windows.

    No-miss results are selected by the caller. Accuracy alone cannot identify
    100/50/slider judgements, hence this is only a model feature.
    """
    values=(accuracy,source_od,target_od,object_count)
    if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite timing reference')
    if not .85<=accuracy<=1 or object_count<2:raise ValueError('Insufficient timing reference')
    # Finite-sample smoothing prevents one short SS from implying perfect timing.
    error_probability=max(1.5*(1-accuracy),.5/object_count)
    z=NormalDist().inv_cdf(1-min(.99,error_probability)/2)
    sigma=max(1.,80-6*source_od)/z
    probability=math.erfc(max(1.,80-6*target_od)/(math.sqrt(2)*sigma))
    return max(0.,min(1.,1-(2/3)*probability))


def quantile(values,weights,q):
    order=np.argsort(values,kind='stable');w=weights[order]
    index=min(len(order)-1,int(np.searchsorted(w.cumsum(),q*w.sum())))
    return float(values[order[index]])


class CapabilityContext(TransferContext):
    feature_names=FEATURES

    def input(self,target,now=None,allowed=None):
        result=super().input(target,now,allowed)
        if result is None:return None
        from datetime import datetime,timezone
        now=now or datetime.now(timezone.utc)
        a=effective_analysis(target)
        group=self.families.get(a['id']) if self.families else '|'.join(song(a))
        eligible=self.times<now.timestamp()
        eligible &= np.asarray([p['map_id']!=a['id'] and p['analysis'].get('checksum')!=a.get('checksum') for p in self.rows])
        if group is not None and str(group):eligible &= self.groups!=group
        if allowed is not None:eligible &= np.isin(self.groups,list(allowed))
        target_class=visibility_class(a['mods'].replace('HD','') or 'NM')
        eligible &= np.asarray([visibility_class(p['mods'].replace('HD','') or 'NM')==target_class for p in self.rows])
        ids=np.flatnonzero(eligible)
        if not len(ids):return {**result,'capability':[result['neighbor']]*4+[0,0,result['neighbor'],result['neighbor'],0,0,result['neighbor'],0,1,1]}
        h=[self.rows[i] for i in ids]
        w=np.asarray(sample_weights(h,now=now))
        # Repeats of a chart cannot manufacture a precise reference distribution.
        keys=[p['analysis'].get('checksum') or p['map_id'] for p in h]
        from collections import Counter
        counts=Counter(keys);w/=np.asarray([counts[k] for k in keys])
        acc=np.asarray([p['accuracy'] for p in h]);curated=self.curated[ids]
        aim=np.asarray([self.maps[i]['aim'] for i in ids]);speed=np.asarray([self.maps[i]['speed'] for i in ids])
        od=np.asarray([self.maps[i]['od'] for i in ids]);ar=np.asarray([self.maps[i]['ar'] for i in ids])
        distance=((aim-a['aim'])/.65)**2+((speed-a['speed'])/.65)**2+((od-a['od'])/1.5)**2+((ar-a['ar'])/1.5)**2
        def near_stats(mask):
            ix=np.flatnonzero(mask);ix=ix[np.argsort(distance[ix],kind='stable')[:30]]
            if not len(ix):return result['neighbor'],result['neighbor'],0.
            ww=w[ix]*np.exp(-.5*np.minimum(distance[ix],40))
            return quantile(acc[ix],ww,.5),quantile(acc[ix],ww,.85),math.log1p(float(ww.sum()))
        ordinary=near_stats(~curated);best=near_stats(curated)
        # References at comparable or greater motor load provide evidence of
        # precision. Their timing transfer is not treated as an output ceiling.
        compatible=(aim>=a['aim']*.8)&(speed>=a['speed']*.8)
        precise=np.asarray([p.get('misses',None)==0 and p['accuracy']>=.9 for p in h])&compatible
        ix=np.flatnonzero(precise)
        ix=ix[np.argsort(distance[ix],kind='stable')[:30]]
        if len(ix):
            timing=np.asarray([timing_accuracy_at_od(acc[i],od[i],a['od'],h[i]['analysis']['object_count']) for i in ix])
            ww=w[ix]*np.exp(-.25*np.minimum(distance[ix],40))
            timing_stats=[quantile(timing,ww,.5),quantile(timing,ww,.85),math.log1p(float(ww.sum())),1.]
        else:timing_stats=[result['neighbor'],result['neighbor'],0.,0.]
        if compatible.any():comfortable=[quantile(acc[compatible],w[compatible],.85),math.log1p(float(w[compatible].sum()))]
        else:comfortable=[result['neighbor'],0.]
        levels=[]
        for values in (aim,speed):levels.append(quantile(values,w,.7))
        capability=[ordinary[0],ordinary[1],best[0],best[1],ordinary[2],best[2],*timing_stats,*comfortable,
                    a['aim']/max(.01,levels[0]),a['speed']/max(.01,levels[1])]
        if not np.isfinite(capability).all():raise ValueError('Nonfinite capability features')
        return {**result,'capability':capability}
