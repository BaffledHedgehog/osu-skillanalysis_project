"""Opt-in local kernel calibration from explicitly supplied player feedback.

Feedback is stored separately from score imports and historical benchmarks.
This is a preview estimate, not a newly validated population or accuracy model.
"""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA='player-error-feedback-v1'
FEATURES=('stars','aim','speed','ar','od','cs','density_log','rhythm_entropy',
          'rhythm_switch','tempo_changes_log','peak_ratio_log','slider_fraction','duration_log')


def vector(a):
    x=[float(a.get('mechanical_stars',a['stars']))/1.2,
       float(a.get('mechanical_aim',a['aim'])),float(a.get('mechanical_speed',a['speed']))/.8,
       float(a['ar'])/1.5,float(a['od'])/2,float(a.get('cs',4))/1.5,
       math.log1p(max(0.,float(a.get('reading_density',0)))),float(a.get('rhythm_entropy',0)),
       float(a.get('rhythm_switch_rate',0))/.5,math.log1p(max(0.,float(a.get('tempo_change_rate',0)))),
       math.log1p(max(0.,float(a.get('peak_to_mean',0))))/.5,float(a.get('slider_fraction',0))/.5,
       math.log1p(max(0.,float(a.get('duration',0)))/60)/2]
    if not np.isfinite(x).all():raise ValueError('Nonfinite feedback input')
    return np.asarray(x)


def artifact_path(database,user):
    key=hashlib.sha256(f'{Path(database).resolve()}:{user.strip().casefold()}'.encode()).hexdigest()[:24]
    return Path(database).resolve().parent/'models'/f'{key}-accuracy-feedback.json'


def fit_feedback(database,user,rows):
    if not rows:raise ValueError('Feedback is empty')
    x=[];y=[];metadata=[];seen=set()
    for row in rows:
        a=row['analysis'];mods=a.get('mods','NM')
        if 'HD' in mods or 'FL' in mods or 'NF' in mods:raise ValueError('Initial feedback must have ordinary visibility and a real completed pass')
        key=(a['checksum'],mods)
        if key in seen:raise ValueError('Duplicate feedback chart/mods')
        seen.add(key)
        predicted=float(row['predicted_accuracy']);actual=float(row['actual_accuracy'])
        if not all(math.isfinite(v) and 0<=v<=1 for v in (predicted,actual)):raise ValueError('Invalid accuracy feedback')
        # Learn a multiplicative change in lost accuracy, never a fixed bonus.
        y.append(math.log(max(.0005,1-actual)/max(.0005,1-predicted)))
        x.append(vector(a))
        metadata.append({'map_id':a['id'],'checksum':a['checksum'],'mods':mods,
                         'predicted_accuracy':predicted,'actual_accuracy':actual,
                         'source':row.get('source','manual_feedback'),'score_id':row.get('score_id')})
    return {'schema':SCHEMA,'features':list(FEATURES),'enabled':True,'preview':True,
            'user':user.strip().casefold(),'database':str(Path(database).resolve()),
            'created_at':datetime.now(timezone.utc).isoformat(),'x':np.asarray(x).tolist(),
            'log_error_residuals':y,'observations':metadata,'regularization':.1,
            'prospective_validation':False,'historical_benchmark_used_as_feedback':False}


def infer(model,a,estimate,reading_adjustment=0.,now=None):
    if not model.get('enabled') or model.get('schema')!=SCHEMA or model.get('features')!=list(FEATURES):return None
    if 'NF' in a.get('mods','') or 'FL' in a.get('mods',''):return None
    now=now or datetime.now(timezone.utc)
    created=datetime.fromisoformat(model['created_at'])
    age=max(0.,(now-created).total_seconds()/86400)
    freshness=2**(-age/30)
    if age>90:return None
    regularization=float(model['regularization'])
    if not math.isfinite(regularization) or regularization<=0:
        raise ValueError('Invalid feedback regularization')
    if not math.isfinite(estimate) or not 0<=estimate<=1 or not math.isfinite(reading_adjustment):
        raise ValueError('Invalid base accuracy')
    # CL changes scoring; parameter mods transfer through effective attributes.
    compatible=np.asarray([('CL' in p['mods'])==('CL' in a.get('mods','')) for p in model['observations']])
    x=np.asarray(model['x'],dtype=float)[compatible]
    y=np.asarray(model['log_error_residuals'],dtype=float)[compatible]
    if not len(x):return None
    point=vector(a)
    if x.shape[1]!=len(point) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid feedback model')
    kernel=np.exp(-.5*np.square(x[:,None,:]-x[None,:,:]).sum(axis=2))
    k=np.exp(-.5*np.square(x-point).sum(axis=1))
    coef=np.linalg.solve(kernel+regularization*np.eye(len(x)),y)
    correction=float(k@coef)*freshness
    if not math.isfinite(correction):raise ValueError('Nonfinite feedback correction')
    base=min(1.,max(0.,estimate-reading_adjustment))
    corrected=min(1.,max(0.,1-max(0.,1-base)*math.exp(max(-8.,min(8.,correction)))+reading_adjustment))
    return {'status':'preview_feedback','accuracy':corrected,'base_accuracy':estimate,
            'adjustment_pp':(corrected-estimate)*100,'samples':int(len(x)),
            'max_similarity':float(k.max()),'error_multiplier':math.exp(max(-8.,min(8.,correction))),
            'target_in_feedback':any(p['checksum']==a['checksum'] and p['mods']==a.get('mods','NM') for p in model['observations']),
            'unverified_feedback':any(not p.get('score_id') for p in model['observations']),
            'prospective_validation':False,'method':'kernel_regression_of_log_accuracy_error'}


@lru_cache(maxsize=8)
def _read(path,mtime,size):return json.loads(Path(path).read_text(encoding='utf-8'))


def predict_feedback(store,user,a,estimate,reading_adjustment=0.):
    if estimate is None:return {'status':'no_base_prediction'}
    path=artifact_path(store.path,user)
    try:
        stat=path.stat();model=_read(str(path),stat.st_mtime_ns,stat.st_size)
        if model.get('user')!=user.strip().casefold() or Path(model['database']).resolve()!=store.path.resolve():return {'status':'wrong_player_or_database'}
        return infer(model,a,estimate,reading_adjustment) or {'status':'unsupported_or_expired'}
    except (OSError,ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError):return {'status':'disabled'}
