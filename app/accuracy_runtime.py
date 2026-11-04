"""Read verified per-player accuracy artifacts; never fit from a prediction request."""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib
import json
import sqlite3
from pathlib import Path

from .accuracy_context import accuracy_input,CONTEXT_FEATURES,SCHEMA
from .accuracy_models import predict_model


def artifact_path(database,user):
    database=Path(database).resolve()
    key=hashlib.sha256(f'{database}:{user.strip().casefold()}'.encode()).hexdigest()[:24]
    return database.parent/'models'/f'{key}-accuracy-v2.json'


@lru_cache(maxsize=16)
def _read(path,mtime,size):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def deploy_personal(source,database=None):
    model=json.loads(Path(source).read_text(encoding='utf-8'))
    if model.get('schema')!=SCHEMA or model.get('features')!=list(CONTEXT_FEATURES) or model.get('accepted') is not True:
        raise ValueError('Accuracy candidate has not passed the release checks')
    deployment=Path(database or model['source_database']).resolve()
    if deployment!=Path(model['source_database']).resolve():
        def ids(path):
            with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
                return {r[0] for r in db.execute('SELECT id FROM plays WHERE user_key=?',(model['user'],))}
        original=ids(model['source_database']);destination=ids(deployment)
        overlap=len(original & destination)
        if overlap<200 or overlap/max(1,min(len(original),len(destination)))<.8:
            raise ValueError('The destination does not contain a matching player history')
        model['deployment_database']=str(deployment)
    path=artifact_path(deployment,model['user'])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
    return path


def predict_personal_accuracy(store,user,target,records):
    path=artifact_path(store.path,user)
    try:
        stat=path.stat();m=_read(str(path),stat.st_mtime_ns,stat.st_size)
        if m.get('schema')!=SCHEMA or m.get('features')!=list(CONTEXT_FEATURES):return {'status':'invalid_artifact'}
        if m.get('accepted') is not True:return {'status':'candidate'}
        if m.get('user')!=user.strip().casefold() or Path(m.get('deployment_database',m['source_database'])).resolve()!=store.path.resolve():return {'status':'wrong_player_or_database'}
        if target['mods'] not in m['supported_mods']:return {'status':'insufficient_mod_data'}
        if not m['stars_range'][0]<=target['stars']<=m['stars_range'][1]:return {'status':'out_of_range'}
        a=accuracy_input(records,target,datetime.now(timezone.utc))
        if a is None:return {'status':'insufficient_history'}
        value=a['neighbor'] if m['model']['algorithm']=='neighbors' else float(predict_model(m['model'],a['x'],a['neighbor'])[0])
        report=m['report'];t=report['test']
        # Show evidence for the active mod, rather than the aggregate of other mods.
        t={**t,**t.get('by_mod',{}).get(target['mods'],{}),**report.get('per_mod_ridge_comparison',{}).get(target['mods'],{})}
        return {'status':'trained','accuracy':value,'error_q90':m['accuracy_error_q90'],'algorithm':m['model']['algorithm'],
                'trained_at':m['trained_at'],'metrics':{k:t[k] for k in ('samples','families','mae_pp','ridge_mae_pp','mean_mae_pp')},
                'cutoff':report['cutoff'],'supported_mods':m['supported_mods'],
                'protocol':report['protocol'],'prospective_validation':False}
    except (OSError,ValueError,KeyError,TypeError,IndexError):
        return {'status':'not_trained'}
