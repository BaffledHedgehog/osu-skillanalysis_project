"""Collect API acceptance evidence on a separate copy of the test stand."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from app.main import create_app
from app.analyzer import parse_map

sys.path.insert(0,str(ROOT/'tests'))
from test_app import chart

OUT = ROOT/'.data/test-evidence/2026-11-04/acceptance'


def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    database=OUT/'acceptance.db'
    if not database.exists():
        source=ROOT/'.data/test-stand-2026-11-04/test.db'
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(database) as dst:src.backup(dst)
    app=create_app(database,public_mode=False);store=app.state.store;client=TestClient(app)
    checks=[]
    def record(name,expected,actual,passed):
        checks.append({'name':name,'expected':expected,'actual':actual,'status':'Pass' if passed else 'Fail'})
        save('checks.json',{'captured_at':datetime.now(timezone.utc).isoformat(),'checks':checks})

    for mods in ('NM','DT'):
        response=client.get('/api/maps/4864861/analysis',params={'mods':mods,'user':'BaffledHedgehog'})
        save(f'map-4864861-{mods}.json',response.json())
    nm=json.loads((OUT/'map-4864861-NM.json').read_text(encoding='utf-8'))
    dt=json.loads((OUT/'map-4864861-DT.json').read_text(encoding='utf-8'))
    record('NM / DT','DT изменяет расчет карты',
        {'nm_stars':nm.get('stars'),'dt_stars':dt.get('stars'),'nm_bpm':nm.get('bpm'),'dt_bpm':dt.get('bpm')},
        nm.get('stars')!=dt.get('stars') and dt.get('bpm',0)>nm.get('bpm',0))

    for i in range(12):
        mid=str(920000+i)
        store.save_map(parse_map(chart(int(mid),spacing=275-i*17)))
        for user,low,high in [('QA_Experienced',7,5),('QA_Beginner',3,1)]:
            count=low if i<6 else high
            for j in range(8):
                passed=j<count
                accuracy=(.99-i*.008 if user=='QA_Experienced' else .94-i*.014) if passed else .58
                store.save_play(user,mid,'NM',accuracy,passed,source='manual',play_id=f'{user}-{mid}-{j}')
    store.save_map(parse_map(chart(929999,spacing=190)))
    predictions={}
    for user in ('QA_Experienced','QA_Beginner','QA_Empty'):
        response=client.get('/api/maps/929999/analysis',params={'user':user})
        predictions[user]=response.json()['prediction']
        save(user+'-prediction.json',response.json())
        save(user+'-summary.json',client.get(f'/api/users/{user}/summary').json())
    a,b=predictions['QA_Experienced'],predictions['QA_Beginner']
    valid=lambda p:p['pass_probability'] is not None and 0<=p['pass_probability']<=100 and 0<=p['expected_accuracy']<=100
    record('Две контрольные истории','Одна карта дает разные персональные прогнозы; значения в допустимом диапазоне',
        {u:{k:p.get(k) for k in ('pass_probability','expected_accuracy','model')} for u,p in predictions.items()},
        valid(a) and valid(b) and a['pass_probability']!=b['pass_probability'] and a['expected_accuracy']!=b['expected_accuracy'])
    empty=predictions['QA_Empty']
    record('Пустая история','Числовой прогноз не выдумывается',
        {k:empty.get(k) for k in ('pass_probability','expected_accuracy','observations','message')},
        empty['pass_probability'] is None and empty['expected_accuracy'] is None)

    response=client.post('/api/users/BaffledHedgehog/model/train')
    training=response.json();save('real-profile-training.json',training)
    accuracy=training.get('accuracy',{})
    record('ML accuracy: реальные данные','При trained MAE ниже базового среднего',
        accuracy,response.status_code==200 and accuracy.get('status')=='trained' and accuracy['metrics']['mae_pp']<accuracy['metrics']['baseline_mae_pp'])

    # The real external request uses the public provider, never browser cookies.
    before=store.play_count('BaffledHedgehog')
    response=client.post('/api/users/import',json={'username':'BaffledHedgehog','force':True,'max_scores':100})
    live={'http_status':response.status_code,'body':response.json()};save('live-profile-import.json',live)
    if response.status_code==200:
        canonical=response.json()['username'];first_count=store.play_count(canonical)
        second=client.post('/api/users/import',json={'username':'BaffledHedgehog','force':False,'max_scores':100})
        repeat={'http_status':second.status_code,'body':second.json(),'count_before':before,
                'count_after_first':first_count,'count_after_second':store.play_count(canonical)}
        save('repeat-profile-import.json',repeat)
        record('Живой импорт профиля','Без ключей получены профиль и доступная история',
            {k:response.json().get(k) for k in ('username','provider','received_recent','received_best','recent_unique','best_unique','requested_per_source','imported')},True)
        record('Повтор импорта','Кеш; число попыток не увеличивается',repeat,
            second.status_code==200 and second.json().get('cached') and second.json().get('imported')==0 and store.play_count(canonical)==first_count)
    else:
        checks.append({'name':'Живой импорт профиля','status':'Blocked','actual':live,'expected':'Источник доступен'})
        save('checks.json',{'captured_at':datetime.now(timezone.utc).isoformat(),'checks':checks})
    print(json.dumps({'checks':[{k:c[k] for k in ('name','status')} for c in checks]},ensure_ascii=True))


if __name__=='__main__':main()
