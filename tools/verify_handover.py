"""Verify a clean installation and SQLite restore without touching player data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / '.data' / 'defense-prep-2026-11-07' / 'clean-install'
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def request(url, data=None, content_type=None):
    payload = data
    if isinstance(data, dict):
        payload = json.dumps(data).encode()
        content_type = 'application/json'
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': content_type} if content_type else {})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def upload(base, mid):
    raw = (ROOT/'deliverables/test-kit/inputs/valid-map.osu').read_bytes()
    raw = re.sub(rb'BeatmapID:\s*\d+', f'BeatmapID: {mid}'.encode(), raw)
    boundary = 'pulsehandover20261107'
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.osu"\r\n'
               'Content-Type: application/octet-stream\r\n\r\n').encode()+raw+f'\r\n--{boundary}--\r\n'.encode()
    return request(base+'/api/maps/upload', payload, 'multipart/form-data; boundary='+boundary)


def start(python, work, database, name):
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env = dict(os.environ, DATABASE_PATH=str(database), PULSE_PUBLIC_MODE='0', PYTHONIOENCODING='utf-8')
    # This fresh copy has no host .env, model artifacts, browser cookies or local history.
    out = (FOLDER/f'{name}.log').open('wb')
    err = (FOLDER/f'{name}-error.log').open('wb')
    process = subprocess.Popen([str(python), '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port)],
                               cwd=work, env=env, stdout=out, stderr=err, creationflags=FLAGS)
    base = f'http://127.0.0.1:{port}'
    for _ in range(100):
        if process.poll() is not None:raise RuntimeError(f'{name}: server exited; see error log')
        try:
            request(base+'/api/health')
            return process,base,(out,err)
        except (urllib.error.URLError,TimeoutError):time.sleep(.2)
    process.terminate();process.wait(timeout=10)
    raise RuntimeError(f'{name}: health timed out')


def stop(process, handles):
    if process.poll() is None:
        process.terminate();process.wait(timeout=10)
    for handle in handles:handle.close()


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    work=FOLDER/'source'
    if work.exists():raise RuntimeError('Fresh install folder already exists; preserve evidence and use a new folder.')
    work.mkdir()
    for name in ('app','static'):
        shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'requirements.txt',work/'requirements.txt')
    venv=FOLDER/'venv'
    subprocess.run([sys.executable,'-m','venv',str(venv)],check=True,creationflags=FLAGS)
    python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    with (FOLDER/'pip-install.log').open('wb') as log:
        subprocess.run([str(python),'-m','pip','install','-r','requirements.txt'],cwd=work,
                       stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=FLAGS)
    with (FOLDER/'installed-packages.txt').open('wb') as log:
        subprocess.run([str(python),'-m','pip','freeze'],stdout=log,check=True,creationflags=FLAGS)
    database=FOLDER/'fresh.db'
    process,base,handles=start(python,work,database,'fresh')
    try:
        assert request(base+'/api/maps')==[]
        for mid in range(900000,900004):
            result=upload(base,mid)
            assert result['maps'][0]['id']==str(mid)
            request(base+'/api/plays',{'user':'Handover_Synthetic_QA','map_id':str(mid),'accuracy':96,'passed':True})
        before=request(base+'/api/maps/900003/analysis?user=Handover_Synthetic_QA')
        assert before['personal_skills']['status']=='available'
        assert before['personal_skills']['maps']==3
        summary=request(base+'/api/users/Handover_Synthetic_QA/summary')
        assert summary['plays']==4
        backup=FOLDER/'backup.db'
        with sqlite3.connect(database) as source,sqlite3.connect(backup) as destination:
            source.backup(destination)
        with sqlite3.connect(backup) as check:assert check.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    finally:stop(process,handles)
    restored=FOLDER/'restored.db';shutil.copy2(backup,restored)
    process,base,handles=start(python,work,restored,'restored')
    try:
        after=request(base+'/api/maps/900003/analysis?user=Handover_Synthetic_QA')
        assert after['personal_skills']==before['personal_skills']
        assert after['aim']==before['aim'] and after['speed']==before['speed']
        assert request(base+'/api/users/Handover_Synthetic_QA/summary')['plays']==4
    finally:stop(process,handles)
    result={'fresh_venv':True,'installed_requirements':True,'new_source_copy_without_private_data':True,
            'health_ok':True,'osu_uploads':4,'synthetic_qa_passes':4,'skill_report_available':True,
            'sqlite_backup_integrity':'ok','restore_health_ok':True,'restore_skill_report_identical':True,
            'player_data_touched':False,'independent_human_run':False,
            'scope':'automated offline install/upload/skill-report and backup/restore; no public API or native replay test'}
    (FOLDER/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
