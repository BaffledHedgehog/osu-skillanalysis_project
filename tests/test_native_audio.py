"""Opt-in native regression check: no map, replay, or audio sample is loaded."""
import json
import os
import subprocess

import pytest


def test_audio_preflight_fails_closed_and_never_loads_a_replay(monkeypatch):
    from pathlib import Path
    from app import replay_health
    calls=[]
    def native_check(command, **kwargs):
        calls.append(command)
        assert command[-2]=='--audio-check'
        # A stale/broken helper that enables audio must be refused.
        Path(command[-1]).write_text(json.dumps({'status':'verified','audio_muted':True,
            'device':'No sound','no_replay_loaded':True,'universal_volume':1,
            'sample_volume':0,'track_volume':0}),encoding='utf-8')
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(replay_health.subprocess,'run',native_check)
    replay_health._verify_silent_engine.cache_clear()
    with pytest.raises(ValueError):replay_health._verify_silent_engine('dotnet','test-helper.dll',-1)
    assert len(calls)==1


@pytest.mark.skipif(os.getenv('PULSE_TEST_NATIVE_AUDIO') != '1', reason='requires installed .NET 10 replay engine')
def test_native_replay_host_is_silent_and_cannot_restore_volume(tmp_path):
    from app.replay_health import _engine
    dotnet, binary = _engine()
    output = tmp_path / 'audio.json'
    process = subprocess.run([dotnet, str(binary), '--audio-check', str(output)],
        cwd=tmp_path, capture_output=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert process.returncode == 0
    result = json.loads(output.read_text(encoding='utf-8-sig'))
    assert result['status'] == 'verified' and result['audio_muted']
    assert result['universal_volume'] == result['sample_volume'] == result['track_volume'] == 0
    assert result['device'] == 'No sound' and result['no_replay_loaded']
