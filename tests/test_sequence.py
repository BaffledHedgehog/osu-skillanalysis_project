import numpy as np
import pytest

from app.analyzer import parse_map, analyze_map
from app.sequence import VIEWS, FEATURES, mirror_map, mirror_cursor, mirror_sequence, encode_map, windows
from app.sequence_data import family_groups, split_families, profile_context, validated_mods
from test_app import chart


def slider_chart():
    return parse_map(chart(900).replace(b'64,64,1000,1,0,0:0:0:0:',
        b'64,64,1000,2,0,B|100:50|200:100|200:100|600:-40,2,250'))


@pytest.mark.parametrize('mods', ['NM', 'HR', 'DT', 'HDHR', 'NCCL'])
@pytest.mark.parametrize('view', VIEWS)
def test_four_views_match_full_map_transform_and_preserve_timing(mods, view):
    parsed = slider_chart()
    analysis = analyze_map(parsed, mods)
    sequence = encode_map(parsed, analysis)
    fast = mirror_sequence(sequence, view)
    full = encode_map(parsed, analysis, view)
    assert np.allclose(fast.values, full.values, rtol=1e-5, atol=1e-6)
    assert np.array_equal(sequence.times_ms, fast.times_ms)
    assert np.array_equal(sequence.end_times_ms, fast.end_times_ms)
    assert sequence.source_checksum == full.source_checksum
    assert len(sequence.values) == len(full.values) and fast.values.shape[1] == len(FEATURES)


def test_mirrors_transform_all_anchors_without_clipping_and_forbid_arbitrary_views():
    parsed = slider_chart()
    reflected = mirror_map(parsed, 'xy')
    assert '-88:424' in reflected.content
    assert reflected.metadata['source_checksum'] == parsed.checksum
    assert '1000,2,0' in reflected.content
    assert np.array_equal(mirror_cursor(np.asarray([[600, -40], [256, 192]]), 'xy'), [[-88, 424], [256, 192]])
    assert parsed.metadata.get('augmentation') is None
    with pytest.raises(ValueError): mirror_map(parsed, 'rotate90')
    with pytest.raises(ValueError): mirror_map(reflected, 'x')


def test_windows_cover_last_note_and_mask_padding():
    parsed = slider_chart()
    sequence = encode_map(parsed, analyze_map(parsed))
    data, masks, starts = windows(sequence, size=32, stride=16)
    assert data.shape == (1, len(FEATURES), 32)
    assert masks.sum() == 24 and not masks[0, 24:].any()
    data, masks, starts = windows(sequence, size=8, stride=6)
    assert starts[-1] == 16
    assert np.array_equal(data[-1, :, -1], sequence.values[-1])


def test_same_song_and_versions_stay_in_one_split():
    one = parse_map(chart(1))
    two = parse_map(chart(2, spacing=100))
    groups = family_groups({'1': one, '2': two})
    assert groups['1'] == groups['2']
    all_groups = [groups['1']] + [f'other-{i}' for i in range(20)]
    split = split_families(all_groups)
    assert split == split_families(list(reversed(all_groups)))
    assert set(split.values()) == {'train', 'validation', 'test'}


def test_profile_never_uses_same_time_future_or_held_out_targets():
    a = analyze_map(parse_map(chart(1)))
    base = {'analysis': a, 'mods': 'NM', 'accuracy': .9, 'pass_mask': True, 'accuracy_mask': True}
    rows = [dict(base, id='1', map_id='train', passed=True, played_at='2026-01-01T00:00:00Z'),
            dict(base, id='2', map_id='heldout', passed=False, played_at='2026-01-01T00:00:00Z'),
            dict(base, id='3', map_id='heldout', passed=True, played_at='2026-01-02T00:00:00Z')]
    context = profile_context(rows, {'train-family'}, {'train': 'train-family', 'heldout': 'test-family'})
    assert np.array_equal(context[0], context[1])
    altered = [dict(r) for r in rows]
    altered[1]['passed'] = True
    assert np.array_equal(context, profile_context(altered, {'train-family'}, {'train': 'train-family', 'heldout': 'test-family'}))
    assert context[2, 0] > context[0, 0]


def test_large_nm_sample_cannot_validate_rare_hidden_mods():
    rows=[{'map_id':str(i),'mods':'NM','passed':i>8,'accuracy_mask':i>8} for i in range(40)]
    rows += [{'map_id':str(40+i),'mods':'HD','passed':True,'accuracy_mask':True} for i in range(2)]
    families={str(i):str(i) for i in range(len(rows))}
    support=validated_mods(rows,families,list(range(len(rows))))
    assert support=={'pass':['NM'],'accuracy':['NM']}


def test_forward_context_excludes_current_and_future_attempt_labels():
    from datetime import datetime,timezone
    from app.sequence_forward import forward_rows
    a=analyze_map(parse_map(chart(1)))
    base={'analysis':a,'mods':'NM','source':'manual','passed_known':1,'complete':1,'accuracy':.9,'passed':True}
    rows=[dict(base,id='past',played_at='2026-01-01T00:00:00Z'),
          dict(base,id='new',played_at='2026-01-03T00:00:00Z'),
          dict(base,id='future',played_at='2026-01-04T00:00:00Z')]
    cutoff=datetime(2026,1,2,tzinfo=timezone.utc)
    _,before=forward_rows(rows,cutoff,{'past'},{a['checksum']})
    altered=[dict(p) for p in rows]
    altered[1]['accuracy']=.1;altered[2]['passed']=False
    _,after=forward_rows(altered,cutoff,{'past'},{a['checksum']})
    assert np.array_equal(before[0],after[0])
    assert before[1]!=after[1]  # The previous attempt is observable on the next day.
