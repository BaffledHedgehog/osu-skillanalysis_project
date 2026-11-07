import numpy as np

from app.analyzer import analyze_map,parse_map
from app.farmability import descriptors,FEATURES
from test_app import chart


def generated(gaps, points=None):
    base=chart(990).decode().split('[HitObjects]')[0]
    positions=points or [(100+(i%2)*200,192) for i in range(len(gaps)+1)]
    times=[1000]
    for gap in gaps:times.append(times[-1]+gap)
    return (base+'[HitObjects]\n'+'\n'.join(f'{x},{y},{t},1,0,0:0:0:0:'
            for (x,y),t in zip(positions,times))).encode()


def extract(raw,mods='NM'):
    p=parse_map(raw)
    return descriptors(p.content,analyze_map(p,mods))


def test_repeated_motifs_and_irregular_rhythms_are_distinguished():
    regular=extract(generated([250]*80))
    rng=np.random.default_rng(42)
    irregular=extract(generated(rng.choice([83,125,167,250,333,375,500,750],80)))
    assert regular['motif_repeat_5']>irregular['motif_repeat_5']
    assert regular['rhythm_longest_run']>irregular['rhythm_longest_run']
    assert regular['rhythm_local_entropy_mean']<irregular['rhythm_local_entropy_mean']


def test_ar_density_and_reaction_pressure_move_in_opposite_directions():
    raw=generated([125]*80)
    low=extract(raw.replace(b'ApproachRate: 9',b'ApproachRate: 5'))
    high=extract(raw.replace(b'ApproachRate: 9',b'ApproachRate: 10'))
    assert low['reading_density_squared_log']>high['reading_density_squared_log']
    assert low['reaction_pressure']<high['reaction_pressure']
    assert low['motif_repeat_5']==high['motif_repeat_5']


def test_motifs_are_unchanged_by_reflections_and_speed_mods():
    points=[(80+(i%3)*150,70+(i%4)*70) for i in range(81)]
    raw=generated([125,250,250,125]*20,points)
    nm=extract(raw)
    keys=('rhythm_repeat_3','motif_repeat_3','motif_repeat_5','spacing_entropy','angle_entropy')
    for xx,yy in [(True,False),(False,True),(True,True)]:
        mirrored=generated([125,250,250,125]*20,[(512-x if xx else x,384-y if yy else y) for x,y in points])
        assert all(extract(mirrored)[k]==nm[k] for k in keys)
    dt=extract(raw,'DT')
    assert all(dt[k]==nm[k] for k in keys)
    assert dt['reaction_pressure']>nm['reaction_pressure']


def test_empty_or_short_pattern_has_finite_features():
    for x in (extract(generated([250])),descriptors('[HitObjects]\n',{})):
        assert list(x)==list(FEATURES)
        assert np.isfinite(list(x.values())).all()
        assert x['motif_repeat_5']==0
