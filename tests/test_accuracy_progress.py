from datetime import timedelta

import numpy as np

from app.accuracy_progress import ProgressContext,relative_coordinates,FEATURES
from app.accuracy_transfer import ALL_FEATURES
from test_effective_accuracy import rows


def test_relative_difficulty_depends_on_ability_at_attempt_date():
    history,a,now=rows(6)
    x=np.ones(len(ALL_FEATURES));b={**a,'stars':4.,'mechanical_stars':4.,'aim':2.,'mechanical_aim':2.,'speed':2.,'mechanical_speed':2.}
    for key in ('stars_level','aim_level','speed_level'):x[ALL_FEATURES.index(key)]=1.
    old=relative_coordinates(b,x)
    for key in ('stars_level','aim_level','speed_level'):x[ALL_FEATURES.index(key)]=4.
    current=relative_coordinates(b,x)
    assert np.all(current<old)
    assert current[0]==0 and old[0]>0


def test_progress_coordinates_do_not_change_from_future_or_own_target_scores():
    history,a,now=rows(12)
    ctx=ProgressContext(history)
    original=ctx.input(a,now)
    assert len(original['capability'])==len(FEATURES)
    added={**history[0],'id':'future','accuracy':.1,'played_at':(now+timedelta(days=1)).isoformat()}
    other=ProgressContext(history+[added]).input(a,now)
    assert np.allclose(original['capability'],other['capability'])
