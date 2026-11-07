from datetime import timedelta
from copy import deepcopy

import numpy as np
import pytest

from app.accuracy_capability import CapabilityContext,timing_accuracy_at_od,FEATURES
from test_effective_accuracy import rows


def test_timing_reference_responds_to_od_and_finite_sample_size():
    assert timing_accuracy_at_od(.96,9,8,1000)>.96
    assert timing_accuracy_at_od(.96,8,9,1000)<.96
    assert timing_accuracy_at_od(1,9,9,100)<timing_accuracy_at_od(1,9,9,1000)<1
    with pytest.raises(ValueError):timing_accuracy_at_od(float('nan'),9,8,1000)


def test_capability_features_are_causal_and_exclude_target_family():
    history,target,now=rows(12)
    for i,p in enumerate(history):p.update(misses=0,accuracy=.94+i*.003)
    ctx=CapabilityContext(history)
    original=ctx.input(target,now)
    assert len(original['capability'])==len(FEATURES)
    poisoned=deepcopy(history[0]);poisoned.update(id='future',accuracy=.1,played_at=(now+timedelta(days=1)).isoformat())
    own=deepcopy(history[0]);own.update(id='own',map_id=target['id'],accuracy=1.,analysis=target)
    altered=CapabilityContext(history+[poisoned,own]).input(target,now)
    assert np.array_equal(original['capability'],altered['capability'])
    empty_allowed=ctx.input(target,now,allowed=set())
    assert empty_allowed is None
