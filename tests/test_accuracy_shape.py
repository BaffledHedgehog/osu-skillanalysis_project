import numpy as np

from app.accuracy_transfer import ALL_FEATURES
from app.farmability import FEATURES as PATTERNS
from app.farmability_lab import predict_base,shape_features


def test_relative_error_curve_extrapolates_smoothly_and_is_monotone():
    x=np.ones((3,len(ALL_FEATURES)+len(PATTERNS)+12))
    x[:,-2]=[-3.,0.,3.];x[:,-1]=[-3.,0.,3.]
    z=shape_features(x)
    model={'algorithm':'monotone_error_logistic','mean':[0]*z.shape[1],'scale':[1]*z.shape[1],
           'coef':[1,1,0,0,0,0,0],'intercept':-2.}
    p=predict_base(model,x,np.zeros(3))
    assert p[0]>.99 and p[0]>p[1]>p[2]>0
    assert np.isfinite(p).all()
