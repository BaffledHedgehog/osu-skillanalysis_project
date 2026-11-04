"""Fit and serialize small regressors as plain JSON, without pickle."""
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DEFAULT_HGB_PARAMS={'loss':'absolute_error','max_iter':160,'max_leaf_nodes':7,'min_samples_leaf':15,
                    'learning_rate':.05,'l2_regularization':10,'early_stopping':False,'random_state':20261104}


def fit_model(name,x,y,base,weights,params=None):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);base=np.asarray(base,dtype=float)
    weights=np.asarray(weights,dtype=float);weights*=len(weights)/weights.sum()
    residual='residual' in name
    target=y-base if residual else y
    if name.startswith('ridge'):
        alpha=float(name.rsplit('_',1)[-1])
        scaler=StandardScaler().fit(x,sample_weight=weights)
        estimator=Ridge(alpha=alpha).fit(scaler.transform(x),target,sample_weight=weights)
        model={'algorithm':'ridge','residual':residual,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
               'coef':estimator.coef_.tolist(),'intercept':float(estimator.intercept_)}
    else:
        estimator=HistGradientBoostingRegressor(**params).fit(x,target,sample_weight=weights)
        trees=[]
        for iteration in estimator._predictors:
            for tree in iteration:
                nodes=[]
                for node in tree.nodes:
                    if node['is_categorical']:raise ValueError('Only numeric trees are supported')
                    nodes.append({'value':float(node['value']),'feature':int(node['feature_idx']),
                                  'threshold':float(node['num_threshold']),'leaf':bool(node['is_leaf']),
                                  'left':int(node['left']),'right':int(node['right'])})
                trees.append(nodes)
        model={'algorithm':'hist_gradient_boosting','residual':residual,
               'intercept':float(estimator._baseline_prediction.reshape(-1)[0]),'trees':trees}
    # Verify exported inference against sklearn before retaining any artifact.
    direct=np.clip(estimator.predict(scaler.transform(x))+(base if residual else 0),0,1) if name.startswith('ridge') else np.clip(estimator.predict(x)+(base if residual else 0),0,1)
    exported=predict_model(model,x,base)
    if not np.allclose(direct,exported,rtol=1e-9,atol=1e-10):raise ValueError('JSON regressor differs from sklearn')
    return model


def predict_model(model,x,base):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x.reshape(1,-1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite regression input')
    if model['algorithm']=='ridge':
        prediction=(x-np.asarray(model['mean']))/np.asarray(model['scale'])@np.asarray(model['coef'])+model['intercept']
    else:
        prediction=np.full(len(x),model['intercept'],dtype=float)
        for tree in model['trees']:
            for i,row in enumerate(x):
                j=0
                while not tree[j]['leaf']:
                    n=tree[j];j=n['left'] if row[n['feature']]<=n['threshold'] else n['right']
                prediction[i]+=tree[j]['value']
    if model['residual']:prediction+=np.asarray(base)
    return np.clip(prediction,0,1)
