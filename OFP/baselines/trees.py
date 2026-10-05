"""Construct the tree baselines with explicit caller-supplied settings."""
def build_tree(name,**parameters):
    if name=='random_forest':
        from sklearn.ensemble import RandomForestClassifier as Model
    elif name=='xgboost':
        from xgboost import XGBClassifier as Model
    elif name=='lightgbm':
        from lightgbm import LGBMClassifier as Model
    elif name=='catboost':
        from catboost import CatBoostClassifier as Model
    else:
        raise ValueError('Choose random_forest, xgboost, lightgbm, or catboost')
    return Model(**parameters)
