import argparse

from digen import Benchmark

# Load a package with DIGEN benchmark
benchmark = Benchmark()

# seedmap=dict(map(lambda x : (x.split('_')[0],x.split('_')[1]), benchmark.list_datasets()))
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
                    required=False, nargs='?')
args = parser.parse_args()

datasets = args.dataset
if args.dataset is None:
    datasets = benchmark.list_datasets()



# Create your default class here or import from the package. As an example, we re benchmarking ExtraTreesClassifier from scikit-learn:
from sklearn.ensemble import ExtraTreesClassifier

est = ExtraTreesClassifier()


# In order to properly benchmark a method, we need to define its parameters and their values.
# Please set the expected range of hyper parameters for your method below. For details, please refer to Optuna.
def params_myParamScope(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'criterion': trial.suggest_categorical(name='criterion', choices=['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
    }
    return params


# Perform optimization of the method on DIGEN datasets
results = benchmark.optimize(est=est, datasets=datasets, parameter_scopes=params_myParamScope,
                             storage='sqlite:///test.db', local_cache_dir='.')
