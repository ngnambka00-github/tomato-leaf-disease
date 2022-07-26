from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from hydra import initialize, compose
from omegaconf import OmegaConf

with initialize(config_path="../config/"):
    data_cfg = compose(config_name="hyper_parameter")
parameter_cfg = OmegaConf.create(data_cfg)

BINS = parameter_cfg.final_variable.hog_bins_feature
N_CLASSES = parameter_cfg.final_variable.n_classes
RANDOM_SEED = parameter_cfg.final_variable.seed
NUM_TREES = parameter_cfg.final_variable.num_trees
MAX_DEPTH = parameter_cfg.final_variable.max_deep

model_classifiers = [
    ("KNN",                             KNeighborsClassifier(n_neighbors=N_CLASSES)),
    ("Decition Tree",                   DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)),
    ("Random Forest",                   RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEPTH, random_state=RANDOM_SEED)),
    ("SVM: {linear}",                   SVC(kernel="linear", C=0.025, random_state=RANDOM_SEED)),
    ("SVM: {gamma=2, C=1}",             SVC(gamma=2, C=1, random_state=RANDOM_SEED)),
    ("SVM: {kernel=rbf}",               SVC(kernel='rbf', random_state=RANDOM_SEED)),
    ("MLP",                             MLPClassifier(alpha=1, max_iter=1000, random_state=RANDOM_SEED)),
    ("Gaussian Naive Baye",             GaussianNB()),
    ("Linear Discriminant Analysis",    LinearDiscriminantAnalysis()),
    ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ("AdaBoost",                        AdaBoostClassifier(random_state=RANDOM_SEED)),
    ("GradientBoosting",                GradientBoostingClassifier(n_estimators=NUM_TREES, learning_rate=1.0, max_depth=MAX_DEPTH)),
    ("XGBoost",                         XGBClassifier(n_estimators=NUM_TREES, random_state=RANDOM_SEED)),
]