import pandas
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
# import lightgbm as lgb
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

import const.lab4Const as CONST
import time
from datetime import datetime

x_train = None
y_train = None
x_test = None
y_test = None


def time_logger(func):
    """Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        start = time.time()
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" начало:{datetime.now()}')
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" завершение: {datetime.now()}')
        print(f'Выполнение функции "{CONST.func_description[func.__name__]}" длительность: {end - start}')
        return result

    return wrap


@time_logger
def init_datasets(train_file, test_file, train_columns, test_column):
    """Инициализация датасета. Датасет для тренировки и тестирования лежат в разных файлах
    поэтому раздельно их и загружаем"""

    # загрузка тренировочного датасета
    df_train = pandas.read_csv(train_file)

    y_train = df_train[test_column]
    x_train = df_train[train_columns]
    # поскольку результирующие данные не числовые,
    # а мультикатегорируемые то проводим преобразования
    le_disease = LabelEncoder()
    y_train = le_disease.fit_transform(y_train)

    # аналогично для тестового набора
    df_test = pandas.read_csv(test_file)
    y_test = df_test[test_column]
    x_test = df_test[train_columns]
    le_disease = LabelEncoder()
    y_test = le_disease.fit_transform(y_test)
    return x_train, y_train, x_test, y_test


def print_report(y_test, y_pred, method):
    """Поскольку для каждого метода выводим отчет,
        то оформляем в виде отдельной функции"""
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f' Report for {method} method')
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)


def learn_model(classifier, x_train, y_train, x_test, y_test, method):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f' Report for {method} method')
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)


@time_logger
def gradient(x_train, y_train, x_test, y_test, params):
    gb_classifier = GradientBoostingClassifier(n_estimators=params['estimators'], max_depth=params['depth'],
                                               learning_rate=params['learning_rate'], random_state=42)
    learn_model(gb_classifier, x_train, y_train, x_test, y_test, 'GradientBoostingClassifier')



@time_logger
def yandex(x_train, y_train, x_test, y_test, params):
    # Создание и обучение классификатора CatBoost
    clf = CatBoostClassifier(iterations=params['iterations'], learning_rate=params['learning_rate'],
                             depth=params['depth'], random_state=100, verbose=params['verbose'])
    learn_model(clf, x_train, y_train, x_test, y_test, 'CatBoostClassifier')



@time_logger
def ada_boost(x_train, y_train, x_test, y_test, params):
    # Создание и обучение классификатора AdaBoost
    AdaBoostClassifier()
    ada_classifier = AdaBoostClassifier(n_estimators=params['estimators'], learning_rate=params['learning_rate'],
                                        random_state=42, algorithm='SAMME')
    learn_model(ada_classifier, x_train, y_train, x_test, y_test, 'AdaBoostClassifier')



@time_logger
def extra_tree(x_train, y_train, x_test, y_test, params):
    # Создание и обучение классификатора Extra Trees
    clf = ExtraTreesClassifier(n_estimators=params['estimators'], max_features='sqrt', random_state=42, )
    learn_model(clf, x_train, y_train, x_test, y_test, 'ExtraTreesClassifier')


@time_logger
def sqr_boost(x_train, y_train, x_test, y_test, params):
    # Создание и обучение классификатора QDA
    clf = QuadraticDiscriminantAnalysis()
    learn_model(clf, x_train, y_train, x_test, y_test, 'QuadraticDiscriminantAnalysis')


@time_logger
def LGBM_method(x_train, y_train, x_test, y_test, params):
    clf = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
    learn_model(clf, x_train, y_train, x_test, y_test, 'LGBMClassifier')


@time_logger
def KNN_method(x_train, y_train, x_test, y_test, params):
    clf = KNeighborsClassifier(n_neighbors=3)  # Задаем количество соседей (K=3)
    learn_model(clf, x_train, y_train, x_test, y_test, 'KNeighborsClassifier')


@time_logger
def decision_tree_method(x_train, y_train, x_test, y_test, params):
    # Создание и обучение модели Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    learn_model(clf, x_train, y_train, x_test, y_test, 'DecisionTreeClassifier')


@time_logger
def extremally_gradient_method(x_train, y_train, x_test, y_test, params):
    # Создание и обучение модели XGBoost
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    learn_model(clf, x_train, y_train, x_test, y_test, 'XGBClassifier')


@time_logger
def dumn_method(x_train, y_train, x_test, y_test, params):
    # Создание и обучение Dummy Classifier
    clf = DummyClassifier(strategy="most_frequent")  # stratified
    learn_model(clf, x_train, y_train, x_test, y_test, 'DummyClassifier')


@time_logger
def SVM_method(x_train, y_train, x_test, y_test, params):
    # Создание и обучение модели SVM
    clf = SVC(kernel='linear', random_state=42)  # альтернатива rbf, poly
    learn_model(clf, x_train, y_train, x_test, y_test, 'SVC')


if __name__ == '__main__':
    model_params = {}
    data_set = []
    x_train, y_train, x_test, y_test = init_datasets(CONST.train_file, CONST.test_file, CONST.train_cols,
                                                     CONST.calc_fied)
    model_params['estimators'] = 300
    model_params['depth'] = 7
    model_params['learning_rate'] = 0.3
    # gradient(x_train, y_train, x_test, y_test, model_params)
    model_params['iterations'] = 700
    model_params['verbose'] = 1
    yandex(x_train, y_train, x_test, y_test, model_params)
    ada_boost(x_train, y_train, x_test, y_test, model_params)
    extra_tree(x_train, y_train, x_test, y_test, model_params)
    sqr_boost(x_train, y_train, x_test, y_test, model_params)
    # LGBM_method(x_train, y_train, x_test, y_test, model_params)
    decision_tree_method(x_train, y_train, x_test, y_test, model_params)
    extremally_gradient_method(x_train, y_train, x_test, y_test, model_params)
    dumn_method(x_train, y_train, x_test, y_test, model_params)
    SVM_method(x_train, y_train, x_test, y_test, model_params)
#
