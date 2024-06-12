import pandas
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier



from util import time_logger, DLlogger
import const.ds_const as CONST


# сразу определяю дата фреймы как глобальные, чтобы потом не перегружать
df_train = None  # Тренировочный датасет
df_test = None  # Тестовый датасет

relevant_columns = []
feature_columns = []


def load_data(train_file, test_file):
    """Загружаем данные из файлов в дата фреймы"""
    global df_train
    global df_test
    df_train = pandas.read_csv(train_file, header=0)
    df_test = pandas.read_csv(test_file, header=0)
    # Поскольку  Light Gradient Boosting Machine требователен к именам переименовываем атрибуты
    iterator = 0
    for col in df_train.columns:
        if col!='Activity' and col!='subject':
            df_train.rename(columns={col: 'Attr_'+str(iterator)}, inplace=True)
        iterator += 1
    iterator = 0
    for col in df_test.columns:
        if col != 'Activity' and col != 'subject':
            df_test.rename(columns={col: 'Attr_' +str(iterator)}, inplace=True)
        iterator += 1



def check_dataset(data_frame):
    """Вывод статистики по дата фрейму"""

    print('***---- head  ------***')
    print(data_frame.head())

    print('***---- shape  ------***')
    print(data_frame.shape)

    print('***---- типы данных  ------***')
    print(data_frame.dtypes)

    print('***---- нулевые значения  ------***')
    print(data_frame.isnull().sum())


def check_attributes(x_train, y_train):
    """Проверка релевантности всех атрибутов дата сета с использованием регрессии рандомного леса"""
    random_forest = RandomForestRegressor(n_estimators=20)
    random_forest.fit(x_train, y_train)
    # Смотрим что получилось
    feature_importances = random_forest.feature_importances_
    feature_names = x_train.columns
    sorted_idx = np.argsort(feature_importances)
    global relevant_columns
    global feature_columns
    iterator = 0
    for item in sorted_idx[::-1]:
        if iterator < 5:  # Пять наиболее значимых атрибутов сохраняем для фича-инженеринга
            iterator += 1
            feature_columns.append(feature_names[item])
        if round(feature_importances[item], 8) != 0:  # Атрибуты с ненулевыми весами считаем релевантными
            relevant_columns.append(feature_names[item])
        print(f'Атрибут  {feature_names[item]} имеет вес: {feature_importances[item]}')
    print(f'Релевантные атрибуты: {relevant_columns}')
    print('--------------------------------------'
          )
    print(f'Атрибуты для фича-инженеринга {feature_columns}')


def get_attributes(data_frame):
    """загрузка дата сета"""
    # определяем что целевой параметр - это активность
    y = data_frame['Activity']
    # параметры для обучения - все кроме целевой, и номера телефона
    x = data_frame.drop(columns=['Activity', 'subject'])
    # поскольку целевые данные не числовые,
    # а мультикатегорируемые то проводим преобразования
    le_disease = LabelEncoder()
    y = le_disease.fit_transform(y)
    return x, y


@time_logger
def init_datasets():
    """Инициализация дата сета. Дата сет для тренировки и тестирования лежат в разных файлах
    поэтому раздельно их и загружаем"""

    # загрузка тренировочного дата сета
    y_train = df_train['Activity']
    x_train = df_train.drop(columns=['Activity', 'subject'])
    # поскольку результирующие данные не числовые,
    # а мультикатегорируемые то проводим преобразования
    le_disease = LabelEncoder()
    y_train = le_disease.fit_transform(y_train)

    # аналогично для тестового набора
    y_test = df_test['Activity']
    x_test = df_test.drop(columns=['Activity', 'subject'])
    le_disease = LabelEncoder()
    y_test = le_disease.fit_transform(y_test)
    return x_train, y_train, x_test, y_test


def feature_engineering(data_frame):
    """Функция генерации данных фича-инженеринга"""
    # Выбираем самые значимые колонки
    df1 = data_frame[relevant_columns]
    df2 = data_frame[feature_columns]
    df2.iloc[:, 0] **= 9
    df2.iloc[:, 1] **= 7
    df2.iloc[:, 2] **= 5
    df2.iloc[:, 3] **= 3
    df2.iloc[:, 4] = df2.iloc[:, 1] * df2.iloc[:, 2]
    return np.hstack((df1, df2))


def init_separated_dataset():
    """Генерация датасетов с помощью фича-инженеринга"""
    # загрузка тренировочного дата сета
    # генерируем тренировочный набор с учетом фича-инженеринга
    x_train = feature_engineering(df_train)

    # аналогично для тестового набора
    x_test = feature_engineering(df_test)
    return x_train, x_test


def learn_model(classifier, x_train, y_train, x_test, y_test, method):
    """Обобщенный метод для обучения моделей"""
    classifier.fit(x_train, y_train)  # Обучаем модель
    y_pred = classifier.predict(x_test)  # Считаем предсказания
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f' Report for {method} method')
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)


@time_logger
def gradient(x_train, y_train, x_test, y_test, params):
    """Создание и обучение модели методом градиетного бустинга"""
    gb_classifier = GradientBoostingClassifier(n_estimators=params['estimators'], max_depth=params['depth'],
                                               learning_rate=params['learning_rate'], random_state=42)
    learn_model(gb_classifier, x_train, y_train, x_test, y_test, 'GradientBoostingClassifier')


@time_logger
def yandex(x_train, y_train, x_test, y_test, params):
    """Создание и обучение классификатора CatBoost"""
    clf = CatBoostClassifier(iterations=params['iterations'], learning_rate=params['learning_rate'],
                             depth=params['depth'], random_state=100, verbose=params['verbose'],
                             loss_function=params['loss_function'])
    learn_model(clf, x_train, y_train, x_test, y_test, 'CatBoostClassifier')


@time_logger
def ada_boost(x_train, y_train, x_test, y_test, params):
    """Создание и обучение классификатора AdaBoost"""
    AdaBoostClassifier()
    ada_classifier = AdaBoostClassifier(n_estimators=params['estimators'], learning_rate=params['learning_rate'],
                                        random_state=42, algorithm='SAMME')
    learn_model(ada_classifier, x_train, y_train, x_test, y_test, 'AdaBoostClassifier')


@time_logger
def extra_tree(x_train, y_train, x_test, y_test, params):
    """Создание и обучение классификатора Extra Trees"""
    clf = ExtraTreesClassifier(n_estimators=params['estimators'], max_features=params['max_features'],
                               random_state=42, )
    learn_model(clf, x_train, y_train, x_test, y_test, 'ExtraTreesClassifier')


@time_logger
def sqr_boost(x_train, y_train, x_test, y_test):
    """Создание и обучение классификатора QDA"""
    clf = QuadraticDiscriminantAnalysis()
    learn_model(clf, x_train, y_train, x_test, y_test, 'QuadraticDiscriminantAnalysis')


@time_logger
def LGBM_method(x_train, y_train, x_test, y_test, params):
    """Создание и обучение классификатора LGBM"""
    clf = lgb.LGBMClassifier(num_leaves=params['num_leaves'], learning_rate=params['learning_rate'],
                             n_estimators=params['estimators'])
    learn_model(clf, x_train, y_train, x_test, y_test, 'LGBMClassifier')


@time_logger
def KNN_method(x_train, y_train, x_test, y_test, params):
    """Создание и обучение классификатора ближайших к-соседей"""
    clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    learn_model(clf, x_train, y_train, x_test, y_test, 'KNeighborsClassifier')


@time_logger
def decision_tree_method(x_train, y_train, x_test, y_test):
    """Создание и обучение модели Decision Tree Classifier"""
    clf = DecisionTreeClassifier(random_state=42)
    learn_model(clf, x_train, y_train, x_test, y_test, 'DecisionTreeClassifier')


@time_logger
def extremally_gradient_method(x_train, y_train, x_test, y_test, params):
    """Создание и обучение модели XGBoost"""
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric=params['eval_metric'])
    learn_model(clf, x_train, y_train, x_test, y_test, 'XGBClassifier')


@time_logger
def dumn_method(x_train, y_train, x_test, y_test, params):
    """Создание и обучение Dummy Classifier"""
    clf = DummyClassifier(strategy=params['strategy'])  # stratified
    learn_model(clf, x_train, y_train, x_test, y_test, 'DummyClassifier')


@time_logger
def SVM_method(x_train, y_train, x_test, y_test, params):
    """Создание и обучение модели SVM"""
    clf = SVC(kernel=params['kernel'], random_state=42)  # альтернатива rbf, poly
    learn_model(clf, x_train, y_train, x_test, y_test, 'SVC')


def calc_all_methods(x_train, y_train, x_test, y_test):
    """Метод последовательно вызывающий все классификационные модели"""

    # Непосредственный вызов моделей
    #Градиентный бустинг
    gradient(x_train, y_train, x_test, y_test, CONST.model_params)
    # CatBoost
    yandex(x_train, y_train, x_test, y_test, CONST.model_params)
    #ADA Boost
    ada_boost(x_train, y_train, x_test, y_test, CONST.model_params)
    # Extra Trees
    extra_tree(x_train, y_train, x_test, y_test, CONST.model_params)
    # QDA Boost
    sqr_boost(x_train, y_train, x_test, y_test)
    #LGBM
    LGBM_method(x_train, y_train, x_test, y_test, CONST.model_params)
    # Decision Tree Classifier
    decision_tree_method(x_train, y_train, x_test, y_test)
    # XGBoost
    extremally_gradient_method(x_train, y_train, x_test, y_test, CONST.model_params)
    # Dummy Classifier
    dumn_method(x_train, y_train, x_test, y_test, CONST.model_params)
    # SVM
    SVM_method(x_train, y_train, x_test, y_test, CONST.model_params)
    # K ближайших соседей
    KNN_method(x_train, y_train, x_test, y_test, CONST.model_params)


if __name__ == '__main__':
    """Основной метод приложения"""
    # Перенаправляем стандартный вывод в файл
    log = DLlogger(CONST.output_file, True)
    print = log.printml

    # Загружаем данные из файла
    load_data(CONST.train_file4lab, CONST.test_file4lab)
    # Собираем статистику по датасету
    print('Статистика тренировочного дата сета')
    check_dataset(df_train)
    print('Статистика тестового дата сета')
    check_dataset(df_test)
    # Инициализируем наборы
    x_train, y_train, x_test, y_test = init_datasets()
    # Запускаем расчет моделей
    calc_all_methods(x_train, y_train, x_test, y_test)
    # Анализируем атрибуты датасета на релевантность
    check_attributes(x_train, y_train)
    # Генерируем новые наборы с помощью фича-инженеринга
    x_train, x_test = init_separated_dataset()
    # Занаво пересчитываем модели
    calc_all_methods(x_train, y_train, x_test, y_test)

#
