from imblearn.under_sampling import NearMiss
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pycaret.classification import *

from utils.util import time_logger, DLlogger
import const.ds_const as CONST


@time_logger
def load_dataset(filename):
    """Загрузка датасета из файла"""
    # загружаем датасет
    loaded_dataset = pd.read_csv(filename)
    # в представленном датасете одно из полей означает сколько было людей с таким набором атрибутов
    # поэтому повторяем каждую строку столько раз сколько необходимо
    # с учетом отсутсвия необходимости имитировать перепись США количество повторов уменьшаем в нужное количество раз
    loaded_dataset = loaded_dataset.loc[
        loaded_dataset.index.repeat(loaded_dataset[CONST.fw_dataset_counter] / CONST.fw_dataset_divider)].reset_index(
        drop=True)
    # удаляем столбец определяющий количество повторений
    loaded_dataset = loaded_dataset.drop(columns=[CONST.fw_dataset_counter])
    # Удаляем столбцы определенные в процессе анализа как бесполезные
    # education.num - поскольку это числовое представление столбца education.
    # (По-хорошему бы делать наоборот, но поскольку это учебная работа - сами будем приводить к чиселкам)
    loaded_dataset = loaded_dataset.drop(columns=['education.num'])

    # relationship - странное поле, зависимость от пола и семейного статуса - только загрязняет датасет
    loaded_dataset = loaded_dataset.drop(columns=['relationship'])
    # перемешиваем датасет
    loaded_dataset = loaded_dataset.sample(frac=1)
    return loaded_dataset


@time_logger
def EDA_report(analyzed_dataset):
    """Проведение EDA-анализа над переданным датасетом"""
    """Вывод статистики по дата фрейму"""
    print('***---- head  ------***')
    print(analyzed_dataset.head())
    print('***---- shape  ------***')
    print(analyzed_dataset.shape)
    print('***---- describe  ------***')
    print(analyzed_dataset.describe())
    print('***---- нулевые значения  ------***')
    print(analyzed_dataset.isnull().sum())
    print('***---- Анализ атрибутов  ------***')
    for col in analyzed_dataset.columns:
        print(f'проводим разбор по атрибуту {col}')
        print(f'тип данных: {analyzed_dataset[col].dtypes}')
        print(f'гранулярность данных {analyzed_dataset[col].nunique()}')
        print('Частота появления каждого значения:')
        print(analyzed_dataset[col].value_counts())
    # выводим значения подмен
    return analyzed_dataset


def analyze_pycaret(dataset, target):
    s = setup(dataset, target=target)
    print(s)
    best = compare_models()
    print(best)
    evaluate_model(best)
    plot_model(best, plot='auc', to_file='auc.png')
    plot_model(best, plot='confusion_matrix',to_file='confusion_matrix.png')
    plot_model(best, plot='class_report', to_file='class_report.png')
    predict_model(best)

@time_logger
def generate_dataframes(census_dataset, predict_attr):
    """Генерируем наборы данных для моделей """
    # убираем дисбаланс dataframe по показателю атрибута
    # Применяем encoder для категориальных (не числовых) атрибутов
    categorical_columns = census_dataset.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        census_dataset[column] = le.fit_transform(census_dataset[column])
        label_encoders[column] = le
    # определяем что целевой параметр
    y = census_dataset[predict_attr]
    # параметры для обучения - все кроме целевой
    x = census_dataset.drop(columns=[predict_attr])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print('До применения метода кол-во меток 1-го класса: {}'.format(sum(y_train == 0)))
    # применяем балансировку
    nm = NearMiss()
    x_train_mean, y_train_mean = nm.fit_resample(x_train, y_train.ravel())
    print('После применения метода кол-во меток  1-го класса: {}'.format(sum(y_train_mean == 0)))
    return x_train_mean, y_train_mean, x_train, y_train, x_test, y_test

@time_logger
def train_model(classifier, x_train, y_train, x_test, y_test, method, state):
    """Обучение модели """
    clf = classifier
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)  # Считаем предсказания
    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f' Report for {method} method. {state} model')
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)


def train_model_dual(classifier, x_train_mean, y_train_mean, x_train, y_train, x_test, y_test, method):
    """Обучаем две модели на несбалансированной и на сбалансированной модели"""
    # несбалансированная
    train_model(classifier, x_train, y_train, x_test, y_test, method, 'disbalanced')
    # сбалансированная
    train_model(classifier, x_train_mean, y_train_mean, x_test, y_test, method, 'balanced')


if __name__ == '__main__':
    # перенаправляем вывод
    log = DLlogger(CONST.fw_output_file, True)
    print = log.printml
    # Получаем данные
    census_dataset = load_dataset(CONST.fw_dataset_file)
    # определяем целевой параметр
    predict_attr = 'marital.status'
    # Проводим анализ Pycaret
    analyze_pycaret(census_dataset, predict_attr)
    # Анализируем данные
    census_dataset = EDA_report(census_dataset)
    # Получаем наборы
    x_train_mean, y_train_mean, x_train, y_train, x_test, y_test = generate_dataframes(census_dataset, predict_attr)
    # Обучаем модели
    for method_name, classifier in CONST.fw_model_dict.items():
        train_model_dual(classifier, x_train_mean, y_train_mean, x_train, y_train, x_test, y_test,
                         method_name)
