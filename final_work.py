import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import ds_utils.analyze_dataset
import utils.util
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
def eda_report(analyzed_dataset):
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


@time_logger
def generate_dataframes(census_ds, predict_attr_df):
    """Генерируем наборы данных для моделей """
    # Убираем дисбаланс dataframe по показателю атрибута.
    # Применяем encoder для категориальных (не числовых) атрибутов
    categorical_columns = census_ds.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        census_ds[column] = le.fit_transform(census_ds[column])
        label_encoders[column] = le
    # определяем что целевой параметр
    y = census_ds[predict_attr_df]

    # параметры для обучения - все кроме целевой
    x = census_ds.drop(columns=[predict_attr_df])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return census_ds, label_encoders, x_train, y_train, x_test, y_test





if __name__ == '__main__':
    # перенаправляем вывод
    log = DLlogger(CONST.fw_output_file, True)
    print = log.printml
    # Получаем данные
    census_dataset = load_dataset(CONST.fw_dataset_file)
    utils.util.create_rep_dir('img')
    census_dataset = eda_report(census_dataset)
    # определяем целевой параметр
    for predict_attr in census_dataset.columns:
        if predict_attr in ['age']:
            continue
        utils.util.create_rep_dir(predict_attr)
        utils.util.create_rep_dir(f'{predict_attr}/img')
        # Проводим анализ Pycaret
        ds_utils.analyze_dataset.analyze_pycaret(census_dataset, predict_attr)
        # Анализируем данные

        # Получаем наборы
        census_dataset, le, x_train, y_train, x_test, y_test = generate_dataframes(
            census_dataset, predict_attr)
        ds_utils.analyze_dataset.show_heatmap(census_dataset)
        ds_utils.analyze_dataset.show_histogram(census_dataset)
        ds_utils.analyze_dataset.analyze_target(census_dataset, predict_attr, dict(
            zip(le[predict_attr].classes_, le[predict_attr].transform(le[predict_attr].classes_))))
        # Обучаем модели
        for method_name, classifier in CONST.fw_model_dict.items():
            train_model(classifier, x_train, y_train, x_test, y_test,
                        method_name)
