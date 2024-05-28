# https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset
import sqlite3
import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def connect_to_db():
    """Создание подключения к базе данных"""
    conn_sql = sqlite3.connect('databases/populationdb.db')
    cursor_sql = conn_sql.cursor()
    return cursor_sql, conn_sql


def create_tables(dbcursor, dbconnection):
    """Создание таблиц в базе данных"""
    try:
        # удаление таблиц если ранее были
        dbcursor.execute('drop table if exists population')
        dbcursor.execute('drop table if exists periods')
        dbcursor.execute('drop table if exists countries')
    except sqlite3.OperationalError:
        print('Tables don''t dropped')

    # создание таблицы-справочника стран
    dbcursor.execute(
        'create table countries(id INTEGER PRIMARY KEY AUTOINCREMENT, code text, name text, capital_name text,'
        'continent text)')

    # создание таблицы-справочника периодов
    dbcursor.execute('create table periods(id INTEGER PRIMARY KEY AUTOINCREMENT, fact_year Integer)')

    # создание таблицы фактов
    dbcursor.execute(
        'create table population(id_country INTEGER, id_period INTEGER, population integer, PRIMARY KEY (id_country, '
        'id_period),'
        ' FOREIGN KEY (id_country)  REFERENCES countries (id),  FOREIGN KEY (id_period)  REFERENCES periods (id) )')
    dbconnection.commit()


def get_countries_id(dbcursor, dbconnection, country_code, country_name, country_capital, country_continent):
    """Функция для получения первичного ключа страны"""
    # ищем в таблице нужную запись
    dbcursor.execute('select id from countries where code = ? and name = ? and capital_name = ? and continent = ?',
                     (country_code, country_name, country_capital, country_continent))
    country_id = dbcursor.fetchone()
    if country_id is None:
        # если не нашли - добавляем и рекурсивно обращаемся к этой же функции, чтобы забрать ID
        dbcursor.execute('insert into countries (code, name, capital_name, continent)values(?, ?, ?, ?)',
                         (country_code, country_name, country_capital, country_continent))
        dbconnection.commit()

        return get_countries_id(dbcursor, dbconnection, country_code, country_name, country_capital, country_continent)
    else:
        # Возвращаем ID
        return country_id[0]


def get_periods_id(dbcursor, dbconnection, year):
    """Функция для получения первичного ключа периода"""
    # ищем в таблице нужную запись
    dbcursor.execute('select id from periods where fact_year = ?', (year,))
    periods_id = dbcursor.fetchone()
    if periods_id is None:
        # если не нашли - добавляем и рекурсивно обращаемся к этой же функции, чтобы забрать ID
        dbcursor.execute('insert into periods (fact_year) values (?)', (year,))
        dbconnection.commit()
        return get_periods_id(dbcursor, dbconnection, year)
    else:
        # Возвращаем ID
        return periods_id[0]


def inidata(dbcursor, dbconnection):
    """загрузка данных из csv датаасета в базу данных"""
    # справочник для маппинга данных в csv (соответствие столбца году)
    mapping_csv = {'2022': 5, '2020': 6, '2015': 7, '2010': 8, '2000': 9, '1990': 10, '1980': 11, '1970': 12}
    # читаем файл
    with open('datasets/world_population.csv', "r", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # пропускаем заголовок
        next(reader, None)
        for row in reader:
            try:
                # вытягиваем данные по стране
                country_code = row[1]
                country_name = row[2]
                country_capital = row[3]
                country_continent = row[4]
                # итерируемся по справочнику
                for year_item in mapping_csv:
                    try:
                        # записываем значения в базу
                        dbcursor.execute('insert into population (id_country, id_period , population) values (?, ?, ?)',
                                         (get_countries_id(dbcursor, dbconnection, country_code, country_name,
                                                           country_capital, country_continent),
                                          get_periods_id(dbcursor, dbconnection, year_item),
                                          row[mapping_csv[year_item]]))
                    except Exception as e:
                        print(f'error:{e}')
            except Exception as e:
                print(f'error:{e}')
            dbconnection.commit()


def show_full_data(dbconnection):
    """Функция для отображения всего содержимого базы данных"""
    # выполняем запрос
    population = pd.read_sql_query(
        'select countries.name, periods.fact_year, population.population '
        'from population, countries, periods '
        'where population.id_country=countries.id and population.id_period= periods.id', dbconnection)
    # на всякий случай выпечатываем первые строки
    print(population.head())
    # строим график
    sns.scatterplot(population, x='fact_year', y='population')
    plt.show()


def show_country_code_data(dbcursor):
    """Метод для выпечатки справочника стран"""
    # выполняем запрос
    dbcursor.execute('select * from countries')
    # забираем данные из курсора
    countries = dbcursor.fetchall()
    # итерируемся по данным и выпечатываем их
    for country in countries:
        print(f' code: {country[1]},  name: {country[2]}, continent: {country[4]}, capital: {country[3]}')


def show_country_data(dbconnection, country_code):
    """Функция для отображения данных по конкретной стране"""
    # выполняем запрос
    population = pd.read_sql_query(
        'select countries.name, periods.fact_year, population.population '
        'from population, countries, periods '
        'where population.id_country=countries.id and population.id_period= periods.id and countries.code =?',
        dbconnection, params=[country_code])
    # на всякий случай выпечатываем первые строки
    print(population.head())
    # строим график
    sns.lineplot(population, x='fact_year', y='population')
    plt.show()


def show_continent_data(dbconnection):
    """Функция отображения аггрегатов по континентам """
    # выполняем запрос
    population = pd.read_sql_query(
        'select countries.continent, periods.fact_year, sum(population.population) as population '
        'from population, countries, periods '
        'where population.id_country=countries.id and population.id_period= periods.id '
        'group by countries.continent, periods.fact_year',
        dbconnection)
    # на всякий случай выпечатываем первые строки
    print(population.head())
    # строим график
    sns.lineplot(population, x='fact_year', y='population', hue='continent')
    plt.show()


if __name__ == '__main__':
    cursor, conn = connect_to_db()
    flag = True
    while flag:
        print(' to recreate the database press 1')
        print(' To view all data press 2')
        print(' To view data for a specific country, press 3')
        print(' To view data by continent 4')
        print(' to view country codes press 9')
        print('To exit press 0')
        choice = input('press key to select: ')
        if choice == '1':
            create_tables(cursor, conn)
            inidata(cursor, conn)
        elif choice == '2':
            show_full_data(conn)
        elif choice == '3':
            show_country_data(conn, input('Enter country code:'))
        elif choice == '4':
            show_continent_data(conn)
        elif choice == '9':
            show_country_code_data(cursor)
        elif choice == '0':
            flag = False
