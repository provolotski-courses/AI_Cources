# https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset
import sqlite3
import csv
import sklearn
connection = None


def connect_to_db():
    conn = sqlite3.connect('populationdb.db')
    cursor = conn.cursor()
    return cursor, conn


def create_tables():
    cursor, conn = connect_to_db()
    try:
        cursor.execute('drop table if exists population')
        cursor.execute('drop table if exists periods')
        cursor.execute('drop table if exists countries')
    except sqlite3.OperationalError:
        print('Tables don''t dropped')
    cursor.execute(
        'create table countries(id INTEGER PRIMARY KEY AUTOINCREMENT, code text, name text, capital_name text,'
        'continent text)')
    cursor.execute('create table periods(id INTEGER PRIMARY KEY AUTOINCREMENT, fact_year text)')
    cursor.execute(
        'create table population(id_country INTEGER, id_period INTEGER, population integer, PRIMARY KEY (id_country, '
        'id_period),'
        ' FOREIGN KEY (id_country)  REFERENCES countries (id),  FOREIGN KEY (id_period)  REFERENCES periods (id) )')
    conn.commit()


def get_countries_id(country_code, country_name, country_capital, country_continent):
    cursor, conn = connect_to_db()
    cursor.execute('select id from countries where code = ? and name = ? and capital_name = ? and continent = ?',
                   (country_code, country_name, country_capital, country_continent))
    country_id = cursor.fetchone()
    if country_id is None:
        cursor.execute('insert into countries (code, name, capital_name, continent)values(?, ?, ?, ?)',
                       (country_code, country_name, country_capital, country_continent))
        conn.commit()
        # connection.close()
        return get_countries_id(country_code, country_name, country_capital, country_continent)
    else:
        cursor.close()
        # connection.close()
        return country_id[0]


def get_periods_id(year):
    cursor, conn = connect_to_db()
    cursor.execute('select id from periods where fact_year = ?', year)
    periods_id = cursor.fetchone()
    if periods_id is None:
        cursor.execute('insert into periods (fact_year) values (?)', year)
        conn.commit()
        cursor.close()
        # connection.close()
        return get_periods_id(year)
    else:
        cursor.close()
        # connection.close()
        return periods_id[0]


def initdata():
    mapping_csv={'2022':5,'2020':6,'2015':7,'2010':8,'2000':9,'1990':10,'1980':11,'1970':12}
    cursor, conn = connect_to_db()
    with open('world_population.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            country_code = row[1]
            country_name = row[2]
            country_capital = row[3]
            country_continent = row[4]
            for dict in mapping_csv:
                cursor.execute('insert into population (id_country, id_period , population) values (?, ?, ?)',
                            (get_countries_id(country_code, country_name, country_capital, country_continent),
                             get_periods_id(dict), row[mapping_csv[dict]]))
            conn.commit()
    cursor.close()


if __name__ == '__main__':
    create_tables()
    # initdata()
    print(get_periods_id('2022'))
