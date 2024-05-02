from datetime import date

myDict = {'Author': input('Input name of author: '), 'Title': input('Input title of book: ')}
flag = True
while flag:
    try:
        myDict['Year'] = int(input('Input publication year: '))
        flag = myDict['Year'] < 1454 or myDict['Year'] > date.today().year
    except ValueError:
        pass
    finally:
        if  flag:
            print('Wrong year')
print(myDict)