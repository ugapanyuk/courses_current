import sys
import math
from typing import NamedTuple

# Алгебраический тип
class SquareRootResult:
    NoRoots = NamedTuple("NoRoots", [])
    OneRoot = NamedTuple("OneRoot", [("root", float)])
    TwoRoots = NamedTuple("TwoRoots", [("root1", float) , ("root2", float)])


def get_coef(index, prompt):
    '''
    Читаем коэффициент из командной строки или вводим с клавиатуры

    Args:
        index (int): Номер параметра в командной строке
        prompt (str): Приглашение для ввода коэффицента

    Returns:
        float: Коэффициент квадратного уравнения
    '''
    try:
        # Пробуем прочитать коэффициент из командной строки
        coef_str = sys.argv[index]
    except:
        # Вводим с клавиатуры
        print(prompt)
        coef_str = input()
    # Переводим строку в действительное число
    coef = float(coef_str)
    return coef


def get_roots(a, b, c):
    '''
    Вычисление корней квадратного уравнения

    Args:
        a (float): коэффициент А
        b (float): коэффициент B
        c (float): коэффициент C

    Returns:
        Список корней в виде типа SquareRootResult
    '''
    D = b*b - 4*a*c
    if D == 0.0:
        root = -b / (2.0*a)
        return SquareRootResult.OneRoot(root)
    elif D > 0.0:
        sqD = math.sqrt(D)
        root1 = (-b + sqD) / (2.0*a)
        root2 = (-b - sqD) / (2.0*a)
        return SquareRootResult.TwoRoots(root1, root2)
    else:
        return SquareRootResult.NoRoots()


def print_roots(roots_tuple):
    '''
    Печать корней квадратного уравнения

    Args:
        Список корней в виде типа SquareRootResult
    '''
    match roots_tuple:
        case SquareRootResult.TwoRoots(root1, root2):
            print(f'Два корня: {root1} и {root2}')
        case SquareRootResult.OneRoot(root):
            print(f'Один корень: {root}')
        case SquareRootResult.NoRoots():
            print('Нет корней')        


def main():
    '''
    Основная функция
    '''
    a = get_coef(1, 'Введите коэффициент А:')
    b = get_coef(2, 'Введите коэффициент B:')
    c = get_coef(3, 'Введите коэффициент C:')
    # Вычисление корней
    roots = get_roots(a,b,c)
    # Вывод корней
    print_roots(roots)


# Если сценарий запущен из командной строки
if __name__ == "__main__":
    main()

# Примеры запуска
# python roots_namedtuple.py 1 0 -4
# Два корня: 2.0 и -2.0

# python roots_namedtuple.py 1 0 0
# Один корень: -0.0

# python roots_namedtuple.py 1 0 4
# Нет корней