from stockfish import Stockfish
from stockfish import StockfishException
import pandas as pd
import numpy as np
import re
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer
import random
import string


import matplotlib.pyplot as plt
import seaborn as sns

def generate_random_key(length=10):
    characters = string.ascii_letters + string.digits  # lettres majuscules, lettres minuscules et chiffres
    random_key = ''.join(random.choice(characters) for i in range(length))
    return random_key


emplacement_actuel = os.getcwd()
print(emplacement_actuel)

nament = 'lichess1500-2000-10.txt'
chemin = os.path.join('pgns', nament)

with open(chemin) as f:
    pos = f.readlines()
pos1 = []
for i in range(len(pos)):
    pos1.append(re.sub('\n', '', pos[i]))
dataset = pos1


# with open('posiciones.txt') as f:
#     pos = f.readlines()
# pos1 = []
# for i in range(len(pos)):
#     pos1.append(re.sub('\n','', pos[i]))
# bl = []
# wh = []
# for i in range(len(pos1)):
#     bl.append(pos1[i] + ' b')
#     wh.append(pos1[i] + ' w')
# dataset = bl + wh



def evaluation(pos, elo, depth):
    stockfish = Stockfish('C:/Users/axel/Git/MT_ChessEngines/stockfish_15_x64_avx2.exe')
    stockfish.set_elo_rating(elo)
    stockfish.set_depth(depth)
    stockfish.set_fen_position(pos)
    ev = stockfish.get_evaluation()
    pos = stockfish.get_top_moves(1)

    return ev, pos


def evaluationdouble(pos1, pos2, elo, depth):
    stockfish = Stockfish('C:/Users/axel/Git/MT_ChessEngines/stockfish_15_x64_avx2.exe')
    stockfish.set_elo_rating(elo)
    stockfish.set_depth(depth)
    stockfish.set_fen_position(pos1)
    ev1 = stockfish.get_evaluation()
    stockfish.set_fen_position(pos2)
    ev2 = stockfish.get_evaluation()
    return ev1, ev2

def split_mate_cp(evaluations):
    pos1s_mate = []
    pos2s_mate = []
    ev1s_mate = []
    ev2s_mate = []
    indexy_mate = []
    pos1s_cp = []
    pos2s_cp = []
    ev1s_cp = []
    ev2s_cp = []
    indexy_cp = []
    index_cp = 0
    index_mate = 0
    for index, eva in evaluations.iterrows():
        if isinstance(eva['evaluation pos1'], list):
            if eva['evaluation pos1'][0].get('Mate')!=None or eva['evaluation pos2'][0].get('type')!=None:
                indexy_mate.append(index_mate)
                index_mate+=1
                pos1s_mate.append(eva['pos1'])
                pos2s_mate.append(eva['pos2'])
                ev1s_mate.append(eva['evaluation pos1'])
                ev2s_mate.append(eva['evaluation pos2'])
            else:
                indexy_cp.append(index_mate)
                index_cp += 1
                pos1s_cp.append(eva['pos1'])
                pos2s_cp.append(eva['pos2'])
                ev1s_cp.append(eva['evaluation pos1'])
                ev2s_cp.append(eva['evaluation pos2'])
        elif eva['evaluation pos1'].get('type')=='mate' or eva['evaluation pos2'].get('type')=='mate':
            indexy_mate.append(index_mate)
            index_mate+=1
            pos1s_mate.append(eva['pos1'])
            pos2s_mate.append(eva['pos2'])
            ev1s_mate.append(eva['evaluation pos1'])
            ev2s_mate.append(eva['evaluation pos2'])
        else:
            indexy_cp.append(index_mate)
            index_cp += 1
            pos1s_cp.append(eva['pos1'])
            pos2s_cp.append(eva['pos2'])
            ev1s_cp.append(eva['evaluation pos1'])
            ev2s_cp.append(eva['evaluation pos2'])
    dmate = {'index': indexy_mate, 'pos1': pos1s_mate, 'evaluation pos1': ev1s_mate,
             'pos2': pos2s_mate, 'evaluation pos2': ev2s_mate}
    dcp = {'index': indexy_cp, 'pos1': pos1s_cp, 'evaluation pos1': ev1s_cp,
             'pos2': pos2s_cp, 'evaluation pos2': ev2s_cp}
    mate_evas = pd.DataFrame(data=dmate)
    cp_evas = pd.DataFrame(data=dcp)
    return cp_evas, mate_evas

def real_positions(evas):
    real = []
    real2 = []
    indexy = []
    ev1s = []
    ev2s = []
    index = 0
    real_pos = dataset[24768:]
    #indexent2 = 0
    for indexent, eva in evas.iterrows():
        for pos in real_pos:
            if eva['pos1'] == pos and eva['pos1'] not in real:
                #print(indexent - indexent2, index)
                #indexent2 = indexent
                indexy.append(index)
                index+=1
                real.append(eva['pos1'])
                real2.append(eva['pos2'])
                ev1s.append(eva['evaluation pos1'])
                ev2s.append(eva['evaluation pos2'])
    d = {'index': indexy, 'pos1': real, 'evaluation pos1': ev1s,
             'pos2': real2, 'evaluation pos2': ev2s}
    df = pd.DataFrame(data=d)
    return df

def analyse_fen(pos,d):
    depth = [i for i in range(1,d+1)]
    evas = []
    for d in depth:
        evas.append(evaluation(pos,50000,d)[0])
    return evas


def saveevas(pos1, pos2, min_d, max_d, type, delta, epsilon,name):
    d = 1
    data1 = []
    data2 = []
    val = []
    if type == 0:
        e1 = evaluation(pos1, 50000, d)[0]
        e2 = evaluation(pos2, 50000, d)[0]
        data1.append(e1)
        data2.append(e2)
        while (not MR_mirror(e1, e2, delta, epsilon) or d < min_d) and d < max_d:
            print(d, end='', flush=True)
            print('\r', end='', flush=True)
            d += 1
            e1 = evaluation(pos1, 50000, d)[0]
            e2 = evaluation(pos2, 50000, d)[0]
            data1.append(e1)
            data2.append(e2)
            if MR_mirror(e1, e2, delta, epsilon):
                val.append(d)
    elif type == 1 or type == 2:
        e1 = evaluation(pos1, 50000, d)[0]
        e2 = evaluation(pos2, 50000, d)[0]
        data1.append(e1)
        data2.append(e2)
        while (not MR_equi(e1, e2, delta, epsilon) or d < min_d) and d < max_d:
            print(d, end='', flush=True)
            print('\r', end='', flush=True)
            d += 1
            e1 = evaluation(pos1, 50000, d)[0]
            e2 = evaluation(pos2, 50000, d)[0]
            data1.append(e1)
            data2.append(e2)
            if MR_equi(e1, e2, delta, epsilon):
                val.append(d)
    elif type == 3:
        e1 = evaluation(pos1, 50000, d)[0]
        e2 = evaluation(pos2, 50000, d)[0]
        data1.append(e1)
        data2.append(e2)
        while (not MR_better(e1, e2, delta, epsilon) or d < min_d) and d < max_d:
            print(d, end='', flush=True)
            print('\r', end='', flush=True)
            d += 1
            e1 = evaluation(pos1, 50000, d)[0]
            e2 = evaluation(pos2, 50000, d)[0]
            data1.append(e1)
            data2.append(e2)
            if MR_better(e1, e2, delta, epsilon):
                val.append(d)
    elif type == 4:
        e1 = evaluation(pos1, 50000, d)[1]
        e2 = evaluation(pos2, 50000, d)[1]
        data1.append(e1[0])
        data2.append(e2[0])
        while (not MR_first(e1, e2, delta, epsilon) or d < min_d) and d < max_d:
            print(d, end='', flush=True)
            print('\r', end='', flush=True)
            d += 1
            e1 = evaluation(pos1, 50000, d)[1]
            e2 = evaluation(pos2, 50000, d)[1]
            data1.append(e1[0])
            data2.append(e2[0])
            if MR_first(e1, e2, delta, epsilon):
                val.append(d)
    print('MR verified at depth = ', val)
    typent = ["sim_mirror", "sim_axis", "sim_diag", "replace", "best_move"][type]
    name = name+".pkl"

    subfolder = 'plotevas'
    chemin = os.path.join(subfolder, name)
    with open(chemin, 'wb') as fichier:
        pickle.dump((data1,data2,val), fichier)


def plotevas(data1, data2, val, type):
    print('MR verified at depth = ', val)
    ind1 = 0
    ind2 = 0
    cp1 = []
    cp2 = []
    mate1 = []
    mate2 = []
    values1cp = []
    values1mate = []
    values2cp = []
    values2mate = []
    if type == 4:
        for data in data1:
            ind1 += 1
            if data.get('Mate') == None:
                cp1.append(ind1)
                values1cp.append(data.get('Centipawn'))
            else:
                mate1.append(ind1)
                values1mate.append(data.get('Mate'))

        for data in data2:
            ind2 += 1
            if data.get('Mate') == None:
                cp2.append(ind2)
                values2cp.append(data.get('Centipawn'))
            else:
                mate2.append(ind2)
                values2mate.append(data.get('Mate'))
    else:
        for data in data1:
            ind1 += 1
            if data.get('type') == 'cp':
                cp1.append(ind1)
                values1cp.append(data.get('value'))
            else:
                values1mate.append(data.get('value'))
                mate1.append(ind1)
        for data in data2:
            ind2 += 1
            if type == 0:
                if data.get('type') == 'cp':
                    cp2.append(ind2)
                    values2cp.append(data.get('value') * (-1))
                else:
                    mate2.append(ind2)
                    values2mate.append(data.get('value') * (-1))

            else:
                if data.get('type') == 'cp':
                    cp2.append(ind2)
                    values2cp.append(data.get('value'))
                else:
                    mate2.append(ind2)
                    values2mate.append(data.get('value'))


    plt.plot(cp1, values1cp, '-', color='#17d1d3')
    plt.plot(cp2, values2cp, '--', color='#405ee3')

    plt.grid(axis='y', color='#D3D3D3', linestyle='-', linewidth=0.5)

    plt.ylabel('Centipawn advantage', color='blue')
    plt.tick_params('y', colors='blue')

    plt.twinx()

    plt.plot(mate1, values1mate, '-', color='#f7766d')
    plt.plot(mate2, values2mate, '--', color='#df2c1f')

    plt.ylabel('Mate in', color='red')
    plt.tick_params('y', colors='red')

    plt.show()


def sorted_gaps(evas,type, delta, epsilon,mate):
    if mate:
        temp = sort_df((getFalses(evas, type, delta, epsilon)),True)
    else:
        temp = sort_df(split_mate_cp(getFalses(evas, type, delta, epsilon))[0])
    temp.reset_index(drop=True, inplace=True)
    return temp

def sort_df(df,mate=False):
    if isinstance(df['evaluation pos1'][0],list) and not mate:
        df = df[df['evaluation pos1'].apply(lambda x: abs(x[0]['Centipawn']) is not None)]
        df['Centipawn'] = df['evaluation pos1'].apply(lambda x: abs(x[0]['Centipawn']))
        df = df.sort_values(by='Centipawn', ascending=True)
        df = df.reset_index(drop=True)
        df = df.drop(columns=['Centipawn'])
        return df
    elif isinstance(df['evaluation pos1'][0],list) and mate:
        df = df[df['evaluation pos1'].apply(lambda x: x[0]['Mate'] is not None)]
        df['Mate'] = df['evaluation pos1'].apply(lambda x: abs(x[0]['Mate']))
        df = df.sort_values(by='Mate', ascending=False)
        df = df.reset_index(drop=True)
        df = df.drop(columns=['Mate'])
        return df
    return df.iloc[df['evaluation pos1'].apply(lambda x: abs(x['value'])).argsort()]

def get_opp(df,type):
    pos1s = []
    pos2s = []
    ev1s = []
    ev2s = []
    indexy = []
    index = 0
    for indexent, data in df.iterrows():
        if type==0:
            if data['evaluation pos1'].get('value') * data['evaluation pos2'].get('value') > 0:
                indexy.append(index)
                index+=1
                pos1s.append(data['pos1'])
                pos2s.append(data['pos2'])
                ev1s.append(data['evaluation pos1'])
                ev2s.append(data['evaluation pos2'])
        elif type==4:
            if data['evaluation pos1'][0].get('Mate') != None:
                if data['evaluation pos2'][0].get('Mate') != None:
                    if data['evaluation pos1'][0].get('Mate')*data['evaluation pos2'][0].get('Mate') < 0:
                        indexy.append(index)
                        index += 1
                        pos1s.append(data['pos1'])
                        pos2s.append(data['pos2'])
                        ev1s.append(data['evaluation pos1'])
                        ev2s.append(data['evaluation pos2'])
                else:
                    if data['evaluation pos1'][0].get('Mate') * data['evaluation pos2'][0].get('Centipawn') < 0:
                        indexy.append(index)
                        index += 1
                        pos1s.append(data['pos1'])
                        pos2s.append(data['pos2'])
                        ev1s.append(data['evaluation pos1'][0])
                        ev2s.append(data['evaluation pos2'][0])
            else:
                if data['evaluation pos2'][0].get('Mate') != None:
                    if data['evaluation pos1'][0].get('Centipawn')*data['evaluation pos2'][0].get('Mate') < 0:
                        indexy.append(index)
                        index += 1
                        pos1s.append(data['pos1'])
                        pos2s.append(data['pos2'])
                        ev1s.append(data['evaluation pos1'])
                        ev2s.append(data['evaluation pos2'])
                else:
                    if data['evaluation pos1'][0].get('Centipawn') * data['evaluation pos2'][0].get('Centipawn') < 0:
                        indexy.append(index)
                        index += 1
                        pos1s.append(data['pos1'])
                        pos2s.append(data['pos2'])
                        ev1s.append(data['evaluation pos1'])
                        ev2s.append(data['evaluation pos2'])
        else:
            if data['evaluation pos1'].get('value')*data['evaluation pos2'].get('value')<0:
                indexy.append(index)
                index += 1
                pos1s.append(data['pos1'])
                pos2s.append(data['pos2'])
                ev1s.append(data['evaluation pos1'])
                ev2s.append(data['evaluation pos2'])
    d = {'index': indexy, 'pos1': pos1s, 'evaluation pos1': ev1s,
         'pos2': pos2s, 'evaluation pos2': ev2s}
    dfent = pd.DataFrame(data=d)
    return dfent

def dif1(val1, val2):
    if abs(val1) + abs(val2) == 0:
        return 0
    else:
        return abs(val1 - val2) / (abs(val1) + abs(val2))


def dif2(val1, val2):
    return float(abs(val1 - val2))


def MR_equi(o1, o2, delta, epsilon):
    final = False
    if o1.get('type') != o2.get('type'):
        final = False
    elif o1.get('type') == o2.get('type') == 'mate' and o1.get('value') != o2.get('value'):
        final = False
    elif o1.get('type') == o2.get('type') == 'cp' and dif1(o1.get('value'), o2.get('value')) > delta and dif2(
            o1.get('value'), o2.get('value')) >= epsilon:
        final = False
    else:
        final = True
    return final


def MR_mirror(o1, o2, delta, epsilon):
    final = False
    if o1.get('type') != o2.get('type'):
        final = False
    elif o1.get('type') == o2.get('type') == 'mate' and o1.get('value') != (-1 * o2.get('value')):
        final = False
    elif o1.get('type') == o2.get('type') == 'cp' and dif1(o1.get('value'), (-1 * o2.get('value'))) > delta and dif2(
            o1.get('value'), (-1 * o2.get('value'))) >= epsilon:
        final = False
    else:
        final = True
    return final


def MR_better(o1, o2, delta, epsilon):
    final = False
    # sachant qu'on améliore des pièces blanches (toujours)
    # si on passe de diff en centipions à mat pour les noirs --> false
    if o1.get('type') == 'cp' and o2.get('type') == 'mate' and o2.get('value') < 0:
        final = False
    # si on passe de mat pour les blancs à diff de centipions --> false
    elif o1.get('type') == 'mate' and o2.get('type') == 'cp' and o1.get('value') > 0:
        final = False
    # ex : si on passe de #7 à #9 --> false
    elif o1.get('type') == o2.get('type') == 'mate' and o2.get('value') > o1.get('value') > 0:
        final = False
    # ex : si on passe de #-9 à #-7 --> false
    elif o1.get('type') == o2.get('type') == 'mate' and 0 > o2.get('value') > o1.get('value'):
        final = False
    # ex : si on passe de #5 à #-5 --> false
    elif o1.get('type') == o2.get('type') == 'mate' and o1.get('value') > 0 > o2.get('value'):
        final = False
    # si diff de centipions les deux, dépend du seuil
    elif o1.get('type') == o2.get('type') == 'cp' and o1.get('value') > o2.get('value') and dif1(o1.get('value'),
                                                                                                 o2.get('value')) > delta and dif2(
            o1.get('value'), o2.get('value')) >= epsilon:
        final = False
    else:
        final = True
    return final


# MR_better d'origine
# def MR_better(o1,o2,delta,epsilon):
#     final = False
#     if o1.get('type') != o2.get('type') and o2.get('type') != 'mate':
#         final = False
#     elif o1.get('type') == o2.get('type') == 'mate' and o1.get('value') < o2.get('value') and o1.get('value') > 0:
#         final = False
#     elif o1.get('type') == o2.get('type') == 'mate' and o1.get('value') > o2.get('value') and o1.get('value') < 0:
#         final = False
#     elif o1.get('type') == o2.get('type') == 'cp' and o1.get('value') > o2.get('value') and dif1(o1.get('value'),o2.get('value')) > delta and dif2(o1.get('value'),o2.get('value')) >= epsilon:
#         final = False
#     else:
#         final = True
#     return final


def MR_first(o1, o2, delta, epsilon):
    final = False
    if o1[0].get('Mate') != o2[0].get('Mate'):
        final = False
    elif o1[0].get('Mate') == o2[0].get('Mate') == None and dif1(o1[0].get('Centipawn'),
                                                                 o2[0].get('Centipawn')) > delta and dif2(
            o1[0].get('Centipawn'), o2[0].get('Centipawn')) >= epsilon:
        final = False
    else:
        final = True
    return final


def sim_axis(pos):
    pos_transformed = []
    col = pos[-1:]
    aux1 = pos[:-2]
    aux2 = aux1.split("/")
    aux3 = []
    for j in range(8):
        aux3.append(aux2[j][::-1])
    aux4 = '/'.join(aux3)
    pos_transformed.append(aux4)
    if col == 'w':
        pos_transformed = pos_transformed[0] + ' w'
    if col == 'b':
        pos_transformed = pos_transformed[0] + ' b'
    return pos_transformed


def sim_diag(pos):
    def Convert(string):
        list1 = []
        list1[:0] = string
        return list1

    pos_transformed = []
    colour = pos[-1:]
    aux = pos.replace('8', '11111111').replace('7', '1111111').replace('6', '111111').replace('5', '11111').replace('4',
                                                                                                                    '1111').replace(
        '3', '111').replace('2', '11')
    aux2 = aux.split('/')
    for i in range(8):
        aux2[i] = Convert(aux2[i])
        aux3 = [] * 8

    for j in reversed(range(8)):
        aux4 = []
        for i in reversed(range(8)):
            aux4.append(aux2[i][j])
        aux3.append(aux4)
    for i in reversed(range(8)):
        aux3[i] = "".join(aux3[i])
    pos_transformed = "/".join(aux3)
    if colour == 'b':
        pos_transformed = pos_transformed + ' b'
    else:
        pos_transformed = pos_transformed + ' w'
    pos_transformed = pos_transformed.replace('11111111', '8').replace('1111111', '7').replace('111111', '6').replace(
        '11111', '5').replace('1111', '4').replace('111', '3').replace('11', '2')

    return pos_transformed


def sim_mirror(pos):
    pos_transformed = []
    colour = pos[-1:]
    aux1 = pos[:-2]
    aux2 = aux1.swapcase()
    aux3 = aux2.split("/")
    aux4 = []
    for j in reversed(aux3):
        aux4.append(j)
        aux5 = '/'.join(aux4)
    pos_transformed.append(aux5)
    if colour == 'b':
        pos_transformed = pos_transformed[0] + ' w'
    else:
        pos_transformed = pos_transformed[0] + ' b'
    return pos_transformed


def replace(pos):
    if pos.count('B') > 0 & pos.count('R') > 0:
        n = random.random()
        if n > 0.5:
            aux = pos.replace('R', 'Q', 1)
        else:
            aux = pos.replace('B', 'Q', 1)
    elif pos.count('B') > 0 & pos.count('R') == 0:
        aux = pos.replace('B', 'Q', 1)
    elif pos.count('R') > 0 & pos.count('B') == 0:
        aux = pos.replace('R', 'Q', 1)
    else:
        aux = pos
    return aux


def checkcheckW(pos):
    def make_matrix(board):  # type(board) == chess.Board()
        pgn = board
        foo = []  # Final board
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        for row in rows:
            foo2 = []  # This is the row I make
            for thing in row:
                if thing.isdigit():
                    for i in range(0, int(thing)):
                        foo2.append('.')
                else:
                    foo2.append(thing)
            foo.append(foo2)
        return foo

    whites = ['Q', 'K', 'R', 'B', 'N', 'P']
    blacks = ['q', 'k', 'r', 'b', 'n', 'p']
    a = make_matrix(pos)
    check = False
    pos = []
    for i in range(8):
        for j in range(8):
            if a[i][j] == 'K':
                pos.append(i)
                pos.append(j)
    for i in list(reversed(range(pos[1]))):
        if a[pos[0]][i] in whites or a[pos[0]][i] in list(blacks[j] for j in [1, 3, 4, 5]):
            break
        elif a[pos[0]][i] == 'q' or a[pos[0]][i] == 'r':
            check = True
            break
    for i in list(range(pos[1] + 1, 8)):
        if a[pos[0]][i] in whites or a[pos[0]][i] in list(blacks[j] for j in [1, 3, 4, 5]):
            break
        elif a[pos[0]][i] == 'q' or a[pos[0]][i] == 'r':
            check = True
            break
    for i in list(reversed(range(pos[0]))):
        if a[i][pos[1]] in whites or a[i][pos[1]] in list(blacks[j] for j in [1, 3, 4, 5]):
            break
        elif a[i][pos[1]] == 'q' or a[i][pos[1]] == 'r':
            check = True
            break
    for i in list(range(pos[0] + 1, 8)):
        if a[i][pos[1]] in whites or a[i][pos[1]] in list(blacks[j] for j in [1, 3, 4, 5]):
            break
        elif a[i][pos[1]] == 'q' or a[i][pos[1]] == 'r':
            check = True
            break
    r = 8 - np.maximum(abs(pos[0]), abs(pos[1]))
    r1 = np.minimum(abs(pos[0]), abs(pos[1]))
    r2 = np.minimum(r, r1)
    for i in list(range(1, r)):
        if a[pos[0] + i][pos[1] + i] in whites or a[pos[0] + i][pos[1] + i] in list(blacks[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] + i][pos[1] + i] == 'b' or a[pos[0] + i][pos[1] + i] == 'q':
            check = True
            break
    for i in list(range(1, r2)):
        if a[pos[0] - i][pos[1] + i] in whites or a[pos[0] - i][pos[1] + i] in list(blacks[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] - i][pos[1] + i] == 'b' or a[pos[0] - i][pos[1] + i] == 'q':
            check = True
            break
    for i in list(range(1, r2)):
        if a[pos[0] - i][pos[1] - i] in whites or a[pos[0] - i][pos[1] - i] in list(blacks[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] - i][pos[1] - i] == 'b' or a[pos[0] - i][pos[1] - i] == 'q':
            check = True
            break
    for i in list(range(1, r)):
        if a[pos[0] + i][pos[1] - i] in whites or a[pos[0] + i][pos[1] - i] in list(blacks[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] + i][pos[1] - i] == 'b' or a[pos[0] + i][pos[1] - i] == 'q':
            check = True
            break
    if pos[0] + 2 < 8 and pos[1] - 1 > 0 and a[pos[0] + 2][pos[1] - 1] == 'n':
        check = True
    if pos[0] + 2 < 8 and pos[1] + 1 < 8 and a[pos[0] + 2][pos[1] + 1] == 'n':
        check = True
    if pos[0] - 2 > 0 and pos[1] - 1 > 0 and a[pos[0] - 2][pos[1] - 1] == 'n':
        check = True
    if pos[0] - 2 > 0 and pos[1] + 1 < 8 and a[pos[0] - 2][pos[1] + 1] == 'n':
        check = True
    if pos[0] + 1 < 8 and pos[1] - 2 > 0 and a[pos[0] + 1][pos[1] - 2] == 'n':
        check = True
    if pos[0] + 1 < 8 and pos[1] + 2 < 8 and a[pos[0] + 1][pos[1] + 2] == 'n':
        check = True
    if pos[0] - 1 > 0 and pos[1] - 2 > 0 and a[pos[0] - 1][pos[1] - 2] == 'n':
        check = True
    if pos[0] - 1 > 0 and pos[1] + 2 < 8 and a[pos[0] - 1][pos[1] + 2] == 'n':
        check = True
    if pos[0] - 1 > 0 and pos[1] - 1 > 0 and a[pos[0] - 1][pos[1] - 1] == 'p':
        check = True
    if pos[0] - 1 > 0 and pos[1] + 1 < 8 and a[pos[0] - 1][pos[1] + 1] == 'p':
        check = True
    return check


def checkcheckB(pos):
    def make_matrix(board):  # type(board) == chess.Board()
        pgn = board
        foo = []  # Final board
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        for row in rows:
            foo2 = []  # This is the row I make
            for thing in row:
                if thing.isdigit():
                    for i in range(0, int(thing)):
                        foo2.append('.')
                else:
                    foo2.append(thing)
            foo.append(foo2)
        return foo

    whites = ['Q', 'K', 'R', 'B', 'N', 'P']
    blacks = ['q', 'k', 'r', 'b', 'n', 'p']
    a = make_matrix(pos)
    check = False
    pos = []
    for i in range(8):
        for j in range(8):
            if a[i][j] == 'k':
                pos.append(i)
                pos.append(j)
    for i in list(reversed(range(pos[1]))):
        if a[pos[0]][i] in blacks or a[pos[0]][i] in list(whites[j] for j in [1, 3, 4, 5]):
            break
        elif a[pos[0]][i] == 'Q' or a[pos[0]][i] == 'R':
            check = True
            break
    for i in list(range(pos[1] + 1, 8)):
        if a[pos[0]][i] in blacks or a[pos[0]][i] in list(whites[j] for j in [1, 3, 4, 5]):
            break
        elif a[pos[0]][i] == 'Q' or a[pos[0]][i] == 'R':
            check = True
            break
    for i in list(reversed(range(pos[0]))):
        if a[i][pos[1]] in blacks or a[i][pos[1]] in list(whites[j] for j in [1, 3, 4, 5]):
            break
        elif a[i][pos[1]] == 'Q' or a[i][pos[1]] == 'R':
            check = True
            break
    for i in list(range(pos[0] + 1, 8)):
        if a[i][pos[1]] in blacks or a[i][pos[1]] in list(whites[j] for j in [1, 3, 4, 5]):
            break
        elif a[i][pos[1]] == 'Q' or a[i][pos[1]] == 'R':
            check = True
            break
    r = 8 - np.maximum(abs(pos[0]), abs(pos[1]))
    r1 = np.minimum(abs(pos[0]), abs(pos[1]))
    r2 = np.minimum(r, r1)
    for i in list(range(1, r)):
        if a[pos[0] + i][pos[1] + i] in blacks or a[pos[0] + i][pos[1] + i] in list(whites[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] + i][pos[1] + i] == 'B' or a[pos[0] + i][pos[1] + i] == 'Q':
            check = True
            break
    for i in list(range(1, r2)):
        if a[pos[0] - i][pos[1] + i] in blacks or a[pos[0] - i][pos[1] + i] in list(whites[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] - i][pos[1] + i] == 'B' or a[pos[0] - i][pos[1] + i] == 'Q':
            check = True
            break
    for i in list(range(1, r2)):
        if a[pos[0] - i][pos[1] - i] in blacks or a[pos[0] - i][pos[1] - i] in list(whites[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] - i][pos[1] - i] == 'B' or a[pos[0] - i][pos[1] - i] == 'Q':
            check = True
            break
    for i in list(range(1, r)):
        if a[pos[0] + i][pos[1] - i] in blacks or a[pos[0] + i][pos[1] - i] in list(whites[j] for j in [1, 2, 4, 5]):
            break
        elif a[pos[0] + i][pos[1] - i] == 'B' or a[pos[0] + i][pos[1] - i] == 'Q':
            check = True
            break
    if pos[0] + 2 < 8 and pos[1] - 1 > 0 and a[pos[0] + 2][pos[1] - 1] == 'N':
        check = True
    if pos[0] + 2 < 8 and pos[1] + 1 < 8 and a[pos[0] + 2][pos[1] + 1] == 'N':
        check = True
    if pos[0] - 2 > 0 and pos[1] - 1 > 0 and a[pos[0] - 2][pos[1] - 1] == 'N':
        check = True
    if pos[0] - 2 > 0 and pos[1] + 1 < 8 and a[pos[0] - 2][pos[1] + 1] == 'N':
        check = True
    if pos[0] + 1 < 8 and pos[1] - 2 > 0 and a[pos[0] + 1][pos[1] - 2] == 'N':
        check = True
    if pos[0] + 1 < 8 and pos[1] + 2 < 8 and a[pos[0] + 1][pos[1] + 2] == 'N':
        check = True
    if pos[0] - 1 > 0 and pos[1] - 2 > 0 and a[pos[0] - 1][pos[1] - 2] == 'N':
        check = True
    if pos[0] - 1 > 0 and pos[1] + 2 < 8 and a[pos[0] - 1][pos[1] + 2] == 'N':
        check = True
    if pos[0] - 1 > 0 and pos[1] - 1 > 0 and a[pos[0] - 1][pos[1] - 1] == 'P':
        check = True
    if pos[0] - 1 > 0 and pos[1] + 1 < 8 and a[pos[0] - 1][pos[1] + 1] == 'P':
        check = True
    return check


def bothcheck(pos):
    both = False
    if checkcheckB(pos) == True and checkcheckW(pos) == True:
        both = True
    return both


def main(dataset, elo, depth, type):
    pos1 = []
    pos2 = []
    results1 = []
    results2 = []
    start = timer()
    # MR = []
    index = []
    if type == 0:
        for i in range(len(dataset)):
            if bothcheck(dataset[i]) == False and bothcheck(sim_mirror(dataset[i])) == False:
                print(i)
                try:
                    ev1 = evaluation(dataset[i], elo, depth)[0]
                    ev2 = evaluation(sim_mirror(dataset[i]), elo, depth)[0]
                    # MR.append(MR_mirror(evaluation(dataset[i],elo,depth)[0],evaluation(sim_mirror(dataset[i]), elo, depth)[0], delta, epsilon)) ## adding MR to dataframe
                    pos1.append(dataset[i])  ## adding original position to dataframe
                    pos2.append(sim_mirror(dataset[i]))  ## adding transformed position to dataframe
                    results1.append(ev1)  ## adding evaluation of original position to dataframe
                    results2.append(ev2)  ## adding evaluation of transformed position to dataframe
                    index.append(i)  ## adding index to dataframe
                except StockfishException:
                    pass
    elif type == 1:
        for i in range(len(dataset)):
            if bothcheck(dataset[i]) == False and bothcheck(sim_axis(dataset[i])) == False:
                print(i)
                try:
                    ev1 = evaluation(dataset[i], elo, depth)[0]
                    ev2 = evaluation(sim_axis(dataset[i]), elo, depth)[0]
                    # MR.append(MR_equi(ev1,ev2, delta, epsilon)) ## adding MR to dataframe
                    pos1.append(dataset[i])  ## adding original position to dataframe
                    pos2.append(sim_axis(dataset[i]))  ## adding transformed position to dataframe
                    results1.append(ev1)  ## adding evaluation of original position to dataframe
                    results2.append(ev2)  ## adding evaluation of transformed position to dataframe
                    index.append(i)  ## adding index to dataframe
                except StockfishException:
                    pass
    elif type == 2:
        for i in range(len(dataset)):
            if bothcheck(dataset[i]) == False and bothcheck(sim_diag(dataset[i])) == False and 'p' not in dataset[
                i] and 'P' not in dataset[i]:
                print(i)
                try:
                    ev1 = evaluation(dataset[i], elo, depth)[0]
                    ev2 = evaluation(sim_diag(dataset[i]), elo, depth)[0]
                    # MR.append(MR_equi(evaluation(dataset[i],elo,depth)[0],evaluation(sim_diag(dataset[i]), elo, depth)[0], delta, epsilon)) ## adding MR to dataframe
                    pos1.append(dataset[i])  ## adding original position to dataframe
                    pos2.append(sim_diag(dataset[i]))  ## adding transformed position to dataframe
                    results1.append(ev1)  ## adding evaluation of original position to dataframe
                    results2.append(ev2)  ## adding evaluation of transformed position to dataframe
                    index.append(i)  ## adding index to dataframe
                except StockfishException:
                    pass
    elif type == 3:
        for i in range(len(dataset)):
            if bothcheck(dataset[i]) == False and bothcheck(replace(dataset[i])) == False and (
                    'B' in dataset[i] or 'R' in dataset[i]):
                print(i)
                try:
                    ev1 = evaluation(dataset[i], elo, depth)[0]
                    ev2 = evaluation(replace(dataset[i]), elo, depth)[0]
                    # MR.append(MR_better(evaluation(dataset[i],elo,depth)[0],evaluation(replace(dataset[i]), elo, depth)[0], delta, epsilon)) ## adding MR to dataframe
                    pos1.append(dataset[i])  ## adding original position to dataframe
                    pos2.append(replace(dataset[i]))  ## adding transformed position to dataframe
                    results1.append(ev1)  ## adding evaluation of original position to dataframe
                    results2.append(ev2)  ## adding evaluation of transformed position to dataframe
                    index.append(i)  ## adding index to dataframe
                except StockfishException:
                    pass

    elif type == 4:
        for i in range(len(dataset)):
            if bothcheck(dataset[i]) == False and bothcheck(sim_axis(dataset[i])) == False:
                print(i)
                try:
                    ev1 = evaluation(dataset[i], elo, depth)[1]
                    ev2 = evaluation(sim_axis(dataset[i]), elo, depth)[1]
                    # ajout d'un len à la fin parce que souci si on essaie de faire jouer un coup sur une position où ça n'est pas possible (mat, pat)
                    if len(ev1) != 0 and len(ev2) != 0:
                        # MR.append(MR_first(evaluation(dataset[i],elo,depth)[1],evaluation(sim_axis(dataset[i]), elo, depth)[1], delta, epsilon)) ## adding MR to dataframe
                        pos1.append(dataset[i])  ## adding original position to dataframe
                        pos2.append(sim_axis(dataset[i]))  ## adding transformed position to dataframe
                        results1.append(ev1)  ## adding evaluation of original position to dataframe
                        results2.append(ev2)  ## adding evaluation of transformed position to dataframe
                        index.append(i)  ## adding index to dataframe
                except StockfishException:
                    pass

    d = {'index': index, 'pos1': pos1, 'evaluation pos1': results1, 'pos2': pos2, 'evaluation pos2': results2}
    # d = {'index': index, 'pos1': pos1, 'evaluation pos1': results1, 'pos2': pos2, 'evaluation pos2': results2,
    #      'MR': MR}
    df = pd.DataFrame(data=d)  ## creating dataframe

    # print(timer()-start)
    return df


def MR(df, type, delta, epsilon):
    ev1s = df["evaluation pos1"]
    ev2s = df["evaluation pos2"]
    MRs = []
    if type == 0:
        for k in range(len(ev1s)):
            MRs.append(MR_mirror(ev1s[k], ev2s[k], delta, epsilon))
    if type == 1 or type == 2:
        for k in range(len(ev1s)):
            MRs.append(MR_equi(ev1s[k], ev2s[k], delta, epsilon))
    if type == 3:
        for k in range(len(ev1s)):
            MRs.append(MR_better(ev1s[k], ev2s[k], delta, epsilon))
    if type == 4:
        for k in range(len(ev1s)):
            MRs.append(MR_first(ev1s[k], ev2s[k], delta, epsilon))
    falses = trues = 0
    for k in range(len(MRs)):
        if MRs[k]:
            trues += 1
        else:
            falses += 1
    print("MR valides : ", trues, "; MR invalides : ", falses, "; Ratio : ", trues / (trues + falses) * 100, "%")
    return trues / (trues + falses) * 100

def tests(evas, type, d):
    results_grid = []

    # Parcourir les valeurs de deltas
    for delta in deltas:
        # Initialiser une liste pour stocker les résultats pour chaque delta
        delta_results = []

        # Parcourir les valeurs de epsilons
        for epsilon in epsilons:
            # Appeler la fonction MR avec les paramètres actuels
            result = 100 - MR(evas, type, delta, epsilon)

            # Ajouter le résultat à la liste des résultats pour ce delta
            delta_results.append(result)

        # Ajouter la liste des résultats pour ce delta à la grille des résultats
        results_grid.append(delta_results)

    # Afficher la grille des résultats
    df_results = pd.DataFrame(results_grid, index=deltas, columns=epsilons)

    # Utiliser seaborn pour créer un heatmap avec un dégradé de couleurs
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_results, cmap='RdYlGn_r', annot=True, fmt=".2f", cbar_kws={'label': 'Valeurs'})
    typent = ["sim_mirror", "sim_axis", "sim_diag", "replace", "best_move"][type]

    label = 'Failure rate, ' + typent + ', depth=' + str(d)

    # Ajouter des titres et labels
    plt.title(label)
    plt.xlabel('Epsilons')
    plt.ylabel('Deltas')

    name = typent + '_d=' + str(d)
    if not os.path.exists(name):
        os.makedirs(name)
    figname = "fig"+generate_random_key()
    chemin = os.path.join(name, figname)

    plt.savefig(chemin)


def getFalses(evas, type, delta, epsilon):
    pos1s = evas["pos1"]
    pos2s = evas["pos2"]
    ev1s = evas["evaluation pos1"]
    ev2s = evas["evaluation pos2"]
    falsespos1s = []
    falsespos2s = []
    falsesev1s = []
    falsesev2s = []
    indexy = []
    index = 0
    if type == 0:
        for k in range(len(ev1s)):
            if not MR_mirror(ev1s[k], ev2s[k], delta, epsilon):
                indexy.append(index)
                index+=1
                falsespos1s.append(pos1s[k])
                falsespos2s.append(pos2s[k])
                falsesev1s.append(ev1s[k])
                falsesev2s.append(ev2s[k])
    if type == 1 or type == 2:
        for k in range(len(ev1s)):
            if not MR_equi(ev1s[k], ev2s[k], delta, epsilon):
                indexy.append(index)
                index+=1
                falsespos1s.append(pos1s[k])
                falsespos2s.append(pos2s[k])
                falsesev1s.append(ev1s[k])
                falsesev2s.append(ev2s[k])
    if type == 3:
        for k in range(len(ev1s)):
            if not MR_better(ev1s[k], ev2s[k], delta, epsilon):
                indexy.append(index)
                index+=1
                falsespos1s.append(pos1s[k])
                falsespos2s.append(pos2s[k])
                falsesev1s.append(ev1s[k])
                falsesev2s.append(ev2s[k])
    if type == 4:
        for k in range(len(ev1s)):
            if not MR_first(ev1s[k], ev2s[k], delta, epsilon):
                indexy.append(index)
                index+=1
                falsespos1s.append(pos1s[k])
                falsespos2s.append(pos2s[k])
                falsesev1s.append(ev1s[k])
                falsesev2s.append(ev2s[k])

    d = {'index': indexy, 'pos1': falsespos1s, 'evaluation pos1': falsesev1s, 'pos2': falsespos2s, 'evaluation pos2': falsesev2s}

    df = pd.DataFrame(data=d)
    return df




def mergedf(dossier_pickle, nouveau_fichier_pickle):
    dataframes_a_fusionner = []

    for fichier in os.listdir(dossier_pickle):
        if fichier.startswith("fr") and fichier.endswith(".pkl"):
            chemin_fichier = os.path.join(dossier_pickle, fichier)
            dataframe = pd.read_pickle(chemin_fichier)
            dataframes_a_fusionner.append(dataframe)

    if dataframes_a_fusionner:
        resultat_fusion = pd.concat(dataframes_a_fusionner, ignore_index=True)
        resultat_fusion.to_pickle(nouveau_fichier_pickle)
        print(f"La fusion a été enregistrée dans {nouveau_fichier_pickle}")
    else:
        print(f"Aucun fichier correspondant trouvé dans {dossier_pickle}.")


def process_data(start_index, end_index, elo, depth, type):
    if len(dataset) <= end_index:
        end_index = len(dataset) - 1
    chunk_data = main(dataset[start_index:end_index], elo, depth, type)

    typent = ["sim_mirror", "sim_axis", "sim_diag", "replace", "best_move"][type]
    name = "fr_" + str(elo)+ typent + '_d=' + str(depth) + "_" + str(start_index) + "_to_" + str(end_index) + ".pkl"

    subfolder = typent + "_d=" + str(depth)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    chemin = os.path.join(subfolder, name)

    with open(chemin, 'wb') as fichier:
        pickle.dump(chunk_data, fichier)

    return chunk_data


def main2(size, threads, elo, depth, type):
    start = timer()
    chunk_size = size // threads
    num_chunks = threads

    with ThreadPoolExecutor() as executor:
        futures = []

        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            futures.append(executor.submit(process_data, start_index, end_index, elo, depth, type))

        results = [future.result() for future in futures]

    merged_dataset = merge_datasets(results)
    print(timer() - start)
    return merged_dataset

    # Reste du traitement avec le jeu de données fusionné


def merge_datasets(dataset_chunks):
    merged_data = pd.DataFrame()
    for chunk_data in dataset_chunks:
        print(chunk_data)
        merged_data = pd.concat([merged_data, chunk_data], ignore_index=True)

    return merged_data


deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
epsilons = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]




depth = 20
type = 1
elo = 50000

print('amogus')
# types:
# 0 : mirror,
# 1 : sim_axis,
# 2 : sim_diag,
# 3 : better,
# 4 : first


evaluations = main2(1000, 8, elo, depth, type)
tests(evaluations, type, depth)

typent = ["sim_mirror", "sim_axis", "sim_diag", "replace", "best_move"][type]
name = "ev_"  + nament.removesuffix('.txt') + '_'+ typent + '_d_' + str(depth) + ".pkl"

subfolder = "reals"
if not os.path.exists(subfolder):
     os.makedirs(subfolder)

chemin = os.path.join(subfolder, name)

with open(chemin, 'wb') as fichier:
    pickle.dump(evaluations, fichier)

# with open(chemin, 'rb') as fichier:
#     evaluations_chargees = pickle.load(fichier)





