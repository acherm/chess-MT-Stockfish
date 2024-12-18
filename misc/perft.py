import subprocess
from time import time

import graphviz




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

#boucle qui redirige les résultats de perft sur les positions du dataset dans des txt
def perft(output_file1,output_file2):
    positions_file = 'valid_positions.txt'

    with open(positions_file, 'r') as f:
        positions = f.readlines()[0:10000]



    results1 = []
    results2 = []
    d=0
    for pos in positions:
        print(d, end='', flush=True)
        print('\r', end='', flush=True)
        d+=1
        command = f"position fen {pos.strip()}"
        result = run_stockfish([command, "go perft 3"])
        results1.append("Position : "+ pos.strip())
        results1.append(result)
        command = f"position fen {sim_axis(pos.strip()).strip()}"
        result = run_stockfish([command, "go perft 3"])
        results2.append("Position : " + sim_axis(pos.strip()))
        results2.append(result)


    with open(output_file1, 'w') as f:
        for result in results1:
            f.write(result + '\n')
    with open(output_file2, 'w') as f:
        for result in results2:
            f.write(result + '\n')

#comparaison de si les txt sont identiques
def comparer_fichiers(file1, file2):
    lignes_diff = []

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        count=0
        for num_ligne, (ligne1, ligne2) in enumerate(zip(f1, f2), start=1):
            if "Nodes searched: " in ligne1 and "Nodes searched: " in ligne2:
                count+=1
                nombre1 = int(ligne1.split("Nodes searched: ")[1].strip())
                nombre2 = int(ligne2.split("Nodes searched: ")[1].strip())

                if nombre1 != nombre2:
                    lignes_diff.append(num_ligne)
    print(count)
    return lignes_diff

#loop perft sur une même position jusqu'à observer une différence (qui n'arrive jamais)
def perft1pos(pos):
    command = f"position fen {pos.strip()}"
    result = run_stockfish([command, "go perft 3"])
    command = f"position fen {pos.strip()}"
    result2 = run_stockfish([command, "go perft 3"])
    t=0
    while result2 == result:
        t+=1
        print(t, end='', flush=True)
        print('\r', end='', flush=True)
        result2 = run_stockfish([command, "go perft 3"])
    return result,result2


#go depth
def depth1pos(pos,pv,d):
    command = [f"position fen {pos.strip()}", f"setoption name MultiPV value {pv}", f"go depth {d}", f"go depth {d}"]
    result = run_stockfish(command)
    return result


#compare les résultats de go depth en ignorant les données inutiles comme les nps (nodes per seconds)
def compare_strings(str1, str2):
    lines1 = str1.split('\n')
    lines2 = str2.split('\n')

    parts1 = []
    parts2 = []

    for line in lines1:
        if " pv" in line and "nps" in line:
            parts1.append(line[:line.index("nps")].strip())
            parts1.append(line[line.index(" pv"):].strip())
        else:
            parts1.append(line.strip())

    for line in lines2:
        if " pv" in line and "nps" in line:
            parts2.append(line[:line.index("nps")].strip())
            parts2.append(line[line.index(" pv"):].strip())
        else:
            parts2.append(line.strip())
    for part1, part2 in zip(parts1, parts2):
        if part1 != part2:
            print(part1,'\n',part2)
            return False

    return True


# loop perft sur une même position jusqu'à observer une différence
# (qui n'arrive jamais, c'était pour vérifier que SF est bien déterministe)
def compare_depth1pos(pos):
    command = [f"position fen {pos.strip()}","setoption name MultiPV value 5", "go depth 3","go depth 3"]
    result = run_stockfish(command)
    result2 = run_stockfish(command)
    t=0
    while compare_strings(result,result2):
        t+=1
        print(t, end='', flush=True)
        print('\r', end='', flush=True)
        result2 = run_stockfish(command)
    return result,result2



# transforme le résultat d'un go depth en liste de noeuds
def nodes(str,max_d):
    lines = str.split('\n')
    nodes = []
    nodes_ev = []
    for d in range(max_d+1):
        for line in lines:
            if f" depth {d}" in line and "currmove" not in line:
                if 'upperbound' not in line and 'lowerbound' not in line:
                    if "cp" in line:
                        ev = ' '+(line[line.index("cp")+3:line.index("nodes")].strip())
                    else:
                        ev = ' #'+(line[line.index("mate") + 5:line.index("nodes")].strip())
                elif 'upperbound' in line:
                    if "cp" in line:
                        ev = ' '+(line[line.index("cp")+3:line.index("upperbound")].strip())
                    else:
                        ev = ' #'+(line[line.index("mate") + 5:line.index("upperbound")].strip())
                elif 'lowerbound' in line:
                    if "cp" in line:
                        ev = ' '+(line[line.index("cp")+3:line.index("lowerbound")].strip())
                    else:
                        ev = ' #'+(line[line.index("mate") + 5:line.index("lowerbound")].strip())
                moves = (line[line.index(" pv")+4:].strip()).split(' ')
                moves[-1]+=ev
                nodes.append(moves)
                nodes_ev.append(ev)
    return nodes, nodes_ev


# transforme une liste de noeuds en arbre sous forme de dictionnaire
def build_tree(data):
    tree = {}
    root_node = "root"
    for sublist in data:
        current_node = tree.setdefault(root_node, {})
        for move in sublist:
            if move not in current_node:
                current_node[move] = {}
            current_node = current_node[move]
    return tree

# pour regarder un arbre dans la console
def visualize(root, indent=0):
    if type(root) == dict:
        for k, v in root.items():
            print(" "*indent + f"{k}:")
            visualize(v, indent+2)
    else:
        print(" "*indent + repr(root))

# dessine deux noeuds et l'arrête qui les relie
def draw(parent_name, child_name,d,ev=None):
    eva = ''
    if ev and '#' in ev:
        eva+=ev
    if ev and '#' not in ev:
        eva+=ev
        eva=str(int(eva)/100)
    if not node_exists(graph,parent_name+str(d)):
        graph.node(parent_name+str(d),label=parent_name[0:4])
    if not node_exists(graph, child_name + str(d+1)) or eva!= '':
        graph.node(child_name + str(d+1), label=eva)
    if not edge_exists(graph,parent_name+str(d),child_name + str(d+1)):
        graph.edge(parent_name+str(d),child_name + str(d+1),label=child_name[0:4])

# dessiner l'arbre en entier
def plot_tree(root, parent=None, d=0):
    if type(root) == dict:
        for k, v in root.items():
            ev=None
            if len(k)>4:
                k,ev=k[0:4],k[5:]
            if isinstance(v,dict):
                if parent:
                    if ev!=None:
                        draw(parent,k+parent,d,ev)
                        plot_tree(v, k+parent, d + 1)
                    else:
                        draw(parent, k + parent, d)
                        plot_tree(v, k + parent, d + 1)
                else:
                    plot_tree(v,k,d+1)

            else:
                draw(parent,k,d)

def edge_exists(graph, source, destination):
    # for existing_edge in graph.get_edges():
    #     if existing_edge.get_source() == source and existing_edge.get_destination() == destination:
    #         return True
    # return False
    for k in graph.body:
        if source in k and destination in k:
            return True
    return False

def node_exists(graph, node_name):
    for k in graph.body:
        if node_name in k:
            return True
    # for existing_node in graph.get_nodes():
    #     if existing_node.get_name() == node_name:
    #         return True
    return False

def run_stockfish(commands):
    stockfish_process = subprocess.Popen(
         ['stockfish_15_x64_avx2'],
        #['stockfish-windows-x86-64-avx2'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )

    # Envoyer les commandes à Stockfish
    for command in commands:
        stockfish_process.stdin.write(command + '\n')

    stockfish_process.stdin.write('quit\n')
    stockfish_process.stdin.flush()  # Assurer que les données sont bien envoyées

    # Récupérer la sortie de Stockfish
    output = stockfish_process.communicate()[0]
    return output




#pos = 'r1b1r1k1/p5pp/2p3q1/2pP1p2/5Bn1/1PNB1K2/P1PQ1PP1/R4R2 b'
pos = '1N5k/3Kb2q/1NP3p1/1R4B1/3R4/1B6/8/n3Q3 w'
pos2 = sim_axis(pos)

pos3 = '3RR3/6k1/K7/8/8/7r/8/8 b'
pos4 = sim_axis(pos3)


depth = 1
mpv = 1

print('En attente de la sortie de Stockfish...', end='', flush=True)
print('\r', end='', flush=True)

a = depth1pos(pos3,mpv,depth)

print('Création des noeuds...', end='', flush=True)
print('\r', end='', flush=True)

nodesent = nodes(a,depth)[0]

print("Création de l'arbre...", end='', flush=True)
print('\r', end='', flush=True)

tree = build_tree(nodesent)

#visualize(tree)

print('Création du graphe...', end='', flush=True)
print('\r', end='', flush=True)

graph = graphviz.Graph()


print('Affichage...', end='', flush=True)
print('\r', end='', flush=True)

plot_tree(tree)

#graph.write_png('example1_graph.png')
print('Rendering...', end='', flush=True)
print('\r', end='', flush=True)


graph.render('graphog', format='png', cleanup=True)

print('', end='', flush=True)