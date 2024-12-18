from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import re
import json
import tqdm


def stockfish_best_move(fen_position: str, depth: int, suffix: str = ""):
    # Start Stockfish engine as a subprocess
    stockfish = subprocess.Popen(f"./stockfish{suffix}",  # Path to the Stockfish executable
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    # Send the 'uci' command to initialize the engine
    stockfish.stdin.write("uci\n")
    stockfish.stdin.flush()

    # Wait for Stockfish to be ready
    while True:
        line = stockfish.stdout.readline().strip()
        if line == "uciok":
            break

    # stockfish.stdin.write("setoption name Skill Level value 2000")
    # stockfish.stdin.flush()

    # Load FEN position
    stockfish.stdin.write(f"position fen {fen_position}\n")
    stockfish.stdin.flush()

    # Go to specified depth
    stockfish.stdin.write(f"go depth {depth}\n")
    stockfish.stdin.flush()

    # Wait for Stockfish to provide the best move
    best_move = None
    evaluation = None
    n = 0
    last_alpha = ""
    while n < 1000:
        line = stockfish.stdout.readline().strip()
        # print(line)
        if line.startswith('bestmove'):
            best_move = line.split(' ')[1]
            break
        elif line.startswith('info depth'):
            match = re.search(r'score (cp|mate) (-?\d+)', line)
            if match:
                score_type = match.group(1)
                score_value = int(match.group(2))
                if score_type == 'cp':
                    evaluation = score_value / 100.0  # Convert centipawns to pawns
                else:
                    # Mate score; convert to signed value
                    evaluation = -100.0 if score_value > 0 else 100.0
                # print("set score:", line)
            # break
        elif "alpha" in line:
            last_alpha = line
        n += 1
    # if last_alpha:
    #     print("[DEBUG]\t", last_alpha)

    # Close the Stockfish process
    stockfish.terminate()

    return best_move, evaluation



def stockfish_moves(fen_position: str, suffix = None):
    # Start Stockfish engine as a subprocess
    stockfish = subprocess.Popen(
        "./stockfish_same" if suffix is None else f"./stockfish_{suffix}",  # Path to the Stockfish executable
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    # Send the 'uci' command to initialize the engine
    stockfish.stdin.write("uci\n")
    stockfish.stdin.flush()

    # Wait for Stockfish to be ready
    while True:
        line = stockfish.stdout.readline().strip()
        if line == "uciok":
            break

    # Load FEN position
    stockfish.stdin.write(f"position fen {fen_position}\n")
    stockfish.stdin.flush()

    # Go to specified depth
    stockfish.stdin.write(f"go depth 2\n")
    stockfish.stdin.flush()

    # Wait for Stockfish to provide the best move
    n = 0
    positions = []
    while n < 1000:
        line = stockfish.stdout.readline().strip()
        if line.startswith('moveList[i]:'):
            pos = line.split(':')[1]
            if len(pos.strip(" \n")) > 3 and pos not in positions:
                positions.append(pos)
        elif line.startswith('end move list'):
            break
        n += 1

    # Close the Stockfish process
    stockfish.terminate()

    return positions


def sim_axis(pos):
    pos_transformed = []
    col = pos[-1:]
    aux1 = pos[:-2]
    aux2 = aux1.split("/")
    aux3 = []
    for j in range(8):
        # if " w " in aux2[j]:
        #     el = aux2[j].split(" w ")
        #     aux3.append(el[0][::-1])
        #     aux3.append(("w " + el[1])[::-1])
        # else:
        aux3.append(aux2[j][::-1])
    aux4 = "/".join(aux3)
    # print("DEBUG:", aux1, "\n", aux2, "\n", aux3, "\n", aux4)
    pos_transformed.append(aux4)
    if col == "w":
        pos_transformed = pos_transformed[0] + " w"
    elif col == "b":
        pos_transformed = pos_transformed[0] + " b"
    return pos_transformed


def sim_diag(pos):
    def Convert(string):
        list1 = []
        list1[:0] = string
        return list1

    pos_transformed = []
    colour = pos[-1:]
    aux = (
        pos.replace("8", "11111111")
        .replace("7", "1111111")
        .replace("6", "111111")
        .replace("5", "11111")
        .replace("4", "1111")
        .replace("3", "111")
        .replace("2", "11")
    )
    aux2 = aux.split("/")
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
    if colour == "b":
        pos_transformed = pos_transformed + " b"
    else:
        pos_transformed = pos_transformed + " w"
    pos_transformed = (
        pos_transformed.replace("11111111", "8")
        .replace("1111111", "7")
        .replace("111111", "6")
        .replace("11111", "5")
        .replace("1111", "4")
        .replace("111", "3")
        .replace("11", "2")
    )

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
        aux5 = "/".join(aux4)
    pos_transformed.append(aux5)
    if colour == "b":
        pos_transformed = pos_transformed[0] + " w"
    else:
        pos_transformed = pos_transformed[0] + " b"
    return pos_transformed


# Example usage


def mirror_move(move: str, axis: str) -> str:
    """
    Mirrors a chess move along a specified axis.

    Parameters:
        move (str): The original chess move in algebraic notation (e.g., 'e2e4').
        axis (str): The axis along which to mirror the move. Should be 'vertical' or 'horizontal'.

    Returns:
        str: The mirrored chess move.
    """
    # Define the mapping of letters for horizontal and vertical axes
    horizontal_axis_map = {
        "a": "h",
        "b": "g",
        "c": "f",
        "d": "e",
        "e": "d",
        "f": "c",
        "g": "b",
        "h": "a",
    }
    vertical_axis_map = {
        "1": "8",
        "2": "7",
        "3": "6",
        "4": "5",
        "5": "4",
        "6": "3",
        "7": "2",
        "8": "1",
    }

    # Validate the axis input
    if axis not in ["vertical", "horizontal"]:
        raise ValueError("Axis should be 'vertical' or 'horizontal'.")

    # Split the move into its components (source square and destination square)
    source_square, dest_square = move[:2], move[2:]

    # Mirror the move along the specified axis
    if axis == "horizontal":
        mirrored_source_square = (
            horizontal_axis_map.get(source_square[0], source_square[0])
            + source_square[1]
        )
        mirrored_dest_square = (
            horizontal_axis_map.get(dest_square[0], dest_square[0]) + dest_square[1]
        )
    else:  # Vertical axis
        mirrored_source_square = source_square[0] + vertical_axis_map.get(
            source_square[1], source_square[1]
        )
        mirrored_dest_square = dest_square[0] + vertical_axis_map.get(
            dest_square[1], dest_square[1]
        )

    # Combine the mirrored squares to form the mirrored move
    mirrored_move = mirrored_source_square + mirrored_dest_square

    return mirrored_move

def mirror_move_diagonal(move: str, axis: str) -> str:
    """
    Mirrors a chess move along a specified diagonal axis.

    Parameters:
        move (str): The original chess move in algebraic notation (e.g., 'e2e4').
        axis (str): The diagonal axis along which to mirror the move. Should be 'main' or 'anti'. (ANTI is not fully supported)

    Returns:
        str: The mirrored chess move.
    """
    # Define the mapping of letters for the diagonal axes
    main_diagonal_map = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8', '1': 'a', '2': 'b', '3': 'c', '4': 'd', '5': 'e', '6': 'f', '7': 'g', '8': 'h'}
    anti_diagonal_map = {'a': '8', 'b': '7', 'c': '6', 'd': '5', 'e': '4', 'f': '3', 'g': '2', 'h': '1'}

    # Validate the axis input
    if axis not in ['main', 'anti']:
        raise ValueError("Axis should be 'main' or 'anti'.")

    # Split the move into its components (source square and destination square)
    source_square, dest_square = move[:2], move[2:]

    # Mirror the move along the specified diagonal axis
    if axis == 'main':
        mirrored_source_square = main_diagonal_map.get(source_square[1], source_square[1]) + main_diagonal_map.get(source_square[0], source_square[0])
        mirrored_dest_square = main_diagonal_map.get(dest_square[1], dest_square[1]) + main_diagonal_map.get(dest_square[0], dest_square[0])
    else:  # 'anti' diagonal axis
        mirrored_source_square = anti_diagonal_map.get(source_square[1], source_square[1]) + anti_diagonal_map.get(source_square[0], source_square[0])
        mirrored_dest_square = anti_diagonal_map.get(dest_square[1], dest_square[1]) + anti_diagonal_map.get(dest_square[0], dest_square[0])

    # Combine the mirrored squares to form the mirrored move
    mirrored_move = mirrored_source_square + mirrored_dest_square

    return mirrored_move
    


def compare(key, depth, evals):
    scores = [stockfish_best_move(fen, depth, va)[1] for fen, va in evals]
    sa = abs(scores[0] - scores[1])
    sb = abs(scores[0] - scores[2])
    return key, sa, sb

if __name__ == "__main__":
    import pickle, os, sys
    
    # path = os.path.join('src', 'chessrepro-main', 'reals','ev_lichess2000-2500-10_nocastlingsim_mirror_d_20.pkl')
    # path = os.path.join('src', 'chessrepro-main', 'sim_mirror_d=10','evaluations50000_sim_mirror_d_10.pkl')
    is_mirror = len(sys.argv) == 4 or int(sys.argv[4]) != 1
    # print(sim_mirror("1b2n3/p1p5/8/5K2/P7/5k2/8/8 w"))
    # assert False
    path = sys.argv[3]
    with open(path, 'rb') as file:
        data = pickle.load(file)
    fens = data["pos1"]

    depth = sys.argv[1]

    diffs = {"depth": sys.argv[1], "classic": [], "invariant": [], "relative": [], "fens": [], "source": path, "is_mirror": is_mirror}
    try:
        with open(f"./stockfish_{depth}.json") as fd:
            dico = json.load(fd)
            if "fens" in dico:
                diffs = dico
    except:
        pass

    fens = [x for x in fens if x not in diffs["fens"]]

    def savedata():
        with open(f"./stockfish_{depth}.json", "w") as fd:
            json.dump(diffs, fd)

    import atexit

    atexit.register(savedata)

    sym_name = "_mirror" if is_mirror else "_axis"
    print("symmetry:", sym_name)

    pool = ProcessPoolExecutor(int(sys.argv[2]))
    futures = []
    pbar = tqdm.tqdm(total=len(fens), smoothing=0)
    for fen in fens:
        sym_fen = sim_mirror(fen) if is_mirror else sim_axis(fen)
        futures.append(pool.submit(compare, fen, depth, ((fen, "_sorted"), (sym_fen, "_sorted"), (sym_fen, sym_name))))
    for future in as_completed(futures):
        fen, sclassic, sinvariant = future.result()

        diffs["classic"].append(sclassic)
        diffs["invariant"].append(sinvariant)
        diffs["relative"].append((sclassic - sinvariant) / max(sclassic, 1e-3))
        diffs["fens"].append(fen)
        pbar.update(1)
    savedata()
    pool.shutdown()
    pbar.close()
