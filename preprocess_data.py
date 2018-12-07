from multiprocessing.dummy import Pool

import chess.uci
import time
from chess.pgn import scan_headers

pieaces = {".": 0,
           "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 10,
           "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -10}


def board_tensor(board):
    res = []
    for c in str(board):
        if c == '\n':
            continue
        elif c == ' ':
            continue
        res.append(pieaces[c] / 10)
    return res


def sample_board(board):
    engine = chess.uci.popen_engine("stockfish")
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)
    engine.position(board)
    engine.go(movetime=50)
    sc, m = info_handler.info["score"][1]
    engine.quit()
    engine.terminate()
    engine.kill()
    sample = board_tensor(board=board)
    if m: sc = 1000
    sc = max([-10, min([10, sc / 100])])
    return sample, sc, 1 if m else 0


pgn = open("data/pgn/KingBase2018-01.pgn")

all_offsets = list(
    offset for offset, headers in scan_headers(pgn))


def sample_game(game):
    board = game.board()
    samples = []
    main_line = list(game.main_line())
    for move in main_line:
        if not board.turn:
            board.push(move)
            continue
        position = board_tensor(board=board)
        samples.append(position)
        board.push(move)
    return samples


games = []
for i, offset in enumerate(all_offsets[:100]):
    pgn.seek(offset)
    game = chess.pgn.read_game(pgn)
    games.append(game)
    print(i, offset)

pool = Pool(5)
step = 10
st = time.time()
with open("data/samples_labels", "w+") as f:
    for chunk in range(0, len(games), step):
        res = pool.map(sample_game, games[chunk:chunk + step])
        print(time.time() - st)
        for game, labels in res:
            f.write("%s:%s\n" % (game, labels))
        print("Done: %s %%" % (100.0 * chunk / i))