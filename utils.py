pieces_ascii = "KQRBNPkqrbnp"
pieces_unicode = "♚♛♜♝♞♟♔♕♖♗♘♙"
translation_rules = str.maketrans(pieces_ascii, pieces_unicode)


def show_board(board):
    # if not board.turn:
    print(str(board).translate(translation_rules)[::-1])
    # else:
    # print(str(board).translate(translation_rules))
    print()
    print()
