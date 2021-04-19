from ImageProssesing import is_image_empty
import numpy as np
import cv2


def init_hog_descripter():
    winSize = (18, 18)
    blockSize = (3, 4)
    blockStride = (1, 2)
    cellSize = (3, 4)
    nbins = 9

    derivAperture = 20
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    return hog


def hog_compute(hog, image):
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (1, 1)
    padding = (3, 3)
    locations = ((8, 8),)
    return hog.compute(image, winStride, padding, locations)


def predict_digit(svm_model, hog, image):
    # image = cv.resize(image, (32, 32))
    # hist = hog.compute(image, winStride, padding, locations)
    hist = hog_compute(hog, image)
    hist = np.reshape(hist, (-1))
    hist = np.matrix([hist]).astype(np.float32)
    retval, results = svm_model.predict(hist)
    return results[0]


def seperate_grid_digits(projected_grid):
    pading = 9
    num_width = 50
    digits_imgs = []
    for i in range(0, 9):
        row = []
        for j in range(0, 9):
            dig = projected_grid[i * num_width + pading:i * num_width + num_width - pading,
                  j * num_width + pading:j * num_width + num_width - pading]
            if not is_image_empty(dig, thresh=20):
                row.append(dig)
            else:
                row.append(None)

        digits_imgs.append(row)
    return digits_imgs


def pridict_digits(digits_imgs, svm_model, hog):
    sudoku_grid = np.zeros((9, 9), np.uint8)
    for i in range(0, 9):
        for j in range(0, 9):
            dig = digits_imgs[i][j]
            if dig is not None:
                sudoku_grid[i, j] = int(predict_digit(svm_model, hog, dig)[0])
    return sudoku_grid


def draw_solved_grid(to_draw_on, old_grid, solved_grid):
    num_width = 50
    pading = 10
    for i in range(0, 9):
        for j in range(0, 9):
            if old_grid[i, j] == 0:
                cv2.putText(to_draw_on, str(solved_grid[i, j]),
                            (j * num_width + pading, i * num_width + pading + int(num_width * 0.5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 5, 5), thickness=2, lineType=cv2.LINE_AA)
    return to_draw_on


def draw_original_grid(to_draw_on, old_grid):
    num_width = 50
    pading = 10
    for i in range(0, 9):
        for j in range(0, 9):
            if (old_grid[i, j] != 0):
                cv2.putText(to_draw_on, str(old_grid[i, j]),
                            (j * num_width + pading + 25, i * num_width + pading + 10 + int(num_width * 0.5)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), thickness=1, lineType=cv2.LINE_4)

    return to_draw_on


def draw_digits(digits):
    image = np.zeros((450, 450), np.uint8)
    image = image * 50
    pading = 9
    num_width = 50
    for i in range(9):
        for j in range(9):
            if digits[i][j] is not None:
                image[i * num_width + pading:i * num_width + num_width - pading,
                j * num_width + pading:j * num_width + num_width - pading] = digits[i][j]
    return image


def measure_different(grid1, grid2):
    m = grid1 - grid2
    d = np.count_nonzero(m)
    return d


# ------------------------------------------------------
class EntryData:
    def __init__(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

    def set_data(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n


# Solve Sudoku using Best-first search
def solve_sudoku(matrix):
    cont = [True]
    # See if it is even possible to have a solution
    for i in range(9):
        for j in range(9):
            if not can_be_correct(matrix, i, j):  # If it is not possible, stop
                return
    sudoku_helper(matrix, cont)  # Otherwise try to solve the Sudoku puzzle


# Helper function - The heart of Best First Search
def sudoku_helper(matrix, cont):
    if not cont[0]:  # Stopping point 1
        return

    # Find the best entry (The one with the least possibilities)
    best_candidate = EntryData(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:  # If it is unfilled
                num_choices = count_choices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)

    # If didn't find any choices, it means...
    if best_candidate.choices == 100:  # Has filled all board, Best-First Search done! Note, whether we have a solution or not depends on whether all Board is non-zero
        cont[0] = False  # Set the flag so that the rest of the recursive calls can stop at "stopping points"
        return

    row = best_candidate.row
    col = best_candidate.col

    # If found the best candidate, try to fill 1-9
    for j in range(1, 10):
        if not cont[0]:  # Stopping point 2
            return

        matrix[row][col] = j

        if can_be_correct(matrix, row, col):
            sudoku_helper(matrix, cont)

    if not cont[0]:  # Stopping point 3
        return
    matrix[row][col] = 0  # Backtrack, mark the current cell empty again


# Count the number of choices haven't been used
def count_choices(matrix, i, j):
    can_pick = [True, True, True, True, True, True, True, True, True, True];  # From 0 to 9 - drop 0

    # Check row
    for k in range(9):
        can_pick[matrix[i][k]] = False

    # Check col
    for k in range(9):
        can_pick[matrix[k][j]] = False;

    # Check 3x3 square
    r = i // 3
    c = j // 3
    for row in range(r * 3, r * 3 + 3):
        for col in range(c * 3, c * 3 + 3):
            can_pick[matrix[row][col]] = False

    # Count
    count = 0
    for k in range(1, 10):  # 1 to 9
        if can_pick[k]:
            count += 1

    return count


# Return true if the current cell doesn't create any violation
def can_be_correct(matrix, row, col):
    # Check row
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False

    # Check column
    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False

    # Check 3x3 square
    r = row // 3
    c = col // 3
    for i in range(r * 3, r * 3 + 3):
        for j in range(c * 3, c * 3 + 3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False

    return True


# Return true if the whole board has been occupied by some non-zero number
# If this happens, the current board is the solution to the original Sudoku
def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True
