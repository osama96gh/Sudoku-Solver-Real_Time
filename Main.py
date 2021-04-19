import copy
from ImageProssesing import *
from SudokoSolver import *

hog = init_hog_descripter()
svm_model = cv2.ml.SVM_load('svm_model.xml')

use_ipwebcam = False
if use_ipwebcam:
    video_url = 'http://192.168.1.102:8080/video'
    video_stream = cv2.VideoCapture(video_url)
else:
    video_url = 0
    video_stream = cv2.VideoCapture(video_url)

solved_grid = np.zeros((9, 9), np.uint8)
not_solved_grid = np.zeros((9, 9), np.uint8)

count = 0
stage = 10
w_x = 200
w_y = 20
while True:
    count += 1
    ret, frame = video_stream.read()
    if not ret:
        continue

    # prepossessing the image
    gray, threshold = get_gray_and_threshold(frame)

    # find contours in threashold
    best_cnt = find_best_contour(threshold)

    if best_cnt is not None:
        approx = approximate_contour(best_cnt)
        if len(approx) == 4:

            projected_grid = project_grid(threshold, approx)

            draw_contour(frame, approx)

            digits_imgs = seperate_grid_digits(projected_grid)

            grid = pridict_digits(digits_imgs, svm_model, hog)
            old_grid = copy.deepcopy(grid)

            if count > 60:
                count = 0
                if measure_different(old_grid, not_solved_grid) != 0:
                    solve_sudoku(grid)
                    if (all_board_non_zero(grid)):
                        not_solved_grid = old_grid
                        solved_grid = grid

            output_img = np.zeros((450, 450, 3), np.uint8)

            output_img = draw_original_grid(output_img, old_grid)

            if all_board_non_zero(solved_grid) and measure_different(old_grid, not_solved_grid) < 5:
                output_img = draw_solved_grid(output_img, not_solved_grid, solved_grid)

            output_img = un_project_grid(output_img, frame, approx, True)

            frame = overlay(frame, output_img)

    cv2.imshow("Frame", frame)
    cv2.moveWindow("Frame", w_x, w_y)

    pressed_key = cv2.waitKey(1)

    if pressed_key == 27 or pressed_key == 23:
        break

cv2.destroyAllWindows()
