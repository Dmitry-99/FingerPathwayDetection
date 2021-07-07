import cv2
import numpy as np

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image): # Принимает в качестве аргументов маску, которая выделяет ладонь на изображении
                               # находим контуры ладони
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def draw_rect(frame): # функция, которая рисует зеленые квадратики 
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    # создаем четыре массива, которые содержат координаты зеленых квадратиков
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle): # проходим по всем квадратикам и рисуем их, total_rectangle = 9
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

# когда пользователь размещает свою руку  - мы считываем пиксели из зеленых прямоугольников и используем их для создания гистограммы HSV
def hand_histogram(frame): # рассчитываем гистограмму оттенка ладони
                           # импользуем ее для поиска областей телесного оттенка на изображении
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # преобразуем входной кадр в HSV
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype) # создаем матрицу размером 90х10 с 3 цветовыми каналами и называем его ROI (Region of Interest)
                                                       # данная матрица будет хранить цвет кожи ладони
    for i in range(total_rectangle): # берем значение пикселей из зеленых прямоугольников и помещаем их в матрицу ROI
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256]) # создаем гистограмму HSV, используя матрицу ROI для цвета кожи
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX) # нормализуем эту матрциу, используя тип нормы cv2.NORM_MINMAX


def hist_masking(frame, hist): # с помощью данной функции ищем компоненты кадра, содержащие кожу
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # с помощью метода calcBackProjection определяем насколько хорошо пиксели данного изображения соответствуют 
    # распределению пикселей в гистограмме оттенков руки
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    cv2.imshow('BackProj', dst) # результат наложения фильтра
    # создаем фильтр для поиска руки
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)


    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)
    # получаем границу, по которой можно отделить ладонь от остального изображения 
    thresh = cv2.merge((thresh, thresh, thresh))

    btws_and = cv2.bitwise_and(frame, thresh) # результат побитового умножения изображения на фтльтр 
    cv2.imshow('Bitwise_and', btws_and) 

    return btws_and #cv2.bitwise_and(frame, thresh)


def centroid(max_contour): # находим центр ладони (розовая точка)
    # print(max_contour)
    moment = cv2.moments(max_contour) # момент функции — это некая скалярная величина, которая характеризует эту функцию и 
                                      # может быть использована для артикуляции её важных свойств
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid): # функция нахождения самой удаленной точки ладони
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid # координаты центра ладони

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float) # координаты х контура ладони
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float) # координаты у контура ладони

        xp = cv2.pow(cv2.subtract(x, cx), 2) # считаем расстояние от центра ладони до точек контура
        yp = cv2.pow(cv2.subtract(y, cy), 2) # считаем расстояние от центра ладони до точек контура
        dist = cv2.sqrt(cv2.add(xp, yp)) # считаем расстояние от центра ладони до точек контура

        dist_max_i = np.argmax(dist) # находим индекс элемента с максимальным расстоянием

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0]) # находим координаты самой дальней точки
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point): # отрисовка точек следа движения
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)

    # удаляем шумы и выделяем передний план
    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image) # находим контуры ладони
    max_cont = max(contour_list, key=cv2.contourArea) # находим массив граничных точек

    cnt_centroid = centroid(max_cont) # находим центроид
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1) # рисуем центроид

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False) # находим многоугольник внутри которого расположена ладонь
        defects = cv2.convexityDefects(max_cont, hull) # находим все дефекты выпуклости
        far_point = farthest_point(defects, max_cont, cnt_centroid) # находим наиболее удаленную точку
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1) # рисуем указатель
        if len(traverse_point) < 20: # храним 20 точек движения
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0) # удаляем первую пройденную точку
            traverse_point.append(far_point) # добавляем последнюю пройденную точку

        draw_circles(frame, traverse_point)


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1) # с клавиши считываем нажатие
        _, frame = capture.read() # считываем картинку
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord('z'): # помещаем ладонь в область квадратиков и нажимаем на клавишу z 
            is_hand_hist_created = True # устанавливаем флаг True, который говорит, что мы создали гистограмму HSV цвета
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist) 

        else:
            frame = draw_rect(frame) # начальное состояние, рисуем зеленые квалратики, куда помещаем ладонь

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()