import cv2
import numpy as np


class Preprocessor():
    """ Осуществляет предобработку изображения.
        Изначально обрабатывает по градиенту, затем нормализует по гистограмме интенсивности цвета.
        Обработанное изображение конверитиркует в 8-битный формат и сохраняет.
    """
    def __init__(self, tiff_name):
        self.processing_img = np.zeros((1152, 1152, 1), dtype=np.uint16)
        self.name = tiff_name

    def process_by_gradient(self, path_input_img):
        """ Обрабатывает изображение по градиенту. """
        img = cv2.imread(path_input_img + self.name, -1)
        # пятиточечная апроксимация первой производной
        for j in range(len(img)):
            for i in range(2, len(img[j]) - 2, 1):
                self.processing_img[i][j] = (-img[i - 2][j] + 8 * img[i - 1][j] -
                                        8 * img[i + 1][j] + img[i + 2][j]) / 12

    def normalize_img(self):
        """ Нормализует изображение по гистограмме интенсивности цвета. """
        hist, bins = np.histogram(self.processing_img.flatten(), 65536, [0, 65536])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')
        self.processing_img = cdf[self.processing_img]

    def convert_to_8bit(self, path_processing_img):
        """ Конвертирует изображение из 16 бит в 8 бит и сохраняет. """
        self.processing_img = (self.processing_img / 256).astype('uint8')
        self.processing_img = cv2.cvtColor(self.processing_img, cv2.COLOR_GRAY2RGB)
        jpg_name = self.name[:-4] + "jpg"
        cv2.imwrite(path_processing_img + jpg_name, self.processing_img)
        return jpg_name
