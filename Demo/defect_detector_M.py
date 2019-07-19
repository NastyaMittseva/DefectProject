import numpy as np
import cv2
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from keras.models import load_model
from FgSegNet_M_S_module import loss, acc
from clear_memory import reset_keras


class DefectDetector():
    """ Осуществляет распознавание дефектов в области шва.
        Изначально получает входные данные.
        Затем предсказывает вероятностную маску дефектов.
        После конвертирует вероятностную маску в бинарную и сохраняет ее.
    """
    def __init__(self):
        self.model_path = './models/defectSegmentation/FgSegNet_M/FgSegNet_M_defect_segmentation.h5'
        self.threshold = 0.45
        self.s1 = []
        self.s2 = []
        self.s3 = []
        self.probs = None
        self.defect_mask = None

    def get_weld_area(self, path_scale_weld, process_img_name):
        """ Получает входное изображение и конвертирует его в три масштаба. """
        x = kImage.load_img(path_scale_weld+process_img_name)
        x = kImage.img_to_array(x)
        self.s1.append(x)
        self.s1 = np.asarray(self.s1)
        pyramid = tuple(pyramid_gaussian(x / 255., max_layer=2, downscale=2))
        self.s2.append(pyramid[1] * 255.)
        self.s3.append(pyramid[2] * 255.)
        self.s2 = np.asarray(self.s2)
        self.s3 = np.asarray(self.s3)

    def predict_defect_mask(self):
        """ Предсказывает вероятностную маску дефектов. """
        model = load_model(self.model_path, custom_objects={'loss': loss, 'acc': acc})
        data = [self.s1, self.s2, self.s3]
        self.probs = model.predict(data, batch_size=1, verbose=1)
        self.probs = self.probs.reshape([self.probs.shape[0], self.probs.shape[1], self.probs.shape[2]])

    def apply_threshold(self):
        """ Преобразовывает вероятностную маску шва в бинарную маску с использованием порога. """
        y = self.probs[0]
        y[y < self.threshold] = 0.
        y[y >= self.threshold] = 1.
        self.defect_mask = y * 255
        self.defect_mask = self.defect_mask.astype(np.uint8)

    def save_defect_mask(self, path_defect_mask, process_img_name):
        """ Сохраняет маску. """
        mask_name = process_img_name
        dir_and_name = path_defect_mask + mask_name
        cv2.imwrite(dir_and_name, self.defect_mask)
        reset_keras()
        return mask_name
