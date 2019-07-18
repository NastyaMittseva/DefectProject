import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
from instance_normalization import InstanceNormalization
from my_upsampling_2d_v2 import MyUpSampling2D
from FgSegNet_v2_module import loss, acc, loss2, acc2
from clear_memory import reset_keras


class WeldSegmentator_FgSegNet_v2():
    """ Осуществляет сегментацию области шва.
        Изначально загружает данные, затем предсказывает вероятностную маску шва.
        После преобразует вероятностную маску в бинарную маску и сохраняет.
    """
    def __init__(self):
        self.model_path = './models/weldSegmentation/FgSegNet_v2/FgSegNet_v2_weld_segmentation.h5'
        self.threshold = 0.5
        self.X = None
        self.probs = None
        self.weld_mask = None

    def get_images(self, path_processing_img, process_img_name):
        """ Получает изображения. """
        self.X = image.load_img(path_processing_img + process_img_name)
        self.X = image.img_to_array(self.X)
        self.X = np.expand_dims(self.X, axis=0)

    def predict_weld_mask(self):
        """ Предсказывает вероятностную маску шва. """
        model = load_model(self.model_path, custom_objects={'MyUpSampling2D': MyUpSampling2D,
                                                            'InstanceNormalization': InstanceNormalization,
                                                            'loss': loss, 'acc': acc, 'loss2': loss2,
                                                            'acc2': acc2})
        self.probs = model.predict(self.X, batch_size=1, verbose=1)
        self.probs = self.probs.reshape([self.probs.shape[1], self.probs.shape[2]])

    def apply_threshold(self):
        """ Преобразовывает вероятностную маску шва в бинарную маску с использованием порога. """
        y = self.probs
        y[y < self.threshold] = 0.
        y[y >= self.threshold] = 1.
        self.weld_mask = y * 255
        self.weld_mask = self.weld_mask.astype(np.uint8)

    def save_weld_masks(self, path_weld_mask, process_img_name):
        """ Сохраняет маску. """
        mask_name = process_img_name[:-3] + 'tiff'
        dir_and_name = path_weld_mask + mask_name
        cv2.imwrite(dir_and_name, self.weld_mask)
        reset_keras()
        return mask_name