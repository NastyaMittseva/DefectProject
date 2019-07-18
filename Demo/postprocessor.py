import cv2
import numpy as np


class Postprocessor():
    """ Осуществляет постобработку изображений.
        Изначально преобразует маску к размеру исходного изображения.
        Затем накладывает маску на исходное изображение и сохраняет.
    """
    def __init__(self):
        self.scaling_mask = np.zeros((1152, 1152, 3), dtype=np.uint8)
        self.alpha = 0.35

    def scale_mask(self, initial_boundar1, initial_boundar2, path_defect_mask, process_img_name):
        """ Преобразует маску к размеру 1152 на 1152, предворительно масштабируя ее к начальной ширине изображения. """
        initial_width = initial_boundar2 - initial_boundar1
        weld_mask = cv2.imread(path_defect_mask + process_img_name)
        weld_mask = cv2.resize(weld_mask, (1152, initial_width))
        self.scaling_mask[initial_boundar1:initial_boundar2, 0:1152] = weld_mask

    def overlay_mask_on_image(self, path_processing_img, process_img_name, path_results):
        """ Накладывает маску на исходное изображение и сохраняет. """
        processing_image = cv2.imread(path_processing_img + process_img_name)
        self.scaling_mask[np.where((self.scaling_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
        self.scaling_mask[np.where((self.scaling_mask == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        result = cv2.addWeighted(self.scaling_mask, self.alpha, processing_image, 1 - self.alpha, 0)
        cv2.imwrite(path_results + process_img_name, result)
        return result
