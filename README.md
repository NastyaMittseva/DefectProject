# DefectProject
Демонстрационная версия проекта по распознаванию дефектов сварных швов по рентгеновским снимкам.

### Структура проекта
Алгоритм распознавания дефектов сварных швов по рентгеновским снимкам состоит из следующих шагов:
- Предобработка данных - [preprocessor.py](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/preprocessor.py)
- Сегментация области шва и околошовного пространства - [weld_segmentator_v2.py](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/weld_segmentator_v2.py)
- Формирование области шва - [intermediate_processor.py](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/intermediate_processor.py)
- Распознавание дефектов в сегментированной области шва - [defect_detector_M.py](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/defect_detector_M.py)
- Постобработка результатов - [postprocessor.py](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/postprocessor.py)

Для сегментации области шва и распознавания дефектов использовались нейронные сети структуры [FgSegNet](https://www.sciencedirect.com/science/article/abs/pii/S0167865518303702), разработанные Long Ang LIM и Hacer YALIM KELES. 
FgSegNet представляет собой структуру кодер-декодер и предназначена для решения задач бинарной сегментации. 

В данном проекте для сегментации области шва использовалась [FgSegNet_v2](https://github.com/lim-anggun/FgSegNet_v2), а для распознавания
дефектов - [FgSegNet_M](https://github.com/lim-anggun/FgSegNet2) с точностью распознавания 0.9983 и 0.9958 соответсвенно по метрике binary accuracy. 

Скачать веса обученных моделей можно по [ссылке](https://drive.google.com/drive/folders/1mSeh2Ln2sGHOjlOMS-zrUH9IRtfb_cRv). 
Затем их необходимо добавить в директории, соблюдая следующую структуру.
```
models/
  weldSegmentation/
    FgSegNet_v2/FgSegNet_v2_weld_segmentation.h5
  defectSegmentation/
    FgSegNet_M/FgSegNet_M_defect_segmentation.h5
```
Запустить алгоритм распознавания дефектов и посмотреть результаты можно с помощью [Demo.ipynb](https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/Demo.ipynb).

### Примеры распознавания дефектов
<img src="https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/results/1.jpg" width="40%" height="40%"> <img src="https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/results/3.jpg" width="40%" height="40%"> 
<img src="https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/results/6.jpg" width="40%" height="40%"> <img src="https://github.com/NastyaMittseva/DefectProject/blob/master/Demo/results/8.jpg" width="40%" height="40%">
