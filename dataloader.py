import os
import cv2


class DataLoader:

    Data = []
    image_size = 70

    def load_data(self, data_set, class_names):
        for c in class_names:
            path_name = os.path.join(data_set, c)
            label = class_names.index(c)
            for img in os.listdir(path_name):
                try:
                    images = cv2.resize(cv2.imread(os.path.join(path_name, img), cv2.IMREAD_GRAYSCALE),
                                        (self.image_size, self.image_size)).flatten()
                    self.Data.append([images, label])
                except Exception as e:
                    pass
        return self.Data

