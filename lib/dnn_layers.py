import numpy as np


class dnn_layers():
    def __int__(self, x):
        self.x = x

    def convolve(self, image, kernel):
        output = np.zeros([100, 676], dtype=float, order='C')
        image_2d = image[:, :].reshape(28, -1)
        for i in range(image.shape[0]):
            for j in range(26):
                for k in range(26):
                    for m in range(3):
                        for n in range(3):
                            output[j][k] += image_2d[j + m][k + n] * kernel[m][n]

        return output