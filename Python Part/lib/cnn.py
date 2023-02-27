import numpy as np


class cnn_layers():
    def __init__(self, n_kernels):
        # Setting the shapes and sizes for convolution operation
        self.kernel = np.random.uniform(0, 1, size=(n_kernels, n_kernels))

        # normalize the kernel
        kernel_sum = np.sum(self.kernel)
        if kernel_sum != 0:
            self.kernel = self.kernel / kernel_sum

    def convolve(self, images):
        all_convolution_images = []
        for k in range(self.kernel.shape[0]):
            print("---convolution start---")
            convolution_images = []
            for image in images:
                image_height, image_width = image.shape
                kernel_height, kernel_width = self.kernel.shape[0], self.kernel.shape[0]
                output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype="float32")
                for i in range(image_height - kernel_height + 1):
                    for j in range(image_width - kernel_width + 1):
                        output[i, j] = (image[i:i + kernel_height, j:j + kernel_width] * self.kernel).sum()
                convolution_images.append(output)

            # update the kernel
            self.kernel = np.random.uniform(0, 1, size=(self.kernel.shape[0], self.kernel.shape[0]))
            print(f"Kernel {k + 1}:")
            print(self.kernel)
            print("--------")

            all_convolution_images.append(convolution_images)
            output2 = np.concatenate(all_convolution_images, axis=1)
            print("###convolution end###")
        return output2

    def pool(self, convolved_images):
        print("---pooling start---")
        images = []
        for image in convolved_images:
            stride = 2
            pool_output = []
            for i in range(0, image.shape[0], stride):
                row = []
                for j in range(0, image.shape[1], stride):
                    row.append(np.amax(image[i:i + stride, j:j + stride]))
                pool_output.append(row)
            images.append(np.array(pool_output))
        print("###pooling end###")
        return np.array(images)
