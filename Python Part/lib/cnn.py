import numpy as np


class cnn_layers():
    def __int__(self, x):
        self.x = x

    def convolve(self, images, kernel):
        print("---convolution start---")
        print(f"will convoluve {images.shape[0]} images with {images.shape[1]} size")
        image_size = int(np.sqrt(images[0].shape[0]))
        print(f"will reshape image to {image_size} * {image_size}")
        convultion_images = []
        for image in images:
            image = image.reshape(image_size, image_size)
            image_height, image_width = image.shape
            kernel_height, kernel_width = kernel.shape[0],kernel.shape[0]

            output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype="float32")
            for i in range(image_height - kernel_height + 1):
                for j in range(image_width - kernel_width + 1):
                    output[i, j] = (image[i:i + kernel_height, j:j + kernel_width] * kernel).sum()
            convultion_images.append(output.flatten())
        print("###convolution end###")

        return np.array(convultion_images)

    def pool(self, convolved_images):
        print("---pooling start---")

        stride = 2
        pool_output = []
        for i in range(0, convolved_images.shape[0], stride):
            row = []
            for j in range(0, convolved_images.shape[1], stride):
                row.append(np.amax(convolved_images[i:i + stride, j:j + stride]))
            pool_output.append(row)
        print("###pooling end###")
        return np.array(pool_output)