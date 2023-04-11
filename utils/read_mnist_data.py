import pandas as pd
from utils.image_helper import reshape_all_images
def read_mnist_data(dataset_path, rows_count):

    data = pd.read_csv(dataset_path, header=0, nrows=rows_count).values
    images, labels = data[:, 1:], data[:, 0]

    # Normalize the image features
    square_images = reshape_all_images(images)
    sample_num = 99
    convert_image_to_array(square_images[sample_num],labels[sample_num])
    print(f"label: {labels[sample_num]}")
    images = images / 255.0
    return images, labels

def convert_image_to_array(image,label):
    c_plus_plus_style = "{"
    for i in range(28):
        c_plus_plus_style += "{"
        for j in range(28):
            c_plus_plus_style += str(int(image[i][j]))
            if j != 27:
                c_plus_plus_style += ", "
        c_plus_plus_style += "}"
        if i != 27:
            c_plus_plus_style += ",\n"

    c_plus_plus_style += "};"

    with open('temp/image.txt', 'w') as f:
        f.write(c_plus_plus_style)
        f.write("\n\n")
        f.write(f"label:{label}")
