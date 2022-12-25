import numpy as np



# Define the arrays to be convolved
array1 = np.arange(16)
array1_2d = array1.reshape(4, 4)
array2 = np.arange(4)
array2_2d = array2.reshape(2, 2)
result = [[0 for i in range(3)] for j in range(3)]

# Perform the convolution
for i in range(3):
    for j in range(3):
        for k in range(2):
            for l in range(1):
                result[i][j] += array1[i+k][j+l] * array2[k][l]


# Print the result
print(result)