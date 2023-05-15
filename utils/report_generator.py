import datetime

import matplotlib.pyplot as plt
from utils.date_helper import get_time_format
import numpy as np

def generate_report(accuracy_train, accuracy_test, epoch_size, training_sample_size, testing_sample_size, alpha, input,
                    hidden, output, model_name, hidden_weights, output_weights):
    print("final training accuracy = %.3f" % accuracy_train[-1])
    print("final test accuracy = %.3f" % accuracy_test[-1])

    plot = plt.figure()
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plot.savefig("temp/results/figures/{}_{}.png".format(model_name, get_time_format()))

    # Open a file for writing the results
    with open('temp/results/text/{}_{}.csv'.format(model_name, get_time_format()), 'w') as f:
        # Write the information about the experiment
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Epoch Size: {epoch_size},")
        f.write(f"Training Sample Size: {training_sample_size},")
        f.write(f"Testing Sample Size: {testing_sample_size},")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Hyper Parameters - Input: {input}, Hidden: {hidden}, Output: {output}\n ")
        f.write("\n\n\n")
        # Write the output_weights in C++ style
        scale = 1000
        f.write("output_weights = {")
        for i in range(output_weights.shape[0]):
            f.write("{")
            for j in range(output_weights.shape[1]):
                f.write(str(output_weights[i, j] * scale))
                if j < output_weights.shape[1] - 1:
                    f.write(",")
            f.write("}")
            if i < output_weights.shape[0] - 1:
                f.write(",")
        f.write("};\n")
        f.write(f"scale:{scale}")
        f.write("\n\n\n")

        # Write the header for the accuracy results
        f.write("Epoch,Train Accuracy,Test Accuracy\n")

        # Write the accuracy results for each epoch
        for i, (acc_train, acc_test) in enumerate(zip(accuracy_train, accuracy_test)):
            f.write(f"{i + 1},{acc_train},{acc_test}\n")



