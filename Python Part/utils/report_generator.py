import datetime

import matplotlib.pyplot as plt
from utils.date_helper import get_time_format


def generate_report(accuracy_train, accuracy_test,epoch_size,training_sample_size,testing_sample_size,alpha):
    print("final training accuracy = %.3f" % accuracy_train[-1])
    print("final test accuracy = %.3f" % accuracy_test[-1])

    plot = plt.figure()
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plot.savefig("temp/results/figures/results_{}.png".format(get_time_format()))

    # Open a file for writing the results
    with open('temp/results/text/results_{}.csv'.format(get_time_format()), 'w') as f:
        # Write the information about the experiment
        f.write(f"Epoch Size: {epoch_size},")
        f.write(f"Training Sample Size: {training_sample_size},")
        f.write(f"Testing Sample Size: {testing_sample_size},")
        f.write(f"Alpha: {alpha}\n")

        # Write the header for the accuracy results
        f.write("Epoch,Train Accuracy,Test Accuracy\n")

        # Write the accuracy results for each epoch
        for i, (acc_train, acc_test) in enumerate(zip(accuracy_train, accuracy_test)):
            f.write(f"{i + 1},{acc_train},{acc_test}\n")

