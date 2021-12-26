import matplotlib.pyplot as plt


def plot_inputs_outputs(inputs, outputs):
    for i in range(inputs.shape[2]):
        plt.plot(inputs[:, 0, i], label=fr"$input_{i + 1}$")
        plt.legend()
    plt.show()

    for i in range(outputs.shape[2]):
        plt.plot(outputs[:, 0, i], label=fr"$output_{i + 1}$")
        plt.legend()
    plt.show()
