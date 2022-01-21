import torch


class DisableSomeNeurons:
    def __init__(self, net: torch.nn.Module):
        self._net = net

    def disable_hidden_weights(self, number_of_neurons: tuple):
        self.disable_output_weights(number_of_neurons)
        i = 0
        for param in self._net.parameters():
            if i == 1:
                for number in number_of_neurons:
                    with torch.no_grad():
                        param[:, number] = 0
                break
            i += 1

    def disable_input_weights(self, number_of_neurons: tuple):
        i = 0
        for param in self._net.parameters():
            if i == 0:
                for number in number_of_neurons:
                    with torch.no_grad():
                        param[number, :] = 0
                break
            i += 1

    def disable_output_weights(self, number_of_neurons: tuple):
        i = 0
        for param in self._net.parameters():
            if i == 2:
                for number in number_of_neurons:
                    with torch.no_grad():
                        param[:, number] = 0
                break
            i += 1
