""" 
Group A: Assignment No. 2
Assignment Title: Generate ANDNOT function using McCulloch-Pitts
neural net by a python program.

Sample input:
1
2
"""
import numpy as np

class McCullochPittsNN:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs)
        self.threshold = 0

    def set_weights(self, weights):
        if len(weights) != len(self.weights):
            raise ValueError("Number of weights must match number of inputs")
        self.weights = np.array(weights)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def activation_function(self, net_input):
        return 1 if net_input > self.threshold else 0

    def forward_pass(self, inputs):
        net_input = np.dot(inputs, self.weights)
        return self.activation_function(net_input)

def generate_ANDNOT():
    # Initialize McCulloch-Pitts neural network with 2 inputs
    mp_nn = McCullochPittsNN(2)
    mp_nn.set_weights([-1, 1])  # Weights for ANDNOT function
    mp_nn.set_threshold(0)

    # Generate truth table for ANDNOT function
    truth_table = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("Truth Table for ANDNOT Function:")
    print("Input1\tInput2\tOutput")
    for inputs in truth_table:
        output = mp_nn.forward_pass(inputs)
        print(f"{inputs[0]}\t{inputs[1]}\t{output}")

def main():
    while True:
        print("\nMenu:")
        print("1. Generate ANDNOT Function")
        print("2. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            generate_ANDNOT()
        elif choice == "2":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
