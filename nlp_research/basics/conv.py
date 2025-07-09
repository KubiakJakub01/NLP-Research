import numpy as np


class Conv2D:  # pylint: disable=too-few-public-methods
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Constructor for the 2D Convolutional layer.

        Args:
            in_channels (int): Number of channels in the input (e.g., 1 for grayscale, 3 for RGB).
            out_channels (int): Number of filters to apply.
            kernel_size (int): The size of the square kernel (e.g., 3 for a 3x3 kernel).
            stride (int): The step size of the convolution.
            padding (int): The amount of zero-padding to add to the input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        # These are the "learnable" parameters. In a real network, they are
        # updated via backpropagation. Here, we initialize them randomly.
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Shape: (out_channels, 1)
        self.biases = np.random.randn(out_channels, 1)

        self.input = None

    def forward(self, input_tensor):
        """
        Performs the forward pass of the convolution.

        Args:
            input_tensor (np.array): The input data. \
            Shape: (n_samples, in_channels, in_height, in_width)

        Returns:
            np.array: The output feature map.
        """
        self.input = input_tensor
        n_samples, _, in_height, in_width = input_tensor.shape

        # 1. Calculate output dimensions
        out_height = int((in_height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        out_width = int((in_width - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # 2. Apply padding to the input tensor
        # The padding is applied only to the height and width dimensions
        # ((0,0), (0,0)) for samples and channels
        # ((self.padding, self.padding), (self.padding, self.padding)) for height and width
        padded_input = np.pad(
            input_tensor,
            pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant',
            constant_values=0,
        )

        # 3. Initialize the output tensor with zeros
        output_tensor = np.zeros((n_samples, self.out_channels, out_height, out_width))

        # 4. Perform the convolution
        # Loop over each sample in the batch
        for i in range(n_samples):
            # Loop over each filter (to create each output channel)
            for f in range(self.out_channels):
                # Loop over the vertical dimension of the output
                for y in range(out_height):
                    # Loop over the horizontal dimension of the output
                    for x in range(out_width):
                        # Define the top-left corner of the sliding window
                        vert_start = y * self.stride
                        vert_end = vert_start + self.kernel_size
                        horiz_start = x * self.stride
                        horiz_end = horiz_start + self.kernel_size

                        # Extract the 3D patch (receptive field) from the padded input
                        input_patch = padded_input[i, :, vert_start:vert_end, horiz_start:horiz_end]

                        # Perform element-wise multiplication and sum
                        # The kernel for this filter (self.weights[f]) is also a 3D volume
                        convolution_sum = np.sum(input_patch * self.weights[f])

                        # Add the bias term
                        output_tensor[i, f, y, x] = convolution_sum + self.biases[f]

        return output_tensor

    def backward(self, output_tensor, learning_rate):
        pass


def main():
    # 1. Create a sample input image (e.g., a 10x10 grayscale image)
    # We add a batch dimension of 1 and a channel dimension of 1
    # Shape: (n_samples, in_channels, height, width) -> (1, 1, 10, 10)
    sample_image = np.zeros((1, 1, 10, 10))
    # Create a vertical line in the middle
    sample_image[:, :, :, 4:6] = 1

    print('--- Input Image (10x10) ---')
    print(sample_image[0, 0])  # Print the 2D image

    # 2. Instantiate the Conv2D layer
    # Input has 1 channel, we want 1 output channel (1 filter)
    # Kernel size is 3x3
    conv_layer = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    # 3. Manually set a kernel to be a vertical edge detector
    # This is what a network would "learn" to do
    vertical_edge_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # Reshape to match the layer's weight dimensions: (out_channels, in_channels, H, W)
    conv_layer.weights = vertical_edge_kernel.reshape(1, 1, 3, 3)  # pylint: disable=too-many-function-args
    # Set bias to zero for clarity
    conv_layer.biases = np.array([[0]])

    # 4. Perform the forward pass
    output_feature_map = conv_layer.forward(sample_image)

    print('\n--- Output Feature Map (10x10) ---')
    # Print the resulting 2D feature map, rounding for clarity
    print(output_feature_map[0, 0].round(2))


if __name__ == '__main__':
    main()
