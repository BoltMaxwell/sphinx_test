""" 
Approximate GP models

Author: Ilias Bilionis

"""

__all__ = ["ApproximateModel1", "ApproximateModel2", "ApproximateModel_DeepCausal", "ApproximateModel_SVDKL", 
           "FCNFeatureExtractor", "CNNFeatureExtractor", "LSTMFeatureExtractor", "FeatureExtractorCausal"]

# Import required packages
import torch
import gpytorch
import torch.nn as nn
import numpy as np
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


# Define a custom Gaussian Process model using approximate inference
class ApproximateModel1(ApproximateGP):
    """Create simple Gaussian process with a constant mean and an RBF kernel
    using approximate inference.

    Demonstrate how math can be used in the docstring: \n
    :math:`f(x) \sim \mathcal{GP}(m(x), k(x, x'))` \n
    :math:`m(x) = c` \n
    :math:`k(x, x') = \sigma^2 \exp(- \\frac{1}{2} (x - x')^T \Lambda (x - x'))`

    Demonstrate how to use a code block in the docstring: \n
    .. code-block:: python

        def hello_world():
            print("Hello, world!")

    """
    def __init__(self, dim, num_inputs, window_size, causal_dim, output_dim, inducing_points=10):
        """Initialize the model.

        Keyword Arguments:
        dim -- The dimension of the input space.
        inducing_points -- The number of inducing points to use.
        """

        if isinstance(inducing_points,int):
            inducing_points = torch.Tensor(np.random.rand(inducing_points, dim))
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super(ApproximateModel1, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        )

    def forward(self, x):
        """Forward pass through the model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def inspect_parameters(self):
        """Print information about the parameters of the model."""
        print("Model Parameters".center(80))
        with torch.no_grad():
            print(f"Constant mean: {self.mean_module.constant.item():1.2f}")
            
            print("Lengthscales")
            for i, l in enumerate(self.covar_module.base_kernel.lengthscale.cpu().numpy().flatten()):
                print(f"{i}\t{l:1.2f}")
            
            print(f"Output scale: {self.covar_module.outputscale.item():1.2f}")

class ApproximateModel2(ApproximateGP):
    """Same as ApproximateModel1 but with a linear mean."""
    def __init__(self, dim, num_inputs, window_size, causal_dim, output_dim, inducing_points=10):
        if isinstance(inducing_points,int):
            inducing_points = torch.Tensor(np.random.rand(inducing_points, dim))
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        

        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super(ApproximateModel2, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.LinearMean(dim)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def inspect_parameters(self):
        """Print information about the parameters of the model."""
        print("Model Parameters".center(80))
        with torch.no_grad():
            print("Lengthscales")
            for i, l in enumerate(self.covar_module.base_kernel.lengthscale.cpu().numpy().flatten()):
                print(f"{i}\t{l:1.2f}")
            
            print(f"Output scale: {self.covar_module.outputscale.item():1.2f}")

class ApproximateModel_DeepCausal(ApproximateGP):
    def __init__(self, dim, num_inputs, window_size, causal_dim, output_dim, inducing_points=10):
        # Convert the inducing_points argument to a tensor if it's an integer
        if isinstance(inducing_points,int):
            inducing_points = torch.Tensor(np.random.rand(inducing_points, dim))
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super(ApproximateModel_DeepCausal, self).__init__(variational_strategy)
        
        # Choose which feature extracter do you want (CNN or DNN or Causal)
        self.feature_extractor = FCNFeatureExtractor(window_size, causal_dim, output_dim) 
        
        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims = output_dim*window_size, ard=True))

    def forward(self, x):

        x_transformed = self.feature_extractor(x)

        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def inspect_parameters(self):
        """Print information about the parameters of the model."""
        print("Model Parameters".center(80))
        with torch.no_grad():
            print("Lengthscales")
            for i, l in enumerate(self.covar_module.base_kernel.lengthscale.cpu().numpy().flatten()):
                print(f"{i}\t{l:1.2f}")
            
            print(f"Output scale: {self.covar_module.outputscale.item():1.2f}")

class GPLayer(ApproximateGP):
    def __init__(self, output_dim, inducing_points):

        if isinstance(inducing_points,int):
            inducing_points = torch.Tensor(np.random.rand(inducing_points, output_dim))
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        variational_strategy = VariationalStrategy(self, 
                                                   inducing_points, 
                                                   variational_distribution, 
                                                   learn_inducing_locations=True)

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims = output_dim, ard=True))


    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def inspect_parameters(self):
        """Print information about the parameters of the model."""
        print("Model Parameters".center(80))
        with torch.no_grad():
            print("Lengthscales")
            for i, l in enumerate(self.covar_module.base_kernel.lengthscale.cpu().numpy().flatten()):
                print(f"{i}\t{l:1.2f}")
            
            print(f"Output scale: {self.covar_module.outputscale.item():1.2f}")
    

## DKL Model 

class ApproximateModel_SVDKL(gpytorch.Module):
    """Approximate Gaussian Process with Deep Kernel.
    
    Adapted from the following GPytorch tutorial:
    https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
    
    """
    def __init__(self, dim, num_inputs, window_size, causal_dim, output_dim, inducing_points):
        super(ApproximateModel_SVDKL, self).__init__()
        self.feature_extractor = FCNFeatureExtractor(dim, output_dim) 
        # self.feature_extractor = CNNFeatureExtractor(num_inputs, window_size, output_dim)
        # self.feature_extractor = LSTMFeatureExtractor(num_inputs, window_size, output_dim)
        # self.feature_extractor = FeatureExtractorCausal(window_size, causal_dim, output_dim)        
        self.gp_layer = GPLayer(output_dim, inducing_points)

    def forward(self, x):
        features = self.feature_extractor(x)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

## Feature Extractors

class FCNFeatureExtractor(nn.Module):
    """A simple fully connected network with 4 layers."""
    def __init__(self, input_dim, output_dim):
        super(FCNFeatureExtractor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class CNNFeatureExtractor(nn.Module):
    """A network with two 1D convolutional layers, a 1D max pooling layer, and fully connected layers."""
    def __init__(self, num_inputs, window_size, output_dim):
        super(CNNFeatureExtractor, self).__init__()
        
        self.num_inputs = num_inputs
        self.window_size = window_size

        self.conv_layers = nn.Sequential(
            # Expects an input shape of dimension (num_samples, num_inputs, window_size)
            nn.Conv1d(num_inputs, 256, kernel_size=5, stride=1, padding=2),
            # Output shape: (num_samples, 256, window_size)
            nn.ReLU(),
            # Output shape: (num_samples, 256, window_size)
            nn.MaxPool1d(kernel_size=2),
            # Output shape: (num_samples, 256, window_size//2)
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            # Output shape: (num_samples, 128, window_size//2)
            nn.ReLU(),
            # Output shape: (num_samples, 128, window_size//2)
        )
        
        # Compute the output size by doing a forward pass with a dummy tensor
        dummy_x = torch.zeros(1, num_inputs, window_size) 
        conv_out = self.conv_layers(dummy_x)
        conv_out_flattened_size = conv_out.numel() // conv_out.shape[0]  
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),   # Flattens the tensor into a 2D tensor with shape (batch_size, -1)
            # Output shape: (num_samples, 128*(window_size//2))
            nn.Linear(conv_out_flattened_size,100),
            # Output shape: (num_samples, 100)
            nn.ReLU(),
            # Output shape: (num_samples, 100)
            nn.Linear(100,50),
            # Output shape: (num_samples, 50)
            nn.ReLU(),
            # Output shape: (num_samples, 50)
            nn.Linear(50,output_dim),
            # Output shape: (num_samples, output_dim)
        )

    def forward(self, x):
        
        # Reshape the input to (batch_size, num_inputs, window_size)
        x = x.view(x.shape[0], self.num_inputs, self.window_size)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class LSTMFeatureExtractor(nn.Module):
    """An LSTM-based feature extractor with a predefined output dimension."""
    def __init__(self, num_inputs, window_size, output_dim):
        super(LSTMFeatureExtractor, self).__init__()
        
        self.num_inputs = num_inputs
        self.window_size = window_size

        self.lstm1 = nn.LSTM(input_size=num_inputs, hidden_size=256, 
                            num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, 
                            num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, 
                            num_layers=1, batch_first=True)
        
        # Linear layer to project LSTM's output to the desired dimension
        self.projection_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        # Reshape the input to (num_samples, window_size, num_inputs)
        x = x.view(x.size(0), self.window_size, self.num_inputs)
        
        # Pass the reshaped data through the first LSTM
        x, _ = self.lstm1(x)
        
        # Pass the reshaped data through the second LSTM
        x, _ = self.lstm2(x)
        
        # Pass the output of the second LSTM through the third LSTM
        x, (h_n, c_n) = self.lstm3(x)
        
        # Use the hidden state of the third LSTM layer
        features = h_n[-1]
        
        # Project features to the desired output dimension
        projected_features = self.projection_layer(features)
        return projected_features
        
class FeatureExtractorCausal(nn.Module):

    def __init__(self, window_size, causal_dim, output_dim):
        super(FeatureExtractorCausal, self).__init__()

        self.path1 = self._make_path(window_size)
        self.path2 = self._make_path(window_size * 2)
        self.path3 = self._make_path(window_size * 3)
        self.path6 = self._make_path(window_size * 6)
        self.path7 = self._make_path(window_size * 7)
        self.path10 = self._make_path(window_size * 10)

        self.combine = nn.Linear(causal_dim * window_size, output_dim * window_size)

    def _make_path(self, window_size):
        return nn.Sequential(
            nn.Linear(window_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, window_size),
            nn.ReLU(),
        )


    def forward(self, x):

        from training.train_approximate_gpr import window_size
        
        N_e = x[:, 0:window_size]
        u_main_soi = x[:, window_size:window_size*2]
        u_main_qty = x[:, window_size*2:window_size*3]
        u_pilot2_soi = x[:, window_size*3:window_size*4]
        u_pilot2_qty = x[:, window_size*4:window_size*5]
        u_post1_soi = x[:, window_size*5:window_size*6]
        u_post1_qty = x[:, window_size*6:window_size*7]
        u_rp = x[:, window_size*7:window_size*8]
        u_vgt = x[:, window_size*8:window_size*9]
        u_egrv = x[:, window_size*9:window_size*10]
        T_egrsys_out = x[:, window_size*10:window_size*11]
        T_intclr_out = x[:, window_size*11:window_size*12]
        W_a = x[:, window_size*12:window_size*13]
        W_egr = x[:, window_size*13:window_size*14]
        InCylO2 = x[:, window_size*14:window_size*15]
        T_tur_in = x[:, window_size*15:window_size*16]

        out1 = self.path1(N_e)
        out2 = self.path1(u_egrv)
        out3 = self.path1(u_vgt)
        out4 = self.path3(torch.cat((out2, W_egr, out3), dim=1))
        out6 = self.path2(torch.cat((W_a, out3),dim=1))
        
        out8 = self.path7(torch.cat((out1,out4,out6,InCylO2), dim=1))     # Input to RBF kernel
        
        out9 = self.path1(u_rp)         
        
        out10 = self.path1(T_egrsys_out)
        out11 = self.path2(torch.cat((T_intclr_out, out10),dim = 1))
        out12 = self.path6(torch.cat((u_main_soi, u_main_qty, u_pilot2_soi, u_pilot2_qty, u_post1_soi, u_post1_qty),dim=1))
        out13 = self.path10(torch.cat((out9, T_tur_in, out11, out12),dim=1))     # Input to RBF kernel
        
        out_combine = torch.cat((out8,u_rp,out13,u_main_soi,u_main_qty,u_pilot2_soi,u_pilot2_qty,u_post1_soi,u_post1_qty),dim=1)
        
        return self.combine(out_combine)

