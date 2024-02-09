"""
Exact GP models with Multi GPU support

Author: Shrenik Zinage

"""
__all__ = ["ExactModel1", "ExactModel2", "ExactModel3", "ExactModel4", "ExactModel5", "FCNFeatureExtractor", "CNNFeatureExtractor", "LSTMFeatureExtractor", "FeatureExtractorCausalFCN", "ExactModelDeepCausalFCN"]


# Importing required packages
import torch
import gpytorch
import torch.nn as nn

class ExactModel1(gpytorch.models.ExactGP):
    """A simple Gaussian process with a zero mean and an RBF kernel."""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModel1, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ZeroMean()
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dim, ard=True))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactModel2(gpytorch.models.ExactGP):
    """Same as ExactModel1 but with a linear mean."""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModel2, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.LinearMean(dim)
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dim, ard=True))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class ExactModel3(gpytorch.models.ExactGP):
    """Same as ExactModel1 but with a Deep RBF kernel (FCN network)"""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModel3, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        self.feature_extractor = FCNFeatureExtractor(num_inputs*window_size, 20) 
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=20, ard=True))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
        

    def forward(self, x):
        
        # Transform the input features with the feature extractor
        x_transformed = self.feature_extractor(x)
        
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class ExactModel4(gpytorch.models.ExactGP):
    """Same as ExactModel1 but with a Deep RBF kernel (LSTM network)"""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModel4, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        self.feature_extractor = LSTMFeatureExtractor(num_inputs, window_size, 20) 
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=20, ard=True))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
        
    def forward(self, x):
        
        # Transform the input features with the feature extractor
        x_transformed = self.feature_extractor(x)
        
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
        
class ExactModel5(gpytorch.models.ExactGP):
    """Same as ExactModel1 but with a Deep RBF kernel (CNN network)"""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModel5, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        self.feature_extractor = CNNFeatureExtractor(num_inputs, window_size, 20) 
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=20, ard=True))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
        

    def forward(self, x):
        
        # Transform the input features with the feature extractor
        x_transformed = self.feature_extractor(x)
        
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class ExactModelDeepCausalFCN(gpytorch.models.ExactGP):
    """Same as ExactModel1 but with a Deep RBF kernel and causal structure"""
    def __init__(self, dim, train_x, train_y, likelihood, n_devices, output_device, num_inputs, window_size):
        super(ExactModelDeepCausalFCN, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()

        self.feature_extractor = FeatureExtractorCausalFCN(window_size, num_inputs)
        
        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=24))
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        
        # Transform the input features with the feature extractor
        x_transformed = self.feature_extractor(x)
        
        mean_x = self.mean_module(x_transformed)
        covar_x = self.covar_module(x_transformed)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
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
        
        
class FeatureExtractorCausalFCN(nn.Module):
    """Fully connected layers while satisfying the causal structure"""
    def __init__(self, window_size, num_inputs):
        super(FeatureExtractorCausalFCN, self).__init__()

        self.path1 = self._make_path(window_size)
        self.path2 = self._make_path(window_size * 2)
        self.path3 = self._make_path(window_size * 3)
        self.path6 = self._make_path(window_size * 6)
        self.path7 = self._make_path(window_size * 7)
        self.path10 = self._make_path(window_size * 10)

        self.combine = nn.Linear(24*window_size, 24)

    def _make_path(self, window_size):
        return nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, window_size),
            nn.ReLU(),
        )

    def forward(self, x):
        
        window_size = 5

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
        
        out8 = self.path7(torch.cat((out1,out4,out6,InCylO2), dim=1))     
        
        out9 = self.path1(u_rp)         
        
        out10 = self.path1(T_egrsys_out)
        out11 = self.path2(torch.cat((T_intclr_out, out10),dim = 1))
        out12 = self.path6(torch.cat((u_main_soi, u_main_qty, u_pilot2_soi, u_pilot2_qty, u_post1_soi, u_post1_qty),dim=1))
        out13 = self.path10(torch.cat((out9, T_tur_in, out11, out12),dim=1))     
        
        out_combine = torch.cat((out8,u_rp,out13,u_main_soi,u_main_qty,u_pilot2_soi,u_pilot2_qty,u_post1_soi,u_post1_qty),dim=1)
        
        return self.combine(out_combine)
        

        












