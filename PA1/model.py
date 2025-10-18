import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Set random seed
torch.manual_seed(42)

# model_config = {
#     p = 2,
#     eps = 1e-12,
#     debug = True,
#     num_clusters = 64,
# }

class VGGFeatureExtractor(nn.Module):
    """
    VGG16 feature extractor that returns conv5_3 feature maps.
    Input : (B, 3, H, W) normalized like ImageNet
    Output: (B, D, H', W') for H=W=224 -> (B, 512, ?, ?)
    """

    def __init__(self, pretrained=True, freeze_features=False, num_layers_last=-1, debug=False              ):
        """VGG16 feature extractor that returns last layer feature maps.

        Args:
            pretrained (bool, optional): pretrained weights. Defaults to True.
            freeze_features (bool, optional): freeze features. Defaults to False.
            num_layers_last (int, optional): Until how many last layers for feature extraction. Defaults to -1.
        """
        super().__init__()
        # load weights robustly across torchvision versions
        try:
            vgg = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None
            )
        # except AttributeError:
        #     vgg = models.vgg16(pretrained=pretrained)
        except Exception as e:
            print(f"Error loading VGG16: {e}")
            vgg = models.vgg16(pretrained=pretrained)
            # raise e
        # take all conv/relu layers up to (but NOT including) the last maxpool
        self.features = nn.Sequential(*list(vgg.features.children())[:num_layers_last])
        # freeze when freeze_features is True
        if freeze_features:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Returns the feature maps from the VGG16 backbone
        Args:
            x: (B, 3, H, W)
        Returns:
            feature_maps: (B, D, H, W)
            where B is the batch size, D is the dimension of the feature maps, H is the height of the feature maps, W is the width of the feature maps
        """
        return self.features(x)


class NetVLADModel(nn.Module):
    """
    Your Task) Complete the function in model.py.

    - Use VGG16 CNN to convert the input image into a high-dimensional feature map.
    - Perform soft-assignment to cluster centroids and aggregate residual vectors into a global descriptor.
    - Normalize the final descriptor for retrieval tasks.

    Returns:
        Tensor: Global image descriptor of shape (N, K*D)
    """

    def __init__(self, num_clusters=16, dim=512, debug=False):
        super().__init__()
        self.K = num_clusters
        self.D = dim
        # TODO:
        # 1. Load a pretrained VGG16 backbone.
        #    * Remove the final pooling/classification layers.
        #    * Keep only convolutional feature extractor.
        self.cnn_feature_extractor = VGGFeatureExtractor(pretrained=True, freeze_features=True, num_layers_last=-1,
                                                          debug=debug)
        # 2. Define a NetVLAD pooling layer with `num_clusters` and feature dimension `dim`.
        #    * Cluster centers as learnable parameters. # c and a  as learnable parameters
        #    * 1x1 convolution for soft-assignment

        self.conv_layer = nn.Conv2d(self.D, self.K, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(self.K, self.D))
        self.debug = debug
        # self.vlad_features = VLAD_Features(num_clusters=num_clusters, dim=dim, debug=debug)

    def get_residual_vector(
        self,
        N: int,
        x_flatten_transpose: torch.Tensor,
        c_k: torch.Tensor,
        a_k: torch.Tensor,
        debug: bool = True,
        B: int = 32,
        D: int = 512,
    ):
        """
        Residual for kth cluster is calculated as follows:
        for each descriptor in the image, we calculate the residual vector as follows:
            residual = (x_flatten_n - c_k) * a_k_n
            then we sum up all the residual vectors to get the residual vector for the kth cluster
            residual = sum(residual)
        return residual
        Args:
            N: number of descriptors in the image
            x_flatten_transpose: (B,D,N)
            c_k: (1,D)
            a_k_n: (B,1,N)
        Returns:
            residual: (B,1,D)

        """
        # initialize residual tensor
        residual = torch.zeros(
            [B, 1, D],
            dtype=x_flatten_transpose.dtype,
            layout=x_flatten_transpose.layout,
            device=x_flatten_transpose.device,
        )
        # for each descriptor in the image [part of formula -> sum of i to N ]
        for n in range(N):  # for each descriptor in the image
            x_flatten_n = x_flatten_transpose[:, n : n + 1, :]  # returns (B,1,D)
            # now lets calculate the residual vector
            a_k_n = a_k[:, 0, n : n + 1].unsqueeze(2) # returns (B,1,1)
            diff = x_flatten_n - c_k # returns (B,1,D)
            diff = diff * a_k_n # returns (B,1,D)
            residual = residual + diff # returns (B,1,D)
        if debug:
            print("--------------------------------")
            print("residual shape:", residual.shape)
            print("x_flatten_transpose shape:", x_flatten_transpose.shape)
            print("x_flatten_n shape:", x_flatten_n.shape)
            print("a_k shape:", a_k.shape)
            print("a_k_n shape:", a_k_n.shape)
            print("c_k shape:", c_k.shape)
            print("residual shape:", residual.shape)

        return residual

    def forward(self, x):
        """
        Parameters:
            x (Tensor): Input image batch (N, 3, H, W)

        Returns:
            Tensor: Normalized global descriptor (N, K*D)
        """
        # TODO:
        # 1. Extract feature maps using backbone.(use models.vgg16() as a backbone)
        x = self.cnn_feature_extractor(x)
        B, D, H, W = x.shape  # batch, VGG_features, height, width
        N = H * W  # number of descriptors in the image
        c = self.centroids  # returns (K, D)
        # 2. Perform soft-assignment to cluster centroids
        # 2.1 Soft assignment
        s = self.conv_layer(x).view(B, self.K, N)  # returns (B, K, N)
        a = F.softmax(s, dim=1)  # returns (B, K, N)
        # Flatten features
        x_flatten = x.view(B, D, -1)  # returns (B, D, N)
        # Vectorized residual computation
        V = torch.zeros(B, self.K, D, device=x.device, dtype=x.dtype)
        x_flatten_transpose = x_flatten.transpose(1, 2)
        # 2.2 Vectorized residual computation
        for k in range(self.K):
            c_k = c[k : k + 1, :]  # returns (D,) kth cluster center
            a_k = a[:, k : k + 1, :]  # returns (B,1,N) kth cluster assignment [All descriptors assigned to kth cluster]
            # get residual vector for each cluster [distance from each image descriptor to the cluster center to cluster center]
            # 2.2.1 get residual vector for each cluster
            residual = self.get_residual_vector(
                N=N,
                x_flatten_transpose=x_flatten_transpose,
                c_k=c_k,
                a_k=a_k,
                debug=self.debug,
                B=B,
                D=D,
            )
            V[:, k : k + 1, :] = residual # returns (B,1,D) kth cluster residual vector
        # 3. Normalization
        # intra normalization across feature dimension D
        V = F.normalize(V, p=2, dim=-1) # Normalizes all residual for each cluster
        # flatten 
        V = V.view(B, -1)
        # Normalization across all elements in the vector for unit length vector -> makes relative comparisons between vectors easier
        V = F.normalize(V, p=2, dim=1)
        # Debug
        if self.debug:
            print("Input Feature Map shape:", x.shape)
            print("Afer Conv Layer shape:", s.shape)
            print("Afer Softmax shape:", a.shape)
            print("x_flatten shape:", x_flatten.shape)
            print("x_flatten_transpose shape:", x_flatten_transpose.shape)
            print("residual shape:", residual.shape)
            print("V shape:", V.shape)
        return V


if __name__ == "__main__":
    x = torch.rand([16, 3, 224, 224])
    print("input shape:", x.shape)
    # Dim should be set as 512 since VGG16 output features are 512 for selected layer 
    netvlad = NetVLADModel(num_clusters=3, dim=512,debug=True)
    out = netvlad(x)
    print("output shape:", out.shape)
