import torch



class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        num_filters = 32

        # To save much space, we can make the output_channels[i] for i=1,..,5
        # into a single function
        def output_i(in_channels, out_channels, num_filters,
             stride1,stride2,pad1,pad2):
            output = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = num_filters,
                    kernel_size = 3,
                    stride = stride1,
                    padding = pad1,
                )
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = num_filters,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = stride2,
                    padding = pad2,
                )
            )
            
            return output

        self.output_0 = nn.Sequential(
            nn.Conv2d(
                in_channels = image_channels,
                out_channels = num_filters,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = num_filters,
                out_channels = num_filters*2,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = num_filters,
                out_channels = num_filters,
                kernel_size = 3,
                stride = 2,
                padding = 1,
            ),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.ReLU(),
        )

        self.output_1 = output_1(output_channels[1],output_channels[2],32*4,1,2,1,1)
        self.output_2 = output_1(output_channels[2],output_channels[3],32*8,1,2,1,1)
        self.output_3 = output_1(output_channels[3],output_channels[4],32*4,1,2,1,1)
        self.output_4 = output_1(output_channels[4],output_channels[5],32*4,1,2,1,1)
        self.output_5 = output_1(output_channels[5],output_channels[6],32*4,1,2,1,1)
        
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.output_0(out_features[0]))
        out_features.append(self.output_1(out_features[0]))
        out_features.append(self.output_2(out_features[0]))
        out_features.append(self.output_3(out_features[0]))
        out_features.append(self.output_4(out_features[0]))
        out_features.append(self.output_5(out_features[0]))
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

