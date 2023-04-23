import numpy as np
from torch import nn

from monopine.pytorch.nn_spec import (
    Conv1dBlockSpec,
    Conv1dChainSpec,
    Conv1dSpec,
    LinearBlockSpec,
    LinearChainSpec,
    LinearSpec,
    ModelSpec,
    Pool1dSpec,
)


def get_conv_spec() -> ModelSpec:
    conv_layer_spec = Conv1dSpec(num_filters=4, kernel_size=5)
    conv_pool_spec = Pool1dSpec(3)
    conv_block_1 = Conv1dBlockSpec(
        conv=conv_layer_spec,
        pool=conv_pool_spec,
        nonlinearity=nn.ReLU(),  # instantiate now
        dropout=0.2,
    )
    conv_block_2 = Conv1dBlockSpec(
        conv=conv_layer_spec,
        pool=conv_pool_spec,
        nonlinearity=nn.ReLU,  # instantiate later
        normalizer=nn.BatchNorm1d,
        dropout=0.2,
    )
    return Conv1dChainSpec([conv_block_1, conv_block_2])


def get_linear_spec() -> ModelSpec:
    linear_layer_spec = LinearSpec(4)
    linear_block_spec = LinearBlockSpec(
        linear=linear_layer_spec,
        nonlinearity=nn.ReLU,
    )
    return LinearChainSpec([linear_block_spec for _ in range(2)])


def test_conv_chain():
    in_channels = 3
    in_features = 50

    conv_layers = get_conv_spec()
    conv_model = conv_layers.build(in_channels=in_channels, in_features=in_features)
    assert all(isinstance(layer, nn.Module) for layer in conv_model)

    linear_layers = get_linear_spec()
    linear_model = linear_layers.build(in_features=np.prod(conv_layers.get_output_shape(in_len=in_features)))
    assert all(isinstance(layer, nn.Module) for layer in linear_model)
