import numpy as np
from torch import nn

from tests.pytorch.test_nn_spec import get_conv_spec, get_linear_spec
from monopine.pytorch.models import Conv1dEncoder, Conv1dEncoderPredictorModel, LinearPredictor
from monopine.pytorch.nn_spec import LinearSpec, LinearBlockSpec


def test_encoder_1d():
    in_channels = 3
    in_features = 50

    conv_spec = get_conv_spec()
    conv_lin_spec = LinearBlockSpec(LinearSpec(8), nonlinearity=nn.ReLU, normalizer=nn.LayerNorm, dropout=0.1)
    conv_encoder = Conv1dEncoder(
        conv_spec=conv_spec,
        in_channels=in_channels,
        linear_spec=conv_lin_spec,
        in_features=in_features,
        normalizer=nn.BatchNorm1d
    )
    assert isinstance(conv_encoder, nn.Module)

    linear_spec = get_linear_spec()
    linear_predictor = LinearPredictor(np.prod(conv_encoder.get_out_shape()), linear_spec)
    assert isinstance(linear_predictor, nn.Module)

    encoder_predictor = Conv1dEncoderPredictorModel(conv_encoder, linear_predictor)
    assert isinstance(encoder_predictor, nn.Module)
