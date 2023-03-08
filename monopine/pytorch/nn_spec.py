from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple, Union

from torch import nn


def get_conv_out_len(in_len: int, kernel_size: int, stride=1, padding=0, dilation=1) -> int:
    return int((in_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class ModelSpecificationError(Exception):
    pass


class ModelSpec:
    @abstractmethod
    def build(self, in_channels: int, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def get_output_len(self, in_len: int) -> int:
        pass

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, dct: dict):
        return cls(**dct)


@dataclass
class Conv1dSpec(ModelSpec):
    num_filters: int
    kernel_size: int
    stride: int = 1
    padding: Union[int, str] = 0
    dilation: int = 1

    def build(self, in_channels: int, **kwargs) -> nn.Module:
        init_kwargs = asdict(self)
        init_kwargs['in_channels'] = in_channels
        init_kwargs['out_channels'] = init_kwargs.pop('num_filters')
        init_kwargs.update(kwargs)
        return nn.Conv1d(**init_kwargs)

    def get_output_len(self, in_len: int) -> int:
        return get_conv_out_len(in_len, kernel_size=self.kernel_size, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def get_output_shape(self, in_len: int) -> Tuple[int, int]:
        return self.num_filters, self.get_output_len(in_len)


@dataclass
class Pool1dSpec(ModelSpec):
    kernel_size: int
    stride: Optional[int] = None
    padding: Union[int, str] = 0
    dilation: int = 1
    mode: str = 'max'

    def build(self, **kwargs) -> nn.Module:
        init_kwargs = asdict(self)
        if not init_kwargs.get('stride'):
            init_kwargs['stride'] = self.kernel_size
        init_kwargs.update(kwargs)
        mode = init_kwargs.pop('mode')
        if mode.lower() == 'max':
            pool_type = nn.MaxPool1d
        elif mode.lower() in ['avg', 'average', 'mean']:
            pool_type = nn.AvgPool1d
        else:
            raise ModelSpecificationError(f"Unrecognized value for pool_mode: {mode}")
        return pool_type(**init_kwargs)

    def get_output_len(self, in_len: int) -> int:
        stride = self.stride or self.kernel_size
        return get_conv_out_len(in_len, kernel_size=self.kernel_size, stride=stride, padding=self.padding,
                                dilation=self.dilation)


@dataclass
class Conv1dBlockSpec(ModelSpec):
    conv: Union[Conv1dSpec, nn.Module]
    pool: Optional[Union[Pool1dSpec, nn.Module]]
    nonlinearity: Optional[nn.Module] = None
    normalizer: Optional[nn.Module] = None
    dropout: float = 0.0

    def build(self, in_channels: int) -> List[nn.Module]:
        layers = []
        if self.normalizer is not None:
            layers.append(self.normalizer)
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        if isinstance(self.conv, Conv1dSpec):
            layers.append(self.conv.build(in_channels))
        else:
            layers.append(self.conv)
        if isinstance(self.pool, Pool1dSpec):
            layers.append(self.pool.build())
        elif self.pool is not None:
            layers.append(self.pool)
        if self.nonlinearity is not None:
            if type(self.nonlinearity) == type:
                self.nonlinearity = self.nonlinearity()
            layers.append(self.nonlinearity)
        return layers

    def get_output_len(self, in_len: int) -> int:
        out_len = self.conv.get_output_len(in_len)
        if self.pool:
            out_len = self.pool.get_output_len(out_len)
        return out_len

    def get_output_shape(self, in_len: int) -> Tuple[int, int]:
        return self.conv.num_filters, self.get_output_len(in_len)

    def __add__(self, other: Union["Conv1dChainSpec", "Conv1dBlockSpec"]) -> "Conv1dChainSpec":
        if isinstance(other, Conv1dBlockSpec):
            return Conv1dChainSpec([self, other])
        elif isinstance(other, Conv1dChainSpec):
            return Conv1dChainSpec([self] + other.blocks)
        else:
            raise TypeError(f"Addition not supported for {type(self).__name__} and {type(other).__name__}")

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, Conv1dBlockSpec):
            return Conv1dChainSpec([other, self])
        elif isinstance(other, Conv1dChainSpec):
            return Conv1dChainSpec(other.blocks + [self])
        else:
            raise TypeError(f"Addition not supported for {type(other).__name__} and {type(self).__name__}")

    def to_json(self) -> dict:
        return {'conv': self.conv.to_json(),
                'pool': self.pool.to_json(),
                'nonlinearity': type(self.nonlinearity).__name__ if self.nonlinearity is not None else None,
                'dropout': self.dropout}

    @classmethod
    def from_json(cls, dct: dict):
        nonlin_type = getattr(nn, dct['nonlinearity'])
        if nonlin_type is not None:
            nonlinearity = nonlin_type()
        else:
            nonlinearity = None
        return cls(conv=Conv1dSpec.from_json(dct['conv']),
                   pool=Pool1dSpec.from_json(dct['pool']),
                   nonlinearity=nonlinearity,
                   dropout=dct['dropout'])


@dataclass
class Conv1dChainSpec(ModelSpec):  # CONVERT TO GENERIC LAYER-BLOCK CHAIN
    blocks: List[Conv1dBlockSpec]

    def build(self, in_channels: int) -> List[nn.Module]:
        layers = []
        next_in_channels = in_channels
        for block in self.blocks:
            layers.extend(block.build(next_in_channels))
            next_in_channels = block.conv.num_filters
        return layers

    def get_output_len(self, in_len: int) -> int:
        out_len = in_len
        for block in self.blocks:
            out_len = block.get_output_len(out_len)
        return out_len

    def get_output_shape(self, in_len: int) -> Tuple[int, int]:
        return self.blocks[-1].conv.num_filters, self.get_output_len(in_len)

    def __add__(self, other: Union["Conv1dChainSpec", Conv1dBlockSpec]) -> "Conv1dChainSpec":
        if isinstance(other, Conv1dBlockSpec):
            return Conv1dChainSpec(self.blocks + [other])
        elif isinstance(other, Conv1dChainSpec):
            return Conv1dChainSpec(self.blocks + other.blocks)
        else:
            raise TypeError(f"Addition not supported for {type(self).__name__} and {type(other).__name__}")

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, Conv1dBlockSpec):
            return Conv1dChainSpec([other] + self.blocks)
        elif isinstance(other, Conv1dChainSpec):
            return Conv1dChainSpec(other.blocks + self.blocks)
        else:
            raise TypeError(f"Addition not supported for {type(other).__name__} and {type(self).__name__}")

    def to_json(self) -> dict:
        return {'blocks': [block.to_json() for block in self.blocks]}

    @classmethod
    def from_json(cls, dct: dict):
        return cls([Conv1dBlockSpec.from_json(block) for block in dct['blocks']])


@dataclass
class LinearSpec(ModelSpec):
    out_features: int = 1
    bias: bool = True

    def build(self, in_features: int, **kwargs) -> nn.Module:
        init_kwargs = asdict(self)
        init_kwargs['in_features'] = in_features
        init_kwargs.update(kwargs)
        return nn.Linear(**init_kwargs)

    def get_output_len(self, in_features=None) -> int:
        return self.out_features


@dataclass
class LinearBlockSpec(ModelSpec):
    linear: LinearSpec
    nonlinearity: Optional[nn.Module] = None
    normalizer: Optional[nn.Module] = None
    dropout: float = 0.0

    def build(self, in_features: int) -> List[nn.Module]:
        layers = []
        if self.normalizer:
            layers.append(self.normalizer)
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(self.linear.build(in_features))
        if self.nonlinearity:
            if type(self.nonlinearity) == type:
                self.nonlinearity = self.nonlinearity()
            layers.append(self.nonlinearity)
        return layers

    def get_output_len(self, in_features=None) -> int:
        return self.linear.get_output_len(in_features)

    def to_json(self):
        dct = {'linear': self.linear.to_json()}
        if self.nonlinearity:
            dct['nonlinearity'] = type(self.nonlinearity).__name__
        dct['dropout'] = self.dropout
        return dct

    @classmethod
    def from_json(cls, dct: dict):
        nonlin_type = dct.get('nonlinearity')
        dct = {'linear': LinearSpec.from_json(dct['linear'])}
        if nonlin_type:
            dct['nonlinearity'] = nonlin_type()
        return cls(**dct)


@dataclass
class LinearChainSpec(ModelSpec):
    blocks: List[LinearBlockSpec]

    def build(self, in_features: int) -> List[nn.Module]:
        layers = []
        next_in_features = in_features
        for block in self.blocks:
            layers.extend(block.build(next_in_features))
            next_in_features = block.linear.out_features
        return layers

    def get_output_len(self, in_features=None) -> int:
        return self.blocks[-1].get_output_len(in_features)

    def to_json(self):
        return {'blocks': [b.to_json() for b in self.blocks]}

    @classmethod
    def from_json(cls, dct: dict):
        return cls(blocks=[LinearBlockSpec.from_json(b) for b in dct['blocks']])
