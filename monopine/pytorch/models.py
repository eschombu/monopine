from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from superleaf.timeseries.datetime_utils import get_hours_minutes_seconds

from monopine.pytorch.nn_spec import ModelSpec, ModelSpecificationError


class SequentialModel(nn.Module):
    def __init__(self, model_layers: Sequence[nn.Module]):
        super().__init__()
        named_layers = self._name_layers(model_layers)
        self.model = nn.Sequential(named_layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _name_layers(layers: Sequence[nn.Module]) -> OrderedDict:
        layer_names = []
        for layer in layers:
            name = f"{type(layer).__name__.lower()}_1"
            while name in layer_names:
                split_name = name.split('_')
                i = int(split_name[-1])
                name = '_'.join(split_name[:-1] + [str(i + 1)])
            layer_names.append(name)
        return OrderedDict(zip(layer_names, layers))


class EncoderBase:
    @abstractmethod
    def get_out_shape(self, in_features: Optional[int] = None):
        pass


class Conv1dEncoder(SequentialModel, EncoderBase):
    def __init__(self, conv_spec: ModelSpec, in_channels=1, linear_spec: Optional[ModelSpec] = None,
                 in_features: Optional[int] = None, normalizer: Optional[nn.Module] = None):
        self._conv_spec = conv_spec
        self.in_channels = in_channels
        self._linear_spec = linear_spec
        layers = []
        if normalizer:
            layers.append(normalizer)
        layers.extend(conv_spec.build(in_channels))
        if linear_spec:
            if not in_features:
                raise ModelSpecificationError("in_features is required if linear layer is present")
            layers.append(nn.Flatten())
            layers.extend(linear_spec.build(in_features=np.prod(conv_spec.get_output_shape(in_features))))
        self.in_features = in_features
        super().__init__(layers)

    def get_out_shape(self, in_features: Optional[int] = None):
        in_features = in_features or self.in_features
        if in_features is None:
            raise ValueError("in_features must be specified to get output shape")
        if self._linear_spec is None:
            return self._conv_spec.get_output_shape(in_features)
        else:
            return self._linear_spec.get_output_len(np.prod(self._conv_spec.get_output_shape(in_features)))


class LinearPredictor(SequentialModel):
    def __init__(self, in_features: int, model_spec: ModelSpec):
        self._model_spec = model_spec
        super().__init__(model_spec.build(in_features))

    def forward(self, x):
        return self.model(x)


class Conv1dEncoderPredictorModel(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module, loss=nn.BCELoss, optimizer=torch.optim.Adam,
                 learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self._default_loss = loss()
        self._default_optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, x):
        e = self.encoder(x)
        p = self.task_predictor(e)
        return p

    def _train_loop(self, dataloader: DataLoader, loss, optimizer) -> torch.Tensor:
        # batches = len(dataloader.dataset)
        losses = []
        for data in dataloader:
            X, y = data
            optimizer.zero_grad()

            # Compute prediction and loss
            pred = self(X)
            batch_loss = loss(pred, y)
            losses.append(batch_loss)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()

        return torch.tensor(losses)

    def fit(self, dataloader: DataLoader, epochs=3, rounds=3, loss=None, optimizer=None,
            test_dataloader=None, test_loss=None) -> Tuple[pd.Series, pd.DataFrame]:
        fit_start = datetime.now()
        loss = loss or self._default_loss
        test_loss = test_loss or loss
        optimizer = optimizer or self._default_optimizer

        train_losses = []
        test_loss_acc = [(0, self.test(test_dataloader, test_loss))]
        try:
            for r in range(rounds):
                print(f"Training round {r + 1} of {rounds}")
                for t in range(epochs):
                    epoch_start = datetime.now()
                    print(f"Epoch {t + 1} of {epochs}\n-------------------------------")
                    self.train()
                    epoch_losses = self._train_loop(dataloader, loss, optimizer)
                    i_start = len(train_losses)
                    i_end = i_start + len(epoch_losses)
                    train_losses.extend(zip(range(i_start, i_end), epoch_losses))
                    if test_dataloader:
                        test_loss_acc.append((i_end, self.test(test_dataloader, test_loss)))
                    epoch_duration = datetime.now() - epoch_start
                    h, m, s = get_hours_minutes_seconds(epoch_duration)
                    epoch_minutes = h * 60 + m + s / 60
                    print(f"Epoch duration: {epoch_minutes:.1f} minutes\n")
        except KeyboardInterrupt:
            print("Training manually halted")

        train_loss_idxs, train_losses = zip(*train_losses)
        test_loss_idxs, test_losses_accs = zip(*test_loss_acc)
        test_losses, test_accs = zip(*test_losses_accs)
        out = (pd.Series(torch.tensor(train_losses).data, index=train_loss_idxs),
               pd.DataFrame({'loss': torch.tensor(test_losses).data,
                             'accuracy': torch.tensor(test_accs).data},
                            index=test_loss_idxs))
        fit_end = datetime.now()
        h, m, s = get_hours_minutes_seconds(fit_start, fit_end)
        print(f"Done! ({h} hours, {m} minutes, {s} seconds)")
        return out

    def test(self, dataloader: DataLoader, loss_fn: torch.nn.modules.loss._Loss, threshold=0.5
             ) -> Tuple[torch.Tensor, torch.Tensor]:  # , OrderedDict]:
        self.eval()
        num_batches = torch.scalar_tensor(len(dataloader))
        loss = None
        correct = None
        count = None
        with torch.no_grad():
            for data in dataloader:
                X, y = data
                pred = self(X)
                if loss is None:
                    loss = loss_fn(pred, y)
                    correct = ((pred >= threshold) == y).type(torch.float).sum()
                    count = len(pred)
                else:
                    loss += loss_fn(pred, y)
                    correct += ((pred >= threshold) == y).type(torch.float).sum()
                    count += len(pred)

        loss /= num_batches
        correct /= count
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}\n")
        return loss, correct  # , self.state_dict()

    def evaluate(self, dataloader: DataLoader, n: Optional[int] = None, threshold=0.5) -> pd.DataFrame:
        self.eval()
        labels = []
        probas = []
        predictions = []
        with torch.no_grad():
            for data in dataloader:
                X, y = data
                pred = self(X)
                labels.extend(y.cpu().numpy().squeeze())
                probas.extend(pred.cpu().numpy().squeeze())
                predictions.extend((pred >= threshold).cpu().numpy().squeeze().astype(float))
                if n is not None and len(labels) >= n:
                    break
        return pd.DataFrame({'label': labels,
                             'probability': probas,
                             'prediction': predictions})

    def save(self, path: str) -> None:
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device=None):
        device = device or torch.device('cpu')
        try:
            self.load_state_dict(torch.load(path))
        except RuntimeError:
            self.load_state_dict(torch.load(path, map_location=device))
        return self
