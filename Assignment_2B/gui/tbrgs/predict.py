"""Predict hourly flow at every site for a chosen day and time with a trained model."""
from pathlib import Path

import numpy as np

from .data import INTERVALS, features


class Predictor:
    def __init__(self, ds, models_dir):
        self.ds, self.dir, self.cache = ds, Path(models_dir), {}
        self.day_index = {d: i for i, d in enumerate(ds.dates)}

    def _load(self, name):
        import torch
        from .models import Net
        if name not in self.cache:
            path = self.dir / f"{name.lower()}.pt"
            if not path.exists():
                raise ValueError(f"Model {name} is not trained yet. Run train.py first.")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            net = Net(ck["kind"], ck["in_size"], ck["hidden"])
            net.load_state_dict(ck["state"])
            self.cache[name] = (net, ck["window"], ck["scale"])
        return self.cache[name]

    def flows(self, name, when):
        """Predicted vehicles per hour at each site: the next 15-minute flow predicted from the
        previous `window` intervals, times 4."""
        from .models import predict
        net, window, scale = self._load(name)
        if when.date() not in self.day_index:
            raise ValueError(f"No data for {when.date()}.")
        idx = self.day_index[when.date()] * INTERVALS + (when.hour * 60 + when.minute) // 15
        if idx < window:
            raise ValueError("Not enough history before this time; choose a later time or day.")
        ids = list(self.ds.series)
        x = np.stack([features(self.ds.series[s][idx - window:idx], idx - window) for s in ids])
        x[:, :, 0] /= scale
        return {s: max(float(p) * scale * 4, 0.0) for s, p in zip(ids, predict(net, x))}
