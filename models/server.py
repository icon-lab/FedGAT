import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import dist
from utils import misc
import copy
from trainer import GATTrainer

class FederatedServer:
    """Federated server for model aggregation."""
    def __init__(self, trainer: GATTrainer, device: torch.device, logger: misc.TensorboardLogger):
        self.trainer = trainer
        self.device = device
        self.logger = logger
        self.best_metrics = {
            "L_mean": float('inf'),
            "L_tail": float('inf'),
            "acc_mean": 0.0,
            "acc_tail": 0.0
        }

    @staticmethod
    def average_models(models: List[Dict]) -> Dict:
        """Federated averaging."""
        avg_state = {}
        for k in models[0].keys():
            avg_state[k] = sum(m[k] for m in models) / len(models)
        return avg_state

    def update(self, models: List[Dict], metrics: List[Dict], epoch: int) -> Dict:
        """Aggregate models and update global model."""
        avg_model = self.average_models(models)
        self.trainer.gat_wo_ddp.load_state_dict(avg_model)

        # Update best metrics
        for m in metrics:
            self.best_metrics["L_mean"] = min(self.best_metrics["L_mean"], m['Lm'])
            self.best_metrics["acc_mean"] = max(self.best_metrics["acc_mean"], m['Accm'])
            if m['Lt'] != -1:
                self.best_metrics["L_tail"] = min(self.best_metrics["L_tail"], m['Lt'])
                self.best_metrics["acc_tail"] = max(self.best_metrics["acc_tail"], m['Acct'])

        return avg_model
