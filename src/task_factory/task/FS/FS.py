import torch
from torch import nn
import pytorch_lightning as pl

class task(pl.LightningModule):
    """Few-shot classification task using prototypical networks."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__()
        self.network = network
        self.lr = getattr(args_task, 'lr', 1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def compute_logits(self, support_x, support_y, query_x):
        z_support = self.network(support_x)
        z_query = self.network(query_x)
        classes = torch.unique(support_y)
        prototypes = []
        for c in classes:
            prototypes.append(z_support[support_y == c].mean(0))
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(z_query, prototypes)
        return -dists

    def step(self, batch, stage):
        logits = self.compute_logits(batch['support_x'], batch['support_y'], batch['query_x'])
        loss = self.loss_fn(logits, batch['query_y'])
        acc = (logits.argmax(dim=1) == batch['query_y']).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
