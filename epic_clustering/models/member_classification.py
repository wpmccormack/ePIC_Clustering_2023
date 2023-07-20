import os
import sys

# 3rd party imports
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import DataLoader, Dataset, Data
from torch.utils.data import random_split
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Local imports
from epic_clustering.utils import make_mlp

sqrt_eps = 1e-12


class MemberClassification(pl.LightningModule):
    
    # ---- The following 100 lines are where the ML actually happens ---- #

    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        torch.manual_seed(0)
        self.network = make_mlp(
            hparams["input_features"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [1],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters(hparams)

    def forward(self, x):

        return torch.sigmoid(self.network(x))


    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Apply the model to the input data
        classification_score = self(batch.x)

        # Apply a loss function
        loss = F.binary_cross_entropy(
            classification_score.squeeze(), batch.y.squeeze()
        )

        self.log("train_loss", loss, batch_size=1, sync_dist=True, on_epoch=True, on_step=True)

        return loss


    def shared_evaluation(self, batch):

        # Apply the model to the input data
        classification_score = self(batch.x)

        # Apply a loss function
        loss = F.binary_cross_entropy(
            classification_score.squeeze(), batch.y.squeeze()
        )

        metrics = self.log_metrics(loss, batch, classification_score)
        
        return {
            "loss": loss,
        }

    def log_metrics(self, loss, batch, classification_score):

        classification_score = classification_score.cpu().detach().numpy()
        
        # Get trues, positives and true positives
        trues = batch.y.bool().cpu().detach().numpy()
        preds = classification_score > 0.5
        true_positives = np.sum(trues.squeeze() & preds.squeeze())

        efficiency = true_positives/np.sum(trues) if np.sum(trues) > 0 else 0
        purity = true_positives/np.sum(preds) if np.sum(preds) > 0 else 0

        current_lr = self.optimizers().param_groups[0]["lr"]
        
        self.log_dict(
            {"val_loss": loss,
                "efficiency": efficiency,
                "purity": purity,
                 "lr": current_lr
                },
            batch_size=1, sync_dist=True
        )

        try:
            auc = roc_auc_score(trues, classification_score)
            self.log_dict(
            {
                "auc": auc
            },
            batch_size=1, sync_dist=True)
        except:
            pass
        
        return {
            "loss": loss,
            "efficiency": efficiency,
            "purity": purity,
        }

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch)
    
    # ----- The below functions simply organize the input data and the optimizer ----- #

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = 1 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][0]
        return DataLoader(
            self.trainset, batch_size=1, num_workers=num_workers
        )

    def val_dataloader(self):
        if self.valset is None:
            return None
        num_workers = 1 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][1]
        return DataLoader(
            self.valset, batch_size=1, num_workers=num_workers, pin_memory=False
        )

    def test_dataloader(self):
        if self.testset is None:
            return None
        num_workers = 1 if ("num_workers" not in self.hparams or self.hparams["num_workers"] is None) else self.hparams["num_workers"][2]
        return DataLoader(
            self.testset, batch_size=1, num_workers=num_workers
        )
    
    def setup(self, stage="fit"):
        """
        The setup logic of the stage.
        1. Setup the data for training, validation and testing.
        2. Run tests to ensure data is of the right format and loaded correctly.
        3. Construct the truth and weighting labels for the model training
        """

        if not hasattr(self, "trainset"):
            self.load_data(self.hparams["input_dir"])
                
    def load_data(self, input_dir):
        """
        Load in the data for training, validation and testing.
        """

        total_dataset = EventDataset(input_dir, sum(self.hparams["data_split"]), hparams = self.hparams)
        self.trainset, self.valset, self.testset = random_split(total_dataset, self.hparams["data_split"])

        print(f"Loaded {len(self.trainset)} training events, {len(self.valset)} validation events and {len(self.testset)} testing events")

    def make_lr_scheduler(self, optimizer):
        warmup_epochs = self.hparams["warmup"]
        lr_decay_factor = self.hparams["factor"]
        patience = self.hparams["patience"]

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # During warm-up, increase the learning rate linearly
                return (epoch + 1) / warmup_epochs
            else:
                # After warm-up, decay the learning rate by lr_decay_factor every 10 epochs
                return lr_decay_factor ** ((epoch - warmup_epochs) // patience)
            
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": self.make_lr_scheduler(optimizer[0]),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

class EventDataset(Dataset):
    """
    The custom default dataset to load CSV events off the disk
    """

    def __init__(self, input_dir, num_events = None, hparams=None, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.hparams = hparams
        self.num_events = num_events
        self.scales = {
                    "E": 30.,
                    "T": 100.,
                    "posx": 200.,
                    "posy": 200.,
                    "posz": 500.,
                }
        
        self.csv_events = self.load_datafiles_in_dir(self.input_dir, self.num_events)

        print("Converting to PyG data objects")
        self.pyg_events = [self.convert_to_pyg(event[1]) for event in tqdm(self.csv_events)]
        
    def load_datafiles_in_dir(self, input_dir, num_events):

        # Each file is 1000 events, so need to load num_events//1000 + 1 files
        csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')][:num_events//1000 + 1]
        events = pd.concat([pd.read_csv(f) for f in csv_files])
        if num_events is not None:
            events = events[events["event"].isin(events["event"].unique()[:num_events])]

        self.scale_features(events)

        return list(events.groupby('event'))

    def convert_to_pyg(self, event):

        # Convert to PyG data object
        event = event.reset_index(drop=True)
        event = event.drop(columns=['event'])

        edge_index = self.create_training_pairs(event).long()
        clusterID = torch.from_numpy(event.clusterID.to_numpy().astype(np.int64))
        y = (clusterID[edge_index[0]] == clusterID[edge_index[1]])
        node_features = torch.from_numpy(event[['posx', 'posy', 'posz', 'E']].to_numpy())
        edge_features = torch.cat([node_features[edge_index[0]], node_features[edge_index[1]]], dim=1)

        data = Data(
                        x = edge_features.float(),
                        y = y.float(),
                        edge_index = edge_index,
                    )

        data.num_nodes = data.x.shape[0]

        return data
        
    def len(self):
        return len(self.pyg_events)

    def get(self, idx):

        return self.pyg_events[idx]

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        for feature in self.scales.keys():
            event[feature] = event[feature]/self.scales[feature]

        return event

    def create_training_pairs(self, event):
        """
        Create the true edge list for the event. This is 
        """

        # Sorted event.E - get first 40
        high_energy_hits = event.sort_values(by="E", ascending=False).iloc[:40]

        # use torch meshgrid to get all pairs between high_energy_hits.hit_number and event.hit_number
        pairs = torch.meshgrid(torch.from_numpy(high_energy_hits.hit_number.values), torch.from_numpy(event.hit_number.values))
        # convert into a 2 x N array
        pairs = torch.stack(pairs).reshape(2, -1)

        return pairs