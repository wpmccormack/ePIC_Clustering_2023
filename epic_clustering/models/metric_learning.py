# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
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
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# Local imports
sys.path.append('../')
from epic_clustering.utils import make_mlp
from epic_clustering.utils import build_edges, graph_intersection
from epic_clustering.scoring import weighted_v_score

sqrt_eps = 1e-12


class MetricLearning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        torch.manual_seed(0)
        self.network = make_mlp(
            hparams["input_features"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters(hparams)

    def forward(self, x):

        x_out = self.network(x)
        x_out = F.normalize(x_out)
        return x_out

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

    def append_hnm_pairs(self, e_spatial, spatial, r_train=None, knn=None):
        if r_train is None:
            r_train = self.hparams["radius"]
        if knn is None:
            knn = self.hparams["knn"]
        
        knn_edges = build_edges(
            query=spatial,
            database=spatial,
            indices=None,
            r_max=r_train,
            k_max=knn,
        )

        e_spatial = torch.cat([e_spatial, knn_edges], dim=-1)

        return e_spatial

    def append_true_pairs(self, batch, edges):

        # Append the bidirectional true edges
        true_edges = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=-1)
        edges = torch.cat(
            [edges, true_edges], dim=-1,
        )

        return edges

    def get_distances(self, embedding, pred_edges, p=1):
        """
        Calculate the distances between the nodes at the edges.
        For p=1, this is the Euclidean distance.
        For p=2, this is the squared Euclidean distance.
        """

        if p == 1:
            d = torch.sqrt(torch.sum((embedding[pred_edges[0]] - embedding[pred_edges[1]]) ** 2, dim=-1) + sqrt_eps)
        elif p == 2:
            d = torch.sum((embedding[pred_edges[0]] - embedding[pred_edges[1]]) ** 2, dim=-1)
        else:
            raise ValueError(f"p={p} not supported")

        return d

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        embedding = self(batch.x)
        training_edges = self.get_training_edges(batch, embedding).long()
        truth = batch.pid[training_edges[0]] == batch.pid[training_edges[1]]
        if "energy_weighting" in self.hparams and self.hparams["energy_weighting"]:
            weighting = (batch.x[training_edges[0], 3] + batch.x[training_edges[1], 3])/2
        else:
            weighting = None

        loss = self.loss_function(embedding, pred_edges=training_edges, truth=truth, weighting=weighting)

        self.log("train_loss", loss, batch_size=1, sync_dist=True)

        return loss

    def get_training_edges(self, batch, embedding):
        
        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        training_edges = self.append_hnm_pairs(training_edges, embedding)

        # Append true signal edges
        training_edges = self.append_true_pairs(batch, training_edges)

        # Remove duplicate edges
        training_edges = torch.unique(training_edges, dim=-1)

        return training_edges

    def loss_function(self, embedding, pred_edges, truth, weighting=None):

        d = self.get_distances(embedding, pred_edges, p=1)

        return self.hinge_loss(truth, d, weighting)

    def hinge_loss(self, truth, d, weighting=None):
        """
        Calculates the hinge loss

        Given a set of edges, we partition into true and false. 
        
        Args:
            truth (``torch.tensor``, required): The truth tensor of composed of 0s and 1s, of shape (E,)
            d (``torch.tensor``, required): The distance tensor between nodes at edges[0] and edges[1] of shape (E,)
        Returns:
            ``torch.tensor`` The weighted hinge loss mean as a tensor
        """
        
        negative_mask = ~truth.bool()

        # Handle negative loss, but don't reduce vector
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            torch.ones_like(d[negative_mask])*-1,
            margin=self.hparams["margin"],
            reduction="none"
        )

        if weighting is not None:
            negative_loss = negative_loss*weighting[negative_mask]

        positive_mask = truth.bool()

        # Handle positive loss, but don't reduce vector
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]),
            margin=self.hparams["margin"],
            reduction="none"
        )

        if weighting is not None:
            positive_loss = positive_loss*weighting[positive_mask]

        total_loss = 0
        if negative_mask.sum() > 0:
            total_loss += negative_loss.mean()
        if positive_mask.sum() > 0:
            total_loss += self.hparams["positive_weight"]*positive_loss.mean()

        return total_loss

    def shared_evaluation(self, batch, knn_radius, knn_num):

        embedding = self(batch.x)
        pred_edges = build_edges(
            query=embedding,
            database=embedding,
            indices=None,
            r_max=knn_radius,
            k_max=knn_num,
        )

        true_edges = torch.cat([batch.edge_index, batch.edge_index.flip(0)], dim=-1)
        truth = batch.pid[pred_edges[0]] == batch.pid[pred_edges[1]]
        d = self.get_distances(embedding, pred_edges)
        if "energy_weighting" in self.hparams and self.hparams["energy_weighting"]:
            weighting = (batch.x[pred_edges[0], 3] + batch.x[pred_edges[1], 3])/2
        else:
            weighting = None
        loss = self.loss_function(embedding, pred_edges, truth, weighting)

        metrics = self.log_metrics(loss, batch, pred_edges, true_edges, truth)
        
        if self.current_epoch % 10 == 0:
            self.log_clustering_metrics(embedding, batch)

        return {
            "loss": loss,
            "distances": d,
            "preds": embedding,
            "truth_graph": true_edges,
            "metrics": metrics,
        }

    def log_metrics(self, loss, batch, pred_edges, true_edges, truth):

        # Get edge lists
        true_pred_edges = pred_edges[:, truth.bool()]

        # Get totals
        all_positive = pred_edges.shape[1]
        all_true = true_edges.shape[1]
        all_true_positive = true_pred_edges.shape[1]

        # Calculate metrics
        eff = all_true_positive / all_true 
        pur = all_true_positive / max(all_positive, 1)
        
        f1 = 2 * eff * pur / max(eff + pur, 1e-12)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"val_loss": loss, "lr": current_lr, "eff": eff, "pur": pur, "f1": f1,
                "all_positive": all_positive, "all_true": all_true, "true_positive": all_true_positive,
                "neighbors_per_hit": pred_edges.shape[1]/batch.x.shape[0],
                },
            batch_size=1, sync_dist=True
        )

        return {
            "eff": eff,
            "pur": pur,
            "f1": f1,
        }

    def log_clustering_metrics(self, embedding, batch):

        """
        Run a clustering loop, and keep track of the best score for each loop
        The loop is to run DBSCAN with a range of eps values, and keep track of the best score
        """

        pid = batch.pid
        energy_weighting = batch.x[:, 3]

        # Get the best clustering score
        best_score = 0
        best_eps = 0
        for eps in np.linspace(0.001, 0.2, 10):
            pred = DBSCAN(eps=eps, min_samples=1, metric="euclidean", n_jobs=-1).fit_predict(embedding.cpu().detach().numpy(), sample_weight=energy_weighting.cpu().numpy())
            score = self.get_clustering_score(pid.squeeze(), pred.squeeze(), energy_weighting*30)
            if score > best_score:
                best_score = score
                best_eps = eps

        print(f"Best clustering score: {best_score} at eps: {best_eps}")

        self.log_dict(
            {"best_clustering_score": best_score,
                "best_clustering_eps": best_eps,
                },
            batch_size=1, sync_dist=True
        )

    def get_clustering_score(self, pid, pred, energy_weighting):
        
        pid = pid.cpu().detach().numpy()
        energy_weighting = energy_weighting.cpu().numpy()

        return weighted_v_score(labels_true=pid, 
                                labels_pred=pred, 
                                labels_weight=energy_weighting)[2]


    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        knn_val = 500 if "knn_val" not in self.hparams else self.hparams["knn_val"]
        outputs = self.shared_evaluation(
            batch, self.hparams["radius"], knn_val
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        return self.shared_evaluation(batch, self.hparams["radius"], 1000)
    

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

        data = Data(
                        x = torch.from_numpy(event[['posx', 'posy', 'posz', 'E', 'T']].to_numpy()).float(),
                        pid = torch.from_numpy(event.clusterID.to_numpy().astype(np.int64)).long(),
                    )

        data.num_nodes = data.x.shape[0]

        # Create the true edge list for the event
        data.edge_index = torch.from_numpy(self.create_true_edge_list(event)).long()

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

    def create_true_edge_list(self, event):
        """
        Create the true edge list for the event. This is 
        """

        particle_groups = event.groupby('clusterID')
        edge_list = [
            np.array(list(itertools.combinations(group.hit_number.values, 2))).T
            for _, group in particle_groups if len(group) > 1
        ]

        return np.concatenate(edge_list, axis=1)