import lightning.pytorch as pl
import yaml
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from epic_clustering.utils import plot_clusters, get_cluster_pos
from epic_clustering.models import MemberClassification
from epic_clustering.scoring import weighted_v_score

def main():
    with open("member_classification.yaml") as f:
        member_classification_config = yaml.safe_load(f)
        
    model = MemberClassification(member_classification_config)
    model.setup(stage="fit")
    
    
    logger = WandbLogger(project=member_classification_config["project"])
    checkpoint_callback = ModelCheckpoint(
        monitor="auc",
        dirpath=member_classification_config["checkpoint_dir"],
        filename="member_classification-{epoch:02d}-{auc:.2f}",
        save_last=True, save_top_k=1,
    )

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=300, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()