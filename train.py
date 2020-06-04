import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import PresenceDataset
from model import PresenceNetwork


def train():
    DATA_PATH = "/media/cimr/DATA_2/few-shot-object-detection/datasets/coco"
    ANNOTAIONS_PATH = "{}/annotations".format(DATA_PATH)

    TRAIN_ANNOTATIONS = "{}/instances_train2017.json".format(ANNOTAIONS_PATH)
    VAL_ANNOTATIONS = "{}/instances_val2017.json".format(ANNOTAIONS_PATH)

    TRAIN_IMAGES = "{}/train2017/".format(DATA_PATH)
    VAL_IMAGES = "{}/val2017/".format(DATA_PATH)

    train_data = PresenceDataset(data_path=TRAIN_IMAGES,
                                 annotation_path=TRAIN_ANNOTATIONS)
    val_data = PresenceDataset(data_path=VAL_IMAGES,
                               annotation_path=VAL_ANNOTATIONS)

    train_iterator = DataLoader(train_data, batch_size=4, num_workers=2,
                                shuffle=False)
    val_iterator = DataLoader(val_data, batch_size=4, num_workers=2,
                              shuffle=False)

    learning_rate = 1e2
    model = PresenceNetwork("resenet50").cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for idx, sample in enumerate(train_iterator):

        scene, target = sample["image_pair"]
        presence = sample["presence"]

        output = model(scene, target)
        loss = criterion(output, presence)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 50000:
            for idx, sample in enumerate(val_iterator):
                scene_val, target_val = sample["image_pair"]
                presence_val = sample["presence"]

                val_out = model(scene_val, target_val)

                val_score = criterion(val_out, presence_val)

                print("Score : {}".format(val_score))


if __name__ == "__main__":
    train()
