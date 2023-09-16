import torch.cuda

from dataclasses import dataclass
from torch.utils.data import DataLoader
from src.config import DinConfig
from src.dataset import AmazonDataset


@dataclass
class TrainingArguments:
    model_config: DinConfig
    train_data_path: str
    test_data_path: str
    max_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    logging_steps: int
    evaluate_steps: int


class Trainer:
    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.args = args
        device_count = torch.cuda.device_count()

        train_dataset = AmazonDataset(data_file=args.train_data_path)
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=max(1, device_count) * args.batch_size,
                                           collate_fn=self._collate,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)

        test_dataset = AmazonDataset(data_file=args.test_data_path)
        self.test_dataloader = DataLoader(dataset=test_dataset,
                                          batch_size=max(1, device_count) * args.batch_size,
                                          collate_fn=self._collate,
                                          num_workers=8,
                                          shuffle=False,
                                          pin_memory=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Totally {len(train_dataset)} / {len(test_dataset)} (train / test) samples, "
              f"{len(self.train_dataloader)} / {len(self.test_dataloader)} (train / test) batches.")

    def _collate(self, batch_inputs):
        batch_user, batch_material, batch_category = [], [], []
        batch_historical_material, batch_historical_category = [], []
        batch_mask, batch_label = [], []

        for data_dict in batch_inputs:
            batch_user.append(data_dict["uid"])
            batch_material.append(data_dict["mid"])
            batch_category.append(data_dict["cat"])

            historical_material = data_dict["historical_mid"]
            padded_historical_material, mask = self._pad_sequence(historical_material, return_mask=True)
            batch_historical_material.append(padded_historical_material)

            historical_category = data_dict["historical_cat"]
            padded_historical_category = self._pad_sequence(historical_category)
            batch_historical_category.append(padded_historical_category)

            batch_mask.append(mask)
            batch_label.append(data_dict["label"])

        return {
            "uid": torch.tensor(batch_user, dtype=torch.long),
            "mid": torch.tensor(batch_material, dtype=torch.long),
            "cat": torch.tensor(batch_category, dtype=torch.long),
            "historical_mid": torch.tensor(batch_historical_material, dtype=torch.long),
            "historical_cat": torch.tensor(batch_historical_category, dtype=torch.long),
            "mask": torch.tensor(batch_mask, dtype=torch.long),
            "label": torch.tensor(batch_label, dtype=torch.long)
        }

    def _pad_sequence(self, seq_data, return_mask=False):
        effective_len = min(self.args.max_len, len(seq_data))
        padded_data = seq_data[-effective_len:] + [0] * (self.args.max_len - effective_len)
        mask = [1] * effective_len + [0] * (self.args.max_len - effective_len)

        return (padded_data, mask) if return_mask else padded_data

    def train(self):
        for epoch_idx in range(self.args.num_epochs):
            for batch_idx, batch_inputs in enumerate(self.train_dataloader, start=1):
                inputs = {k: v.to(self.device) for k, v in batch_inputs.items() if k != "label"}
                labels = batch_inputs["label"].to(self.device)

                print(inputs["uid"].shape)
                print(inputs["mid"].shape)
                print(inputs["cat"].shape)
                print(inputs["historical_mid"].shape)
                print(inputs["historical_cat"].shape)
                print(inputs["mask"].shape)
                print(labels.shape)
                break
            break
