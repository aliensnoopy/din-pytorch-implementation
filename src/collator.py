import torch


class Collator:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(self, batch_inputs):
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
        effective_len = min(self.max_len, len(seq_data))
        padded_data = seq_data[-effective_len:] + [0] * (self.max_len - effective_len)
        mask = [1] * effective_len + [0] * (self.max_len - effective_len)

        return (padded_data, mask) if return_mask else padded_data
