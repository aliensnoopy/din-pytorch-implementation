import pandas as pd
from torch.utils.data import Dataset


DATA_COLUMNS = [
    "uid",
    "mid",
    "cat",
    "historical_mid",
    "historical_cat",
    "label"
]


class AmazonDataset(Dataset):
    def __init__(self, data_file: str):
        super().__init__()
        self.data = pd.read_parquet(data_file, columns=DATA_COLUMNS).to_dict(orient="records")
        for row in self.data:
            row["historical_mid"] = row["historical_mid"].tolist()
            row["historical_cat"] = row["historical_cat"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


