from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class Reader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        pass


class PandasDelimitedFileReader(Reader):

    def __init__(self,
                 train_edges: Path,
                 valid_edges: Path = None,
                 test_edges: Path = None,
                 columns: list = [0, 1, 2],
                 header_length: int = 0,
                 delim: str = "\t",
                 dtype: str = "int32"
                 ):

        super().__init__()

        self.train_edges = train_edges
        self.valid_edges = valid_edges
        self.test_edges = test_edges
        self.columns = columns
        self.header_length = header_length

        self.delim = delim
        self.dtype = dtype

        if len(self.columns) == 2:
            self.has_rels = False
        elif len(self.columns) == 3:
            self.has_rels = True
        else:
            raise RuntimeError(
                "Incorrect number of columns specified, expected length 2 or 3, received {}".format(len(self.columns)))

    def read(self):
        train_edges_df: pd.DataFrame = None
        valid_edges_df: pd.DataFrame = None
        test_edges_df: pd.DataFrame = None

        if self.valid_edges is None and self.test_edges is None:
            # no validation or test edges supplied

            # read in training edge list
            train_edges_df = pd.read_csv(self.train_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
            train_edges_df = train_edges_df[train_edges_df.columns[self.columns]]
        else:
            # predefined valid and test edges.
            train_edges_df = pd.read_csv(self.train_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
            valid_edges_df = pd.read_csv(self.valid_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
            test_edges_df = pd.read_csv(self.test_edges, delimiter=self.delim, skiprows=self.header_length, header=None)

            train_edges_df = train_edges_df[train_edges_df.columns[self.columns]]
            valid_edges_df = valid_edges_df[valid_edges_df.columns[self.columns]]
            test_edges_df = test_edges_df[test_edges_df.columns[self.columns]]

        return train_edges_df, valid_edges_df, test_edges_df