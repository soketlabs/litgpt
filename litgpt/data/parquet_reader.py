####### first version.         ##############################
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from litgpt.data import DataModule
from litgpt.data.text_files import validate_tokenizer
from litgpt.tokenizer import Tokenizer



def tokenize_parquet(filename: str, tokenizer: Tokenizer):
    """
    Reads a parquet file and yields tokenized documents.
    CPT policy:
      - bos = False
      - eos = True
    """
    df = pd.read_parquet(filename)

    global_rank = int(os.environ["DATA_OPTIMIZER_GLOBAL_RANK"])
    num_workers = int(os.environ["DATA_OPTIMIZER_NUM_WORKERS"])
    local_rank = global_rank % num_workers

    for idx, text in enumerate(tqdm(df["text"], position=local_rank)):
        if not isinstance(text, str):
            continue

        text = text.strip()
        if not text:
            continue

        tokens = tokenizer.encode(
            text,
            bos=False,   
            eos=True    
        )
        
        if global_rank == 0 and idx < 3:
            print("Sample %d", idx)
            print("TEXT: %s...", text[:])
            print("TOKENS (head): %s", tokens[:10])
            print("TOKENS (tail): %s", tokens[-5:])
            print("Starts with BOS: %s", tokens[0] == tokenizer.bos_id)
            print("Ends with EOS: %s", tokens[-1] == tokenizer.eos_id)
        yield tokens



@dataclass
class ParquetCPT(DataModule):

    data_path: Path = Path("/home/sayantan/soket_litgpt/litgpt/data/data_folder/")
    parquet_file: str = "cleaned_output.parquet"
    val_split_fraction: float = 0.05
    num_workers: int = 4
    seed: int = 42 

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        self.train_dir = self.data_path / "train"
        self.val_dir = self.data_path / "val"
        self.train_parquet = self.data_path / "train_split.parquet"
        self.val_parquet = self.data_path / "val_split.parquet"


    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: int = -1,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # +1 for next-token prediction
        self.max_seq_length = max_seq_length + 1

    def prepare_data(self) -> None:
        from litdata import TokensLoader, optimize

        validate_tokenizer(self.tokenizer)

        parquet_path = self.data_path / self.parquet_file
        assert parquet_path.exists(), f"{parquet_path} not found"

        # Split data into train and val if not already done
        if not self.train_parquet.exists() or not self.val_parquet.exists():
            print(f"Splitting {parquet_path} into train and validation sets...")
            print(f"Train split: {1 - self.val_split_fraction:.1%}, Val split: {self.val_split_fraction:.1%}")
            
            df = pd.read_parquet(parquet_path)
            
            # Shuffle and split
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            split_idx = int(len(df) * (1 - self.val_split_fraction))
            
            train_df = df[:split_idx]
            val_df = df[split_idx:]
            
            print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
            
            # Save splits
            train_df.to_parquet(self.train_parquet, index=False)
            val_df.to_parquet(self.val_parquet, index=False)
            print(f"Saved train split to {self.train_parquet}")
            print(f"Saved validation split to {self.val_parquet}")
        else:
            print(f"Train and validation splits already exist, skipping data splitting.")

        # Tokenize training data
        if self.train_dir.exists():
            print(f"{self.train_dir} already exists, skipping train tokenization.")
        else:
            print("Tokenizing training data...")
            optimize(
                fn=partial(tokenize_parquet, tokenizer=self.tokenizer),
                inputs=[str(self.train_parquet)],
                output_dir=str(self.train_dir),
                num_workers=max(os.cpu_count() - 1, 1),
                chunk_bytes="200MB",
                item_loader=TokensLoader(),
            )

        # Tokenize validation data
        if self.val_dir.exists():
            print(f"{self.val_dir} already exists, skipping validation tokenization.")
        else:
            print("Tokenizing validation data...")
            optimize(
                fn=partial(tokenize_parquet, tokenizer=self.tokenizer),
                inputs=[str(self.train_parquet)],  #self.val_parquet
                output_dir=str(self.val_dir),
                num_workers=max(os.cpu_count() - 1, 1),
                chunk_bytes="200MB",
                item_loader=TokensLoader(),
            )


    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, StreamingDataLoader, TokensLoader

        dataset = StreamingDataset(
            input_dir=str(self.train_dir),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns validation dataloader.
        """
        from litdata.streaming import StreamingDataset, StreamingDataLoader, TokensLoader

        dataset = StreamingDataset(
            input_dir=str(self.val_dir),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=False,  # Don't shuffle validation data
        )

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,  # Keep all validation samples
        )





