# """
# Unified SFT DataModule (Parquet-based)
# Reads local Parquet file instead of Hugging Face datasets.

# Parquet columns:
# ['custom_id', 'prompt', 'system_instruction', 'thoughts', 'advisory']

# Mapping:
# - instruction = prompt
# - output = <unused0> thoughts <unused1>\n\n advisory
# """

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer



def format_parquet_agri(df: pd.DataFrame) -> List[Dict]:
    """
    Convert Parquet rows into SFT format.
    """
    formatted = []
    max_len = 0

    for _, row in df.iterrows():
        prompt = str(row.get("prompt", "")).strip()
        thoughts = str(row.get("thoughts", "")).strip()
        advisory = str(row.get("advisory", "")).strip()

        # Skip broken rows
        if not prompt or not advisory:
            continue

        output = (
            "<unused0>"
            f"{thoughts}"
            "<unused1>\n\n"
            f"{advisory}"
        )

        formatted.append({
            "instruction": prompt,
            "input": "",
            "output": output,
        })
        total_text = prompt + output
        max_len = max(max_len, len(total_text))
    
    print(f"Max text length in dataset: {max_len} characters")
        #print(f"Formatted row: instruction={prompt}, output={output}")
    return formatted



@dataclass
class ParquetSFTDataModule(DataModule):
    """
    SFT DataModule reading from a local Parquet file.
    """

    parquet_path: str = "/projects/data/teams/tts_team/agri_training/data_folder/merged_all.parquet"

    mask_prompt: bool = False
    val_split_fraction: float = 0.1
    prompt_style: Union[str, PromptStyle] = "gemma"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 1

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

   
    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

 
    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

   
    def prepare_data(self) -> None:
        """
        Validate Parquet file presence.
        """
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        #print(f"Found Parquet file: {self.parquet_path}")


    def setup(self, stage: str = "") -> None:
        """
        Load Parquet, format, split, and build SFT datasets.
        """
        print("Loading Parquet dataset...")
        df = pd.read_parquet(self.parquet_path)

        # print("Formatting dataset...")
        formatted_data = format_parquet_agri(df)
        print(f"Total formatted samples: {len(formatted_data)}")

        if not formatted_data:
            raise RuntimeError("No valid samples found after formatting")

        # Debug: print one row
        # print("\nSample formatted row:")
        # print(formatted_data[0])

        # Train / validation split
        train_data, val_data = random_split(
            formatted_data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.train_dataset = SFTDataset(
            data=list(train_data),
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        #print(f"the train dataset is : {train_data.dataset}")

        self.test_dataset  = SFTDataset(
            data=list(val_data),
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        #print(f"the test dataset is : {val_data.dataset}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset ,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )






















# # ################# limit check ######################

# # """
# # Unified SFT DataModule (Parquet-based)
# # Reads local Parquet file instead of Hugging Face datasets.

# # Parquet columns:
# # ['custom_id', 'prompt', 'system_instruction', 'thoughts', 'advisory']

# # Mapping:
# # - instruction = prompt
# # - output = <unused0> thoughts <unused1>\n\n advisory
# # """

# # import os
# # from dataclasses import dataclass, field
# # from typing import List, Dict, Optional, Union

# # import pandas as pd
# # import torch
# # from torch.utils.data import DataLoader, random_split

# # from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# # from litgpt.prompts import PromptStyle
# # from litgpt.tokenizer import Tokenizer

# # # =========================
# # # TRAINING CAP (used ONLY for SFT)
# # # =========================
# # TRAIN_MAX_SEQ_LEN = 14000


# # # =========================
# # # FORMATTER
# # # =========================
# # def format_parquet_agri(df: pd.DataFrame) -> List[Dict]:
# #     formatted = []
# #     max_len = 0

# #     for _, row in df.iterrows():
# #         prompt = str(row.get("prompt", "")).strip()
# #         thoughts = str(row.get("thoughts", "")).strip()
# #         advisory = str(row.get("advisory", "")).strip()

# #         if not prompt or not advisory:
# #             continue

# #         output = (
# #             "<unused0>"
# #             f"{thoughts}"
# #             "<unused1>\n\n"
# #             f"{advisory}"
# #         )

# #         formatted.append({
# #             "instruction": prompt,
# #             "input": "",
# #             "output": output,
# #         })

# #         max_len = max(max_len, len(prompt + output))

# #     print(f"Max text length (characters): {max_len}")
# #     return formatted


# # # =========================
# # # DATAMODULE
# # # =========================
# # @dataclass
# class ParquetSFTDataModule(DataModule):
#     parquet_path: str

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "gemma"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 1

#     tokenizer: Optional[Tokenizer] = field(default=None, init=False)
#     batch_size: int = field(default=1, init=False)
#     max_seq_length: int = field(default=-1, init=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False)
#     val_dataset: Optional[SFTDataset] = field(default=None, init=False)

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#     def connect(
#         self,
#         tokenizer: Tokenizer,
#         batch_size: int = 1,
#         max_seq_length: Optional[int] = None,
#     ):
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = (
#             min(max_seq_length, TRAIN_MAX_SEQ_LEN)
#             if max_seq_length is not None
#             else TRAIN_MAX_SEQ_LEN
#         )
#         print(f"[SFT] Using max_seq_length = {self.max_seq_length}")

#     def prepare_data(self):
#         if not os.path.exists(self.parquet_path):
#             raise FileNotFoundError(self.parquet_path)

#     def setup(self, stage: str = ""):
#         df = pd.read_parquet(self.parquet_path)
#         formatted = format_parquet_agri(df)

#         n_total = len(formatted)
#         n_val = int(n_total * self.val_split_fraction)
#         n_train = n_total - n_val

#         train_data, val_data = random_split(
#             formatted,
#             [n_train, n_val],
#             generator=torch.Generator().manual_seed(self.seed),
#         )

#         self.train_dataset = SFTDataset(
#             data=list(train_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         self.val_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         print(f"Training samples: {len(train_data)}")
#         print(f"Validation samples: {len(val_data)}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )



# if __name__ == "__main__":
#     # from pathlib import Path
    

#     # parquet_path = "/home/sayantan/soket_litgpt/litgpt/data/data_folder/parsed_batch_part_001.parquet"
#     # checkpoint_dir = Path(
#     #     "/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it"
#     # )

#     # tokenizer = Tokenizer(checkpoint_dir)

#     # df = pd.read_parquet(parquet_path)

#     # max_tokens = 0
#     # lengths = []

#     # print("\n🔍 Computing TRUE token lengths (no truncation)...")

#     # for _, row in df.iterrows():
#     #     prompt = str(row.get("prompt", "")).strip()
#     #     thoughts = str(row.get("thoughts", "")).strip()
#     #     advisory = str(row.get("advisory", "")).strip()

#     #     if not prompt or not advisory:
#     #         continue

#     #     text = (
#     #         prompt
#     #         + "<unused0>"
#     #         + thoughts
#     #         + "<unused1>\n\n"
#     #         + advisory
#     #     )

#     #     tokens = tokenizer.encode(text, bos=True, eos=True)
#     #     n = len(tokens)
#     #     lengths.append(n)
#     #     max_tokens = max(max_tokens, n)

#     # lengths = sorted(lengths)

#     # def pct(p):
#     #     return lengths[int(len(lengths) * p)]
    
    

#     # print(" TRUE TOKEN LENGTH STATS (NO TRUNCATION)")
#     # print(f"Samples analyzed : {len(lengths)}")
#     # print(f"Max tokens       : {max_tokens}")
#     # print(f"P50 tokens       : {pct(0.50)}")
#     # print(f"P90 tokens       : {pct(0.90)}")
#     # print(f"P95 tokens       : {pct(0.95)}")
#     # print(f"P99 tokens       : {pct(0.99)}")

#     # print("\nAnalysis complete. You can now choose TRAIN_MAX_SEQ_LEN safely.")
    
    
#     ########## second try ##########
    
    
#     #     from pathlib import Path

#     #     parquet_path = "/home/sayantan/soket_litgpt/litgpt/data/data_folder/parsed_batch_part_001.parquet"
#     #     checkpoint_dir = Path(
#     #     "/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it"
#     # )

#     #     tokenizer = Tokenizer(checkpoint_dir)
#     #     df = pd.read_parquet(parquet_path)

#     #     lengths = []
#     #     records = []

#     #     print("\n🔍 Computing TRUE token lengths (no truncation)...")

#     #     for idx, row in df.iterrows():
#     #         prompt = str(row.get("prompt", "")).strip()
#     #         thoughts = str(row.get("thoughts", "")).strip()
#     #         advisory = str(row.get("advisory", "")).strip()

#     #         if not prompt or not advisory:
#     #             continue

#     #         text = (
#     #             prompt
#     #             + "<unused0>"
#     #             + thoughts
#     #             + "<unused1>\n\n"
#     #             + advisory
#     #         )

#     #         tokens = tokenizer.encode(text, bos=True, eos=True)
#     #         n = len(tokens)

#     #         lengths.append(n)
#     #         records.append((idx, n, prompt, thoughts, advisory))

#     #     lengths.sort()

#     #     def pct(p):
#     #         return lengths[int(len(lengths) * p)]

#     #     print("\n TRUE TOKEN LENGTH STATS (NO TRUNCATION)")
#     #     print(f"Samples analyzed : {len(lengths)}")
#     #     print(f"Max tokens       : {max(lengths)}")
#     #     print(f"P50 tokens       : {pct(0.50)}")
#     #     print(f"P90 tokens       : {pct(0.90)}")
#     #     print(f"P95 tokens       : {pct(0.95)}")
#     #     print(f"P99 tokens       : {pct(0.99)}")

#     #     # =========================
#     #     # 🔥 TOP-K LONGEST SAMPLES
#     #     # =========================
#     #     TOP_K = 5
#     #     records.sort(key=lambda x: x[1], reverse=True)

#     #     print("\n🔥 TOP LONGEST SAMPLES")
#     #     for rank, (idx, n, prompt, thoughts, advisory) in enumerate(records[:TOP_K], 1):
#     #         print(f"\nRank #{rank}")
#     #         print(f"Row index   : {idx}")
#     #         print(f"Token count : {n}")
#     #         print(f"Prompt      : {prompt[:]}...")
#     #         print(f"Thoughts    : {thoughts[:]}...")
#     #         print(f"Advisory    : {advisory[:]}...")

#     #     # =========================
#     #     # 💾 SAVE EXTREME OUTLIERS
#     #     # =========================
#     #     OUTLIER_THRESHOLD = 10000

#     #     outliers = [
#     #         {
#     #             "row_index": idx,
#     #             "token_count": n,
#     #             "prompt": prompt,
#     #             "thoughts": thoughts,
#     #             "advisory": advisory,
#     #         }
#     #         for idx, n, prompt, thoughts, advisory in records
#     #         if n > OUTLIER_THRESHOLD
#     #     ]

#     #     if outliers:
#     #         outlier_df = pd.DataFrame(outliers)
#     #         outlier_df.to_csv("token_outliers.csv", index=False)
#     #         print(f"\n💾 Saved {len(outliers)} outliers to token_outliers.csv")
#     #     else:
#     #         print("\n✅ No extreme outliers found")

#     #     print("\n✅ Analysis complete. You can now safely choose TRAIN_MAX_SEQ_LEN.")
        
        
        
#     #     max_tokens = max(lengths)

#     #     count_max = sum(1 for n in lengths if n == max_tokens)

#     #     print("\n📌 MAX TOKEN OCCURRENCE")
#     #     print(f"Max token length : {max_tokens}")
#     #     print(f"Number of samples with max length : {count_max}")
#     #     print(f"Percentage of dataset : {100 * count_max / len(lengths):.4f}%")


# ####### third try ##########
#     from pathlib import Path

#     parquet_path = "/home/sayantan/soket_litgpt/litgpt/data/data_folder/merged_all.parquet"
#     checkpoint_dir = Path(
#         "/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it"
#     )

#     tokenizer = Tokenizer(checkpoint_dir)
#     df = pd.read_parquet(parquet_path)

#     lengths = []
#     records = []

#     print("\n🔍 Computing TRUE token lengths (no truncation)...")

#     for idx, row in df.iterrows():
#         prompt = str(row.get("prompt", "")).strip()
#         thoughts = str(row.get("thoughts", "")).strip()
#         advisory = str(row.get("advisory", "")).strip()

#         if not prompt or not advisory:
#             continue

#         text = (
#             prompt
#             + "<unused0>"
#             + thoughts
#             + "<unused1>\n\n"
#             + advisory
#         )

#         n_tokens = len(tokenizer.encode(text, bos=True, eos=True))

#         lengths.append(n_tokens)
#         records.append((idx, n_tokens, prompt, thoughts, advisory))

#     lengths.sort()

#     def pct(p):
#         return lengths[int(len(lengths) * p)]

#     max_tokens = max(lengths)
#     count_max = sum(1 for n in lengths if n == max_tokens)

#     print("\n📊 TRUE TOKEN LENGTH STATS")
#     print(f"Samples analyzed : {len(lengths)}")
#     print(f"Max tokens       : {max_tokens}")
#     print(f"P50 tokens       : {pct(0.50)}")
#     print(f"P90 tokens       : {pct(0.90)}")
#     print(f"P95 tokens       : {pct(0.95)}")
#     print(f"P99 tokens       : {pct(0.99)}")

#     print("\n📌 MAX TOKEN OCCURRENCE")
#     print(f"Number of samples with max length : {count_max}")
#     print(f"Percentage of dataset             : {100 * count_max / len(lengths):.4f}%")

#     # =========================
#     # 💾 SAVE ONLY MAX-TOKEN SAMPLES
#     # =========================
#     max_token_samples = [
#         {
#             "row_index": idx,
#             "token_count": n,
#             "prompt": prompt,
#             "thoughts": thoughts,
#             "advisory": advisory,
#         }
#         for idx, n, prompt, thoughts, advisory in records
#         if n == max_tokens
#     ]

#     df_max = pd.DataFrame(max_token_samples)
#     print(df_max['advisory'][:])
#     df_max.to_csv("max_token_samples.csv", index=False)

#     print(f"\n💾 Saved {len(df_max)} sample(s) to max_token_samples.csv")
#     print("\n✅ Analysis complete.")