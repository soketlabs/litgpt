# # Copyright Lightning AI. Licensed under the Apache License 2.0
# """Unified SFT DataModule for Medical-O1 and Indic-Instruct"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal
# from venv import logger

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer


# # ---------------------------------------------------------------------
# # Dataset formatters
# # ---------------------------------------------------------------------

# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_indic_instruct(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["messages"]

#         if include_multiturn:
#             for i in range(0, len(convo) - 1, 2):
#                 formatted.append({
#                     "instruction": convo[i]["content"],
#                     "input": "",
#                     "output": convo[i + 1]["content"],
#                 })
#         else:
#             formatted.append({
#                 "instruction": convo[0]["content"],
#                 "input": "",
#                 "output": convo[1]["content"],
#             })

#     return formatted


# def data_collator(self, batch):
#     logger.debug(f"Collating batch of size {len(batch)}")


# # ---------------------------------------------------------------------
# # Unified DataModule
# # ---------------------------------------------------------------------

# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified DataModule with dataset switch.

#     dataset_type:
#         - "medical_o1"
#         - "indic_instruct"
#     """

#     # ---- Switch ----
#     dataset_type: Literal["medical_o1", "indic_instruct"] = "medical_o1"

#     # ---- Common training params ----
#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True

#     # ---- Dataset repos ----
#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     indic_repo_id: str = "ai4bharat/indic-instruct-data-v0.1"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

#     # ---- Runtime ----
#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

#     # -----------------------------------------------------------------

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {"medical_o1", "indic_instruct"}:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 1,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length

#     # -----------------------------------------------------------------

#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)
#         else:
#             load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)

#     # -----------------------------------------------------------------

#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             dataset = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(dataset["train"])

#         else:
#             dataset = load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)
#             data = format_indic_instruct(
#                 dataset["hi"],
#                 self.include_multiturn_conversations,
#             )

#         # ---- Train / Val split ----
#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
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

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#     # -----------------------------------------------------------------

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )




############################## three dataset ###################



# """Unified SFT DataModule for Medical-O1, Indic-Instruct, and KissanAI"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal
# from venv import logger

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer




# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_indic_instruct(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["messages"]

#         if include_multiturn:
#             for i in range(0, len(convo) - 1, 2):
#                 formatted.append({
#                     "instruction": convo[i]["content"],
#                     "input": "",
#                     "output": convo[i + 1]["content"],
#                 })
#         else:
#             formatted.append({
#                 "instruction": convo[0]["content"],
#                 "input": "",
#                 "output": convo[1]["content"],
#             })

#     return formatted


# def format_kissanai(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["conversations"]

#         user_turns = [x["value"] for x in convo if x.get("from") == "user"]
#         assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

#         if not user_turns or not assistant_turns:
#             continue

#         def replace_think_tokens(text: str) -> str:
#             return (
#                 text
#                 .replace("<think>", "<unused0>")
#                 .replace("</think>", "<unused1>")
#             )

#         if include_multiturn:
#             for u, a in zip(user_turns, assistant_turns):
#                 formatted.append({
#                     "instruction": u.strip(),
#                     "input": "",
#                     "output": replace_think_tokens(a.strip()),
#                 })
#         else:
#             formatted.append({
#                 "instruction": user_turns[0].strip(),
#                 "input": "",
#                 "output": replace_think_tokens(assistant_turns[0].strip()),
#             })

#     if formatted:
#         logger.info(f"formatted_ds[0]: {formatted[0]}")

#     return formatted


# def data_collator(self, batch):
#     logger.debug(f"Collating batch of size {len(batch)}")



# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified SFT DataModule.

#     dataset_type:
#       - medical_o1
#       - indic_instruct
#       - kissanai
#     """

#     dataset_type: Literal[
#         "medical_o1",
#         "indic_instruct",
#         "kissanai",
#     ] = "medical_o1"

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True


#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     indic_repo_id: str = "ai4bharat/indic-instruct-data-v0.1"
#     kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))


#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)



#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {
#             "medical_o1",
#             "indic_instruct",
#             "kissanai",
#         }:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 2,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length


#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)

#         elif self.dataset_type == "indic_instruct":
#             load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)

#         else:
#             load_dataset(self.kissanai_repo_id, token=self.access_token)



#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(ds["train"])

#         elif self.dataset_type == "indic_instruct":
#             ds = load_dataset(self.indic_repo_id, "anudesh", token=self.access_token)
#             data = format_indic_instruct(
#                 ds["hi"],
#                 self.include_multiturn_conversations,
#             )

#         else:
#             ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
#             data = format_kissanai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )

#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
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

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )


#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )



# ############################## two dataset  with <unused0> and <unused1>###################

# """Unified SFT DataModule for Medical-O1 and KissanAI"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal
# from venv import logger

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer


# # ---------------------------------------------------------------------
# # Dataset formatters
# # ---------------------------------------------------------------------

# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_kissanai(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["conversations"]

#         user_turns = [x["value"] for x in convo if x.get("from") == "user"]
#         assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

#         if not user_turns or not assistant_turns:
#             continue

#         def replace_think_tokens(text: str) -> str:
#             return (
#                 text
#                 .replace("<think>", "<unused0>")
#                 .replace("</think>", "<unused1>")
#             )

#         if include_multiturn:
#             for u, a in zip(user_turns, assistant_turns):
#                 formatted.append({
#                     "instruction": u.strip(),
#                     "input": "",
#                     "output": replace_think_tokens(a.strip()),
#                 })
#         else:
#             formatted.append({
#                 "instruction": user_turns[0].strip(),
#                 "input": "",
#                 "output": replace_think_tokens(assistant_turns[0].strip()),
#             })

#     if formatted:
#         logger.info(f"formatted_ds[0]: {formatted[0]}")

#     return formatted


# # ---------------------------------------------------------------------
# # Unified DataModule
# # ---------------------------------------------------------------------

# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified SFT DataModule.

#     dataset_type:
#       - medical_o1
#       - kissanai
#     """

#     dataset_type: Literal[
#         "medical_o1",
#         "kissanai",
#     ] = "medical_o1"

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True

#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {"medical_o1", "kissanai"}:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 2,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length

#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)
#         else:
#             load_dataset(self.kissanai_repo_id, token=self.access_token)

#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(ds["train"])
#         else:
#             ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
#             data = format_kissanai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )

#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
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

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )





###################################################### four datatset  with <unused0> and <unused1>###################




# """Unified SFT DataModule for Medical-O1, KissanAI, AgThoughts, and RespiratoryAI"""

# import os
# import re
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer


# # ---------------------------------------------------------------------
# # Dataset formatters
# # ---------------------------------------------------------------------

# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_kissanai(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["conversations"]

#         user_turns = [x["value"] for x in convo if x.get("from") == "user"]
#         assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

#         if not user_turns or not assistant_turns:
#             continue

#         def replace_think_tokens(text: str) -> str:
#             return (
#                 text
#                 .replace("<think>", "<unused0>")
#                 .replace("</think>", "<unused1>")
#             )

#         if include_multiturn:
#             for u, a in zip(user_turns, assistant_turns):
#                 formatted.append({
#                     "instruction": u.strip(),
#                     "input": "",
#                     "output": replace_think_tokens(a.strip()),
#                 })
#         else:
#             formatted.append({
#                 "instruction": user_turns[0].strip(),
#                 "input": "",
#                 "output": replace_think_tokens(assistant_turns[0].strip()),
#             })

#     return formatted


# def format_agthoughts(dataset_partition: List[Dict]) -> List[Dict]:
#     """Format AgThoughts dataset with <unused0> reasoning <unused1> answer pattern"""
#     formatted_ds = []

#     for entry in dataset_partition:
#         question = (entry.get("Question") or "").strip()
#         reasoning = (entry.get("Reasoning Traces") or "").strip()
#         answer = (entry.get("Answer") or "").strip()

#         # Skip completely broken rows
#         if not question or not answer:
#             continue

#         output_text = (
#             "<unused0>"
#             f"{reasoning}"
#             "<unused1>\n\n"
#             f"{answer}"
#         )

#         formatted_ds.append({
#             "instruction": question,
#             "input": "",
#             "output": output_text,
#         })

#     return formatted_ds


# def format_respiratory_ai(
#     dataset_partition: List[Dict], 
#     include_multi_turn_conversations: bool = True
# ) -> List[Dict]:
#     """Format RespiratoryAI dataset (big-reasoning-traces)"""
#     formatted_ds = []
    
#     for entry in dataset_partition:
#         # Extract data from the new dataset structure
#         text = entry.get("text", "")
#         prompt = entry.get("prompt", "")
#         response = entry.get("response", "")
        
#         # Skip if essential fields are missing
#         if not prompt or not response:
#             continue
        
#         def extract_and_replace_think_content(text: str) -> str:
#             """Extract content between <think> tags and replace with special tokens."""
#             # Find all think content
#             think_pattern = r"<think>(.*?)</think>"
#             think_matches = re.findall(think_pattern, text, re.DOTALL)
            
#             # Find all answer content
#             answer_pattern = r"<answer>(.*?)</answer>"
#             answer_matches = re.findall(answer_pattern, text, re.DOTALL)
            
#             # Combine think content (with token replacement) and answer content
#             result_parts = []
            
#             # Add think content with token replacement
#             for think_content in think_matches:
#                 replaced_think = think_content.replace("<think>", "<unused0>").replace("</think>", "<unused1>")
#                 result_parts.append(replaced_think.strip())
            
#             # Add answer content
#             for answer_content in answer_matches:
#                 result_parts.append(answer_content.strip())
            
#             # Join all parts with spaces
#             return " ".join(result_parts) if result_parts else text
        
#         # Process the response to extract and format required content
#         formatted_output = extract_and_replace_think_content(response)
        
#         # Create the formatted entry
#         formatted_ds.append({
#             "instruction": prompt.strip(),
#             "input": "",  
#             "output": formatted_output.strip(),
#         })
    
#     return formatted_ds


# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified SFT DataModule for agricultural and reasoning datasets.

#     dataset_type:
#       - medical_o1: Medical reasoning with CoT
#       - kissanai: Climate/agricultural thinking dataset
#       - agthoughts: Agricultural Q&A with reasoning traces
#       - respiratory_ai: RespiratoryAI reasoning dataset
#     """

#     dataset_type: Literal[
#         "medical_o1",
#         "kissanai",
#         "agthoughts",
#         "respiratory_ai",
#     ] = "agthoughts"  # Default to agricultural dataset

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "pragna-1b"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True

#     # Dataset repository IDs
#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"
#     agthoughts_repo_id: str = "BGLab/AgThoughts"
#     respiratory_repo_id: str = "allenai/big-reasoning-traces"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {"medical_o1", "kissanai", "agthoughts", "respiratory_ai"}:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 2,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length

#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)
#         elif self.dataset_type == "kissanai":
#             load_dataset(self.kissanai_repo_id, token=self.access_token)
#         elif self.dataset_type == "agthoughts":
#             load_dataset(self.agthoughts_repo_id, token=self.access_token)
#         else:  # respiratory_ai
#             load_dataset(self.respiratory_repo_id, "DeepSeek", token=self.access_token)

#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(ds["train"])
#         elif self.dataset_type == "kissanai":
#             ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
#             data = format_kissanai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )
#         elif self.dataset_type == "agthoughts":
#             ds = load_dataset(self.agthoughts_repo_id, token=self.access_token)
#             data = format_agthoughts(ds["train"])
#         else:  # respiratory_ai
#             ds = load_dataset(self.respiratory_repo_id, "DeepSeek", token=self.access_token)
#             data = format_respiratory_ai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )

#         # Print dataset statistics
#         print(f"Loaded {len(data)} examples from {self.dataset_type} dataset")
#         if data:
#             print(f"First example structure: {list(data[0].keys())}")
#             print(f"Instruction sample: {data[0]['instruction'][:100]}...")
#             print(f"Output sample: {data[0]['output'][:100]}...")

#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
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

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         print(f"Training samples: {len(train_data)}")
#         print(f"Validation samples: {len(val_data)}")

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )













########################################## three dataset  with <unused0> and <unused1>###################

# # Copyright Lightning AI. Licensed under the Apache License 2.0
# """Unified SFT DataModule for Medical-O1, KissanAI, and AgThoughts"""

# import os
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Union, Literal

# import torch
# from torch.utils.data import DataLoader, random_split

# from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
# from litgpt.prompts import PromptStyle
# from litgpt.tokenizer import Tokenizer




# def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
#     """Question + <unused0> CoT <unused1> + Final Answer"""
#     formatted = []

#     for entry in dataset_partition:
#         output = (
#             "<unused0>"
#             f"{entry['Complex_CoT'].strip()}"
#             "<unused1>\n\n"
#             f"{entry['Response'].strip()}"
#         )

#         formatted.append({
#             "instruction": entry["Question"].strip(),
#             "input": "",
#             "output": output,
#         })

#     return formatted


# def format_kissanai(
#     dataset_partition: List[Dict],
#     include_multiturn: bool,
# ) -> List[Dict]:
#     formatted = []

#     for entry in dataset_partition:
#         convo = entry["conversations"]

#         user_turns = [x["value"] for x in convo if x.get("from") == "user"]
#         assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

#         if not user_turns or not assistant_turns:
#             continue

#         def replace_think_tokens(text: str) -> str:
#             return (
#                 text
#                 .replace("<think>", "<unused0>")
#                 .replace("</think>", "<unused1>")
#             )

#         if include_multiturn:
#             for u, a in zip(user_turns, assistant_turns):
#                 formatted.append({
#                     "instruction": u.strip(),
#                     "input": "",
#                     "output": replace_think_tokens(a.strip()),
#                 })
#         else:
#             formatted.append({
#                 "instruction": user_turns[0].strip(),
#                 "input": "",
#                 "output": replace_think_tokens(assistant_turns[0].strip()),
#             })

#     return formatted


# def format_agthoughts(dataset_partition: List[Dict]) -> List[Dict]:
#     """Format AgThoughts dataset with <unused0> reasoning <unused1> answer pattern"""
#     formatted_ds = []

#     for entry in dataset_partition:
#         question = (entry.get("Question") or "").strip()
#         reasoning = (entry.get("Reasoning Traces") or "").strip()
#         answer = (entry.get("Answer") or "").strip()

#         # Skip completely broken rows
#         if not question or not answer:
#             continue

#         output_text = (
#             "<unused0>"
#             f"{reasoning}"
#             "<unused1>\n\n"
#             f"{answer}"
#         )

#         formatted_ds.append({
#             "instruction": question,
#             "input": "",
#             "output": output_text,
#         })

#     return formatted_ds




# @dataclass
# class UnifiedSFTDataModule(DataModule):
#     """
#     Unified SFT DataModule for agricultural reasoning datasets.

#     dataset_type:
#       - medical_o1: Medical reasoning with CoT
#       - kissanai: Climate/agricultural thinking dataset
#       - agthoughts: Agricultural Q&A with reasoning traces
#     """

#     dataset_type: Literal[
#         "medical_o1",
#         "kissanai",
#         "agthoughts",
#     ] = "agthoughts"  

#     mask_prompt: bool = False
#     val_split_fraction: float = 0.1
#     prompt_style: Union[str, PromptStyle] = "gemma"
#     ignore_index: int = -100
#     seed: int = 42
#     num_workers: int = 4
#     include_multiturn_conversations: bool = True

#     # Dataset repository IDs
#     medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
#     kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"
#     agthoughts_repo_id: str = "BGLab/AgThoughts"

#     access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

#     tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
#     batch_size: int = field(default=1, init=False, repr=False)
#     max_seq_length: int = field(default=-1, init=False, repr=False)

#     train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
#     test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

#     def __post_init__(self):
#         super().__init__()
#         if isinstance(self.prompt_style, str):
#             self.prompt_style = PromptStyle.from_name(self.prompt_style)

#         if self.dataset_type not in {"medical_o1", "kissanai", "agthoughts"}:
#             raise ValueError(f"Invalid dataset_type: {self.dataset_type}")

#     def connect(
#         self,
#         tokenizer: Optional[Tokenizer] = None,
#         batch_size: int = 2,
#         max_seq_length: Optional[int] = None,
#     ) -> None:
#         self.tokenizer = tokenizer
#         self.batch_size = batch_size
#         self.max_seq_length = -1 if max_seq_length is None else max_seq_length

#     def prepare_data(self) -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             load_dataset(self.medical_repo_id, "en", token=self.access_token)
#         elif self.dataset_type == "kissanai":
#             load_dataset(self.kissanai_repo_id, token=self.access_token)
#         else: 
#             load_dataset(self.agthoughts_repo_id, token=self.access_token)

#     def setup(self, stage: str = "") -> None:
#         from datasets import load_dataset

#         if self.dataset_type == "medical_o1":
#             ds = load_dataset(self.medical_repo_id, "en", token=self.access_token)
#             data = format_medical_o1(ds["train"])
#         elif self.dataset_type == "kissanai":
#             ds = load_dataset(self.kissanai_repo_id, token=self.access_token)
#             data = format_kissanai(
#                 ds["train"],
#                 self.include_multiturn_conversations,
#             )
#         else: 
#             ds = load_dataset(self.agthoughts_repo_id, token=self.access_token)
#             data = format_agthoughts(ds["train"])

#         # Print dataset statistics
#         print(f"Loaded {len(data)} examples from {self.dataset_type} dataset")
#         if data:
#             print(f"First example structure: {list(data[0].keys())}")
#             print(f"Instruction sample: {data[0]['instruction'][:100]}...")
#             print(f"Output sample: {data[0]['output'][:100]}...")

#         train_data, val_data = random_split(
#             data,
#             [1.0 - self.val_split_fraction, self.val_split_fraction],
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

#         self.test_dataset = SFTDataset(
#             data=list(val_data),
#             tokenizer=self.tokenizer,
#             prompt_style=self.prompt_style,
#             max_seq_length=self.max_seq_length,
#             mask_prompt=self.mask_prompt,
#             ignore_index=self.ignore_index,
#         )

#         print(f"Training samples: {len(train_data)}")
#         print(f"Validation samples: {len(val_data)}")

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             generator=torch.Generator().manual_seed(self.seed),
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=get_sft_collate_fn(
#                 max_seq_length=self.max_seq_length,
#                 ignore_index=self.ignore_index,
#             ),
#         )



# Copyright Lightning AI. Licensed under the Apache License 2.0
"""Unified SFT DataModule for Medical-O1, KissanAI, and AgThoughts - Combined Version"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


def format_medical_o1(dataset_partition: List[Dict]) -> List[Dict]:
    """Question + <unused0> CoT <unused1> + Final Answer"""
    formatted = []

    for entry in dataset_partition:
        output = (
            "<unused0>"
            f"{entry['Complex_CoT'].strip()}"
            "<unused1>\n\n"
            f"{entry['Response'].strip()}"
        )

        formatted.append({
            "instruction": entry["Question"].strip(),
            "input": "",
            "output": output,
        })

    return formatted


def format_kissanai(
    dataset_partition: List[Dict],
    include_multiturn: bool,
) -> List[Dict]:
    formatted = []

    for entry in dataset_partition:
        convo = entry["conversations"]

        user_turns = [x["value"] for x in convo if x.get("from") == "user"]
        assistant_turns = [x["value"] for x in convo if x.get("from") == "assistant"]

        if not user_turns or not assistant_turns:
            continue

        def replace_think_tokens(text: str) -> str:
            return (
                text
                .replace("<think>", "<unused0>")
                .replace("</think>", "<unused1>")
            )

        if include_multiturn:
            for u, a in zip(user_turns, assistant_turns):
                formatted.append({
                    "instruction": u.strip(),
                    "input": "",
                    "output": replace_think_tokens(a.strip()),
                })
        else:
            formatted.append({
                "instruction": user_turns[0].strip(),
                "input": "",
                "output": replace_think_tokens(assistant_turns[0].strip()),
            })

    return formatted


def format_agthoughts(dataset_partition: List[Dict]) -> List[Dict]:
    """Format AgThoughts dataset with <unused0> reasoning <unused1> answer pattern"""
    formatted_ds = []

    for entry in dataset_partition:
        question = (entry.get("Question") or "").strip()
        reasoning = (entry.get("Reasoning Traces") or "").strip()
        answer = (entry.get("Answer") or "").strip()

        # Skip completely broken rows
        if not question or not answer:
            continue

        output_text = (
            "<unused0>"
            f"{reasoning}"
            "<unused1>\n\n"
            f"{answer}"
        )

        formatted_ds.append({
            "instruction": question,
            "input": "",
            "output": output_text,
        })

    return formatted_ds


@dataclass
class CombinedSFTDataModule(DataModule):
    """
    Combined SFT DataModule that merges all three agricultural reasoning datasets:
    - medical_o1: Medical reasoning with CoT
    - kissanai: Climate/agricultural thinking dataset
    - agthoughts: Agricultural Q&A with reasoning traces
    
    All datasets are loaded, formatted, and combined into a single training set.
    """

    mask_prompt: bool = False
    val_split_fraction: float = 0.1
    prompt_style: Union[str, PromptStyle] = "gemma"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    include_multiturn_conversations: bool = True

    # Dataset repository IDs
    medical_repo_id: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    kissanai_repo_id: str = "KissanAI/Thinking-climate-100k"
    agthoughts_repo_id: str = "BGLab/AgThoughts"

    # Optional: Control which datasets to include
    include_medical_o1: bool = True
    include_kissanai: bool = True
    include_agthoughts: bool = True

    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 2,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        """Download all enabled datasets"""
        from datasets import load_dataset

        if self.include_medical_o1:
            print("Preparing Medical-O1 dataset")
            load_dataset(self.medical_repo_id, "en", token=self.access_token)
        
        if self.include_kissanai:
            print("Preparing KissanAI dataset")
            load_dataset(self.kissanai_repo_id, token=self.access_token)
        
        if self.include_agthoughts:
            print("Preparing AgThoughts dataset")
            load_dataset(self.agthoughts_repo_id, token=self.access_token)

    def setup(self, stage: str = "") -> None:
        """Load, format, and combine all datasets"""
        from datasets import load_dataset

        combined_data = []

        # Load and format Medical-O1
        if self.include_medical_o1:
            print("Loading and formatting Medical-O1 dataset")
            ds_medical = load_dataset(self.medical_repo_id, "en", token=self.access_token)
            medical_data = format_medical_o1(ds_medical["train"])
            print(f"  Medical-O1: {len(medical_data)} examples")
            combined_data.extend(medical_data)

        # Load and format KissanAI
        if self.include_kissanai:
            print("Loading and formatting KissanAI dataset")
            ds_kissanai = load_dataset(self.kissanai_repo_id, token=self.access_token)
            kissanai_data = format_kissanai(
                ds_kissanai["train"],
                self.include_multiturn_conversations,
            )
            print(f"  KissanAI: {len(kissanai_data)} examples")
            combined_data.extend(kissanai_data)

        # Load and format AgThoughts
        if self.include_agthoughts:
            print("Loading and formatting AgThoughts dataset")
            ds_agthoughts = load_dataset(self.agthoughts_repo_id, token=self.access_token)
            agthoughts_data = format_agthoughts(ds_agthoughts["train"])
            print(f"  AgThoughts: {len(agthoughts_data)} examples")
            combined_data.extend(agthoughts_data)

        # Print combined dataset statistics
        print(f"\n{'='*60}")
        print(f"COMBINED DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total examples: {len(combined_data)}")
        if combined_data:
            print(f"First example structure: {list(combined_data[0].keys())}")
            print(f"Instruction sample: {combined_data[0]['instruction'][:100]}")
            print(f"Output sample: {combined_data[0]['output'][:100]}")
        print(f"{'='*60}\n")

        # Split into train and validation
        train_data, val_data = random_split(
            combined_data,
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

        self.test_dataset = SFTDataset(
            data=list(val_data),
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )