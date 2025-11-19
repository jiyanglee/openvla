"""
vamos_dataset.py

Custom PyTorch Dataset for VAMOS navigation dataset (parquet format).
Extracts 2D velocities (v_x, v_y) from trajectory_3d data.
"""

import io
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import IGNORE_INDEX

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class VamosDataset(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        train: bool = True,
        predict_stop_token: bool = True,
    ) -> None:
        """
        Custom Dataset for VAMOS navigation dataset.

        Args:
            data_root_dir: Path to directory containing parquet files (data/ subdirectory)
            action_tokenizer: Action tokenizer for discretizing velocities
            base_tokenizer: Base LLM tokenizer
            image_transform: Image transformation function
            prompt_builder_fn: Prompt builder class
            train: Whether to use training split (True) or validation split (False)
            predict_stop_token: Whether to predict stop token
        """
        self.data_root_dir = Path(data_root_dir)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token

        # Load parquet files
        data_dir = self.data_root_dir / "data"
        if train:
            parquet_files = sorted(list(data_dir.glob("train-*.parquet")))
        else:
            parquet_files = sorted(list(data_dir.glob("validation-*.parquet")))

        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found in {data_dir} for split={'train' if train else 'validation'}")

        # Store parquet file paths for lazy loading (to avoid OOM)
        print(f"Found {len(parquet_files)} parquet files (will load lazily)...")
        self.parquet_files = parquet_files
        
        # Count total samples without loading all data
        print("Counting total samples...")
        total_samples = 0
        for parquet_file in parquet_files[:10]:  # Sample first 10 files to estimate
            df_sample = pd.read_parquet(parquet_file)
            total_samples += len(df_sample)
        avg_samples_per_file = total_samples / min(10, len(parquet_files))
        estimated_total = int(avg_samples_per_file * len(parquet_files))
        print(f"Estimated {estimated_total} samples from VAMOS dataset")
        
        # LRU cache for dataframes (max 2 files in memory at once to save RAM)
        self.dataframe_cache = OrderedDict()
        self.max_cache_size = 2  # Keep only 2 files in memory

        # Compute dataset statistics for normalization
        self.dataset_statistics = self._compute_statistics()

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute action statistics (q01, q99) for normalization using sampling."""
        print("Computing dataset statistics (sampling from dataset to avoid OOM)...")
        all_velocities = []
        sample_size = min(10000, len(self.parquet_files) * 50)  # Sample up to 10K trajectories
        
        # Sample from multiple files
        files_to_sample = self.parquet_files[:min(50, len(self.parquet_files))]  # Sample from first 50 files
        trajectories_sampled = 0
        
        for parquet_file in files_to_sample:
            if trajectories_sampled >= sample_size:
                break
            df = pd.read_parquet(parquet_file)
            # Sample up to 200 trajectories per file
            sample_per_file = min(200, len(df), sample_size - trajectories_sampled)
            df_sample = df.sample(n=sample_per_file, random_state=42) if len(df) > sample_per_file else df
            
            for idx in range(len(df_sample)):
                if trajectories_sampled >= sample_size:
                    break
                row = df_sample.iloc[idx]
                traj_3d = np.array([np.array(x) for x in row["trajectory_3d"]])
                velocities = np.diff(traj_3d, axis=0)[:, :2]  # Only v_x, v_y
                all_velocities.append(velocities)
                trajectories_sampled += 1

        all_velocities = np.concatenate(all_velocities, axis=0)
        print(f"Computed statistics from {len(all_velocities)} velocity samples (sampled from {trajectories_sampled} trajectories)")
        print(f"v_x range: [{all_velocities[:, 0].min():.4f}, {all_velocities[:, 0].max():.4f}]")
        print(f"v_y range: [{all_velocities[:, 1].min():.4f}, {all_velocities[:, 1].max():.4f}]")

        return {
            "vamos_dataset": {
                "action": {
                    "q01": np.quantile(all_velocities, 0.01, axis=0).astype(np.float32).tolist(),
                    "q99": np.quantile(all_velocities, 0.99, axis=0).astype(np.float32).tolist(),
                    "mean": all_velocities.mean(axis=0).astype(np.float32).tolist(),
                    "std": all_velocities.std(axis=0).astype(np.float32).tolist(),
                    "min": all_velocities.min(axis=0).astype(np.float32).tolist(),
                    "max": all_velocities.max(axis=0).astype(np.float32).tolist(),
                },
                "num_transitions": int(len(all_velocities) * (len(self.parquet_files) / len(files_to_sample))),  # Estimate
                "num_trajectories": trajectories_sampled * (len(self.parquet_files) / len(files_to_sample)),  # Estimate
            }
        }

    def __len__(self):
        """Return total number of transitions (not trajectories)."""
        if not hasattr(self, "index_map"):
            self._build_index()
        return len(self.index_map)

    def _load_single_file(self, file_idx: int) -> pd.DataFrame:
        """Load a single parquet file with LRU caching (max 2 files in memory)."""
        # Check if already in cache
        if file_idx in self.dataframe_cache:
            # Move to end (most recently used)
            self.dataframe_cache.move_to_end(file_idx)
            return self.dataframe_cache[file_idx]
        
        # Load new file
        df = pd.read_parquet(self.parquet_files[file_idx])
        
        # Add to cache
        self.dataframe_cache[file_idx] = df
        
        # Remove oldest if cache is full
        if len(self.dataframe_cache) > self.max_cache_size:
            oldest_key = next(iter(self.dataframe_cache))
            del self.dataframe_cache[oldest_key]
            import gc
            gc.collect()  # Force garbage collection
        
        return df
    
    def _build_index(self):
        """Pre-compute index mapping for efficient access without loading all data."""
        if not hasattr(self, 'index_map'):
            print("Building index (this may take a while but uses less memory)...")
            self.index_map = []  # List of (file_idx, traj_idx_in_file, timestep) tuples
            
            # Build index by loading files one at a time, then releasing memory
            for file_idx, parquet_file in enumerate(self.parquet_files):
                if (file_idx + 1) % 50 == 0:
                    print(f"Indexing file {file_idx + 1}/{len(self.parquet_files)}...")
                
                # Load file temporarily just to get trajectory lengths
                df = pd.read_parquet(parquet_file)
                
                # Build index for this file
                for traj_idx in range(len(df)):
                    row = df.iloc[traj_idx]
                    traj_3d = np.array([np.array(x) for x in row["trajectory_3d"]])
                    num_transitions = len(traj_3d) - 1
                    for timestep in range(num_transitions):
                        self.index_map.append((file_idx, traj_idx, timestep))
                
                # Don't cache the dataframe - release memory immediately
                # It will be loaded on-demand in __getitem__
                del df
                import gc
                gc.collect()
            
            print(f"Index built: {len(self.index_map)} total transitions")

    def __getitem__(self, idx):
        """
        Get a single transition (image, velocity action, instruction).
        """
        # Build index on first access
        if not hasattr(self, "index_map"):
            self._build_index()

        file_idx, traj_idx, timestep = self.index_map[idx]

        # Load the specific file if not already loaded
        df = self._load_single_file(file_idx)
        row = df.iloc[traj_idx]
        traj_3d = np.array([np.array(x) for x in row["trajectory_3d"]])

        # Compute velocity at this timestep (v_x, v_y only)
        velocity = traj_3d[timestep + 1, :2] - traj_3d[timestep, :2]  # [v_x, v_y]
        action = velocity.astype(np.float32)

        # Get image (same image for all timesteps in a trajectory)
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get instruction
        instruction = row.get("lang_goal", row.get("text", "navigate")).lower()

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name="vamos_dataset",
        )

