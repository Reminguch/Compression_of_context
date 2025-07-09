from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoTokenizer
import gc
import random


class DatasetCreator:
    def __init__(self, tokenizer, chat_percentage: float = 0.2, reasoning_seed: int = 3407, sampling_seed: int = 2407):
        """
        Initialize the DatasetCreator.
        
        Args:
            tokenizer: The tokenizer to use for applying chat templates
            chat_percentage: Percentage of chat vs reasoning data in the final dataset
            reasoning_seed: Seed for shuffling the final dataset
            sampling_seed: Seed for sampling the non-reasoning subset
        """
        self.tokenizer = tokenizer
        self.chat_percentage = chat_percentage
        self.reasoning_seed = reasoning_seed
        self.sampling_seed = sampling_seed
        
    def _ensure_dataset(self, dataset):
        """Convert any dataset type to a regular Dataset object."""
        try:
            # If it's already a Dataset, return as-is
            if isinstance(dataset, Dataset):
                return dataset
            
            # If it has to_list method (IterableDataset), convert
            if hasattr(dataset, 'to_list'):
                return Dataset.from_list(list(dataset))
            
            # If it's a DatasetDict, get the first split
            if hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys')):
                keys = list(dataset.keys())
                if keys:
                    return self._ensure_dataset(dataset[keys[0]])
            
            # Try to iterate and convert to list
            data_list = []
            for item in dataset:
                # Ensure item is a dict
                if isinstance(item, dict):
                    data_list.append(item)
                else:
                    # Convert to dict if possible
                    try:
                        data_list.append(dict(item))
                    except:
                        # Skip invalid items
                        continue
                if len(data_list) >= 100000:  # Limit to prevent memory issues
                    break
            return Dataset.from_list(data_list)
            
        except Exception as e:
            print(f"Warning: Could not convert dataset type {type(dataset)}. Error: {e}")
            # Return as-is and hope for the best
            return dataset
        
    def load_non_reasoning_dataset_sample(self, sample_size: int):
        """Load and immediately sample from the non-reasoning dataset to save memory."""
        # Load the dataset with streaming enabled
        full_dataset = load_dataset("mlabonne/FineTome-100k", split="train", streaming=True)
        
        # Convert streaming dataset to list with limited size
        data_list = []
        for i, item in enumerate(full_dataset):
            if i >= sample_size:
                break
            # Ensure item is a dict
            if isinstance(item, dict):
                data_list.append(item)
            else:
                try:
                    data_list.append(dict(item))
                except:
                    continue
        
        # Convert to regular dataset
        sampled_dataset = Dataset.from_list(data_list)
        
        # Clear the data list from memory
        del data_list, full_dataset
        gc.collect()
        
        return sampled_dataset
    
    def load_reasoning_dataset(self):
        """Load only the reasoning dataset when needed."""
        streaming_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot", streaming=True)
        
        # Convert streaming dataset to regular dataset
        data_list = []
        for item in streaming_dataset:
            # Ensure item is a dict
            if isinstance(item, dict):
                data_list.append(item)
            else:
                try:
                    data_list.append(dict(item))
                except:
                    continue
        
        # Convert to regular dataset
        dataset = Dataset.from_list(data_list)
        del data_list, streaming_dataset
        gc.collect()
        
        return dataset
    
    def generate_conversation(self, examples):
        """Convert problem-solution pairs into conversation format."""
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
        return {"conversations": conversations}
    
    def create_reasoning_conversations(self):
        """Process reasoning dataset into conversations."""
        reasoning_dataset = self.load_reasoning_dataset()
        
        # Convert to regular dataset if it's an iterable dataset
        if hasattr(reasoning_dataset, 'to_list'):
            reasoning_dataset = Dataset.from_list(list(reasoning_dataset))
            
        reasoning_conversations_dataset = reasoning_dataset.map(
            self.generate_conversation, 
            batched=True
        )
        
        conversations = reasoning_conversations_dataset["conversations"]
        
        # Apply chat template in batches for better performance
        reasoning_conversations = []
        batch_size = 1000  # Process in smaller batches
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i+batch_size]
            batch_formatted = self.tokenizer.apply_chat_template(
                batch,
                tokenize=False,
            )
            reasoning_conversations.extend(batch_formatted)
        
        # Clear intermediate datasets
        del reasoning_dataset, reasoning_conversations_dataset, conversations
        gc.collect()
        
        return reasoning_conversations
    
    def create_non_reasoning_conversations(self, sample_size: int):
        """Process sampled non-reasoning dataset into conversations."""
        # Load only the sampled data
        sampled_dataset = self.load_non_reasoning_dataset_sample(sample_size)
        
        # Standardize the sampled dataset
        dataset = standardize_sharegpt(sampled_dataset)
        
        # Convert to regular dataset if it's an iterable dataset
        if hasattr(dataset, 'to_list'):
            dataset = Dataset.from_list(list(dataset))
        
        # Apply chat template in batches for better performance
        conversations = dataset["conversations"]
        non_reasoning_conversations = []
        batch_size = 1000  # Process in smaller batches
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i+batch_size]
            batch_formatted = self.tokenizer.apply_chat_template(
                batch,
                tokenize=False,
            )
            non_reasoning_conversations.extend(batch_formatted)
        
        # Clear intermediate datasets
        del sampled_dataset, dataset, conversations
        gc.collect()
        
        return non_reasoning_conversations
    
    def create_combined_dataset(self, max_length: int = 4096):
        """Combine reasoning and non-reasoning data into final dataset."""
        # Create reasoning conversations first
        reasoning_conversations = self.create_reasoning_conversations()
        
        # Calculate required sample size for non-reasoning data
        sample_size = int(len(reasoning_conversations) * (self.chat_percentage / (1 - self.chat_percentage)))
        
        # Create non-reasoning conversations with pre-calculated sample size
        non_reasoning_conversations = self.create_non_reasoning_conversations(sample_size)
        
        # Combine the data more efficiently
        all_conversations = reasoning_conversations + non_reasoning_conversations
        
        # Filter out extremely long sequences that slow down tokenization
        print(f"Filtering conversations longer than {max_length} tokens...")
        filtered_conversations = []
        for text in all_conversations:
            # Quick token count estimation (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(text) // 4
            if estimated_tokens <= max_length and len(text.strip()) > 10:  # Also filter very short texts
                filtered_conversations.append(text)
        
        print(f"Kept {len(filtered_conversations)}/{len(all_conversations)} conversations after length filtering")
        
        # Create final dataset directly without pandas conversion
        combined_dataset = Dataset.from_dict({"text": filtered_conversations})
        combined_dataset = combined_dataset.shuffle(seed=self.reasoning_seed)
        
        # Clear intermediate data
        del reasoning_conversations, non_reasoning_conversations, all_conversations, filtered_conversations
        gc.collect()
        
        return combined_dataset
    
    def create_dataset(self, max_length: int = 4096):
        """Main method to create the complete dataset."""
        return self.create_combined_dataset(max_length=max_length)


# Example usage:
# tokenizer = AutoTokenizer.from_pretrained("your-model-name")
# creator = DatasetCreator(tokenizer, chat_percentage=0.3)
# dataset = creator.create_dataset()