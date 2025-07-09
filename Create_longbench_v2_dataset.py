from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Union
import pandas as pd


class LongBenchV2DatasetCreator:
    """
    Dataset creator specifically for LongBench v2 dataset.
    Supports streaming and is compatible with the existing DatasetCreator structure.
    """
    
    def __init__(self, tokenizer, use_streaming: bool = True, seed: int = 42):
        """
        Initialize the LongBench v2 DatasetCreator.
        
        Args:
            tokenizer: The tokenizer to use for applying chat templates
            use_streaming: Whether to use streaming for the dataset
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.use_streaming = use_streaming
        self.seed = seed
        self.dataset: Optional[Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict]] = None
        
    def load_dataset(self):
        """Load the LongBench v2 dataset."""
        print("Loading LongBench v2 dataset...")
        self.dataset = load_dataset(
            "THUDM/LongBench-v2", 
            split="train",
            streaming=self.use_streaming
        )
        print(f"Dataset loaded with streaming={'enabled' if self.use_streaming else 'disabled'}")
        return self.dataset
    
    def format_multiple_choice_question(self, example: Dict[str, Any]) -> str:
        """
        Format a LongBench v2 example into a multiple choice question string.
        
        Args:
            example: Single example from LongBench v2 dataset
            
        Returns:
            Formatted question string
        """
        question = example["question"]
        choices = [
            f"A. {example['choice_A']}",
            f"B. {example['choice_B']}",
            f"C. {example['choice_C']}",
            f"D. {example['choice_D']}"
        ]
        
        formatted_question = f"""Question: {question}

{chr(10).join(choices)}

Please select the correct answer (A, B, C, or D)."""
        
        return formatted_question
    
    def create_conversation_from_example(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert a LongBench v2 example into conversation format.
        
        Args:
            example: Single example from LongBench v2 dataset
            
        Returns:
            Conversation in the format expected by chat templates
        """
        # Format the context and question
        context = example["context"]
        question = self.format_multiple_choice_question(example)
        answer = example["answer"]
        
        # Create the user message with context and question
        user_content = f"""Context:
{context}

{question}"""
        
        # Create the assistant response
        assistant_content = f"The correct answer is {answer}."
        
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        return conversation
    
    def process_examples_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Process a batch of examples into conversation format.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Dictionary with conversations list
        """
        conversations = []
        
        # Process each example in the batch
        num_examples = len(examples["_id"])
        for i in range(num_examples):
            example = {key: examples[key][i] for key in examples.keys()}
            conversation = self.create_conversation_from_example(example)
            conversations.append(conversation)
        
        return {"conversations": conversations}
    
    def apply_chat_template_to_conversations(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """
        Apply chat template to a list of conversations.
        
        Args:
            conversations: List of conversations
            
        Returns:
            List of formatted text strings
        """
        try:
            formatted_texts = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted_texts
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # Fallback: process conversations individually
            formatted_texts = []
            for conversation in conversations:
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    formatted_texts.append(formatted_text)
                except Exception as inner_e:
                    print(f"Error processing individual conversation: {inner_e}")
                    # Create a simple fallback format
                    formatted_text = f"{conversation[0]['content']}\n\n{conversation[1]['content']}"
                    formatted_texts.append(formatted_text)
            return formatted_texts
    
    def create_dataset(self, max_examples: Optional[int] = None) -> Dataset:
        """
        Create the complete LongBench v2 dataset with chat templates applied.
        
        Args:
            max_examples: Maximum number of examples to process (for testing/debugging)
            
        Returns:
            Processed dataset with text column
        """
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset is None:
            raise ValueError("Failed to load dataset")
        
        print("Processing LongBench v2 dataset...")
        
        if self.use_streaming:
            # Handle streaming dataset - process examples one by one without loading all into memory
            print("Processing streaming dataset...")
            all_formatted_texts = []
            count = 0
            
            for example in self.dataset:
                # Process single example (ensure it's the right type)
                example_dict = dict(example) if not isinstance(example, dict) else example
                conversation = self.create_conversation_from_example(example_dict)
                
                # Apply chat template to single conversation
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception as e:
                    print(f"Error applying chat template to example {count}: {e}")
                    # Fallback format
                    formatted_text = f"{conversation[0]['content']}\n\n{conversation[1]['content']}"
                
                all_formatted_texts.append(formatted_text)
                count += 1
                
                if max_examples and count >= max_examples:
                    break
                    
                if count % 100 == 0:
                    print(f"Processed {count} examples...")
            
            print(f"Processed {len(all_formatted_texts)} examples from streaming dataset")
        else:
            # Handle non-streaming dataset
            if isinstance(self.dataset, Dataset):
                regular_dataset = self.dataset
                if max_examples:
                    regular_dataset = regular_dataset.select(range(min(max_examples, len(regular_dataset))))
            else:
                # Convert any other dataset type to list first
                examples_list = list(self.dataset)
                if max_examples:
                    examples_list = examples_list[:max_examples]
                regular_dataset = Dataset.from_list(examples_list)
            
            # Process the dataset in batches
            print("Converting examples to conversations...")
            
            # Get column names safely - all Dataset objects should have column_names
            columns_to_remove = regular_dataset.column_names
            
            conversation_dataset = regular_dataset.map(
                self.process_examples_batch,
                batched=True,
                batch_size=50,  # Process in smaller batches to manage memory
                remove_columns=columns_to_remove
            )
            
            # Apply chat templates
            print("Applying chat templates...")
            conversations = conversation_dataset["conversations"]
            
            # Process conversations in chunks to manage memory
            chunk_size = 100
            all_formatted_texts = []
            
            for i in range(0, len(conversations), chunk_size):
                chunk = conversations[i:i + chunk_size]
                formatted_chunk = self.apply_chat_template_to_conversations(chunk)
                all_formatted_texts.extend(formatted_chunk)
                
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"Processed {i + len(chunk)} conversations...")
            
            print(f"Generated {len(all_formatted_texts)} formatted texts")
        
        # Create final dataset
        final_dataset = Dataset.from_dict({"text": all_formatted_texts})
        
        # Shuffle the dataset
        final_dataset = final_dataset.shuffle(seed=self.seed)
        
        print(f"Created final dataset with {len(final_dataset)} examples")
        return final_dataset
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the LongBench v2 dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self.dataset is None:
            self.load_dataset()
        
        info = {
            "name": "LongBench v2",
            "description": "Challenging long-context benchmark with 503 multiple-choice questions",
            "streaming": self.use_streaming,
            "tasks": [
                "single-document QA",
                "multi-document QA", 
                "long in-context learning",
                "long-dialogue history understanding",
                "code repo understanding",
                "long structured data understanding"
            ],
            "context_length_range": "8k to 2M words",
            "format": "multiple-choice questions"
        }
        
        if not self.use_streaming and self.dataset is not None:
            try:
                # Only try len() on Dataset objects, not IterableDataset
                if hasattr(self.dataset, '__len__') and isinstance(self.dataset, Dataset):
                    info["total_examples"] = len(self.dataset)
                else:
                    info["total_examples"] = "Unknown (streaming or unsupported dataset type)"
            except (TypeError, AttributeError):
                # Some dataset types don't support len()
                info["total_examples"] = "Unknown (streaming or unsupported dataset type)"
        
        return info


# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset creator
    creator = LongBenchV2DatasetCreator(
        tokenizer=tokenizer,
        use_streaming=True,  # Enable streaming for better memory efficiency
        seed=42
    )
    
    # Get dataset info
    info = creator.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dataset (use max_examples for testing)
    dataset = creator.create_dataset(max_examples=10)  # Remove max_examples for full dataset
    
    print(f"\nSample from created dataset:")
    print(dataset[0]["text"][:500] + "...") 