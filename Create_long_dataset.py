from datasets import load_dataset, Dataset, IterableDataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Union
import pandas as pd


class DatasetCreator:
    """
    Dataset creator for LongBench v2 and Super-NaturalInstructions datasets.
    Supports streaming and is compatible with the existing DatasetCreator structure.
    """
    
    def __init__(self, tokenizer, dataset_name: str = "longbench", use_streaming: bool = True, seed: int = 42):
        """
        Initialize the DatasetCreator.
        
        Args:
            tokenizer: The tokenizer to use for applying chat templates
            dataset_name: Either "longbench" or "supernatural" for dataset selection
            use_streaming: Whether to use streaming for the dataset
            seed: Random seed for shuffling
        """
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.use_streaming = use_streaming
        self.seed = seed
        self.dataset: Optional[Union[Dataset, IterableDataset, DatasetDict, IterableDatasetDict]] = None
        
        if self.dataset_name not in ["longbench", "supernatural"]:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}. Must be 'longbench' or 'supernatural'")
        
    def load_dataset(self):
        """Load the specified dataset."""
        if self.dataset_name == "longbench":
            print("Loading LongBench v2 dataset...")
            self.dataset = load_dataset(
                "THUDM/LongBench-v2", 
                split="train",
                streaming=self.use_streaming
            )
        elif self.dataset_name == "supernatural":
            print("Loading Super-NaturalInstructions dataset...")
            self.dataset = load_dataset(
                "Muennighoff/natural-instructions",
                split="train",
                streaming=self.use_streaming
            )
        
        print(f"Dataset loaded with streaming={'enabled' if self.use_streaming else 'disabled'}")
        return self.dataset
    
    def format_single_choice_question(self, example: Dict[str, Any]) -> str:
        """
        Format a LongBench v2 example into a single choice question string.
        
        Args:
            example: Single example from LongBench v2 dataset
            
        Returns:
            Formatted question string
        """
        question = example["question"]
        
        formatted_question = f"""Question: {question}

Please provide your answer."""
        
        return formatted_question
        
    def format_supernatural_task(self, example: Dict[str, Any]) -> str:
        """
        Format a Super-NaturalInstructions example into an instruction string.
        
        Args:
            example: Single example from Super-NaturalInstructions dataset
            
        Returns:
            Formatted instruction string
        """
        definition = example["definition"]
        inputs = example["inputs"]
        
        formatted_instruction = f"""{definition}

Input: {inputs}

Please provide your answer."""
        
        return formatted_instruction
    
    def create_conversation_from_example(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert an example into conversation format.
        
        Args:
            example: Single example from the dataset
            
        Returns:
            Conversation in the format expected by chat templates
        """
        if self.dataset_name == "longbench":
            return self._create_longbench_conversation(example)
        elif self.dataset_name == "supernatural":
            return self._create_supernatural_conversation(example)
            
    def _create_longbench_conversation(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create conversation format for LongBench v2 examples."""
        # Format the context and question
        context = example["context"]
        question = self.format_single_choice_question(example)
        answer = example["answer"]
        
        # Get the actual answer text based on the answer choice
        answer_mapping = {
            'A': example["choice_A"],
            'B': example["choice_B"], 
            'C': example["choice_C"],
            'D': example["choice_D"]
        }
        answer_text = answer_mapping.get(answer, answer)
        
        # Create the user message with context and question
        user_content = f"""Context:
{context}

{question}"""
        
        # Create the assistant response with the actual answer text
        assistant_content = answer_text
        
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        return conversation
        
    def _create_supernatural_conversation(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create conversation format for Super-NaturalInstructions examples."""
        # Format the instruction
        instruction = self.format_supernatural_task(example)
        targets = example["targets"]
        
        # Create the user message with instruction
        user_content = instruction
        
        # Create the assistant response with targets
        assistant_content = targets
        
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
        if self.dataset_name == "longbench":
            id_key = "_id"
        elif self.dataset_name == "supernatural":
            id_key = "id"
            
        num_examples = len(examples[id_key])
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
        Create the complete dataset with chat templates applied.
        
        Args:
            max_examples: Maximum number of examples to process (for testing/debugging)
            
        Returns:
            Processed dataset with text column
        """
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset is None:
            raise ValueError("Failed to load dataset")
        
        print(f"Processing {self.dataset_name} dataset...")
        
        if self.use_streaming:
            # Handle streaming dataset
            examples_list = []
            count = 0
            
            for example in self.dataset:
                examples_list.append(example)
                count += 1
                
                if max_examples and count >= max_examples:
                    break
                    
                if count % 1000 == 0:
                    print(f"Processed {count} examples...")
            
            print(f"Collected {len(examples_list)} examples from streaming dataset")
            
            # Convert to regular dataset for processing
            regular_dataset = Dataset.from_list(examples_list)
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
        Get information about the selected dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset_name == "longbench":
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
        elif self.dataset_name == "supernatural":
            info = {
                "name": "Super-NaturalInstructions",
                "description": "Large collection of diverse NLP tasks with natural language instructions",
                "streaming": self.use_streaming,
                "tasks": [
                    "text classification",
                    "question answering",
                    "text generation",
                    "reading comprehension",
                    "sentiment analysis",
                    "named entity recognition",
                    "and many more"
                ],
                "format": "instruction-following tasks"
            }
        
        if not self.use_streaming and self.dataset is not None:
            # Only try len() on Dataset and DatasetDict types that support it
            if isinstance(self.dataset, (Dataset, DatasetDict)):
                info["total_examples"] = len(self.dataset)
            else:
                info["total_examples"] = "Unknown (streaming or iterable dataset type)"
        
        return info


# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset creator for LongBench v2
    creator_longbench = LongBenchV2DatasetCreator(
        tokenizer=tokenizer,
        dataset_name="longbench",  # Use LongBench v2
        use_streaming=True,
        seed=42
    )
    
    # Create dataset creator for Super-NaturalInstructions
    creator_supernatural = LongBenchV2DatasetCreator(
        tokenizer=tokenizer,
        dataset_name="supernatural",  # Use Super-NaturalInstructions
        use_streaming=True,
        seed=42
    )
    
    # Get dataset info for both
    print("LongBench v2 Dataset Info:")
    info_longbench = creator_longbench.get_dataset_info()
    for key, value in info_longbench.items():
        print(f"  {key}: {value}")
    
    print("\nSuper-NaturalInstructions Dataset Info:")
    info_supernatural = creator_supernatural.get_dataset_info()
    for key, value in info_supernatural.items():
        print(f"  {key}: {value}")
    
    # Create datasets (use max_examples for testing)
    print("\nCreating LongBench v2 dataset:")
    dataset_longbench = creator_longbench.create_dataset(max_examples=5)
    print(f"Sample from LongBench v2 dataset:")
    print(dataset_longbench[0]["text"][:500] + "...")
    
    print("\nCreating Super-NaturalInstructions dataset:")
    dataset_supernatural = creator_supernatural.create_dataset(max_examples=5)
    print(f"Sample from Super-NaturalInstructions dataset:")
    print(dataset_supernatural[0]["text"][:500] + "...") 