from dataclasses import dataclass
 
@dataclass
class dataset:
    file: str = None
    training_size: float = 1
    encoder_decoder: bool = False
    dataset_path: str = None

    # Distillation
    generated_by: str = None