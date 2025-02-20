import wandb
from transformers import (
    TrainerCallback, 
    TrainerState, 
    TrainerControl, 
    TrainingArguments,
    PreTrainedModel
)

from datasets import Dataset
import matplotlib.pyplot as plt


class VisualizeGeneratedMEGCallback(TrainerCallback):
    def __init__(
        self,
        model: PreTrainedModel,
        eval_dataset: Dataset,
        eval_batch_size=8,
        forecasting_length=0
    ):
        super().__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.forecasting_length = forecasting_length

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Called when evaluation is run:
          - Model is already in eval mode
          - Gather predictions & references for entries in a random subset of eval dataset
          - Log figures to W&B
        """
        # Get a random subset of the evaluation dataset
        eval_subset = self.eval_dataset.shuffle().select(range(1))
        device = self.model.device

        for batch in eval_subset:
            # Generates predictions and logs them to wandb over the course of training
            meg_generated, _ = self.model.generate(
                batch["input_ids"].unsqueeze(0).to(device),
                max_length=self.forecasting_length  # setting forecasting to 0, only translating
            )
             
            labels = batch["labels"].unsqueeze(0)
        
            plt.ioff()
            fig, ax = plt.subplots(1, ncols=2)
            ax[0].imshow(meg_generated.squeeze().detach().cpu().numpy().T)
            ax[1].imshow(labels.detach().cpu().numpy().T)
            
            # Add titles to the subplots
            ax[0].set_title('Generated MEG')
            ax[1].set_title('Ground Truth')
            ax[0].set_xlabel('Timesteps')
            ax[1].set_xlabel('Timesteps')
            ax[0].set_ylabel('Channels')
            ax[1].set_ylabel('Channels')

            fig.tight_layout()

            # Log the figure to wandb
            wandb.log({
                "MEG_Comparison": wandb.Image(fig),
                "epoch": state.epoch
            })
        
        return control
            
