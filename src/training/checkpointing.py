from ignite.handlers import Checkpoint, DiskSaver
from ignite.engine import Events

def setup_checkpointing(trainer, model, optimizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    to_save = {
        "model": model,
        "optimizer": optimizer,
    }

    checkpoint = Checkpoint(
        to_save,
        DiskSaver(output_dir, create_dir=True),
        n_saved=3,
        filename_prefix="checkpoint",
        global_step_transform=lambda *_: trainer.state.epoch,  #  CRITICAL FIX
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    print("Checkpointing enabled")