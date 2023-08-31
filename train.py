import torch
import lightning as L

from voicebox_pytorch import (
    VoiceBox,
    ConditionalFlowMatcherWrapper
)

MAX_STEPS = 25000
BATCH_SIZE = 16
NUM_SAMPLES = 1000

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

rand_x = torch.randn(1024, 512)
rand_phoneme = torch.randint(0, 256, (1024,))
rand_mask = torch.randint(0, 2, (1024,)).bool()

class MahDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
        # self.list_of_x = [rand_x for _ in range(NUM_SAMPLES)]
        # self.list_of_phonemes = [rand_phoneme for _ in range(NUM_SAMPLES)]
        # self.list_of_mask = [rand_mask for _ in range(NUM_SAMPLES)]

    def __len__(self):
        return NUM_SAMPLES
    
    # def __getitem__(self, idx):
    #     x = self.list_of_x[idx]
    #     phonemes = self.list_of_phonemes[idx]
    #     mask = self.list_of_mask[idx]
    #     return x, phonemes, mask

    def __getitem__(self, idx):
        return rand_x, rand_phoneme, rand_mask

class LitVoiceBox(L.LightningModule):
    def __init__(self):
        super().__init__()

        model = VoiceBox(
            dim = 512,
            num_phoneme_tokens = 256,
            depth = 2,
            dim_head = 64,
            heads = 16
        )

        self.cfm_wrapper = ConditionalFlowMatcherWrapper(
            voicebox = model,
            use_torchode = False
        )

    def training_step(self, batch, batch_idx):
        x, phonemes, mask = batch
        # x = x.view(x.size(0), -1) # Needed?
        loss = self.cfm_wrapper(
            x,
            phoneme_ids = phonemes,
            cond = x,
            mask = mask)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

def main():
    trainer = L.Trainer()
    dataset = MahDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    trainer.fit(LitVoiceBox(), dataloader)

if __name__ == '__main__':
    main()