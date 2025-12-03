import torch
import torchaudio
import torchaudio.transforms as ta
from torch.utils.data import Dataset

class SERAudioDataset(Dataset):
    def __init__(self, 
                 df, 
                 sample_rate=16000, 
                 feature_type="mfcc",
                 win_ms=25,
                 hop_ms=10, 
                 n_mels=64,
                 n_mfcc=20,
                 mono=True, 
                 normalize_waveform=True
                 ):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.mono = mono
        self.normalize_waveform = normalize_waveform

        self.win_length = int(sample_rate * win_ms /1000)
        self.hop_length = int(sample_rate * hop_ms / 1000)

        self.mfcc_tf = None
        if self.feature_type == "mfcc":
           n_fft = 1 << (self.win_length - 1).bit_length()
           self.mfcc_tf = ta.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc,
                melkwargs=dict(
                    n_fft=n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_mels=n_mels,
                    f_min=0,
                    f_max=None,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                ),
            ) 
        
        print("dataset created with", len(df), "files!")

    def __len__(self):
        return len(self.df)
        
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["Path"]
        label = int(row["Emotion"])

        waveform, sr = torchaudio.load(path)
        
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = ta.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        if self.normalize_waveform:
            max_abs = waveform.abs().amax()
            if max_abs > 0:
                waveform = waveform / max_abs
        
        if self.feature_type == "mfcc":
            x = self.mfcc_tf(waveform).squeeze(0)
        elif self.feature_type == "waveform":
            x = waveform.squeeze(0)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return x, label, path

   
def collate_mfcc(batch):
        xs, ys, paths = zip(*batch)
        lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.int64)
        T_max = int(lengths.max())
        xs_padded = torch.stack([
            torch.nn.functional.pad(x, (0, T_max - x.shape[1]))
            for x in xs
        ], dim=0)
        ys = torch.tensor(ys, dtype=torch.long)
        return xs_padded, lengths, ys, paths


