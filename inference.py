import torch
import torchaudio

import utils
import commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

config_path = "./configs/mix_base.json"
model_path = "logs/mix_base/100000.pth"

hps_ms = utils.get_hparams_from_file(config_path)
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
net_g_ms.eval()
utils.load_checkpoint(model_path, net_g_ms, None)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


if __name__ == "__main__":

    sid = torch.LongTensor([10])  # speaker identity
    stn_tst = get_text(
        "[GD]宜家唔系事必要你讲，但系你所讲嘅说话将会变成呈堂证供。[GD]", hps_ms)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid,
                               noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0]

    print(audio, audio.dtype, audio.shape)
    torchaudio.save("data/gen.wav", audio, hps_ms.data.sampling_rate)
