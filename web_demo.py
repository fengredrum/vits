import streamlit as st
import torch
import numpy as np

import utils
import commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

def load_model_configs():
    config_path = "./configs/mix_base.json"
    model_path = "./saved/G_118000.pth"

    hps_ms = utils.get_hparams_from_file(config_path)
    net_g_ms = SynthesizerTrn(
        len(symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    net_g_ms.eval()
    utils.load_checkpoint(model_path, net_g_ms, None)
    return net_g_ms, hps_ms

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference(net_g_ms, hps_ms, text_norm, sid=4):
    sid = torch.LongTensor([sid])  # speaker identity
    stn_tst = get_text(text_norm, hps_ms)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid,
                            noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0]
    return audio.data.cpu().float().numpy()

samples = [
        "宜家唔系事必要你讲，但系你所讲嘅说话将会变成呈堂证供。",
        "各个国家有各个国家嘅国歌",
        "各个国家有各个国家的国歌",
        "为研判未来科技发展趋势、前瞻谋划和布局前沿科技领域与方向提供依据",
    ]

if __name__ == "__main__":

    st.write("# 语音合成演示")
    selected_dialog = st.selectbox(
        "请选择模板或直接输入文字",
        ('无', '粤语模板1', '粤语模板2', '普通话模板1', '普通话模板2', '混读'))

    mandarin_text = None
    cantonese_text = None
    if selected_dialog == '粤语模板1':
        cantonese_text = samples[0]
    elif selected_dialog == '粤语模板2':
        cantonese_text = samples[1]
    elif selected_dialog == '普通话模板1':
        mandarin_text = samples[2]
    elif selected_dialog == '普通话模板2':
        mandarin_text = samples[3]
    elif selected_dialog == '混读':
        cantonese_text = samples[1]
        mandarin_text = samples[2]

    mandarin_text = st.text_area(
        "普通话",
        mandarin_text,
        height=50,
    )
    cantonese_text = st.text_area(
        "粤语",
        cantonese_text,
        height=50,
    )
    
    if mandarin_text != 'None':
        mandarin_text = '[ZH]' + mandarin_text + '[ZH]'
    else:
        mandarin_text = ''

    if cantonese_text != 'None':
        cantonese_text = '[GD]' + cantonese_text + '[GD]'
    else:
        cantonese_text = ''
    text_norm = mandarin_text + cantonese_text

    speakers = []
    for i in range(100):
        if i < 10:
            speaker = '0' + str(i)
        else:
            speaker = str(i)
        speakers.append(speaker)
    
    sid = st.radio(
        "请选择说话人",
        speakers,
        horizontal=True
    )

    button_submit = st.button("生成")
    if button_submit:
        if len(text_norm) > 0:
            with st.spinner('推理中...'):
                net_g_ms, hps_ms = load_model_configs()
                audio = inference(net_g_ms, hps_ms, text_norm, sid=int(sid))
            st.audio(audio, sample_rate=hps_ms.data.sampling_rate)

    
    