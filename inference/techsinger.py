import os
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.text.text_encoder import build_token_encoder, is_sil_phoneme
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.plot.plot import spec_to_figure
from resemblyzer import VoiceEncoder
from singing.svs.base_gen_task import AuxDecoderMIDITask,f0_to_figure,mel2ph_to_dur
from singing.svs.module.rf_singer import RFSinger, RFPostnet, RF_CFG_Postnet
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls


def process_align(ph_durs, mel, item, hop_size ,audio_sample_rate):
    mel2ph = np.zeros([mel.shape[0]], int)
    startTime = 0

    for i_ph in range(len(ph_durs)):
        start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
        mel2ph[start_frame:end_frame] = i_ph + 1
        startTime = startTime + ph_durs[i_ph]

    # item['mel2ph'] = mel2ph
    return mel2ph


class techinfer(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = RFSinger(dict_size, self.hparams)
        model.eval()
        load_ckpt(model, hparams['fs2_ckpt_dir'], strict=True)     
        self.model_post=RF_CFG_Postnet()
        
        load_ckpt(self.model_post, os.path.join('checkpoints', hparams['exp_name']), strict=True)
        self.model_post.eval()
        self.model_post.to(self.device)

        binary_data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_token_encoder(f'{binary_data_dir}/phone_set.json')
        return model

    def build_vocoder(self):
        vocoder = get_vocoder_cls(hparams["vocoder"])()
        return vocoder

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        # txt_tokens_gen = sample['txt_tokens_gen']
        # txt_tokens_prompt = sample['txt_tokens_prompt']
        txt_lengths = sample['txt_lengths']
        # mels = sample['mels']  # [B, T_s, 80]
        # mel2ph_prompt = sample['mel2ph']
        # mel_prompt = mels
        notes, note_durs,note_types = sample["notes"], sample["note_durs"],sample['note_types']
        mel2ph=None
        spk_id=sample['spk_id']
        mix,falsetto,breathy=sample['mix'],sample['falsetto'],sample['breathy']
        pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
        output = {}

        # Run model
        with torch.no_grad():
            umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
            upharyngeal, uvibrato, uglissando = torch.ones_like(pharyngeal, dtype=pharyngeal.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id,
                            target=None,ph_lengths=txt_lengths, infer=True, 
                            mix=mix, falsetto=falsetto, breathy=breathy, pharyngeal=pharyngeal,vibrato=vibrato,glissando=glissando,
                            note=notes, note_dur=note_durs, note_type=note_types)
            zero_tech = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id,
                            target=None,ph_lengths=txt_lengths, infer=True, 
                            mix=umix, falsetto=ufalsetto, breathy=ubreathy, pharyngeal=upharyngeal,vibrato=uvibrato,glissando=uglissando,
                            note=notes, note_dur=note_durs, note_type=note_types)       
            self.model_post(tgt_mels=None, infer=True, ret=output, spk_embed=None, zero_tech=zero_tech, cfg_scale=hparams['cfg_scale'],  noise=None)
            mel_out =  output['mel_out'][0]
            pred_f0 = output.get('f0_denorm_pred')[0]
            wav_out = self.vocoder.spec2wav(mel_out.cpu(),f0=pred_f0.cpu())

        return wav_out, mel_out
    

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        ph_gen=' '.join(inp['text_gen'])
        ph_token = self.ph_encoder.encode(ph_gen)
        # ph_token = ph_token_prompt + ph_token_gen
        # txt = txt + txt_gen
        note=inp['note_gen']
        note_dur=inp['note_dur_gen']
        note_type=inp['note_type_gen']
        
        tech_list=inp['tech_list']

        mix=[]
        falsetto=[]
        breathy=[]
        pharyngeal=[]
        vibrato=[]
        glissando=[]    
        for element in tech_list:
            mix.append(1 if '1' in element else 0)
            falsetto.append(1 if '2' in element else 0)
            breathy.append(1 if '3' in element else 0)
            pharyngeal.append(1 if '4' in element else 0)
            vibrato.append(1 if '5' in element else 0)
            glissando.append(1 if '6' in element else 0)

        item = {'item_name': inp['gen'], 'text': inp['text_gen'], 'ph': inp['text_gen'],
                'ph_token': ph_token, 'spk_id':inp['spk_id'],
                'mel2ph': None, 'note':note, 'note_dur':note_dur,'note_type':note_type,
                'mix_tech': mix, 'falsetto_tech': falsetto, 'breathy_tech': breathy,
                'pharyngeal_tech':pharyngeal , 'vibrato_tech':vibrato,'glissando_tech':glissando
                }
        
        # item['mel2ph']=mel2ph=process_align(inp["ph_durs"], mel, item,hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate'])
        # print(mel2ph)
        # assert False
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]])[None, :].to(self.device)
        
        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'][:hparams['max_input_tokens']])[None, :].to(self.device)

        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)

        # if hparams['use_spk_id']:
        spk_id= torch.LongTensor([item['spk_id']]).to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            # 'txt_tokens_gen': txt_tokens_gen,
            # 'txt_tokens_prompt': txt_tokens_prompt,
            'txt_lengths': txt_lengths,
            'spk_id': spk_id,
            # 'sent_txt_lengths': sent_txt_lengths,
            # 'mels': mels,
            # 'mel2ph': mel2ph,
            # 'mel_lengths': mel_lengths,
            'notes': note,
            'note_durs': note_dur,
            'note_types': note_type
        }

        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy
        batch['pharyngeal'],batch['vibrato'],batch['glissando']=pharyngeal,vibrato,glissando

        return batch

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav

        set_hparams()
        exp_name = hparams['exp_name'].split('/')[-1]
        tech2id ={
            'control': ['0'] * 35,
            'mix': ['1'] * 35,
            'falsetto': ['2'] * 35,
            'breathy': ['3'] * 35,
            'vibrato': ['5'] * 35,
            'glissando': ['6'] * 35,
            'control_mix':  ['0'] * 15 + ['1'] * 20,
            'control_falsetto':  ['0'] * 15 + ['2'] * 20,
        }
        infer_ins = cls(hp)
        for tech, ref_tech in tech2id.items():
            inp = {
                'tech_list': ref_tech,
                'spk_id': hparams['ref_id'],
                'gen': "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#一次就好#Mixed_Voice_Group#0000"
            }
            import json
            items_list = json.load(open(f"{hparams['processed_data_dir']}/metadata.json"))
            print(f"{hparams['processed_data_dir']}/metadata.json")
            for item in items_list:
                if inp['gen'] in item['item_name']:
                    
                    inp['text_gen']=item['ph']
                    inp['note_gen']=item['ep_pitches']
                    inp['note_dur_gen'] =item['ep_notedurs']
                    inp['note_type_gen']=item['ep_types']  
                    break       

            out = infer_ins.infer_once(inp)
            wav_out, mel_out = out
            os.makedirs(f'infer_out/{exp_name}', exist_ok=True)
            save_wav(wav_out, f'infer_out/{exp_name}/test_{tech}.wav', hp['audio_sample_rate'])
            f0 = extract_pitch_simple(wav_out)
            spec_to_figure(mel_out, vmin=-6, vmax=1.5, f0s=f0)
            plt.savefig(f'infer_out/{exp_name}/gen_mel_{tech}.png')
            print('enjoy')

if __name__ == '__main__':
    techinfer.example_run()
