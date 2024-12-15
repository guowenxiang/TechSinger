# midi singer
import torch.nn as nn
from modules.tts.fs import FastSpeech
import math
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
import torch
import torch.nn.functional as F
from singing.svs.module.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion, GaussianMultinomialDiffusionx0
from singing.svs.module.diff.gaussian_multinomial_reflow import GaussianMultinomialReflow
from singing.svs.module.diff.net import DiffNet, F0DiffNet, DDiffNet, MDiffNet
# from singing.svs.module.diff.ConcatDit import ConcatDiTF0

from utils.commons.hparams import hparams
from singing.svs.module.diff.diff_f0 import GaussianDiffusionF0, GaussianDiffusionx0
from singing.svs.module.diff.reflow_f0 import ReflowF0
from modules.commons.nar_tts_modules import PitchPredictor
from singing.svs.module.vqvae.vqvae import VectorQuantizedVAE
from modules.commons.layers import Embedding


DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}


# 处理note_tokens, note_durs and note types
class NoteEncoder(nn.Module):
    def __init__(self, n_vocab, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels, padding_idx=0)
        self.type_emb = nn.Embedding(5, hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.type_emb.weight, 0.0, hidden_channels ** -0.5)
        self.dur_ln = nn.Linear(1, hidden_channels)

    def forward(self, note_tokens, note_durs, note_types):
        x = self.emb(note_tokens) * math.sqrt(self.hidden_channels)
        types = self.type_emb(note_types) * math.sqrt(self.hidden_channels)
        durs = self.dur_ln(note_durs.unsqueeze(dim=-1))
        x = x + durs + types
        return x
    
class TechEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.mix_emb = nn.Embedding(3, hidden_channels, padding_idx=2)
        self.falsetto_emb = nn.Embedding(3, hidden_channels, padding_idx=2)
        self.breathy_emb = nn.Embedding(3, hidden_channels, padding_idx=2)
        self.pharyngeal_emb = nn.Embedding(3, hidden_channels, padding_idx=2)
        self.vibrato_emb = nn.Embedding(3, hidden_channels, padding_idx=2)
        self.glissando_emb = nn.Embedding(3, hidden_channels, padding_idx=2)

        nn.init.normal_(self.mix_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.falsetto_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.breathy_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.pharyngeal_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.vibrato_emb.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.glissando_emb.weight, 0.0, hidden_channels ** -0.5)

    def forward(self, mix,falsetto,breathy,pharyngeal,vibrato,glissando):
        mix = self.mix_emb(mix) * math.sqrt(self.hidden_channels)
        falsetto = self.falsetto_emb(falsetto) * math.sqrt(self.hidden_channels)
        breathy = self.breathy_emb(breathy) * math.sqrt(self.hidden_channels)
        pharyngeal = self.pharyngeal_emb(pharyngeal) * math.sqrt(self.hidden_channels)
        vibrato = self.vibrato_emb(vibrato) * math.sqrt(self.hidden_channels)
        glissando = self.glissando_emb(glissando) * math.sqrt(self.hidden_channels)

        x=mix+falsetto+breathy+pharyngeal+vibrato+glissando

        return x


class RFSinger(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        print(hparams['num_spk'])
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)
        self.tech_encoder = TechEncoder(hidden_channels=self.hidden_size)
        if hparams["f0_gen"] == "diff":
            self.uv_predictor = PitchPredictor(
                self.hidden_size, n_chans=128,
                n_layers=5, dropout_rate=0.1, odim=2,
                kernel_size=hparams['predictor_kernel'])
            self.pitch_flow_diffnet = F0DiffNet(in_dims=1)
            if hparams["param_"] == "x0":
                self.f0_gen = GaussianDiffusionx0(out_dims=1, denoise_fn=self.pitch_flow_diffnet, timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
            else:
                self.f0_gen = GaussianDiffusionF0(out_dims=1, denoise_fn=self.pitch_flow_diffnet, timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
        elif hparams["f0_gen"] == "gmdiff":
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            if hparams["param_"] == "x0":
                self.f0_gen = GaussianMultinomialDiffusionx0(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
            else:
                self.f0_gen = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
        elif hparams["f0_gen"] == "flow":
            self.uv_predictor = PitchPredictor(
                self.hidden_size, n_chans=128,
                n_layers=5, dropout_rate=0.1, odim=2,
                kernel_size=hparams['predictor_kernel'])
            self.pitch_flow_diffnet = F0DiffNet(in_dims=1)
            # self.pitch_flow_diffnet = ConcatDiTF0(in_dims=1)
            self.f0_gen = ReflowF0(out_dims=1, denoise_fn=self.pitch_flow_diffnet, timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
        elif hparams["f0_gen"] == "gmflow":
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            self.f0_gen = GaussianMultinomialReflow(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"], f0_K_step=hparams["f0_K_step"])
        elif hparams["f0_gen"] == "orig":
            self.uv_predictor = PitchPredictor(
                self.hidden_size, n_chans=128,
                n_layers=5, dropout_rate=0.1, odim=2,
                kernel_size=hparams['predictor_kernel'])
        if hparams.get('use_spk_prompt', False):
            self.vqvae = VectorQuantizedVAE(hparams)    
        if hparams.get("load_tts_ckpt",'') != '':
            from modules.tts.fs import FS_ENCODERS
            from utils.commons.ckpt_utils import load_ckpt
            assert hparams['encoder_type'] == 'rel_fft'
            ckpt_base_dir = hparams['load_tts_ckpt']
            self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
            load_ckpt(self.encoder, ckpt_base_dir=ckpt_base_dir, model_name='model.encoder')
            print(f"load tts ckpt from {ckpt_base_dir}")

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None, mel_prompt=None,target=None,ph_lengths=None,
                f0=None, uv=None, infer=False, 
                mix=None,falsetto=None,breathy=None,
                pharyngeal=None,vibrato=None,glissando=None,note=None, note_dur=None, note_type=None):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        encoder_out = encoder_out + note_out
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        if spk_embed!=None or spk_id!=None:
            ret['spk_embed'] = style_embed = self.forward_style_embed(spk_embed, spk_id)
        else:
            ret['spk_embed'] = style_embed = self.vqvae.encode_spk_embed(target.transpose(1, 2)).transpose(1, 2)
        tech=self.tech_encoder(mix,falsetto,breathy,pharyngeal,vibrato,glissando)
        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)
        in_nonpadding = (mel2ph > 0).float()[:, :, None]  

        ret['tech'] = tech = expand_states(tech, mel2ph) * tgt_nonpadding
        ret['mel2ph']=mel2ph

        # add pitch embed
        midi_notes = None
        pitch_inp = (decoder_inp + style_embed + tech) * tgt_nonpadding
        if infer:
            f0, uv = None, None
            midi_notes = expand_states(note[:, :, None], mel2ph)
        decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out, midi_notes=midi_notes)
        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed + tech) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer)

        return ret
    
    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_pred_inp = decoder_inp
        pitch_padding = mel2ph == 0
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if hparams["f0_gen"] == "diff":
            f0, uv = self.add_diff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)
        elif hparams["f0_gen"] == "gmdiff":
            f0, uv = self.add_gmdiff_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)
        elif hparams["f0_gen"] == "gmflow":
            f0, uv = self.add_gmflow_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)
        elif hparams["f0_gen"] == "flow":
            f0, uv = self.add_flow_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)
        elif hparams["f0_gen"] == "orig":
            f0, uv = self.add_orig_pitch(pitch_pred_inp, f0, uv, mel2ph, ret, **kwargs)
        
        f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm_pred'] = f0_denorm
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed
    
    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            dur = (dur.exp() - 1).clamp(min=0)
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph
    
    def add_orig_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)

        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = uv
            f0 = uv_pred[:, :, 1]
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (mel2ph > 0).float()
            f0_pred = uv_pred[:, :, 1] 
            ret["fdiff"] = (F.mse_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
        return f0, uv
    
    def add_diff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = uv
            lower_bound = midi_notes - 3
            upper_bound = midi_notes + 3
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # 
            # f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer)
            f0 = f0[:, :, 0]
            f0 = minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["fdiff"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0, nonpadding.unsqueeze(dim=1), ret, infer)
        return f0, uv

    def add_flow_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = uv
            lower_bound = midi_notes - 3
            upper_bound = midi_notes + 3
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # 
            # f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer)
            f0 = f0[:, :, 0]
            f0 = minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            norm_f0 = minmax_norm(f0)
            ret["fdiff"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0, nonpadding.unsqueeze(dim=1), ret, infer)
        return f0, uv
    
    def add_gmdiff_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            # uv = uv
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            lower_bound = midi_notes - 3 # 1 for good gtdur F0RMSE
            upper_bound = midi_notes + 3 # 1 for good gtdur F0RMSE
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            pitch_pred = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)
        return f0, uv
    
    def add_gmflow_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs):
        pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x> x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x
        if infer:
            # uv = uv
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            lower_bound = midi_notes - 3 # 1 for good gtdur F0RMSE
            upper_bound = midi_notes + 3 # 1 for good gtdur F0RMSE
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound-69)/12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound-69)/12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            pitch_pred = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, None, ret, infer, dyn_clip=[lower_norm_f0, upper_norm_f0]) # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)
        return f0, uv
    
    def forward_decoder(self, decoder_inp,tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

class RFPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        # from singing.svs.module.diff.shallow_diffusion_tts import GaussianDiffusionPostnet
        from singing.svs.module.diff.reflow import CFM_Postnet
        cond_hs=80+hparams['hidden_size']*3
        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        self.postdiff = CFM_Postnet(
            phone_encoder=None,
            out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
    
    def forward(self, tgt_mels, infer, ret, spk_embed=None, spk_id=None, noise=None):
        x_recon = ret['mel_out']
        g = x_recon.detach()
        B, T, _ = g.shape
        decoder_inp=ret['decoder_inp']
        g = torch.cat([g, decoder_inp], dim=-1)
        spk_embed=ret['spk_embed']
        spk_embed = spk_embed.repeat(1, T, 1)
        tech=ret['tech']
        g = torch.cat([g, spk_embed, tech], dim=-1)
        g = self.ln_proj(g)
        self.postdiff(g, tgt_mels, x_recon, ret, infer, noise)


class RF_CFG_Postnet(nn.Module):
    def __init__(self):
        super().__init__()
        from singing.svs.module.diff.reflow import CFM_CFG_Postnet
        cond_hs=80+hparams['hidden_size']*3
        
        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        self.postdiff = CFM_CFG_Postnet(
            phone_encoder=None,
            out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
    
    def forward(self, tgt_mels, infer, ret, spk_embed=None, spk_id=None, zero_tech=None, cfg_scale=1.0, noise=None):
        g = self.get_condition(ret)
        ucond = None
        if zero_tech!=None and infer:
            ucond = self.get_condition(zero_tech)
            if ucond.shape[1] > g.shape[1]:
                ucond = ucond[:, :g.shape[1], :]
            elif ucond.shape[1] <  g.shape[1]:
                padding_size = g.shape[1] - ucond.shape[1]  
                padding_tensor = torch.zeros(g.shape[0], padding_size, g.shape[2])  
                ucond = torch.cat((ucond, padding_tensor), dim=1)  
            # print(g.shape, ucond.shape)
        self.postdiff(g, tgt_mels, ret['mel_out'], ret, infer, ucond, noise, cfg_scale)
    
    def get_condition(self, ret):
        x_recon = ret['mel_out']
        g = x_recon.detach()
        B, T, _ = g.shape
        decoder_inp=ret['decoder_inp']
        g = torch.cat([g, decoder_inp], dim=-1)
        spk_embed=ret['spk_embed']
        spk_embed = spk_embed.repeat(1, T, 1)
        tech=ret['tech']
        g = torch.cat([g, spk_embed, tech], dim=-1)
        g = self.ln_proj(g)
        return g