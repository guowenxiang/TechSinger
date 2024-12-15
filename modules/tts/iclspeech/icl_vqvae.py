import math
import torch
from torch import nn

from modules.commons.layers import Embedding
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from modules.tts.fs import FS_DECODERS, FastSpeech
from modules.tts.iclspeech.vqvae.vqvae import VectorQuantizedVAE,SpatialNorm,MoVQDecoder
from utils.commons.hparams import hparams
from singing.svs.module.diff.diff_f0 import GaussianDiffusionF0
from singing.svs.module.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from singing.svs.module.diff.multinomial_diffusion import MultinomialDiffusion
from singing.svs.module.diff.net import DiffNet, F0DiffNet, DDiffNet, MDiffNet
from modules.commons.nar_tts_modules import LengthRegulator, PitchPredictor
from utils.audio.pitch.utils import f0_to_coarse, denorm_f0, coarse_to_f0

# def expand_states(h, mel2token):
#     h = F.pad(h, [0, 0, 1, 0])
#     mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
#     h = torch.gather(h, 1, mel2token_)  # [B, T, H]
#     return h

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

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}

class ICLVectorQuantizedVAE(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        # del self.decoder
        # del self.mel_out
        self.vqvae = VectorQuantizedVAE(hparams)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size)
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)

        if hparams["use_f0"]:
            # self.pitch_predictor = PitchPredictor(
            #     self.hidden_size, n_chans=self.hidden_size,
            #     n_layers=5, dropout_rate=0.1, odim=2,
            #     kernel_size=hparams['predictor_kernel'])
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            self.f0_gen = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"])
        
        self.decoder=MoVQDecoder( hparams['hidden_size'], hparams['dec_layers'], hparams['dec_ffn_kernel_size'],num_heads= hparams['num_heads'])

        if hparams['post'] == 'diff':
            cond_hs = 80 + hparams['hidden_size'] * 2
            # if hparams.get('use_txt_cond', True):
            #     cond_hs = cond_hs + hparams['hidden_size']
            from singing.svs.module.diff.shallow_diffusion_tts import GaussianDiffusionPostnet

            self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
            self.postdiff = GaussianDiffusionPostnet(
                phone_encoder=None,
                out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
                timesteps=hparams['timesteps'],
                K_step=hparams['K_step'],
                loss_type=hparams['diff_loss_type'],
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
            )


    def forward(self, txt_tokens, mel2ph=None, tgt_mels=None, mel_prompt=None, mel2ph_prompt=None,
                ph_lengths=None, use_gt_mel2ph=True, spk_embed_prompt=None, note=None, note_dur=None, note_type=None,f0=None,uv=None, infer=False, *args, **kwargs):
        ret = {}
    
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        encoder_out = encoder_out + note_out
        # src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        # ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        ph_encoder_out = encoder_out * src_nonpadding

        # add dur
        if use_gt_mel2ph:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            nonpadding_prompt = (mel2ph_prompt > 0).float()[:, :, None]
            # with open('b.txt','w+') as f:
            #     f.write(str([tgt_mels.shape,mel_prompt.shape,in_nonpadding.shape]))  
            # ph_z_e_x, ph_z_q_x, ph_z_q_x_st, global_z_q_x_st,indices = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)
            ph_z_q_x_st, global_z_q_x_st,vq_loss,indices = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)            
            ret['vq_loss']=vq_loss
            # ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            # with open('a.txt','w+') as f:
            #     f.write(str([ph_vqcode.shape]))  

            if spk_embed_prompt != None:
                global_z_q_x_st = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
                
            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + global_z_q_x_st.transpose(1,2)) * src_nonpadding
            else:
                dur_inp = (ph_encoder_out + ph_z_q_x_st.transpose(1,2)) * src_nonpadding

            out_mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
            out_nonpadding = in_nonpadding
            txt_cond = expand_states(ph_encoder_out, out_mel2ph)
            midi_notes = None
            if infer:
                midi_notes = expand_states(note[:, :, None], out_mel2ph)
            # pitch_inp = (txt_cond+ ph_z_q_x_st.transpose(1,2)) * out_nonpadding
            txt_cond = txt_cond * out_nonpadding
            pitch_inp = (txt_cond + expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph)+global_z_q_x_st.transpose(1, 2))* out_nonpadding
            if hparams['use_f0']:
                pitch = self.add_gmdiff_pitch(pitch_inp, f0, uv, out_mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs)
                global_inp= global_z_q_x_st + pitch.transpose(1, 2)
            else:
                global_inp=global_z_q_x_st
                # global_z_q_x_st= global_z_q_x_st + self.add_gmdiff_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs).transpose(1, 2)
            # ret['prosody']=expand_states(ph_z_e_x.transpose(1, 2), mel2ph)
            #     # global_z_q_x_st= global_z_q_x_st + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out).transpose(1, 2)
            # with open('a.txt','w+') as w:
            #     w.write(str([tgt_mels.shape,mel2ph.shape,ret['f0_denorm_pred'].shape,ph_code.shape]))
            # print(ph_encoder_out.shape,ph_z_q_x_st.transpose(1,2).shape,txt_cond.shape,global_z_q_x_st.shape)
            # x_tilde = self.vqvae.forward_second_stage(txt_cond, ph_z_q_x_st, global_inp, out_nonpadding.transpose(1,2), out_mel2ph)
            # with open('b.txt','w+') as f:
            #     f.write(str([x_tilde.shape,ph_z_q_x_st.shape,expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph).transpose(1, 2).shape]))
            #     import sys
            #     sys.exit()
        else:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            nonpadding_prompt = (mel2ph_prompt > 0).float()[:, :, None]
            # ph_z_e_x, ph_z_q_x, ph_z_q_x_st, global_z_q_x_st,indices = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)
            ph_z_q_x_st, global_z_q_x_st,vq_loss,indices = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)
            ret['vq_loss']=vq_loss
            if spk_embed_prompt != None:
                global_z_q_x_st = self.spk_embed_proj(spk_embed_prompt)[:,:,None]

            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + global_z_q_x_st.transpose(1,2)) * src_nonpadding
            else:
                dur_inp = (ph_encoder_out + ph_z_q_x_st.transpose(1,2)) * src_nonpadding

            out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
            out_nonpadding = (out_mel2ph > 0).float()[:, :, None]
            txt_cond = expand_states(ph_encoder_out, out_mel2ph)
            midi_notes = None

            if infer:
                midi_notes = expand_states(note[:, :, None], out_mel2ph)
            # pitch_inp = (txt_cond+ ph_z_q_x_st.transpose(1,2)) * out_nonpadding
            pitch_inp = (txt_cond + expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph)+global_z_q_x_st.transpose(1, 2))* out_nonpadding
            txt_cond = txt_cond * out_nonpadding
            if hparams['use_f0']:
                pitch=  self.add_gmdiff_pitch(pitch_inp, f0, uv, out_mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs)
                global_inp= global_z_q_x_st + pitch.transpose(1, 2)
            else:
                global_inp=global_z_q_x_st
            # x_tilde = self.vqvae.forward_second_stage(txt_cond, ph_z_q_x_st, global_inp, out_nonpadding.transpose(1,2), out_mel2ph)

        # ret['x_tilde'], ret['z_e_x'], ret['z_q_x'],ret['ph_code'] = x_tilde, [ph_z_e_x], [ph_z_q_x],expand_states(indices.unsqueeze(-1), out_mel2ph)
        # ret['x_tilde'] = x_tilde
        ret['prosody'] =expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph)
        ret['timbre']=global_z_q_x_st.transpose(1, 2)
        # with open('a.txt','w+') as f:
        #     torch.set_printoptions(edgeitems=10)
        #     f.write(str([ret['prosody'],ret['timbre'],pitch,txt_cond]))
        #     import sys
        #     sys.exit()
        # ret['decoder_inp']=(txt_cond+ret['prosody']+ret['timbre'])* out_nonpadding
        ret['decoder_inp']=(txt_cond+ret['prosody']+ret['timbre']+pitch)* out_nonpadding
        # ret['decoder_inp']=self.vqvae.forward_decoder_inp(txt_cond, ph_z_q_x_st, global_inp, out_nonpadding.transpose(1,2), out_mel2ph).transpose(1,2)
        ret['x_tilde'] = self.forward_decoder(ret['decoder_inp'], ret['prosody'], out_nonpadding, ret, infer=infer, **kwargs)  

        is_training = self.training
        # postflow
        if hparams['post']=='diff':
            self.run_post_diff(tgt_mels.transpose(1, 2), infer, is_training, ret)
        # ret['x_tilde'] = ret['mel_out'].transpose(1, 2)

        return ret
    
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
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def get_text_cond(self, txt_tokens):
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        return ph_encoder_out
    
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
            # ret["fdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(decoder_inp.transpose(-1, -2), norm_f0.unsqueeze(dim=1), uv, nonpadding, ret, infer)
        f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        ret['f0_denorm_pred'] = f0_denorm
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed
    
    def run_post_diff(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['x_tilde']
        g = x_recon.detach()
        B, T, _ = g.shape
        # with open('b.txt','w+') as f:
        #     f.write(str([g.shape,ret['decoder_inp'].shape,ret['timbre'].shape,ret['prosody'].shape]))
        # if hparams.get('use_txt_cond', True):
        #     g = torch.cat([g, ret['decoder_inp']], -1)
        g_spk_embed = ret['timbre'].repeat(1, T, 1)
        g_pro_embed = ret['prosody']
        g = torch.cat([g, g_spk_embed, g_pro_embed], dim=-1)     
        g = self.ln_proj(g)
        # if not infer:
        #     if is_training:
        #         self.train()
        self.postdiff(g, tgt_mels, x_recon, ret, infer)

    def forward_decoder(self, decoder_inp,cond, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x,cond=cond)
        x = self.mel_out(x)
        return x * tgt_nonpadding