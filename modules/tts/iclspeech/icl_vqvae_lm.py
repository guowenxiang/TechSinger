import math
import random
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.utils.data as tud
import torch.nn.functional as F

from modules.commons.layers import Embedding
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.tts.fs import FS_ENCODERS, FastSpeech
from modules.tts.iclspeech.vqvae.vqvae import VectorQuantizedVAE
from modules.tts.iclspeech.attention.simple_attention import SimpleAttention
from modules.tts.iclspeech.reltransformer_controlnet import RelTransformerEncoder_ControlNet
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.layers import LayerNorm, Linear
# from modules.tts.iclspeech.flash_transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from modules.commons.transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from utils.commons.hparams import hparams
from singing.svs.module.diff.diff_f0 import GaussianDiffusionF0
from singing.svs.module.diff.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from singing.svs.module.diff.multinomial_diffusion import MultinomialDiffusion
from singing.svs.module.diff.net import DiffNet, F0DiffNet, DDiffNet, MDiffNet
from modules.commons.nar_tts_modules import LengthRegulator, PitchPredictor
from utils.audio.pitch.utils import f0_to_coarse, denorm_f0, coarse_to_f0

def beam_search(
    model, 
    ph_tokens, 
    prev_vq_code, 
    spk_embed, note=None, note_dur=None,note_type=None,
    predictions = 20,
    beam_width = 3,
    batch_size = 1, 
    progress_bar = 0
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.
        next_probabilities = model.forward(ph_tokens, prev_vq_code, spk_embed,note=note, note_dur=note_dur,note_type=note_type)[:, -predictions, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        prev_vq_code = prev_vq_code.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        ph_tokens = ph_tokens.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        prev_vq_code[:, -predictions] = idx
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            next_probabilities = model.forward(ph_tokens, prev_vq_code, spk_embed,note=note, note_dur=note_dur)[:, -predictions+i+1, :].squeeze().log_softmax(-1)
            # next_probabilities = next_probabilities.reshape(
                # (-1, beam_width, next_probabilities.shape[-1])
            # )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            # next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                # .unsqueeze(-1)
            # best_candidates = (idx / vocabulary_size).long()
            # best_candidates += torch.arange(
                # X.shape[0] // beam_width, 
                # device = X.device
            # ).unsqueeze(-1) * beam_width
            # X = X[best_candidates].flatten(end_dim = -2)
            prev_vq_code = prev_vq_code.repeat((beam_width, 1, 1)).transpose(0, 1)
            print(prev_vq_code[:, :, -predictions+i+1].shape)
            print(idx.shape)
            prev_vq_code[:, :, -predictions+i+1] = idx
            prev_vq_code = prev_vq_code.flatten(end_dim = -2)
            ph_tokens = ph_tokens.repeat((beam_width, 1, 1)).transpose(0, 1)\
                .flatten(end_dim = -2)
            probabilities = probabilities.flatten()
        print(prev_vq_code.shape)
        import sys
        sys.exit(0)
        torch.argmax(F.softmax(vq_pred, dim=-1), -1)
        return prev_vq_code

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

class VQLanguageModel(nn.Module):
    def __init__(self, dict_size):
        super().__init__()
        self.hidden_size = hidden_size = hparams['lm_hidden_size']
        self.ph_encoder = RelTransformerEncoder(
        dict_size, hidden_size, hidden_size,
        hidden_size*4, hparams['num_heads'], hparams['enc_layers'],
        hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.vqcode_emb = Embedding(hparams['vqvae_ph_codebook_dim'] + 2, hidden_size, 0)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        dec_num_layers = 8
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(hidden_size, 0., kernel_size=5, num_heads=8) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)
        self.project_out_dim = Linear(hidden_size, hparams['vqvae_ph_codebook_dim'] + 1, bias=True)

        # Speaker embed related
        self.spk_embed_proj = Linear(hparams['hidden_size'], hidden_size, bias=True)
        self.spk_mode = 'direct' # 'direct' or 'attn'
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)
        if hparams['use_text'] is True:
            self.text_emb= nn.Embedding(hparams['num_spk']+1, hidden_size)
        # self.text_embed_proj = Linear(hparams['hidden_size'], hidden_size, bias=True)

    def forward(self, ph_tokens, prev_vq_code, spk_embed,note=None, note_dur=None, note_type=None, spk_ids=None,incremental_state=None, ret=None):
        # run encoder
        x = self.vqcode_emb(prev_vq_code)
        src_nonpadding = (ph_tokens > 0).float()[:, :, None]

        encoder_out = self.ph_encoder(ph_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur, note_type)
        ph_embed = (encoder_out + note_out) * src_nonpadding
        
        # ph_embed = self.ph_encoder(ph_tokens) * src_nonpadding

        # with open('a.txt','w+') as f:
        #     f.write(str([spk_ids.shape,self.text_emb(spk_ids).shape,ph_embed.shape]))
        #     import sys
        #     sys.exit()

        if hparams['use_text'] is True:
            text=self.text_emb(spk_ids)[:, None, :]
            ph_embed=ph_embed+text
            ph_embed = ph_embed * src_nonpadding
        
        if self.spk_mode == 'direct':
            # Currently we only support one-sentence prompt based zero-shot generation
            # The spk_embed is obtained from the one-sentence mel prompt
            # Thus, we do not use attention mechanics here
            ph_embed = ph_embed + self.spk_embed_proj(spk_embed)
            ph_embed = ph_embed * src_nonpadding

        # run decoder
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            ph_embed = ph_embed[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            positions = positions[:, -1:]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            self_attn_padding_mask = ph_tokens.eq(0).data

        x += positions
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        ph_embed = ph_embed.transpose(0, 1)
        x = x + ph_embed

        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)
        return x

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(self.fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def infer(self, ph_tokens, ph_vqcode, spk_embed, prompt_length, ret,note=None, note_dur=None, note_type=None,spk_ids=None, mode='argmax'):
        # mode = one-sentence prompt, zero-shot generation
        incremental_state = None
        # Add prompt
        vq_decoded = torch.zeros_like(ph_tokens)
        vq_decoded[:, :prompt_length] = ph_vqcode[:, :prompt_length]
        # Start Decode
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
        if mode == 'argmax':
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                print(f'{step}/{vq_decoded.shape[1] - 1}')
                vq_pred = self(ph_tokens, vq_decoded[:, :-1], spk_embed,note=note, note_dur=note_dur, note_type=note_type,spk_ids=spk_ids,
                            incremental_state=incremental_state, ret=ret)
                vq_pred = torch.argmax(F.softmax(vq_pred, dim=-1), -1)
                vq_decoded[:, step + 1] = vq_pred[:, step]
        elif mode == 'topk':
            K = 10
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                print(f'{step}/{vq_decoded.shape[1] - 1}')
                vq_pred = self(ph_tokens, vq_decoded[:, :-1], spk_embed,note=note, note_dur=note_dur, note_type=note_type,spk_ids=spk_ids,
                            incremental_state=incremental_state, ret=ret)
                _, idx = F.softmax(vq_pred, dim=-1).topk(k = K, axis = -1)
                rand_idx = random.randint(0,K-1)
                vq_decoded[:, step + 1] = idx[:, step, rand_idx]
        else:
            # Buggy
            predictions = beam_search(self, ph_tokens, vq_decoded[:, :-1], spk_embed, predictions=vq_decoded.shape[1]-prompt_length)
        return vq_decoded[:, 1:]

    def fill_with_neg_inf2(self, t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(-1e8).type_as(t)

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins'])
}

class ICLVectorQuantizedVAELM(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        del self.decoder
        del self.mel_out
        self.vqvae = VectorQuantizedVAE(hparams)
        self.vq_lm = VQLanguageModel(ph_dict_size)
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size)

        if hparams["use_f0"]:
            # self.pitch_predictor = PitchPredictor(
            #     self.hidden_size, n_chans=self.hidden_size,
            #     n_layers=5, dropout_rate=0.1, odim=2,
            #     kernel_size=hparams['predictor_kernel'])
            self.gm_diffnet = DDiffNet(in_dims=1, num_classes=2)
            self.f0_gen = GaussianMultinomialDiffusion(num_classes=2, denoise_fn=self.gm_diffnet, num_timesteps=hparams["f0_timesteps"])
        if hparams['post'] == 'diff':
            cond_hs = 80 + hparams['hidden_size'] * 2
            if hparams.get('use_txt_cond', True):
                cond_hs = cond_hs + hparams['hidden_size']
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

    def forward(self, txt_tokens, txt_tokens_gen, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,
                mel_prompt=None, spk_embed_prompt=None, global_step=None, use_gt_mel2ph=True, note=None, note_dur=None, note_type=None,f0=None,uv=None,*args,spk_ids=None, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # Forward LM
        if not infer:
            
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # Get GT VQCode
            ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            if spk_embed_prompt != None:
                spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # Forward VQ LM
            ph_vqcode = (ph_vqcode.detach() + 1) * src_nonpadding.squeeze(-1).long()
            prev_ph_vqcode = F.pad(ph_vqcode[:, :-1], [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
            vq_codes_pred = self.vq_lm(txt_tokens, prev_ph_vqcode, spk_embed.transpose(1,2), note=note,note_dur=note_dur,note_type=note_type, spk_ids=spk_ids, ret=ret)

        else:
            # # Infer with pred VQCode
            in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
            # Get GT VQCode for the first sentence
            ph_vqcode = self.vqvae.encode_ph_vqcode(mel_prompt, in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            if spk_embed_prompt != None:
                spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # Infer VQCode for the second sentence
            ph_vqcode = (ph_vqcode.detach() + 1)
            vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1],note=note,note_dur=note_dur,note_type=note_type, spk_ids=spk_ids,ret=ret)
            z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
            
            # Infer with GT VQCode
            # in_mel2ph = mel2ph
            # in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            # spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            # if spk_embed_prompt != None:
            #     spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # # ph_vqcode = (ph_vqcode.detach() + 1)
            # vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
            # # vq_codes_pred = (vq_codes_pred - 1).clamp_min(0)
            # z_q_x_bar = self.vqvae.vqcode_to_latent(ph_vqcode)
            
            # Infer mel with pred VQCode

            encoder_out = self.encoder(txt_tokens)  # [B, T, C]
            note_out = self.note_encoder(note, note_dur, note_type)
            ph_encoder_out = encoder_out + note_out
            # src_nonpadding = (txt_tokens > 0).float()[:, :, None]

            # src_nonpadding = (txt_tokens > 0).float()[:, :, None]
            # ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
            # ph_encoder_out = encoder_out * src_nonpadding
            # ph_encoder_out = self.encoder(txt_tokens)
            if use_gt_mel2ph:
                out_mel2ph = mel2ph
            else:
                if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                    dur_inp = (ph_encoder_out + spk_embed.transpose(1,2)) * src_nonpadding
                else:
                    dur_inp = (ph_encoder_out + z_q_x_bar.transpose(1,2)) * src_nonpadding
                out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
                ret['out_mel2ph'] = out_mel2ph
            out_nonpadding = (out_mel2ph > 0).float()[:, :, None]
            txt_cond = expand_states(ph_encoder_out, out_mel2ph)
            txt_cond = txt_cond * out_nonpadding
            pitch_inp = (txt_cond + expand_states(z_q_x_bar.transpose(1, 2), out_mel2ph)+spk_embed.transpose(1, 2))* out_nonpadding
            midi_notes = expand_states(note[:, :, None], out_mel2ph)
            # with open('b.txt','w+') as f:
            #     f.write(str([spk_embed.shape,z_q_x_bar.shape,expand_states(z_q_x_bar.transpose(1, 2), out_mel2ph).shape,txt_cond.shape]))
            if hparams['use_f0']:
                pitch = self.add_gmdiff_pitch(pitch_inp, f0, uv, out_mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs)
                spk_inp=spk_embed+pitch.transpose(1, 2)
                # spk_embed = spk_embed +self.add_gmdiff_pitch(pitch_inp, f0, uv, out_mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs).transpose(1, 2)
            x_tilde = self.vqvae.forward_second_stage(txt_cond, z_q_x_bar, spk_inp, out_nonpadding.transpose(1,2), out_mel2ph)
            ret['x_tilde'] = x_tilde

            ret['prosody'] =expand_states(z_q_x_bar.transpose(1, 2), out_mel2ph)
            ret['timbre']=spk_embed.transpose(1, 2)
            ret['decoder_inp']=(txt_cond+ret['prosody']+ret['timbre']+pitch)* out_nonpadding
            is_training = self.training
            # postflow
            if hparams['post']=='diff':
                self.run_post_diff(tgt_mels.transpose(1, 2), infer, is_training, ret)

        ret['vq_codes_pred'], ret['vq_codes'] = vq_codes_pred, ph_vqcode
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


    def infer(self, txt_tokens, txt_tokens_gen, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,
                mel_prompt=None, spk_embed_prompt=None, global_step=None, use_gt_mel2ph=True, note=None, note_dur=None,note_type=None,f0=None,uv=None,spk_ids=None, *args, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # # Infer with pred VQCode
        in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
        # Get GT VQCode for the first sentence
        ph_vqcode = self.vqvae.encode_ph_vqcode(mel_prompt, in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
        spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
        if spk_embed_prompt != None:
            spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
        # Infer VQCode for the second sentence
        ph_vqcode = (ph_vqcode.detach() + 1)
        vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1],ret= ret,note=note,note_dur=note_dur,note_type=note_type,spk_ids=spk_ids)
        z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
        
        # Infer with GT VQCode
        # in_mel2ph = mel2ph
        # in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
        # ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
        # spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
        # if spk_embed_prompt != None:
        #     spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
        # # ph_vqcode = (ph_vqcode.detach() + 1)
        # vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
        # # vq_codes_pred = (vq_codes_pred - 1).clamp_min(0)
        # z_q_x_bar = self.vqvae.vqcode_to_latent(ph_vqcode)
        
        # Infer mel with pred VQCode
        # z_q_x_bar = z_q_x_bar[:, :, txt_tokens_prompt.shape[1]:]

        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        note_out = self.note_encoder(note, note_dur,note_type)
        ph_encoder_out = encoder_out + note_out

        # ph_encoder_out = self.encoder(txt_tokens_gen)
        if use_gt_mel2ph:
            out_mel2ph = mel2ph
        else:
            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + spk_embed.transpose(1,2))
            else:
                dur_inp = (ph_encoder_out + z_q_x_bar.transpose(1,2))
            out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
            ret['out_mel2ph'] = out_mel2ph
        out_nonpadding = (out_mel2ph > 0).float()[:, :, None]
        txt_cond = expand_states(ph_encoder_out, out_mel2ph)
        txt_cond = txt_cond * out_nonpadding
        pitch_inp = (txt_cond + expand_states(z_q_x_bar.transpose(1, 2), out_mel2ph)+spk_embed.transpose(1, 2))* out_nonpadding
        midi_notes = expand_states(note[:, :, None], out_mel2ph)
        if hparams['use_f0']:
            pitch = self.add_gmdiff_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out,midi_notes=midi_notes, **kwargs)
            spk_inp=spk_embed+pitch.transpose(1, 2)
        x_tilde = self.vqvae.forward_second_stage(txt_cond, z_q_x_bar, spk_inp, out_nonpadding.transpose(1,2), out_mel2ph)
        ret['x_tilde'] = x_tilde

        ret['prosody'] =expand_states(z_q_x_bar.transpose(1, 2), out_mel2ph)
        ret['timbre']=spk_embed.transpose(1, 2)
        ret['decoder_inp']=(txt_cond+ret['prosody']+ret['timbre']+pitch)* out_nonpadding
        is_training = self.training
        # postflow
        if hparams['post']=='diff':
            self.run_post_diff(tgt_mels.transpose(1, 2), infer, is_training, ret)

        ret['vq_codes_pred'], ret['vq_codes'],ret['ph_code'] = vq_codes_pred, ph_vqcode,expand_states((vq_codes_pred - 1).clamp_min(0).unsqueeze(-1), out_mel2ph)
        return ret
    
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
        if hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp']], -1)
        g_spk_embed = ret['timbre'].repeat(1, T, 1)
        g_pro_embed = ret['prosody']
        g = torch.cat([g, g_spk_embed, g_pro_embed], dim=-1)     
        g = self.ln_proj(g)
        if not infer:
            if is_training:
                self.train()
        self.postdiff(g, tgt_mels, x_recon, ret, infer)