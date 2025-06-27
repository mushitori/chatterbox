from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import librosa
import torch
# import perth  # --- MODIFIED: Removed watermarking
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
        ("‘", "'"), ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Optional[Conditionals] = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # --- ADDED: Caching mechanism state ---
        self._cached_audio_prompt_path: Optional[str] = None
        # --- MODIFIED: Removed watermarker initialization ---
        # self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available. Falling back to CPU.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        # --- ADDED: Write to the cache ---
        self._cached_audio_prompt_path = str(Path(wav_fpath).resolve())
        print(f"INFO: Conditionals processed and cached for audio: {self._cached_audio_prompt_path}")

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        # stream
        tokens_per_slice: int = 1000,
        remove_milliseconds: int = 45,
        remove_milliseconds_start: int = 25,
        chunk_overlap_method: Literal["zero", "full"] = "zero",
    ):
        # --- ADDED: caching and conditional management logic ---
        if audio_prompt_path:
            normalized_provided_path = str(Path(audio_prompt_path).resolve())
            if normalized_provided_path != self._cached_audio_prompt_path:
                print(f"INFO: New or different audio prompt. Processing: {normalized_provided_path}")
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                print(f"INFO: Audio prompt '{normalized_provided_path}' matches cache. Reusing conditionals.")
        
        if self.conds is None:
            raise ValueError("No audio prompt provided, and no default conditionals are loaded. Please provide `audio_prompt_path`.")

        current_exaggeration_tensor = exaggeration * torch.ones(1, 1, 1, device=self.device)
        if not torch.equal(self.conds.t3.emotion_adv, current_exaggeration_tensor):
            print(f"INFO: Updating emotion exaggeration to: {exaggeration}")
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=current_exaggeration_tensor,
            ).to(device=self.device)
        
        # --- Original Streaming Logic ---
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot, eot = self.t3.hp.start_text_token, self.t3.hp.stop_text_token
        text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)

        with torch.inference_mode():
            def _t3_infer():
                # Determine optimal max_new_tokens based on GPU
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name().lower()
                    cuda_capability = torch.cuda.get_device_capability()[0]
                    
                    if "l4" in device_name or "rtx 4090" in device_name or "rtx 4080" in device_name:
                        max_tokens = 1000  # Maximum for L4 and RTX 40 series
                    elif "t4" in device_name or "rtx 3090" in device_name or "rtx 3080" in device_name:
                        max_tokens = 800   # High for T4 and RTX 30 series
                    elif cuda_capability >= 8:
                        max_tokens = 600   # High for modern GPUs
                    elif cuda_capability >= 7:
                        max_tokens = 400   # Medium for mid-range GPUs
                    elif cuda_capability >= 6:
                        max_tokens = 200   # Conservative for older GPUs
                    else:
                        max_tokens = 100   # Very conservative for legacy GPUs
                else:
                    max_tokens = 100  # Conservative for CPU
                
                for token in self.t3.inference(
                    t3_cond=self.conds.t3, text_tokens=text_tokens,
                    max_new_tokens=max_tokens, temperature=temperature, cfg_weight=cfg_weight,
                ):
                    yield token
            
            def speech_to_wav(speech_tokens, previous_length=0):
                speech_tokens = speech_tokens[0]
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens, ref_dict=self.conds.gen,
                    no_trim=tokens_per_slice < 1000,
                )
                
                # --- Personal Optimization: Keeps everything on the GPU ---
                
                # 1. Squeeze and detach, but keep it on the original device (GPU)
                wav_gpu = wav.squeeze(0).detach()

                if chunk_overlap_method == "full":
                    # Note: This slicing I did not did not use myself, but if it is, it works on tensors too
                    wav_gpu = wav_gpu[previous_length:]
                
                # 2. Perform trimming directly on the GPU tensor
                if remove_milliseconds > 0:
                    trim_samples = int(self.sr * remove_milliseconds / 1000)
                    wav_gpu = wav_gpu[:-trim_samples]
                if remove_milliseconds_start > 0:
                    trim_samples_start = int(self.sr * remove_milliseconds_start / 1000)
                    wav_gpu = wav_gpu[trim_samples_start:]

                # 3. Return the GPU tensor directly (with a batch dimension)
                return wav_gpu.unsqueeze(0), wav_gpu.shape[0] + previous_length

            eos_token = torch.tensor([self.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)

            def chunked():
                token_stream = []
                for batch in _t3_infer():
                    token_stream.extend(batch.squeeze(0))
                    while len(token_stream) >= tokens_per_slice:
                        yield token_stream[:tokens_per_slice]
                        token_stream = token_stream[tokens_per_slice:]
                if token_stream:
                    yield token_stream

            def accumulating_chunks():
                accumulated = []
                for batch in _t3_infer():
                    accumulated.extend(batch.squeeze(0))
                    if len(accumulated) % tokens_per_slice == 0 and len(accumulated) > 0:
                        yield accumulated.copy()
                if accumulated and len(accumulated) % tokens_per_slice != 0:
                    yield accumulated.copy()

            previous_length = 0
            iterator = chunked() if chunk_overlap_method == "zero" else accumulating_chunks()
            for slice_tokens in iterator:
                tokens = torch.stack(slice_tokens).unsqueeze(0)
                tokens_with_eos = torch.cat([tokens, eos_token], dim=1)
                wav, previous_length = speech_to_wav(tokens_with_eos, previous_length)
                yield wav

    def configure_t3_compilation(self, mode="reduce-overhead", dynamic=True, fullgraph=True, backend="inductor"):
        """
        Configure T3 model compilation for optimal performance.
        
        Args:
            mode (str): Compilation mode. Options:
                - "reduce-overhead": Fastest compilation, good for inference
                - "max-autotune": Slower compilation, best performance
                - "max-autotune-no-cudagraphs": Good balance
            dynamic (bool): Enable dynamic shapes for better performance
            fullgraph (bool): Compile the entire model graph
            backend (str): Compilation backend ("inductor", "aot_eager", "aot_ts")
        """
        self.t3.configure_compilation(mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
        print(f"T3 compilation configured: mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}, backend={backend}")

    def get_t3_compilation_info(self):
        """
        Get information about T3 model compilation status and settings.
        """
        return self.t3.get_compilation_info()

    def check_gpu_compatibility(self):
        """
        Check GPU compatibility for different compilation backends.
        """
        return self.t3.check_gpu_compatibility()

    def auto_configure_t3_compilation(self):
        """
        Automatically configure T3 compilation based on GPU compatibility.
        """
        self.t3.auto_configure_compilation()
        print("T3 compilation auto-configured based on GPU compatibility.")

    def configure_t3_cache_settings(self, cache_size_limit=600, suppress_errors=False):
        """
        Configure torch._dynamo cache settings for optimal performance.
        
        Args:
            cache_size_limit (int): Maximum number of compiled functions to cache (default: 600)
            suppress_errors (bool): Whether to suppress compilation errors (default: False)
        """
        self.t3.configure_cache_settings(cache_size_limit=cache_size_limit, suppress_errors=suppress_errors)
        print(f"T3 cache settings configured: cache_size_limit={cache_size_limit}, suppress_errors={suppress_errors}")

    def warmup_t3_model(self, text="Hello world", audio_prompt_path=None):
        """
        Warm up the T3 model by running a short inference to trigger compilation.
        This should be called once after model loading for optimal performance.
        
        Args:
            text (str): Short text for warmup
            audio_prompt_path (str): Audio prompt path for warmup
        """
        print("Warming up T3 model for optimal performance...")
        
        if audio_prompt_path and self.conds is None:
            self.prepare_conditionals(audio_prompt_path)
        
        if self.conds is None:
            print("Warning: No conditionals available for warmup. Please provide audio_prompt_path.")
            return
        
        # Run a short warmup inference
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        sot, eot = self.t3.hp.start_text_token, self.t3.hp.stop_text_token
        text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)
        
        # Trigger compilation with a short inference
        with torch.inference_mode():
            for _ in self.t3.inference(
                t3_cond=self.conds.t3, 
                text_tokens=text_tokens,
                max_new_tokens=10,  # Very short for warmup
                temperature=0.8,
                cfg_weight=0.0,
            ):
                break  # Just trigger compilation, don't need full generation
        
        print("T3 model warmup completed.")
