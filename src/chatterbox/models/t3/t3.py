# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List
import threading

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, DynamicCache
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer


logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        
        # --- ADDED: Compilation state and thread safety ---
        self.compiled = False
        self.compile_lock = threading.Lock()
        self.patched_model = None
        self.compiled_model = None
        
        # --- ADDED: Compilation configuration ---
        self.compile_mode = "reduce-overhead"  # Options: "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        self.compile_dynamic = True  # Enable dynamic shapes for better performance
        self.compile_fullgraph = True  # Compile the entire model graph

    @property
    def device(self):
        return self.speech_head.weight.device

    def _compile_model(self):
        """
        Compile the T3 model for optimized inference performance.
        Uses torch.compile with optimized settings for TTS workloads.
        """
        if self.compiled:
            return
            
        with self.compile_lock:
            if self.compiled:  # Double-check after acquiring lock
                return
                
            logger.info("Compiling T3 model for optimized inference...")
            
            try:
                # --- ADDED: Configure torch._dynamo cache size ---
                import torch._dynamo
                # Increase cache size to prevent cache_size_limit errors
                torch._dynamo.config.cache_size_limit = 600
                # Disable recompilation warnings for cleaner logs
                torch._dynamo.config.suppress_errors = False
                # Enable more aggressive caching
                torch._dynamo.config.cache_size_limit = 600
                
                logger.info(f"Configured torch._dynamo cache_size_limit: {torch._dynamo.config.cache_size_limit}")
                
                # Create the patched model for HuggingFace compatibility
                patched_model = T3HuggingfaceBackend(
                    config=self.cfg,
                    llama=self.tfmr,
                    speech_enc=self.speech_emb,
                    speech_head=self.speech_head,
                )
                
                # Get the configured backend or use default
                backend = getattr(self, '_compile_backend', 'inductor')
                
                # --- ADDED: GPU capability check for inductor backend ---
                if backend == "inductor" and torch.cuda.is_available():
                    cuda_capability = torch.cuda.get_device_capability()
                    cuda_capability_major = cuda_capability[0]
                    
                    if cuda_capability_major < 7:
                        logger.warning(
                            f"GPU CUDA capability {cuda_capability_major}.{cuda_capability[1]} "
                            f"is too low for inductor backend (requires >= 7.0). "
                            f"Falling back to aot_eager backend."
                        )
                        backend = "aot_eager"
                        self._compile_backend = backend
                
                # Compile the model with optimized settings
                compiled_model = torch.compile(
                    patched_model,
                    mode=self.compile_mode,
                    dynamic=self.compile_dynamic,
                    fullgraph=self.compile_fullgraph,
                    backend=backend,
                )
                
                self.patched_model = patched_model
                self.compiled_model = compiled_model
                self.compiled = True
                
                logger.info(f"T3 model compiled successfully with mode: {self.compile_mode}, backend: {backend}")
                
                # Log compilation info for debugging
                compilation_info = self.get_compilation_info()
                logger.info(f"Compilation settings: {compilation_info}")
                
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Falling back to uncompiled model.")
                # Fallback to uncompiled model
                self.patched_model = patched_model
                self.compiled_model = patched_model
                self.compiled = True

    def _get_compiled_model(self):
        """
        Get the compiled model, compiling if necessary.
        """
        if not self.compiled:
            self._compile_model()
        return self.compiled_model

    def configure_compilation(self, mode="reduce-overhead", dynamic=True, fullgraph=True, backend="inductor"):
        """
        Configure compilation settings for optimal performance.
        
        Args:
            mode (str): Compilation mode. Options:
                - "reduce-overhead": Fastest compilation, good for inference
                - "max-autotune": Slower compilation, best performance
                - "max-autotune-no-cudagraphs": Good balance
            dynamic (bool): Enable dynamic shapes for better performance
            fullgraph (bool): Compile the entire model graph
            backend (str): Compilation backend ("inductor", "aot_eager", "aot_ts")
        """
        if self.compiled:
            logger.warning("Model already compiled. Recompilation will occur on next inference.")
            self.compiled = False
            
        self.compile_mode = mode
        self.compile_dynamic = dynamic
        self.compile_fullgraph = fullgraph
        
        # Store backend for compilation
        self._compile_backend = backend
        
        logger.info(f"Compilation configured: mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}, backend={backend}")

    def get_compilation_info(self):
        """
        Get information about the current compilation status and settings.
        """
        return {
            "compiled": self.compiled,
            "mode": self.compile_mode,
            "dynamic": self.compile_dynamic,
            "fullgraph": self.compile_fullgraph,
            "backend": getattr(self, '_compile_backend', 'inductor'),
        }

    def check_gpu_compatibility(self):
        """
        Check GPU compatibility for different compilation backends.
        Returns a dictionary with compatibility information.
        """
        compatibility_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_name": None,
            "cuda_capability": None,
            "supported_backends": [],
            "recommended_backend": None,
            "gpu_type": None,
            "optimization_level": None,
        }
        
        if torch.cuda.is_available():
            compatibility_info["device_name"] = torch.cuda.get_device_name()
            compatibility_info["cuda_capability"] = torch.cuda.get_device_capability()
            
            cuda_major = compatibility_info["cuda_capability"][0]
            device_name = compatibility_info["device_name"].lower()
            
            # Detect specific GPU types for specialized optimization
            if "t4" in device_name:
                compatibility_info["gpu_type"] = "T4"
                compatibility_info["optimization_level"] = "high"
                # T4 has CUDA 7.5 capability, supports all backends
                compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                compatibility_info["recommended_backend"] = "inductor"
                
            elif "l4" in device_name:
                compatibility_info["gpu_type"] = "L4"
                compatibility_info["optimization_level"] = "maximum"
                # L4 has CUDA 8.9 capability, excellent for all optimizations
                compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                compatibility_info["recommended_backend"] = "inductor"
                
            elif "rtx" in device_name and ("4090" in device_name or "4080" in device_name or "3090" in device_name):
                compatibility_info["gpu_type"] = "RTX_40/30_Series"
                compatibility_info["optimization_level"] = "maximum"
                compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                compatibility_info["recommended_backend"] = "inductor"
                
            elif "rtx" in device_name or "gtx" in device_name:
                if cuda_major >= 8:
                    compatibility_info["gpu_type"] = "RTX_20/30_Series"
                    compatibility_info["optimization_level"] = "high"
                    compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "inductor"
                elif cuda_major >= 7:
                    compatibility_info["gpu_type"] = "GTX_16/20_Series"
                    compatibility_info["optimization_level"] = "medium"
                    compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "inductor"
                else:
                    compatibility_info["gpu_type"] = "GTX_10_Series"
                    compatibility_info["optimization_level"] = "low"
                    compatibility_info["supported_backends"] = ["aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "aot_eager"
            else:
                # Generic detection based on CUDA capability
                if cuda_major >= 8:
                    compatibility_info["gpu_type"] = "High_End"
                    compatibility_info["optimization_level"] = "maximum"
                    compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "inductor"
                elif cuda_major >= 7:
                    compatibility_info["gpu_type"] = "Mid_Range"
                    compatibility_info["optimization_level"] = "high"
                    compatibility_info["supported_backends"] = ["inductor", "aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "inductor"
                elif cuda_major >= 6:
                    compatibility_info["gpu_type"] = "Entry_Level"
                    compatibility_info["optimization_level"] = "low"
                    compatibility_info["supported_backends"] = ["aot_eager", "aot_ts"]
                    compatibility_info["recommended_backend"] = "aot_eager"
                else:
                    compatibility_info["gpu_type"] = "Legacy"
                    compatibility_info["optimization_level"] = "minimal"
                    compatibility_info["supported_backends"] = ["aot_eager"]
                    compatibility_info["recommended_backend"] = "aot_eager"
        else:
            compatibility_info["supported_backends"] = ["aot_eager"]
            compatibility_info["recommended_backend"] = "aot_eager"
            compatibility_info["gpu_type"] = "CPU"
            compatibility_info["optimization_level"] = "minimal"
        
        return compatibility_info

    def auto_configure_compilation(self):
        """
        Automatically configure compilation based on GPU compatibility.
        """
        compatibility = self.check_gpu_compatibility()
        
        if not compatibility["cuda_available"]:
            logger.info("CUDA not available. Using CPU-optimized compilation.")
            self.configure_compilation(mode="reduce-overhead", backend="aot_eager")
            return
        
        device_name = compatibility["device_name"]
        cuda_capability = compatibility["cuda_capability"]
        recommended_backend = compatibility["recommended_backend"]
        gpu_type = compatibility["gpu_type"]
        optimization_level = compatibility["optimization_level"]
        
        logger.info(f"GPU: {device_name}")
        logger.info(f"GPU Type: {gpu_type}")
        logger.info(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
        logger.info(f"Optimization Level: {optimization_level}")
        logger.info(f"Recommended backend: {recommended_backend}")
        
        # Configure based on GPU type and optimization level
        if optimization_level == "maximum":
            # For L4, RTX 40/30 series - maximum performance
            self.configure_compilation(mode="max-autotune", backend="inductor")
            logger.info("Configured for maximum performance (max-autotune + inductor)")
            
        elif optimization_level == "high":
            # For T4, RTX 20/30 series - high performance
            self.configure_compilation(mode="reduce-overhead", backend="inductor")
            logger.info("Configured for high performance (reduce-overhead + inductor)")
            
        elif optimization_level == "medium":
            # For GTX 16/20 series - medium performance
            self.configure_compilation(mode="reduce-overhead", backend="inductor")
            logger.info("Configured for medium performance (reduce-overhead + inductor)")
            
        elif optimization_level == "low":
            # For GTX 10 series - conservative performance
            self.configure_compilation(mode="reduce-overhead", backend="aot_eager")
            logger.info("Configured for conservative performance (reduce-overhead + aot_eager)")
            
        else:
            # Fallback for minimal optimization
            self.configure_compilation(mode="reduce-overhead", backend="aot_eager")
            logger.info("Configured for minimal optimization (reduce-overhead + aot_eager)")
            
        logger.info(f"Auto-configured compilation for {gpu_type} with {optimization_level} optimization")

    def configure_cache_settings(self, cache_size_limit=600, suppress_errors=False):
        """
        Configure torch._dynamo cache settings for optimal performance.
        
        Args:
            cache_size_limit (int): Maximum number of compiled functions to cache (default: 600)
            suppress_errors (bool): Whether to suppress compilation errors (default: False)
        """
        import torch._dynamo
        
        # Configure cache settings
        torch._dynamo.config.cache_size_limit = cache_size_limit
        torch._dynamo.config.suppress_errors = suppress_errors
        
        logger.info(f"Configured torch._dynamo cache_size_limit: {cache_size_limit}")
        logger.info(f"Configured torch._dynamo suppress_errors: {suppress_errors}")
        
        # Reset compilation state to apply new settings
        if self.compiled:
            logger.info("Resetting compilation state to apply new cache settings...")
            self.compiled = False

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        length_penalty=1.0,
        repetition_penalty=2.0,
        cfg_weight=0,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # --- MODIFIED: Use compiled model for inference ---
        compiled_model = self._get_compiled_model()

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input if cfg_weight > 0
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        # --- MODIFIED: Optimize max_new_tokens for different GPUs ---
        if max_new_tokens is None:
            # Set default tokens based on GPU type and capability
            if torch.cuda.is_available():
                cuda_capability = torch.cuda.get_device_capability()[0]
                device_name = torch.cuda.get_device_name().lower()
                
                if "l4" in device_name or "rtx 4090" in device_name or "rtx 4080" in device_name:
                    max_new_tokens = 1000  # Maximum for L4 and RTX 40 series
                elif "t4" in device_name or "rtx 3090" in device_name or "rtx 3080" in device_name:
                    max_new_tokens = 800   # High for T4 and RTX 30 series
                elif cuda_capability >= 8:
                    max_new_tokens = 600   # High for modern GPUs
                elif cuda_capability >= 7:
                    max_new_tokens = 400   # Medium for mid-range GPUs
                elif cuda_capability >= 6:
                    max_new_tokens = 200   # Conservative for older GPUs
                else:
                    max_new_tokens = 100   # Very conservative for legacy GPUs
            else:
                max_new_tokens = 100  # Conservative for CPU
        else:
            # Limit max_new_tokens based on GPU capability to prevent OOM
            if torch.cuda.is_available():
                cuda_capability = torch.cuda.get_device_capability()[0]
                device_name = torch.cuda.get_device_name().lower()
                
                if "l4" in device_name or "rtx 4090" in device_name or "rtx 4080" in device_name:
                    max_limit = 1200  # Very high limit for L4 and RTX 40 series
                elif "t4" in device_name or "rtx 3090" in device_name or "rtx 3080" in device_name:
                    max_limit = 1000  # High limit for T4 and RTX 30 series
                elif cuda_capability >= 8:
                    max_limit = 800   # High limit for modern GPUs
                elif cuda_capability >= 7:
                    max_limit = 600   # Medium limit for mid-range GPUs
                elif cuda_capability >= 6:
                    max_limit = 300   # Conservative limit for older GPUs
                else:
                    max_limit = 150   # Very conservative limit for legacy GPUs
                
                if max_new_tokens > max_limit:
                    logger.warning(f"Limiting max_new_tokens from {max_new_tokens} to {max_limit} for {device_name}")
                    max_new_tokens = max_limit

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = compiled_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = DynamicCache.from_legacy_cache(output.past_key_values)

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits = output.logits[:, -1, :]

            # CFG
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.squeeze(1)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering.
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = top_p_warper(None, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = compiled_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

            yield next_token
