import torchaudio as ta
import torch
import time
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "In the heart of a bustling metropolis, where skyscrapers kissed the clouds and neon lights painted the night sky, a lone inventor worked tirelessly."

# Benchmark first run (default voice)
print("\n=== Benchmarking First Run (Default Voice) ===")
start_time = time.time()
chunks = list(model.generate(text))
end_time = time.time()
generation_time = end_time - start_time

wav = torch.cat(chunks, dim=-1).cpu()  # Concatenate all chunks into one tensor
ta.save("test-1.wav", wav, model.sr)

print(f"Generation time: {generation_time:.2f} seconds")
print(f"Audio length: {wav.shape[-1] / model.sr:.2f} seconds")
print(f"Real-time factor: {generation_time / (wav.shape[-1] / model.sr):.2f}x")

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"

# Benchmark second run (with audio prompt)
print("\n=== Benchmarking Second Run (With Audio Prompt) ===")
start_time = time.time()
chunks = list(model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH))
end_time = time.time()
generation_time_with_prompt = end_time - start_time

wav = torch.cat(chunks, dim=-1).cpu()
ta.save("test-2.wav", wav, model.sr)

print(f"Generation time: {generation_time_with_prompt:.2f} seconds")
print(f"Audio length: {wav.shape[-1] / model.sr:.2f} seconds")
print(f"Real-time factor: {generation_time_with_prompt / (wav.shape[-1] / model.sr):.2f}x")

# Compare performance
print("\n=== Performance Comparison ===")
print(f"Default voice vs Audio prompt: {generation_time_with_prompt - generation_time:+.2f} seconds difference")
if generation_time_with_prompt > generation_time:
    print(f"Audio prompt is {generation_time_with_prompt / generation_time:.2f}x slower")
else:
    print(f"Audio prompt is {generation_time / generation_time_with_prompt:.2f}x faster")
