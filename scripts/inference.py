import os
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def run_inference(
    unet_config_path: str,
    inference_ckpt_path: str,
    video_path: str,
    audio_path: str,
    video_out_path: str,
    inference_steps: int = 20,
    guidance_scale: float = 1.0,
    seed: int = 1247,
    enable_deepcache: bool = True,
    temp_dir: str = "temp"
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path '{video_path}' not found")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path '{audio_path}' not found")

    config = OmegaConf.load(unet_config_path)

    # Float16 if available
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained("configs")

    # Whisper model path selection
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    # Audio encoder
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # UNet
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to("cuda", dtype=dtype)

    # Pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # DeepCache
    if enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=1, cache_branch_id=0)
        helper.enable()

    if seed != -1:
        set_seed(seed)

    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=temp_dir,
    )
