import os
import torch
import traceback
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
import gc

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
    try:
        print("[INFO] Проверка путей...")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path '{video_path}' not found")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path '{audio_path}' not found")

        print("[INFO] Загрузка конфига UNet...")
        config = OmegaConf.load(unet_config_path)

        print("[INFO] Определение типа данных (FP16/FP32)...")
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        print(f"[INFO] FP16 supported: {is_fp16_supported}, dtype: {dtype}")

        print("[INFO] Загрузка DDIM Scheduler...")
        scheduler_config_path = "configs/scheduler_config.json"
        if not os.path.exists(scheduler_config_path):
            raise FileNotFoundError(f"Scheduler config not found at {scheduler_config_path}")
        scheduler = DDIMScheduler.from_config(scheduler_config_path)

        print("[INFO] Подбор Whisper модели...")
        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        print("[INFO] Загрузка аудио энкодера...")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        print("[INFO] Загрузка VAE...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        print("[INFO] Загрузка UNet...")
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            inference_ckpt_path,
            device="cpu",
        )
        unet = unet.to(dtype=dtype)

        print("[INFO] Сборка пайплайна...")
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")

        if enable_deepcache:
            print("[INFO] Включение DeepCache...")
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()

        if seed != -1:
            print(f"[INFO] Установка seed: {seed}")
            set_seed(seed)

        print("[INFO] Запуск пайплайна...")
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
        print(f"[INFO] Инференс завершен. Файл сохранен: {video_out_path}")
        
        print("[INFO] Очистка ресурсов...")
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"[FATAL] Ошибка в run_inference: {e}")
        traceback.print_exc()
        
        print("[INFO] Очистка ресурсов...")
        torch.cuda.empty_cache()
        gc.collect()
        raise
