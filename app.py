from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
from flask import send_from_directory
import requests

app = Flask(__name__)

# Папка для хранения данных
BASE_DIR = "uploads"
os.makedirs(BASE_DIR, exist_ok=True)

# Импорт вашей функции
from scripts.inference import run_inference  # Замените на актуальный путь

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}: {response.status_code}")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@app.route('/process', methods=['POST'])
def process_media():
    name = request.form.get('name')
    inference_steps = request.form.get('inference_steps')
    guidance_scale = request.form.get('guidance_scale', default=1.0)
    audio_url = request.form.get('audio_url')
    video_url = request.form.get('video_url')

    if not name or not audio_url or not video_url:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        inference_steps = int(inference_steps)
        guidance_scale = float(guidance_scale)
    except ValueError:
        return jsonify({'error': 'Invalid parameter types'}), 400

    save_dir = os.path.join(BASE_DIR, secure_filename(name))
    os.makedirs(save_dir, exist_ok=True)

    # Скачивание файлов
    try:
        audio_filename = f"audio_{uuid.uuid4().hex}.mp3"
        video_filename = f"video_{uuid.uuid4().hex}.mp4"

        audio_path = os.path.join(save_dir, audio_filename)
        video_path = os.path.join(save_dir, video_filename)

        download_file(audio_url, audio_path)
        download_file(video_url, video_path)
    except Exception as e:
        return jsonify({'error': f'Failed to download media: {str(e)}'}), 400

    video_out_path = os.path.join(save_dir, f"output_{uuid.uuid4().hex}.mp4")

    try:
        run_inference(
            unet_config_path="configs/unet/stage2.yaml",
            inference_ckpt_path="checkpoints/latentsync_unet.pt",
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=1247,
            enable_deepcache=True,
            temp_dir=os.path.join(save_dir, "temp")
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'video_out_path': video_out_path}), 200

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5001)
