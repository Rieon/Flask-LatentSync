from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
from flask import send_from_directory

app = Flask(__name__)

# Папка для хранения данных
BASE_DIR = "uploads"
os.makedirs(BASE_DIR, exist_ok=True)

# Импорт вашей функции
from scripts.inference import run_inference  # Замените на актуальный путь

@app.route('/process', methods=['POST'])
def process_media():
    # Проверка обязательных параметров
    name = request.form.get('name')
    inference_steps = request.form.get('inference_steps')
    guidance_scale = request.form.get('guidance_scale', default=1.0)

    if not name or 'audio' not in request.files or 'video' not in request.files:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        inference_steps = int(inference_steps)
        guidance_scale = float(guidance_scale)
    except ValueError:
        return jsonify({'error': 'Invalid parameter types'}), 400

    # Создание директории для имени
    save_dir = os.path.join(BASE_DIR, secure_filename(name))
    os.makedirs(save_dir, exist_ok=True)

    # Сохранение файлов
    audio_file = request.files['audio']
    video_file = request.files['video']

    audio_path = os.path.join(save_dir, secure_filename(audio_file.filename))
    video_path = os.path.join(save_dir, secure_filename(video_file.filename))

    audio_file.save(audio_path)
    video_file.save(video_path)

    # Генерация пути для выходного видео
    video_out_path = os.path.join(save_dir, f"output_{uuid.uuid4().hex}.mp4")

    # Вызов инференса
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
