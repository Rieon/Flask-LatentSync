from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
from flask import send_from_directory
import requests
import traceback
import faulthandler
import threading
import time
import json
faulthandler.enable()

app = Flask(__name__)

# Папка для хранения данных
BASE_DIR = "uploads"
os.makedirs(BASE_DIR, exist_ok=True)

# Импорт вашей функции
#try:
#    from scripts.inference import run_inference
#    print("[INFO] run_inference импортирован успешно")
#except Exception as e:
#    print(f"[ERROR] Не удалось импортировать run_inference: {e}")
#    raise

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}: {response.status_code}")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

QUEUE_FILE = "queue.jsonl"
QUEUE_LOCK = threading.Lock()

def append_task_to_queue(task_dict):
    with QUEUE_LOCK:
        with open(QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(task_dict) + "\n")

def read_queue():
    """Считать все задачи из файла"""
    if not os.path.exists(QUEUE_FILE):
        return []
    with QUEUE_LOCK:
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    return tasks

def write_queue(tasks):
    """Перезаписать файл очереди"""
    with QUEUE_LOCK:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

def worker():
    from scripts.inference import run_inference  # импорт один раз для оптимизации

    while True:
        tasks = read_queue()
        if tasks:
            task = tasks[0]
            print(f"[INFO] Запускаю задачу: {task['name']}")

            try:
                run_inference(
                    unet_config_path=task["unet_config_path"],
                    inference_ckpt_path=task["inference_ckpt_path"],
                    video_path=task["video_path"],
                    audio_path=task["audio_path"],
                    video_out_path=task["video_out_path"],
                    inference_steps=task["inference_steps"],
                    guidance_scale=task["guidance_scale"],
                    seed=task.get("seed", 1247),
                    enable_deepcache=task.get("enable_deepcache", True),
                    temp_dir=task["temp_dir"]
                )
                print(f"[INFO] Задача {task['name']} выполнена успешно")
            except Exception as e:
                print(f"[ERROR] Ошибка во время инференса задачи {task['name']}: {e}")
                traceback.print_exc()
            finally:
                # Удаляем задачу из очереди
                tasks = read_queue()  # перечитываем на случай, если очередь изменилась
                if tasks and tasks[0]["name"] == task["name"]:
                    tasks.pop(0)
                    write_queue(tasks)

        else:
            # Нет задач, ждем
            time.sleep(3)

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
    temp_dir = os.path.join(save_dir, "temp")

    # Сохраняем задачу в очередь
    task = {
        "name": name,
        "unet_config_path": "configs/unet/stage2.yaml",
        "inference_ckpt_path": "checkpoints/latentsync_unet.pt",
        "video_path": video_path,
        "audio_path": audio_path,
        "video_out_path": video_out_path,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": 1247,
        "enable_deepcache": True,
        "temp_dir": temp_dir
    }

    append_task_to_queue(task)

    return jsonify({
        'message': f'Задача {name} добавлена в очередь',
        'polling_result_path': video_out_path
    }), 200

if __name__ == '__main__':
    # Запускаем worker в отдельном потоке
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    app.run(host="0.0.0.0", port=5001)
