import asyncio
import glob
import json

from fastapi import FastAPI, WebSocket
from starlette.responses import FileResponse

from evaluation import Monitor
from test_tracks.track_3 import country_balls_amount, track_data
from trackers import tracker_soft
from utils import NpEncoder

app = FastAPI(title="Tracker assignment")
images_list = glob.glob("imgs/*")
country_balls = [{"cb_id": x, "img": images_list[x % len(images_list)]} for x in range(country_balls_amount)]
print("Started")


@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.get("/imgs/{image_path:path}")
async def get_image(image_path: str):
    image_file_path = f"imgs/{image_path}"
    return FileResponse(image_file_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection...")
    await websocket.accept()

    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))

    metrics_monitor = Monitor()
    for el in track_data:
        await asyncio.sleep(0.1)

        # Part 1
        if el["frame_id"] == 1:
            id_info = {}
            num = 0
        try:
            el_soft, id_info, num = tracker_soft(el, id_info, num)
            el_soft = json.loads(json.dumps(el_soft, cls=NpEncoder))
            await websocket.send_json(el_soft)
            metrics_monitor.update(el_soft)
        except IndexError:
            continue

    print(metrics_monitor.calculate_track_metrics())
    print("Bye..")
