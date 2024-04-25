import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

deepsort_tracker = DeepSort(max_age=5)


def get_centroid(bbox: list) -> np.ndarray:
    bbox = np.array(bbox)
    return np.mean(bbox.reshape(2, 2), axis=0).astype(int)


def tracker_soft(el, id_info, num):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """

    el["data"] = [item for item in el["data"] if any(item["bounding_box"])]

    if el["frame_id"] == 1:
        for i, item in enumerate(el["data"]):
            el["data"][i]["track_id"] = i
            id_info[i] = get_centroid(item["bounding_box"])
        num = len(el["data"])
        return el, id_info, num

    current_centroids = np.array([get_centroid(item["bounding_box"]) for item in el["data"]])
    previous_ids, previous_centroids = np.array(list(id_info.keys())), np.array(list(id_info.values()))

    distances = np.linalg.norm(previous_centroids[:, None, :] - current_centroids[None, :, :], axis=2)

    for i, item in enumerate(el["data"]):
        if distances.size > 0:
            closest_id_index = np.argmin(distances[:, i])
            closest_id = previous_ids[closest_id_index]
            el["data"][i]["track_id"] = closest_id
            id_info[closest_id] = current_centroids[i]

            distances[closest_id_index, :] = np.inf
        else:
            el["data"][i]["track_id"] = num
            id_info[num] = current_centroids[i]
            num += 1

    return el, id_info, num


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    boxes = []
    frame = cv2.cvtColor(cv2.imread("imgs/frames/" + str(el["frame_id"]) + ".png"), cv2.COLOR_BGR2RGB)
    for data in filter(lambda x: x["bounding_box"], el["data"]):
        bbox_corners = data["bounding_box"]
        bbox_args = np.array(
            [
                bbox_corners[0],
                bbox_corners[1],
                bbox_corners[2] - bbox_corners[0],
                bbox_corners[3] - bbox_corners[1],
            ]
        )  # [left, top, w, h]
        boxes.append((bbox_args, 1, 0))

    tracks = deepsort_tracker.update_tracks(boxes, frame=frame)
    boxes_track = {i.track_id: get_centroid(i.to_ltrb()) for i in tracks}

    if not len(boxes_track):
        return el

    for i, data in enumerate(el["data"]):
        if not data["bounding_box"]:
            continue
        el_bbox = get_centroid(data["bounding_box"])
        idx, _ = min(boxes_track.items(), key=lambda x: np.linalg.norm(x[1] - el_bbox))
        el["data"][i]["track_id"] = idx
        del boxes_track[idx]

    return el
