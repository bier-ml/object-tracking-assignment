import numpy as np


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

    def get_centroid(bbox: list):
        bbox = np.array(bbox)
        return np.mean(bbox.reshape(2, 2), axis=0).astype(int)

    el["data"] = [item for item in el["data"] if any(item["bounding_box"])]

    if el["frame_id"] == 1:
        for i, item in enumerate(el["data"]):
            el["data"][i]["track_id"] = i
            id_info[i] = get_centroid(item["bounding_box"])
        num = len(el["data"])
        return el, id_info, num

    current_centroids = np.array([get_centroid(item["bounding_box"]) for item in el["data"]])
    previous_ids = np.array(list(id_info.keys()))
    previous_centroids = np.array(list(id_info.values()))

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
    return el
