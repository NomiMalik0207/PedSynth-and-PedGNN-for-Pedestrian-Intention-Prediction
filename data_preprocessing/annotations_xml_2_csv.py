import os
import xmltodict
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable, Dict, List, OrderedDict


class AnnotationsXml2Csv():
    # This is to be used for mapping xml annotations to csv mapping
    def __init__(self, path_annotations_folder: str, output_path: str):
        self.path_annotations_folder = path_annotations_folder
        self.output_path = output_path

        self._attributes_annotations = 'annotations_attributes'
        self._sets = sorted([d for d in os.listdir(os.path.join(path_annotations_folder, self._attributes_annotations))
                            if os.path.isdir(os.path.join(path_annotations_folder, self._attributes_annotations, d))])

        # default basic annotations
        self._annotations = {
            'annotations': (self._process_annotations_file, ('set_name', 'video', 'id', 'frame')),
            'annotations_vehicle': (self._process_annotations_vehicle_file, ('set_name', 'video', 'frame')),
        }

    @staticmethod
    def _check_and_add_to_dict(dictionary: Dict, key, value) -> Dict:
        if not key in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)

        return dictionary

    def _process_box(self, box, video, set_name, processing_dict, video_width, video_height):
        self._check_and_add_to_dict(
            dictionary=processing_dict, key="video", value=video
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict, key="video_width", value=video_width
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict, key="video_height", value=video_height
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict, key="set_name", value=set_name
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict,
            key="frame",
            value=box["@frame"],
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict,
            key="x2",
            value=float(box["@xbr"]),
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict,
            key="y2",
            value=float(box["@ybr"]),
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict,
            key="x1",
            value=float(box["@xtl"]),
        )
        self._check_and_add_to_dict(
            dictionary=processing_dict,
            key="y1",
            value=float(box["@ytl"]),
        )
        attrs = box["attribute"] if isinstance(box["attribute"], list) else [box["attribute"]]
        for attribute in attrs:
            name, value = attribute["@name"], attribute["#text"]

            # add attributes of the pedestrian
            if name == "id":
                self._check_and_add_to_dict(
                    dictionary=processing_dict,
                    key=name,
                    value=value,
                )

                if value in self.dict_aggregated_attributes.keys():
                    for key, value in self.dict_aggregated_attributes[
                        value
                    ].items():
                        self._check_and_add_to_dict(
                            dictionary=processing_dict,
                            key=key,
                            value=value,
                        )
                else:
                    for key, value in self._extract_pedestrian_attributes(None).items():
                        self._check_and_add_to_dict(
                            dictionary=processing_dict,
                            key=key,
                            value=value,
                        )
                break

    def _process_folder(self, folder: str, file_processing_callback: Callable):
        print(f"Processing: {folder} ")

        processing_dict = OrderedDict()

        # iterate over sets
        for set_name in self._sets:
            set_folder = os.path.join(self.path_annotations_folder, folder, set_name)
            # iterate over files in the folder
            for file in tqdm(
                sorted([file for file in os.listdir(set_folder) if file.endswith(".xml")]),
                desc=f"{set_name}",
            ):
                # get file path
                path_file = os.path.join(
                    set_folder,
                    file,
                )

                with open(path_file) as f:
                    root = xmltodict.parse(f.read())

                file_processing_callback(file, root, set_name, processing_dict)

        return processing_dict

    def _extract_annotations_attributes(self, track, video_id, set_name, processing_dict, video_width, video_height):
        # iterate over boxes within each track
        for box in track["box"]:
            self._process_box(
                box=box,
                video=video_id,
                set_name=set_name,
                processing_dict=processing_dict,
                video_width=video_width,
                video_height=video_height,
            )

    def _process_annotations_file(self, file, root, set_name, processing_dict):
        # if there are no pedestians in the video -> skip the video
        # (there are some videos where there are no pedestians)
        if not "track" in root["annotations"].keys():
            return

        video_id = root["annotations"]["meta"]["task"]["name"]
        video_width = int(root["annotations"]["meta"]["task"]["original_size"]["width"])
        video_height = int(root["annotations"]["meta"]["task"]
                           ["original_size"]["height"])
        if set_name:
            video_id = video_id.replace(f"{set_name}_", "")

        # iterate over tracks in the file
        if isinstance(
            root["annotations"]["track"], List
        ):  # there are multiple predestians (tracks)
            for track in root["annotations"]["track"]:
                self._extract_annotations_attributes(
                    track, video_id, set_name, processing_dict, video_width, video_height)

        else:  # there is only one track
            track = root["annotations"]["track"]
            self._extract_annotations_attributes(
                track, video_id, set_name, processing_dict, video_width, video_height)

    def _extract_pedestrian_attributes(self, pedestrian=None):
        if pedestrian is not None:
            #print(pedestrian["@age"])
            #print(pedestrian["@crossing"])
            #kkk
            return {
                "age": pedestrian["@age"],
                "crossing": int(pedestrian["@crossing"]),
                "crossing_point": int(pedestrian["@crossing_point"]),
                "gender": pedestrian["@gender"],
            }
        else:
            return {
                "age": "n/a",
                "crossing": -1,
                "crossing_point": -1,
                "gender": "n/a",
            }

    def _process_annotations_attributes_file(self, file, root, set_name, processing_dict):
        # there are no pedestians
        if root["ped_attributes"] is None:
            return

        if isinstance(
            root["ped_attributes"]["pedestrian"], List
        ):  # there are multiple pedestrians (tracks)
            for pedestrian in root["ped_attributes"]["pedestrian"]:
                processing_dict[pedestrian["@id"]
                                ] = self._extract_pedestrian_attributes(pedestrian)
        else:  # single pedestrian
            pedestrian = root["ped_attributes"]["pedestrian"]
            processing_dict[
                pedestrian["@id"]
            ] = self._extract_pedestrian_attributes(pedestrian)

    def _extract_vehicle_attributes(self, frame):
        return {
            "frame": frame["@id"]
        }

    def _process_annotations_vehicle_file(self, file, root, set_name, processing_dict):
        # there are no pedestians
        if root["vehicle_info"] is None:
            return

        for frame in root["vehicle_info"]["frame"]:
            self._check_and_add_to_dict(
                dictionary=processing_dict,
                key="video",
                value=file[:10],
            )
            self._check_and_add_to_dict(
                dictionary=processing_dict,
                key="set_name",
                value=set_name,
            )
            for key, value in self._extract_vehicle_attributes(frame).items():
                self._check_and_add_to_dict(
                    dictionary=processing_dict,
                    key=key,
                    value=value,
                )
            path_to_videos = os.path.join(
                self.path_annotations_folder, "videos", set_name)
            video_file_name = file[:10] + ".mp4"
            path_to_video_file = os.path.join(path_to_videos, video_file_name)
            self._check_and_add_to_dict(
                dictionary=processing_dict,
                key="video_path",
                value=path_to_video_file,
            )

    def get_complete_annotations(self):
        self.dict_aggregated_attributes = self._process_folder(
            folder=self._attributes_annotations,
            file_processing_callback=self._process_annotations_attributes_file,
        )

        dfs = []
        for folder, (file_processing_callback, index) in self._annotations.items():
            folder_dict = self._process_folder(
                folder=folder,
                file_processing_callback=file_processing_callback,
            )
            df = pd.DataFrame.from_dict(folder_dict)
            df.set_index(list(index), inplace=True)
            dfs.append(df)

        df_full: pd.DataFrame = dfs[0]
        for df in dfs[1:]:
            df_full: pd.DataFrame = df_full.join(df, how="left")

        self.df_full: pd.DataFrame = df_full.reset_index(drop=False)

        return self.df_full

    def generate_df(self):
        dirname = os.path.dirname(self.output_path)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.get_complete_annotations()
        self.df_full.to_csv(self.output_path, index=False)

        print(
            f"Annotations generated and stored in {self.output_path}. You should now move them to the dataset folder.")
