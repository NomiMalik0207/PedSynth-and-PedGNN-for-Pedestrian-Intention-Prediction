from mapper_xml_to_csv import AnnotationsXml2Csv


class JAADAnnotationsXml2Csv(AnnotationsXml2Csv):
    def __init__(self, path_annotations_folder: str = "/datasets/JAAD", output_path: str = "/outputs/JAAD/annotations.csv"):#change path
        super().__init__(path_annotations_folder, output_path)

        # we're not using this data at the moment, no point in bloating CSV with it
        # self._annotations.update({
        #     'annotations_appearance': (self._process_annotations_appearance_file, ('id', 'frame')),
        # })

        self._sets = ['']  # JAAD doesn't have sets

    def _extract_annotations_attributes(self, track, video_id, set_name, processing_dict, video_width, video_height):
        # iterate over boxes within each track
        for box in track["box"]:
            self._check_and_add_to_dict(
                dictionary=processing_dict, key="beh", value=(track["@label"] == "pedestrian")
            )
            self._process_box(
                box=box,
                video=video_id,
                set_name=set_name,
                processing_dict=processing_dict,
                video_width=video_width,
                video_height=video_height,
            )

    def _extract_appearance_attributes(self, track, processing_dict):
        if (
            track["@label"] == "pedestrian"
        ):  # only for pedestrians not bypassers (denoted as peds)
            for box in track["box"]:
                self._check_and_add_to_dict(
                    dictionary=processing_dict,
                    key="id",
                    value=track["@id"],
                )
                for key, value in box.items():
                    # this includes frame number
                    self._check_and_add_to_dict(
                        dictionary=processing_dict,
                        key=key[1:],
                        value=value,
                    )

    def _process_annotations_appearance_file(self, file, root, set_name, processing_dict):
        # there are no pedestians
        if root["pedestrian_appearance"] is None:
            return

        # iterate over tracks in the file
        if isinstance(
            root["pedestrian_appearance"]["track"], list
        ):  # there are multiple pedestrians (tracks)
            for track in root["pedestrian_appearance"]["track"]:
                self._extract_appearance_attributes(track, processing_dict)
        else:  # there is only one track
            track = root["pedestrian_appearance"]["track"]
            self._extract_appearance_attributes(track, processing_dict)

    def _extract_pedestrian_attributes(self, pedestrian=None):
        attrs = super()._extract_pedestrian_attributes(pedestrian)
        if pedestrian is not None:
            attrs.update({
                "decision_point": int(pedestrian["@decision_point"]),
                "group_size": int(pedestrian["@group_size"]),
            })
        else:
            attrs.update({
                "decision_point": -1,
                "group_size": float("nan"),
            })
        return attrs

    def _extract_vehicle_attributes(self, vehicle_frame):
        attrs = super()._extract_vehicle_attributes(vehicle_frame)
        attrs.update({
            "speed": vehicle_frame["@action"]
        })
        return attrs


if __name__ == "__main__":
    annotations_xml_2_csv = JAADAnnotationsXml2Csv()
    annotations_xml_2_csv.generate_df()