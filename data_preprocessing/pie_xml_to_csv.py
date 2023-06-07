from mapper_xml_to_csv import AnnotationsXml2Csv


class PIEAnnotationsXml2Csv(AnnotationsXml2Csv):
    # This is to be used for PIE xml annotations to csv mapping
    def __init__(self, path_annotations_folder: str = "/media/nriaz/NaveedData/PIE_dataset/", output_path: str = "/media/nriaz/NaveedData/PIE_dataset/annotations.csv"):
        super().__init__(path_annotations_folder, output_path)

    def _extract_pedestrian_attributes(self, pedestrian):
        attrs = super()._extract_pedestrian_attributes(pedestrian)
        if pedestrian is not None:
            attrs.update({
                "critical_point": int(pedestrian["@critical_point"]),
                "exp_start_point": int(pedestrian["@exp_start_point"]),
            })
        else:
            attrs.update({
                "critical_point": -1,
                "exp_start_point": 0,
            })
        return attrs

    def _extract_vehicle_attributes(self, vehicle_frame):
        attrs = super()._extract_vehicle_attributes(vehicle_frame)
        attrs.update({
            "speed": vehicle_frame["@GPS_speed"]
        })
        return attrs


if __name__ == "__main__":
    annotations_xml_2_csv = PIEAnnotationsXml2Csv()
    annotations_xml_2_csv.generate_df()