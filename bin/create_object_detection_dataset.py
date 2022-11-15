import argparse
import random
import pandas as pd
import numpy as np
import os
from PIL import Image
import requests
import pdsc
import cv2
import tqdm
import shutil
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import math
from datetime import datetime
import warnings

# Let's assume that Mars is a sphere
# Mars equitorial radius in km
# 6794 / 2
# Mars polar radius in km
# 6752 / 2
# Let's use equitorial radius since
# most impacts are along this region
# meter radius
MARS_RADIUS = 6794 / 2 * 1000

class CreateObjectDetectionDataset:
    def __init__(self, latlon_file, pdsc_table_path, output_dir,
                 get_positives, get_negatives, n_negative_samples, random_seed, mars_radius=MARS_RADIUS):
        self.mars_radius = mars_radius
        self.read_latlon_data(latlon_file)
        self.output_dir = output_dir
        self.get_positives = get_positives
        self.get_negatives = get_negatives
        self.n_negative_samples = n_negative_samples
        self.client = pdsc.PdsClient(pdsc_table_path)

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.positive_output_dir = os.path.join(self.output_dir, 'positive')
        self.pos_raw_dir = os.path.join(self.positive_output_dir, 'raw')
        self.pos_enhanced_dir = os.path.join(self.positive_output_dir, 'enhanced')

        self.negative_output_dir = os.path.join(self.output_dir, 'negative')
        self.neg_raw_dir = os.path.join(self.negative_output_dir, 'raw')
        self.neg_enhanced_dir = os.path.join(self.negative_output_dir, 'enhanced')

        self.pos_dataframe_path = os.path.join(self.positive_output_dir, 'positive_output.csv')
        self.neg_dataframe_path = os.path.join(self.negative_output_dir, 'negative_output.csv')
        self.combined_dataframe_path = os.path.join(self.output_dir, 'combined_output.csv')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if get_positives:
            if os.path.exists(self.positive_output_dir):
                shutil.rmtree(self.positive_output_dir)
            os.makedirs(self.positive_output_dir)
            os.makedirs(self.pos_raw_dir)
            os.makedirs(self.pos_enhanced_dir)
        if get_negatives:
            if os.path.exists(self.negative_output_dir):
                shutil.rmtree(self.negative_output_dir)
            os.makedirs(self.negative_output_dir)
            os.makedirs(self.neg_raw_dir)
            os.makedirs(self.neg_enhanced_dir)

        self.positive_df = pd.DataFrame(columns=['impact_id', 'observation_id', 'lat', 'lon', 'diameter_in_m',
                                          'xmin', 'xmax', 'ymin', 'ymax', 'xmin_px', 'xmax_px', 'ymin_px', 'ymax_px',
                                          'bb_width_px', 'bb_height_px', 'date_discovered', 'image_path', 'class'])
        self.negative_df = pd.DataFrame(columns=['impact_id', 'observation_id', 'lat', 'lon', 'diameter_in_m',
                                          'xmin', 'xmax', 'ymin', 'ymax', 'xmin_px', 'xmax_px', 'ymin_px', 'ymax_px',
                                          'bb_width_px', 'bb_height_px', 'date_discovered', 'image_path', 'class'])

    def process_data(self):
        if self.get_positives:
            self.process_positive_data()
            self.positive_df.to_csv(self.pos_dataframe_path, index=False)
        if self.get_negatives:
            self.process_negative_data()
            self.negative_df.to_csv(self.neg_dataframe_path, index=False)
        if not self.get_positives and os.path.exists(self.pos_dataframe_path):
            self.positive_df = pd.read_csv(self.pos_dataframe_path)
        if not self.get_negatives and os.path.exists(self.neg_dataframe_path):
            self.negative_df = pd.read_csv(self.neg_dataframe_path)
        self.combined_df = pd.concat([self.positive_df, self.negative_df])
        self.combined_df.to_csv(self.combined_dataframe_path, index=False)

    def read_latlon_data(self, latlon_file):
        latlon_df = pd.read_csv(latlon_file)
        cleaned_cols = []
        for col in latlon_df:
            cleaned_col = col.strip()
            cleaned_col = cleaned_col.replace('\'', '')
            cleaned_cols.append(cleaned_col)

        latlon_df.columns = cleaned_cols
        latlon_df = latlon_df[['Latitude (deg E, centric)', 'Longitude (deg N)', 'Diameter (m)', 'After Image Date']]
        self.latlon_df = latlon_df

    def parse_file_name_to_url(self, file_name_specification):
        # Get path to map projected image by using .NOMAP.
        file_name_specification = file_name_specification.replace('COLOR', 'RGB')
        url_suffix = file_name_specification.split(".")[0] + ".NOMAP.browse.jpg"
        url = "https://hirise-pds.lpl.arizona.edu/PDS/EXTRAS/" + url_suffix
        return url

    def get_image_from_metadata(self, metadata):
        url = self.parse_file_name_to_url(metadata.file_name_specification)
        response = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def get_m_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = radians(lat1), radians(lon1), radians(lat2), radians(lon2)
        latlon1 = np.array([lat1, lon1])
        latlon2 = np.array([lat2, lon2])
        return haversine_distances([latlon1, latlon2]) * self.mars_radius

    def get_xy_resolution_per_pixel(self, metadata, image_height, image_width):

        lat_c1 = metadata.corner1_latitude
        lon_c1 = metadata.corner1_longitude
        lat_c2 = metadata.corner2_latitude
        lon_c2 = metadata.corner2_longitude
        lat_c3 = metadata.corner3_latitude
        lon_c3 = metadata.corner3_longitude
        lat_c4 = metadata.corner4_latitude
        lon_c4 = metadata.corner4_longitude

        dist1 = self.get_m_distance(lat_c1, lon_c1, lat_c2, lon_c2)
        dist2 = self.get_m_distance(lat_c2, lon_c2, lat_c3, lon_c3)
        dist3 = self.get_m_distance(lat_c3, lon_c3, lat_c4, lon_c4)
        dist4 = self.get_m_distance(lat_c3, lon_c3, lat_c1, lon_c1)

        x_pixel_resolution = np.mean([dist1, dist3]) / image_width
        y_pixel_resolution = np.mean([dist2, dist4]) / image_height

        return x_pixel_resolution, y_pixel_resolution


    def process_object(self, current_impact_id, observation_id, curr_metadata, lat, lon, diameter, after_img_date, is_positive=True):
        if is_positive and after_img_date > curr_metadata.start_time:
            return

        if is_positive:
            current_raw_dir = self.pos_raw_dir
            current_enh_dir = self.pos_enhanced_dir
        else:
            current_raw_dir = self.neg_raw_dir
            current_enh_dir = self.neg_enhanced_dir
        y, x = self.get_yx_from_metadata(curr_metadata, lat, lon)
        image = self.get_image_from_metadata(curr_metadata)
        h, w = image.shape[0], image.shape[1]
        y, x = int(y * h), int(x * w)
        image = image[:, :, ::-1]
        image_raw = Image.fromarray(image)

        # compute meter to pixel resolution
        x_pixel_resolution, y_pixel_resolution = self.get_xy_resolution_per_pixel(curr_metadata, h, w)
        # x_pixel_resolution, y_pixel_resolution = 1.0, 1.0
        scaling_factor = 20
        diameter_y = math.ceil(diameter / y_pixel_resolution) * scaling_factor
        diameter_x = math.ceil(diameter / x_pixel_resolution) * scaling_factor

        red = np.array([255, 0, 0])

        bb_size_y = diameter_y
        bb_size_x = diameter_x
        bb_thickness = 5
        bb_x_left_max = min(w - 1 - bb_thickness, max(bb_thickness, x - bb_size_x))
        bb_x_right_max = max(bb_thickness, min(w - 1 - bb_thickness, x + bb_size_x))
        bb_y_top_max = min(h - 1 - bb_thickness, max(bb_thickness, y - bb_size_y))
        bb_y_bottom_max = max(bb_thickness, min(h - 1 - bb_thickness, y + bb_size_y))
        if bb_x_left_max == bb_x_right_max or bb_y_top_max == bb_y_bottom_max:
            return

        # Crop out subimage
        image_crop = image_raw.crop((bb_x_left_max, bb_y_top_max, bb_x_right_max, bb_y_bottom_max))

        # Draw bounding box
        image[bb_y_bottom_max:bb_y_bottom_max + bb_thickness, bb_x_left_max:bb_x_right_max, :] = red
        image[bb_y_top_max:bb_y_top_max + bb_thickness, bb_x_left_max:bb_x_right_max, :] = red
        image[bb_y_top_max:bb_y_bottom_max, bb_x_left_max:bb_x_left_max + bb_thickness, :] = red
        image[bb_y_top_max:bb_y_bottom_max, bb_x_right_max : bb_x_right_max + bb_thickness, :] = red
        image_bb = Image.fromarray(image)

        raw_imagepath = os.path.join(current_raw_dir, observation_id + '.png')
        cropped_imagepath = os.path.join(current_enh_dir, observation_id + '_cropped.png')
        bb_imagepath = os.path.join(current_enh_dir, observation_id + '_bounding_box.png')

        new_row = {'impact_id': current_impact_id, 'observation_id': observation_id,
                                      'lat': lat, 'lon': lon, 'diameter_in_m': diameter, 'xmin': bb_x_left_max / w,
                                      'xmax': bb_x_right_max / w, 'ymin': bb_y_top_max / h, 'ymax': bb_y_bottom_max / h,
                                      'xmin_px': bb_x_left_max, 'xmax_px': bb_x_right_max, 'ymin_px': bb_y_top_max,
                                      'ymax_px': bb_y_bottom_max, 'bb_height_px': bb_size_y, 'bb_width_px': bb_size_x,
                                      'date_discovered': after_img_date,
                                      'image_path': raw_imagepath,
                                      'class': 1 if is_positive else 0}

        if is_positive:
            self.positive_df = self.positive_df.append(new_row, ignore_index=True)
        else:
            self.negative_df = self.negative_df.append(new_row, ignore_index=True)

        # Save images
        image_raw.save(raw_imagepath)
        image_crop.save(cropped_imagepath)
        image_bb.save(bb_imagepath)

    def latlon_to_images(self, lat, lon, diameter, after_img_date, current_impact_id):
        observation_ids = self.get_all_overlapping_observations(lat, lon)
        metadata = self.client.query_by_observation_id("hirise_rdr", observation_ids)
        after_img_date = datetime.strptime(after_img_date, '%Y-%m-%d')
        for i, observation_id in enumerate(observation_ids):
            self.process_object(current_impact_id, observation_id, metadata[i], lat, lon,
                                diameter, after_img_date, is_positive=True)

    def get_yx_from_metadata(self, metadata, lat, lon):
        metadata.map_projection_type = metadata.map_projection_type.strip()
        rdr_localizer = pdsc.get_localizer(metadata, nomap=True)
        y, x = rdr_localizer.latlon_to_pixel(lat, lon)
        return y, x

    def get_latlon_from_metadata(self, metadata, y, x):
        metadata.map_projection_type = metadata.map_projection_type.strip()
        rdr_localizer = pdsc.get_localizer(metadata, nomap=True)
        lat, lon = rdr_localizer.pixel_to_latlon(y, x)
        return lat, lon

    def get_all_overlapping_observations(self, lat, lon):
        observation_ids = self.client.find_observations_of_latlon('hirise_rdr', lat, lon)
        return observation_ids

    def process_positive_data(self):
        # disable silly sklearn warnings so we can see progress more easily
        warnings.filterwarnings("ignore")
        num_rows = self.latlon_df.shape[0]

        for i in tqdm.tqdm(range(num_rows)):
            row = self.latlon_df.iloc[i]
            lat = row['Latitude (deg E, centric)']
            lon = row['Longitude (deg N)']
            diameter = row['Diameter (m)']
            after_img_date = row['After Image Date']
            current_impact_id = i
            self.latlon_to_images(lat, lon, diameter, after_img_date, current_impact_id)

    def process_negative_data(self):
        random_metadata = self.get_random_metadata_items(self.n_negative_samples)
        warnings.filterwarnings("ignore")

        for i in tqdm.tqdm(range(self.n_negative_samples)):
            current_impact_id = -i - 1
            metadata = random_metadata[i]
            image = self.get_image_from_metadata(metadata)
            h, w = image.shape[0], image.shape[1]
            padding_size = 50
            x = random.randint(padding_size, w - 1 - padding_size)
            y = random.randint(padding_size, h - 1 - padding_size)
            lat, lon = self.get_latlon_from_metadata(metadata, y / h , x / w)
            after_img_date = "None"
            radius = random.randint(5, 10)
            self.process_object(current_impact_id, metadata.observation_id, metadata, lat, lon,
                                radius, after_img_date, is_positive=False)


    def get_random_indices(self, n, index_range):
        random_indices = (np.random.rand(n) * index_range) // 1
        return random_indices.astype(int)

    def get_rgb_metadata_items(self, metadata_list):
        return [m for m in metadata_list if 'COLOR' in m.product_id]

    def get_random_metadata_items(self, n_samples):
        """
            Get a random sample of metadata items
        """
        # Get the full list of metadat, filter out the non-RED items,
        # then return n random samples.
        metadata_list = self.client.query('hirise_rdr')
        red_metadata_list = self.get_rgb_metadata_items(metadata_list)
        random_indices = self.get_random_indices(n_samples, len(red_metadata_list))
        observation_id_list = np.array([m.observation_id for m in red_metadata_list])
        random_observation_ids = observation_id_list[random_indices]
        random_metadata_items = self.client.query_by_observation_id("hirise_rdr", random_observation_ids)
        return self.get_rgb_metadata_items(random_metadata_items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latlon_file', type=str, default='latlon_data.csv')
    parser.add_argument('--pdsc_table_path', type=str, default='data/pdsc_tables')
    parser.add_argument('--output_dir', type=str, default='data/test_set')
    parser.add_argument('--get_positives', action='store_true', default=False)
    parser.add_argument('--get_negatives', action='store_true', default=False)
    parser.add_argument('--n_negative_samples', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    dataset = CreateObjectDetectionDataset(args.latlon_file, args.pdsc_table_path, args.output_dir,
                                                    args.get_positives, args.get_negatives, args.n_negative_samples,
                                           args.random_seed)
    dataset.process_data()

