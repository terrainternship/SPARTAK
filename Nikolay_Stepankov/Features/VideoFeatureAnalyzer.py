import os
import time
import shutil
import gzip
import pickle
import tempfile
import cv2
import numpy as np
import re
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from PIL import Image
from google.colab import drive
import logging

class VideoFeatureAnalyzer:
    def __init__(self, video_path, features_path, debug=False, step=1, mode='inception'):
        self.video_path = video_path
        self.features_path = features_path
        self.features_data_path = os.path.join(features_path, mode)
        self.features_compare_path = os.path.join(features_path, f'{mode}-compare')
        self.all_frames_data = self.load_all_frames_data()
        self.step = step

        log_level = logging.DEBUG if debug else logging.INFO
        self.logger = logging.getLogger('my_logger')
        self.logger.propagate = False
        self.logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler('debug.log')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def mount(self):
        drive.mount('/content/drive/')

    def load_all_frames_data(self):
        data = {}
        try:
            all_frames_path = os.path.join(self.features_path, 'data', f'all_frames.1.gz')
            with gzip.open(all_frames_path, 'rb') as gz_file:
                data = pickle.load(gz_file)
        except FileNotFoundError:
            self.logger.error("Файл с данными не найден.")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных фреймов: {e}")

        return data

    def get_cached_file_path(self, original_file_path, cache_folder, copy_files=True):
        file_name = os.path.basename(original_file_path)
        last_folder_name = os.path.basename(os.path.dirname(original_file_path))
        cached_folder_path = os.path.join(cache_folder, last_folder_name)
        cached_file_path = os.path.join(cached_folder_path, file_name)

        if not os.path.isdir(cached_folder_path):
            os.makedirs(cached_folder_path)

        if os.path.isfile(cached_file_path):
            self.logger.info(f'Файл уже скопирован: {cached_file_path}')
        elif copy_files:
            start_time = time.time()
            shutil.copy2(original_file_path, cached_file_path)
            end_time = time.time()
            self.logger.info(f'Файл скопирован из {original_file_path} в {cached_file_path} за {end_time - start_time:.2f} секунд.')
        else:
            return original_file_path

        return cached_file_path

    def copy_file_and_measure_time(self, src_path, dst_path):
        start_time = time.time()

        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if os.path.isfile(dst_path):
            self.logger.info(f'Файл уже скопирован: {dst_path}')
        else:
            with tempfile.NamedTemporaryFile(dir=dst_dir, delete=False) as temp_file:
                shutil.copy2(src_path, temp_file.name)
                temp_file_path = temp_file.name

            shutil.move(temp_file_path, dst_path)

            elapsed_time = time.time() - start_time
            self.logger.info(f'Файл {src_path} скопирован в {dst_path} за {elapsed_time:.2f} секунд')

    def get_feature_filename(self, video_folder_name: str):
        return os.path.join(self.features_data_path, f"{video_folder_name}.{self.step}.gz")

    def load_features(self, file_path, delete_invalid=False):
        if not os.path.isfile(file_path):
            self.logger.error(f'Файл не найден: {file_path}')
            return None

        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            self.logger.error(f'Файл поврежден или ошибка чтения: {file_path}')
            self.logger.error(f'Ошибка: {e}')
            if delete_invalid:
                self.logger.info(f'Удаление поврежденного файла: {file_path}')
                os.remove(file_path)
            return None

    def extract_video_name_from_path(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def extract_number_from_path(self, filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'\d{4}', filename)
        return int(match.group()) if match else 9999

    def build_frame_array(self, videos_features):
        return [(self.extract_video_name_from_path(video_info['video_path']), frame_number)
                for video_info in videos_features
                for frame_number in video_info['frames']]

    def compare_feature_files_cpu(self, video_folder1, video_folder2,
                                  calculate_euclidean=False, calculate_manhattan=False,
                                  frames_start1=0, frames_count1=None, frames_start2=0, frames_count2=None):
        feature_file_path1 = self.get_feature_filename(video_folder1)
        feature_file_path2 = self.get_feature_filename(video_folder2)

        videos_features1 = self.load_features(feature_file_path1)
        videos_features2 = self.load_features(feature_file_path2)

        if videos_features1 is None or videos_features2 is None:
            self.logger.error('Ошибка загрузки признаков')
            return None

        videos_features1.sort(key=lambda x: self.extract_number_from_path(x['video_path']))
        videos_features2.sort(key=lambda x: self.extract_number_from_path(x['video_path']))

        frames_array1 = self.build_frame_array(videos_features1)
        frames_array2 = self.build_frame_array(videos_features2)

        features_array1 = np.vstack([np.array(video_info['features']) for video_info in videos_features1])
        features_array2 = np.vstack([np.array(video_info['features']) for video_info in videos_features2])

        features_array1 = features_array1.reshape(features_array1.shape[0], -1)
        features_array2 = features_array2.reshape(features_array2.shape[0], -1)

        features_array1 = features_array1[frames_start1:frames_start1 + frames_count1 if frames_count1 else None]
        frames_array1 = frames_array1[frames_start1:frames_start1 + frames_count1 if frames_count1 else None]
        features_array2 = features_array2[frames_start2:frames_start2 + frames_count2 if frames_count2 else None]
        frames_array2 = frames_array2[frames_start2:frames_start2 + frames_count2 if frames_count2 else None]

        cos_similarities = cosine_similarity(features_array1, features_array2)

        if calculate_euclidean:
            euc_distances = pairwise_distances(features_array1, features_array2, metric='euclidean')
            euc_similarities = 1 / (1 + euc_distances)

        if calculate_manhattan:
            man_distances = pairwise_distances(features_array1, features_array2, metric='manhattan')
            man_similarities = 1 / (1 + man_distances)

        cos_results = []
        euc_results = []
        man_results = []

        for frame_index1 in range(len(frames_array1)):
            video_name1, frame_number1 = frames_array1[frame_index1]

            most_similar_cos = np.argmax(cos_similarities[frame_index1])
            cos_results.append([
                video_name1,
                frame_number1,
                frames_array2[most_similar_cos][0],
                frames_array2[most_similar_cos][1],
                cos_similarities[frame_index1][most_similar_cos]
            ])

            if calculate_euclidean:
                most_similar_euc = np.argmin(euc_distances[frame_index1])
                euc_results.append([
                    video_name1,
                    frame_number1,
                    frames_array2[most_similar_euc][0],
                    frames_array2[most_similar_euc][1],
                    euc_similarities[frame_index1][most_similar_euc]
                ])

            if calculate_manhattan:
                most_similar_man = np.argmin(man_distances[frame_index1])
                man_results.append([
                    video_name1,
                    frame_number1,
                    frames_array2[most_similar_man][0],
                    frames_array2[most_similar_man][1],
                    man_similarities[frame_index1][most_similar_man]
                ])

        return {
            "folder1": video_folder1,
            "folder2": video_folder2,
            "cos_similarity": cos_results,
            "euc_similarity": euc_results,
            "man_similarity": man_results
        }

    def process_video_frames(self, start_frame, frames_count, frames_folder, break_folder=None, extract_frames=False, single_folder=False):
        processed_folders = set()
        frames_info = {}
        pattern = re.compile(r'(\s*\d+ \d{2}\.\d{2}\.\d{4}\s*)-(\s*\d+ \d{2}\.\d{2}\.\d{4}\s*)\.(\d+)\.gz')

        files = os.listdir(self.features_compare_path)
        self.logger.debug(f'frames: {len(files)}')
        files.sort(reverse=True)

        for filename in files:
            self.logger.debug(f'processing: {filename}')
            source = os.path.join(self.features_compare_path, filename)
            loaded_results = self.load_compared_results(source)
            filtered_results = self.get_selected_frames_results(loaded_results, start_frame, frames_count)
            result = self.analyze_frame_sequences(filtered_results['cos_similarity'])

            match = pattern.match(filename)
            if match:
                folder1, folder2, step = match.groups()
                self.logger.debug(f'matched folders: {folder1}, {folder2}, {step}')

                if folder1 not in processed_folders:
                    start_frame1 = result['frame_number1']['min']
                    frames_info[folder1] = start_frame1
                    if extract_frames:
                        frames1 = self.load_specific_frames(f'{folder1}.{step}', start_frame1, frames_count)
                        self.extract_and_save_frames(frames1, folder1, frames_folder, single_folder)
                    processed_folders.add(folder1)

                if folder2 not in processed_folders:
                    start_frame2 = result['frame_number2']['min']
                    frames_info[folder2] = start_frame2
                    if extract_frames:
                        frames2 = self.load_specific_frames(f'{folder2}.{step}', start_frame2, frames_count)
                        self.extract_and_save_frames(frames2, folder2, frames_folder, single_folder)
                    processed_folders.add(folder2)

                start_frame = result['frame_number2']['min']
                if break_folder is not None and (folder1 == break_folder or folder2 == break_folder):
                    break
            else:
                self.logger.error(f'No match found {filename}')

        return frames_info

    def find_featured_frames(self, start_frames, start_folder, start_frame, frames_count, compare_mode):
        featured_frames = {}
        folder1 = start_folder
        start_frames[start_folder] = start_frame

        self.logger.debug(f'start frames lenght: {len(start_frames)}')
        for folder2, folder2_frame in list(start_frames.items()):
            if folder2 == start_folder:
                featured_frames[folder2] = [start_frame, start_frame, start_frame]
                self.logger.debug(f"folder(1) found: {folder1},'{compare_mode}',{start_frame},{start_frame},{start_frame}")
            else:
                compare_result = self.compare_feature_files_cpu(folder1, folder2, True, True,
                                                               start_frame, 1, folder2_frame, frames_count)

                cos_frame = compare_result['cos_similarity'][0][3]
                euc_frame = compare_result['euc_similarity'][0][3]
                man_frame = compare_result['man_similarity'][0][3]
                featured_frames[folder2] = [cos_frame, euc_frame, man_frame]
                self.logger.debug(f"folder(2) found: {folder2},'{compare_mode}',{cos_frame},{euc_frame},{man_frame}")

                if compare_mode == 'cos':
                    start_frame = cos_frame
                elif compare_mode == 'euc':
                    start_frame = euc_frame
                elif compare_mode == 'man':
                    start_frame = man_frame

                folder1 = folder2

        return featured_frames

    def create_video_from_images(self, folder_path, output_path, fps=30, frame_size=(1920, 1080)):
        images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
        images.sort()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        for image in tqdm(images):
            img_path = os.path.join(folder_path, image)
            img = cv2.imread(img_path)
            img = cv2.resize(img, frame_size)
            video.write(img)

        video.release()
        cv2.destroyAllWindows()

        self.logger.info(f'result video: {output_path}')

    def process_and_analyze_frames(self, main_frame, start_folder, break_folder, frames_folder, result_folder, compare_mode, frames_count=300, extract_frames=False):
        start_frames = self.process_video_frames(start_frame=main_frame, frames_count=frames_count, frames_folder=frames_folder, break_folder=break_folder, extract_frames=extract_frames, single_folder=False)
        featured_frames = self.find_featured_frames(start_frames, start_folder, main_frame, frames_count, compare_mode)

        compare_index = 0
        if compare_mode == 'cos':
            compare_index = 0
        elif compare_mode == 'euc':
            compare_index = 1
        elif compare_mode == 'man':
            compare_index = 2

        for folder, data in list(featured_frames.items()):
            self.logger.debug(f'featured frames: {folder}, {data[compare_index]}')
            frames = self.load_specific_frames(f"{folder}.1", data[compare_index], 1)
            self.extract_and_save_frames(frames, folder, result_folder, single_folder=True)

        self.create_video_from_images(result_folder, f'{result_folder}.mp4', fps=3, frame_size=(1920, 1080))

    def load_compared_results(self, filename):
        if not os.path.isfile(filename):
            self.logger.error(f"Файл не найден: {filename}")
            return None

        with gzip.open(filename, 'rb') as file:
            try:
                results = pickle.load(file)
                return results
            except Exception as e:
                self.logger.error(f"Ошибка загрузки {filename}: {e}")
                return None

    def analyze_frame_sequences(self, cos_similarity):
        frame_numbers1 = [item[1] for item in cos_similarity]
        frame_numbers2 = [item[3] for item in cos_similarity]

        frame_counts = Counter(frame_numbers2)
        most_common_frame = frame_counts.most_common(1)[0]
        most_common_frame_number, most_common_frame_count = most_common_frame

        half_length1 = len(frame_numbers1) // 2
        sequence_start = max(0, most_common_frame_number - half_length1)
        sequence_end = sequence_start + len(frame_numbers1)

        frame_numbers2 = list(range(sequence_start, sequence_end))

        return {
            "frame_number1": {"min": min(frame_numbers1), "max": max(frame_numbers1), "sequence": frame_numbers1},
            "frame_number2": {"min": min(frame_numbers2), "max": max(frame_numbers2), "sequence": frame_numbers2},
        }

    def get_selected_frames_results(self, results, start_frame, count):
        filtered_results = {
            "folder1": results["folder1"],
            "folder2": results["folder2"],
            "cos_similarity": [],
            "euc_similarity": [],
            "man_similarity": []
        }

        for i in range(len(results["cos_similarity"])):
            if start_frame <= i < start_frame + count:
                filtered_results["cos_similarity"].append(results["cos_similarity"][i])

                if results["euc_similarity"]:
                    filtered_results["euc_similarity"].append(results["euc_similarity"][i])

                if results["man_similarity"]:
                    filtered_results["man_similarity"].append(results["man_similarity"][i])

        return filtered_results

    def check_file_path(self, folder, file_name):
        for ext in ['.mp4', '.mov']:
            file_path = os.path.join(folder, file_name + ext)
            if os.path.exists(file_path):
                return file_path
            file_path = os.path.join(folder, file_name + ext.upper())
            if os.path.exists(file_path):
                return file_path

        raise FileNotFoundError(f"Не найден файл: {file_name} с .MP4 или .MOV расширением в папке {folder}")

    def extract_and_save_frames(self, data, folder_name, frames_folder, single_folder=False):
        last_file_name = ""
        cap = None

        for file_name, frame_number in tqdm(data):
            if file_name != last_file_name:
                if cap is not None:
                    cap.release()

                folder2 = os.path.join(self.video_path, folder_name)
                file_path = self.check_file_path(folder2, file_name)
                cap = cv2.VideoCapture(file_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                last_file_name = file_name

            self.save_frame(cap, frame_number, frames_folder, folder_name, single_folder)

        if cap is not None:
            cap.release()

    def save_frame(self, cap, frame_number, frames_folder, folder_name, single_folder):
        ret, frame = cap.read()

        if ret:
            output_folder = ""
            frame_file_name = ""
            if single_folder:
                output_folder = frames_folder
                frame_file_name = f"{folder_name}-{str(frame_number).zfill(6)}.jpg"
            else:
                output_folder = os.path.join(frames_folder, folder_name)
                frame_file_name = f"{str(frame_number).zfill(6)}.jpg"

            os.makedirs(output_folder, exist_ok=True)

            frame_path = os.path.join(output_folder, frame_file_name)
            if not os.path.isfile(frame_path):
                cv2.imwrite(frame_path, frame)

    def load_specific_frames(self, target_file, start_frame, num_frames):
        if target_file in self.all_frames_data:
            videos_features = self.all_frames_data[target_file]

            videos_features.sort(key=lambda x: self.extract_number_from_path(x['video_path']))
            frames_array = self.build_frame_array(videos_features)

            end_frame = start_frame + num_frames

            return frames_array[start_frame:end_frame]

        return None

    def copy_features(self, copy_features, new_path='/content/features'):
        if copy_features:
            features_path2 = new_path
            files = os.listdir(self.features_data_path)
            files.sort(reverse=True)
            for filename in files:
                source = os.path.join(self.features_data_path, filename)
                destination = os.path.join(features_path2, filename)
                self.copy_file_and_measure_time(source, destination)

            self.features_data_path = features_path2

    def validate_features(self, validate_features):
        invalid_features = []
        if validate_features:
            sorted_files = sorted(os.listdir(self.features_data_path))
            for file_name in sorted_files:
                file_path = os.path.join(self.features_data_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.gz'):
                    features = self.load_features(file_path, delete_invalid=False)
                    if features is None:
                        invalid_features.append(file_path)
                    else:
                        for feature in features:
                            self.logger.info(f"{feature['video_path']}, {len(feature['features'])}")

            print(invalid_features)

    def validate_frames(self, validate_frames, frame_list_file):
        if validate_frames:
            invalid_files = []
            with open(frame_list_file, 'r') as file:
                for line in file:
                    parts = line.strip().rsplit(maxsplit=1)
                    if len(parts) != 2:
                        self.logger.error(f"Неверный формат строки: {line}")
                        continue

                    video_path, expected_frames_str = parts
                    try:
                        expected_frames = int(expected_frames_str)
                    except ValueError:
                        self.logger.error(f"Некорректное количество фреймов: {expected_frames_str} в строке: {line}")
                        continue

                    if not os.path.exists(video_path):
                        self.logger.error(f"Файл {video_path} не найден.")
                        continue

                    cap = cv2.VideoCapture(video_path)
                    real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if real_frames == expected_frames:
                        self.logger.info(f"Видео {video_path}: количество фреймов соответствует ({real_frames}).")
                    else:
                        invalid_files.append(video_path)
                        self.logger.error(f"-- Видео {video_path}: расхождение в количестве фреймов (ожидалось: {expected_frames}, нашлось: {real_frames}).")

                    cap.release()

            print(invalid_files)

    def check_video_subfolders(self):
        video_subfolders = [f.name for f in os.scandir(self.video_path) if f.is_dir()]
        for folder in video_subfolders:
            feature_file = self.get_feature_filename(folder)
            if not os.path.exists(os.path.join(self.features_data_path, feature_file)):
                self.logger.error(f"Файл признаков для '{folder}' отсутствует.")

    def compare_and_save(self, folder1, folder2):
        output_file = os.path.join(self.features_compare_path, f"{folder1}-{folder2}.{self.step}.gz")

        if os.path.isfile(output_file):
            self.logger.info(f'Результаты уже есть {output_file}.')
            return

        self.logger.info(f'Сравнение папки {folder1} и {folder2}')

        start_time = time.time()
        results = self.compare_feature_files_cpu(folder1, folder2)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Время выполнения: {elapsed_time} секунд.")

        if results is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                with gzip.GzipFile(fileobj=tmp_file, mode='wb') as gzf:
                    pickle.dump(results, gzf)

            shutil.move(tmp_file.name, output_file)

    def compare_folders_and_save_results(self):
        folders = [folder for folder in os.listdir(self.video_path) if os.path.isdir(os.path.join(self.video_path, folder))]
        folders.sort(reverse=True)

        for i in range(len(folders) - 1):
            self.compare_and_save(folders[i], folders[i + 1])

    def calculate_compare_results(self, calculate_compare_results):
        if calculate_compare_results:
            self.compare_folders_and_save_results(self.video_path)

    def run_analysis(self):
            for main_frame in range(5300, 5301, 300):
                frames_folder = f'/content/drive/MyDrive/internship/result/K13-CLIP/{main_frame}'
                result_folder = f'{frames_folder}COS'
                print(frames_folder)

                self.process_and_analyze_frames(
                    self.features_compare_path,
                    main_frame=main_frame,
                    start_folder='182 27.04.2023',
                    break_folder='142 27.07.2022',
                    frames_folder=frames_folder,
                    result_folder=result_folder,
                    compare_mode='cos',  # cos, euc, man
                )

