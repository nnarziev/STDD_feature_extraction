# This script is for Feature Extraction of STDD season 2

import datetime
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import HTTPError

import statistics

NUMBER_OF_EMA = 6
LOCATION_HOME = "HOME"
LOCATION_LIBRARY = "LIBRARY"
LOCATION_UNIVERSITY = "UNIV"

UNLOCK_DURATION = "UNLOCK_DURATION"
CALLS = "CALLS"
ACTIVITY_TRANSITION = "ACTIVITY_TRANSITION"
ACTIVITY_RECOGNITION = "ACTIVITY_RECOGNITION"
AUDIO_LOUDNESS = "AUDIO_LOUDNESS"
TOTAL_DIST_COVERED = "TOTAL_DIST_COVERED"
MAX_DIST_TWO_LOCATIONS = "MAX_DIST_TWO_LOCATIONS"
RADIUS_OF_GYRATION = "RADIUS_OF_GYRATION"
MAX_DIST_FROM_HOME = "MAX_DIST_FROM_HOME"
NUM_OF_DIF_PLACES = "NUM_OF_DIF_PLACES"
GEOFENCE = "GEOFENCE"
SCREEN_ON_OFF = "SCREEN_ON_OFF"
APPLICATION_USAGE = "APPLICATION_USAGE"
SURVEY_EMA = "SURVEY_EMA"

APP_PCKG_TOCATEGORY_MAP_FILENAME = "package_to_category_map.csv"

pckg_to_cat_map = {}
cat_list = pd.read_csv('Cat_group.csv')


def in_range(number, start, end):
    if start <= number <= end:
        return True
    else:
        return False


def get_filename_from_data_src(filenames, data_src, username):
    for filename in filenames:
        if username in filename and data_src in filename:
            return filename


def get_unlock_result(filename, start_time, end_time, username_id):
    result = {
        "duration": 0,
        "number": 0
    }
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]

    for line_unlock in rows.itertuples(index=False):
        start = line_unlock.timestamp_start
        end = line_unlock.timestamp_end
        duration = line_unlock.duration
        if in_range(int(start) / 1000, start_time, end_time) and in_range(int(end) / 1000, start_time, end_time):
            result['duration'] += int(duration)
            result['number'] += 1

    if result['number'] == 0:
        result['duration'] = "-"
        result['number'] = "-"

    return result


def get_unlock_duration_at_location(filename_geofence, filename_unlock, start_time, end_time, location_name, username_id):
    result = {
        "duration": 0,
        "number": 0
    }
    geofence_rows = pd.read_csv(filename_geofence)
    geofence_rows = geofence_rows.loc[geofence_rows['username_id'] == username_id]

    for line_geofence in geofence_rows.itertuples(index=False):
        enter_time = line_geofence.timestamp_enter
        exit_time = line_geofence.timestamp_exit
        location = line_geofence.location

        if in_range(int(enter_time) / 1000, start_time, end_time) and location == location_name:
            unlock_rows = pd.read_csv(filename_unlock)
            unlock_rows = unlock_rows.loc[unlock_rows['username_id'] == username_id]

            for line_unlock in unlock_rows.itertuples(index=False):
                start = line_unlock.timestamp_start
                end = line_unlock.timestamp_end
                duration = line_unlock.duration
                if in_range(int(start) / 1000, int(enter_time) / 1000, int(exit_time) / 1000) and in_range(int(end) / 1000, int(enter_time) / 1000, int(exit_time) / 1000):
                    result['duration'] += int(duration)
                    result['number'] += 1

    if result['number'] == 0:
        result['duration'] = "-"
        result['number'] = "-"

    return result


def get_total_distance(filename, start_time, end_time, username_id):
    result = 0.0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        distance = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = float(distance)

    return result if result > 0.0 else "-"


def get_std_of_displacement(filename, start_time, end_time, username_id):
    result = 0.0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        value = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = float(value)

    return result if result > 0.0 else "-"


def get_steps(filename, start_time, end_time, username_id):
    result = 0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        timestamp = row.timestamp
        if in_range(int(timestamp) / 1000, start_time, end_time):
            result += 1

    return result if result > 0 else "-"


def get_sig_motion(filename, start_time, end_time, username_id):
    result = 0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        timestamp = row.timestamp
        if in_range(int(timestamp) / 1000, start_time, end_time):
            result += 1

    return result if result > 0 else "-"


def get_radius_of_gyration(filename, start_time, end_time, username_id):
    result = 0.0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        value = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = float(value)

    return result if result > 0.0 else "-"


def get_phonecall(filename, start_time, end_time, username_id):
    result = {
        "in_duration": 0,
        "out_duration": 0,
        "in_number": 0,
        "out_number": 0
    }

    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        call_type = row.call_type
        duration = row.duration
        if in_range(int(end) / 1000, start_time, end_time):
            if call_type == "IN":
                result["in_duration"] += int(duration)
                result["in_number"] += 1
            elif call_type == "OUT":
                result["out_duration"] += int(duration)
                result["out_number"] += 1

    if result["in_number"] == 0:
        result["in_duration"] = "-"
        result["in_number"] = "-"

    if result["out_number"] == 0:
        result["out_duration"] = "-"
        result["out_number"] = "-"

    return result


def get_num_of_dif_places(filename, start_time, end_time, username_id):
    result = 0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        value = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = int(value)

    return result if result > 0 else "-"


def get_max_dist_two_locations(filename, start_time, end_time, username_id):
    result = 0.0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        value = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = float(value)

    return result if result > 0.0 else "-"


def get_max_dist_home(filename, start_time, end_time, username_id):
    result = 0.0
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.timestamp_start
        end = row.timestamp_end
        value = row.value
        if in_range(int(start) / 1000, start_time, end_time):
            result = float(value)

    return result if result > 0.0 else "-"


def get_light(filename, start_time, end_time, username_id):
    result = {
        'min': 0,
        'max': 0,
        'avg': 0
    }
    light_data = []
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        timestamp = row.timestamp
        value = row.value
        if in_range(int(timestamp) / 1000, start_time, end_time):
            light_data.append(int(value))

    if light_data.__len__() > 0:
        result['min'] = min(light_data)
        result['max'] = max(light_data)
        result['avg'] = statistics.mean(light_data)
    else:
        result['min'] = "-"
        result['max'] = "-"
        result['avg'] = "-"

    return result


def get_hrm(filename, start_time, end_time, username_id):
    result = {
        'min': 0,
        'max': 0,
        'avg': 0
    }
    hrm_data = []
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        timestamp = row.timestamp
        value = row.value
        if in_range(int(timestamp) / 1000, start_time, end_time):
            hrm_data.append(int(value))

    if hrm_data.__len__() > 0:
        result['min'] = min(hrm_data)
        result['max'] = max(hrm_data)
        result['avg'] = statistics.mean(hrm_data)
    else:
        result['min'] = "-"
        result['max'] = "-"
        result['avg'] = "-"

    return result


def get_num_of_dif_activities(filename, start_time, end_time, username_id):
    result = {
        "still": 0,
        "walking": 0,
        "running": 0,
        "on_bicycle": 0,
        "in_vehicle": 0,
        "on_foot": 0,
        "tilting": 0,
        "unknown": 0
    }

    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        timestamp = row.timestamp
        activity_type = row.activity_type
        confidence = row.confidence
        if in_range(int(timestamp) / 1000, start_time, end_time):
            if float(confidence) > 0.80:
                if activity_type == 'STILL':
                    result['still'] += 1
                elif activity_type == 'WALKING':
                    result['walking'] += 1
                elif activity_type == 'RUNNING':
                    result['running'] += 1
                elif activity_type == 'ON_BICYCLE':
                    result['on_bicycle'] += 1
                elif activity_type == 'IN_VEHICLE':
                    result['in_vehicle'] += 1
                elif activity_type == 'ON_FOOT':
                    result['on_foot'] += 1
                elif activity_type == 'TILTING':
                    result['tilting'] += 1
                elif activity_type == 'UNKNOWN':
                    result['unknown'] += 1

    if result['still'] == 0:
        result['still'] = "-"
    if result['walking'] == 0:
        result['walking'] = "-"
    if result['running'] == 0:
        result['running'] = "-"
    if result['on_bicycle'] == 0:
        result['on_bicycle'] = "-"
    if result['in_vehicle'] == 0:
        result['in_vehicle'] = "-"
    if result['on_foot'] == 0:
        result['on_foot'] = "-"
    if result['tilting'] == 0:
        result['tilting'] = "-"
    if result['unknown'] == 0:
        result['unknown'] = "-"

    return result


def get_app_category_usage(filename, start_time, end_time, username_id):
    result = {
        "Entertainment & Music": 0,
        "Utilities": 0,
        "Shopping": 0,
        "Games & Comics": 0,
        "Others": 0,
        "Health & Wellness": 0,
        "Social & Communication": 0,
        "Education": 0,
        "Travel": 0,
        "Art & Design & Photo": 0,
        "News & Magazine": 0,
        "Food & Drink": 0,
        "Unknown & Background": 0
    }

    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    for row in rows.itertuples(index=False):
        start = row.start_timestamp
        end = row.end_timestamp
        pckg_name = row.package_name
        duration = int(end) - int(start)
        if in_range(int(start), start_time, end_time) and in_range(int(end), start_time, end_time) and duration > 0:
            if pckg_name in pckg_to_cat_map:
                category = pckg_to_cat_map[pckg_name]
            else:
                category = get_google_category(pckg_name)
                pckg_to_cat_map[pckg_name] = category

            if category == "Entertainment & Music":
                result['Entertainment & Music'] += duration
            elif category == "Utilities":
                result['Utilities'] += duration
            elif category == "Shopping":
                result['Shopping'] += duration
            elif category == "Games & Comics":
                result['Games & Comics'] += duration
            elif category == "Others":
                result['Others'] += duration
            elif category == "Health & Wellness":
                result['Health & Wellness'] += duration
            elif category == "Social & Communication":
                result['Social & Communication'] += duration
            elif category == "Education":
                result['Education'] += duration
            elif category == "Travel":
                result['Travel'] += duration
            elif category == "Art & Design & Photo":
                result['Art & Design & Photo'] += duration
            elif category == "News & Magazine":
                result['News & Magazine'] += duration
            elif category == "Food & Drink":
                result['Food & Drink'] += duration
            elif category == "Unknown & Background":
                result['Unknown & Background'] += duration

    if result['Entertainment & Music'] == 0:
        result['Entertainment & Music'] = "-"
    if result['Utilities'] == 0:
        result['Utilities'] = "-"
    if result['Shopping'] == 0:
        result['Shopping'] = "-"
    if result['Games & Comics'] == 0:
        result['Games & Comics'] = "-"
    if result['Others'] == 0:
        result['Others'] = "-"
    if result['Health & Wellness'] == 0:
        result['Health & Wellness'] = "-"
    if result['Social & Communication'] == 0:
        result['Social & Communication'] = "-"
    if result['Education'] == 0:
        result['Education'] = "-"
    if result['Travel'] == 0:
        result['Travel'] = "-"
    if result['Art & Design & Photo'] == 0:
        result['Art & Design & Photo'] = "-"
    if result['News & Magazine'] == 0:
        result['News & Magazine'] = "-"
    if result['Food & Drink'] == 0:
        result['Food & Drink'] = "-"
    if result['Unknown & Background'] == 0:
        result['Unknown & Background'] = "-"

    return result


def get_sleep_duration(filename, start_time, end_time, username_id):
    result = 0

    durations = []
    rows = pd.read_csv(filename)
    rows = rows.loc[rows['username_id'] == username_id]
    row_iterator = rows.iterrows()
    _, current_row = next(row_iterator)
    for idx, next_row in row_iterator:
        if in_range(int(current_row.timestamp_start / 1000), start_time, end_time):
            durations.append(int((int(next_row.timestamp_start) - int(current_row.timestamp_end)) / 1000))
        current_row = next_row

    if durations:
        result = max(durations)
    return result if result > 0 else "-"


def get_google_category(app_package):
    url = "https://play.google.com/store/apps/details?id=" + app_package
    grouped_Category = ""
    try:
        html = urlopen(url)
        source = html.read()
        html.close()

        soup = BeautifulSoup(source, 'html.parser')
        table = soup.find_all("a", {'itemprop': 'genre'})

        genre = table[0].get_text()

        grouped = cat_list[cat_list['App Category'] == genre]['Grouped Category'].values
        # print(grouped)

        if len(grouped) > 0:
            grouped_Category = grouped[0]
        else:
            grouped_Category = 'NotMapped'
    except HTTPError as e:
        grouped_Category = 'Unknown or Background'

    finally:
        # print("Pckg: ", App, ";   Category: ", grouped_Category)
        return grouped_Category


def extract_features():
    try:
        columns = [
            'user_id',
            'day_num',
            'ema',
            'phq1',
            'phq2',
            'phq3',
            'phq4',
            'phq5',
            'phq6',
            'phq7',
            'phq8',
            'phq9',
            'unlock_duration',
            'unlock_number',
            'unlock_duration_home',
            'unlock_number_home',
            'unlock_duration_univ',
            'unlock_number_univ',
            'unlock_duration_library',
            'unlock_number_library',
            'total_distance',
            'std_displacement',
            'steps',
            'significant_motion',
            'radius_of_gyration',
            'in_call_duration',
            'in_call_number',
            'out_call_duration',
            'out_call_number',
            'num_of_dif_places',
            'max_dist_btw_two_locations',
            'max_dist_home',
            'light_min',
            'light_max',
            'light_avg',
            'hrm_min',
            'hrm_max',
            'hrm_avg',
            'still_number',
            'walking_number',
            'running_number',
            'on_bicycle_number',
            'in_vehicle_number',
            'on_foot_number',
            'tilting_number',
            'unknown_number',
            'app_entertainment_music',
            'app_utilities',
            'app_shopping',
            'app_games_comics',
            'app_health_wellness',
            'app_social_communication',
            'app_education',
            'app_travel',
            'app_art_design_photo',
            'app_news_magazine',
            'app_food_drink',
            'app_unknown_background',
            'app_others',
            'sleep_duration'
        ]

        header = True
        ema_rows = pd.read_csv('edd_data/edd_new_data/ema_responses.csv')
        print("Start time: {}".format(datetime.datetime.now()))
        for idx, row in enumerate(ema_rows.itertuples(index=False)):
            print("Iteration: {}".format(idx + 1))
            if int(row.time_expected) != 0 and int(row.day_num) <= 35:
                end_time = int(row.time_expected)
                start_time = end_time - 14400  # 14400sec = 4 hours before each EMA
                if start_time < 0:
                    continue

                print(1)
                unlock_data = get_unlock_result("edd_data/edd_new_data/unlock_duration.csv", start_time, end_time, row.username_id)
                print(2)
                unlock_at_home_data = get_unlock_duration_at_location("edd_data/edd_new_data/geofencing.csv", "edd_data/edd_new_data/unlock_duration.csv", start_time, end_time, LOCATION_HOME, row.username_id)
                print(3)
                unlock_at_univ_data = get_unlock_duration_at_location("edd_data/edd_new_data/geofencing.csv", "edd_data/edd_new_data/unlock_duration.csv", start_time, end_time, LOCATION_UNIVERSITY, row.username_id)
                print(4)
                unlock_at_library_data = get_unlock_duration_at_location("edd_data/edd_new_data/geofencing.csv", "edd_data/edd_new_data/unlock_duration.csv", start_time, end_time, LOCATION_LIBRARY, row.username_id)
                print(5)
                total_distance_data = get_total_distance("edd_data/edd_new_data/total_dist_covered.csv", start_time, end_time, row.username_id)
                print(6)
                std_displacement_data = get_std_of_displacement("edd_data/edd_new_data/std_of_displacement.csv", start_time, end_time, row.username_id)
                print(7)
                steps_data = get_steps("edd_data/edd_new_data/steps.csv", start_time, end_time, row.username_id)
                print(8)
                sig_motion_data = get_sig_motion("edd_data/edd_new_data/significant_motion.csv", start_time, end_time, row.username_id)
                print(9)
                rad_of_gyration_data = get_radius_of_gyration("edd_data/edd_new_data/radius_of_gyration.csv", start_time, end_time, row.username_id)
                print(10)
                calls_data = get_phonecall("edd_data/edd_new_data/phone_calls.csv", start_time, end_time, row.username_id)
                print(11)
                num_of_dif_places_data = get_num_of_dif_places("edd_data/edd_new_data/num_of_dif_places.csv", start_time, end_time, row.username_id)
                print(12)
                max_dist_two_locations_data = get_max_dist_two_locations("edd_data/edd_new_data/max_dist_two_locations.csv", start_time, end_time, row.username_id)
                print(13)
                max_dist_home_data = get_max_dist_home("edd_data/max_dist_from_home.csv", start_time, end_time, row.username_id)
                print(14)
                light_data = get_light("edd_data/edd_new_data/light.csv", start_time, end_time, row.username_id)
                print(15)
                hrm_data = get_hrm("edd_data/edd_new_data/hrm.csv", start_time, end_time, row.username_id)
                print(16)
                activity_number_data = get_num_of_dif_activities("edd_data/edd_new_data/activities.csv", start_time, end_time, row.username_id)
                print(17)
                app_usage_data = get_app_category_usage("edd_data/edd_new_data/app_usage.csv", start_time, end_time, row.username_id)

                day_hour_start = 18
                day_hour_end = 11
                date_start = datetime.datetime.fromtimestamp(end_time)
                date_start = date_start - datetime.timedelta(days=1)
                date_start = date_start.replace(hour=day_hour_start, minute=0, second=0)
                date_end = datetime.datetime.fromtimestamp(end_time)
                date_end = date_end.replace(hour=day_hour_end, minute=0, second=0)
                print(18)
                sleep_duration = get_sleep_duration("edd_data/edd_new_data/unlock_duration.csv", date_start.timestamp(), date_end.timestamp(), row.username_id)
                data = {
                    'user_id': row.username_id,
                    'day_num': row.day_num,
                    'ema': row.ema_order,
                    'phq1': row.interest,
                    'phq2': row.mood,
                    'phq3': row.sleep,
                    'phq4': row.fatigue,
                    'phq5': row.weight,
                    'phq6': row.worthlessness,
                    'phq7': row.concentrate,
                    'phq8': row.restlessness,
                    'phq9': row.suicide,
                    'unlock_duration': unlock_data['duration'],
                    'unlock_number': unlock_data['number'],
                    'unlock_duration_home': unlock_at_home_data['duration'],
                    'unlock_number_home': unlock_at_home_data['number'],
                    'unlock_duration_univ': unlock_at_univ_data['duration'],
                    'unlock_number_univ': unlock_at_univ_data['number'],
                    'unlock_duration_library': unlock_at_library_data['duration'],
                    'unlock_number_library': unlock_at_library_data['number'],
                    'total_distance': total_distance_data,
                    'std_displacement': std_displacement_data,
                    'steps': steps_data,
                    'significant_motion': sig_motion_data,
                    'radius_of_gyration': rad_of_gyration_data,
                    'in_call_duration': calls_data['in_duration'],
                    'in_call_number': calls_data['in_number'],
                    'out_call_duration': calls_data['out_duration'],
                    'out_call_number': calls_data['out_number'],
                    'num_of_dif_places': num_of_dif_places_data,
                    'max_dist_btw_two_locations': max_dist_two_locations_data,
                    'max_dist_home': max_dist_home_data,
                    'light_min': light_data['min'],
                    'light_max': light_data['max'],
                    'light_avg': light_data['avg'],
                    'hrm_min': hrm_data['min'],
                    'hrm_max': hrm_data['max'],
                    'hrm_avg': hrm_data['avg'],
                    'still_number': activity_number_data['still'],
                    'walking_number': activity_number_data['walking'],
                    'running_number': activity_number_data['running'],
                    'on_bicycle_number': activity_number_data['on_bicycle'],
                    'in_vehicle_number': activity_number_data['in_vehicle'],
                    'on_foot_number': activity_number_data['on_foot'],
                    'tilting_number': activity_number_data['tilting'],
                    'unknown_number': activity_number_data['unknown'],
                    'app_entertainment_music': app_usage_data['Entertainment & Music'],
                    'app_utilities': app_usage_data['Utilities'],
                    'app_shopping': app_usage_data['Shopping'],
                    'app_games_comics': app_usage_data['Games & Comics'],
                    'app_health_wellness': app_usage_data['Health & Wellness'],
                    'app_social_communication': app_usage_data['Social & Communication'],
                    'app_education': app_usage_data['Education'],
                    'app_travel': app_usage_data['Travel'],
                    'app_art_design_photo': app_usage_data['Art & Design & Photo'],
                    'app_news_magazine': app_usage_data['News & Magazine'],
                    'app_food_drink': app_usage_data['Food & Drink'],
                    'app_unknown_background': app_usage_data['Unknown & Background'],
                    'app_others': app_usage_data['Others'],
                    'sleep_duration': sleep_duration
                }

                df = pd.DataFrame(data, index=[0])
                df = df[columns]
                mode = 'w' if header else 'a'

                df.to_csv('features_edd.csv', encoding='utf-8', mode=mode, header=header, index=False)
                header = False

                # datasets.append(data)  # dataset of rows

        # (pd.DataFrame.from_dict(data=pckg_to_cat_map, orient='index')).to_csv('pckg_to_category.csv', header=False)
        print("End time: {}".format(datetime.datetime.now()))
    except Exception as e:
        print("Ex: ", e)


def main():
    extract_features()


if __name__ == "__main__":
    main()
