# This script is for Feature Extraction of STDD study season 3

import os

import sys
import pandas as pd
import datetime
import numpy as np
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import HTTPError
import statistics
import math

NUMBER_OF_EMA = 6
SOUND_SAMPLING_RATE = 11025
QUIET_STATE_THRESHOLD = 0  # TODO: fix quiet threshold
NOISY_STATE_THRESHOLD = 0  # TODO: fix noisy threshold
LOCATION_HOME = "HOME"


ACTIVITY_RECOGNITION = "ACTIVITY_RECOGNITION"
ANDROID_PRESSURE = "ANDROID_PRESSURE"
ANDROID_SIGNIFICANT_MOTION = "ANDROID_SIGNIFICANT_MOTION"
ANDROID_STEP_DETECTOR = "ANDROID_STEP_DETECTOR"
APPLICATION_USAGE = "APPLICATION_USAGE"
CALLS = "CALLS"
GEOFENCE = "GEOFENCE"
KEYSTROKE_LOG = "KEYSTROKE_LOG"
LOCATIONS_MANUAL = "LOCATIONS_MANUAL"
LOCATION_GPS = "LOCATION_GPS"
NETWORK_USAGE = "NETWORK_USAGE"
NOTIFICATIONS = "NOTIFICATIONS"
REWARD_POINTS = "REWARD_POINTS"
SCREEN_STATE = "SCREEN_STATE"
SMS = "SMS"
SOUND_DATA = "SOUND_DATA"
SURVEY_EMA = "SURVEY_EMA"
TYPING = "TYPING"
UNLOCK_STATE = "UNLOCK_STATE"

pckg_to_cat_map = {}
cat_list = pd.read_csv('Cat_group.csv')


def in_range(number, start, end):
    if start <= number <= end:
        return True
    else:
        return False


def get_distance(lat1, lng1, lat2, lng2):
    earth_radius = 6371000  # in meters
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLng / 2) * math.sin(dLng / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = float(earth_radius * c)
    return dist


def get_filename_for_data_src(filenames, data_src, username):
    for filename in filenames:
        if username in filename and data_src in filename:
            return filename


def is_file_not_empty(file):
    if os.fstat(file.fileno()).st_size > 0:
        return True
    else:
        return False


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


# Done unlock duration and number
def get_unlock_result(filename, start_time, end_time):
    result = {
        "duration": 0,
        "number": 0
    }
    unlock_sum = 0
    lock_sum = 0
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            lines = f.readlines()
            if lines.__len__() > 1:
                for index, line in enumerate(lines):
                    values = re.sub('"', '', lines[index].split(",")[1])
                    time, type = values[:-1].split(" ")
                    if in_range(int(time), start_time, end_time):
                        if type == "UNLOCK":
                            unlock_sum += int(time)
                            result['number'] += 1
                        elif type == "LOCK":
                            lock_sum += int(time)

                if lines[0] == 'LOCK':
                    unlock_sum += start_time

                if lines[lines.__len__() - 1] == 'UNLOCK':
                    lock_sum += end_time

    if lock_sum > unlock_sum:
        result['duration'] = int((lock_sum - unlock_sum) / 1000)

    if result['number'] == 0:
        result['duration'] = "-"
        result['number'] = "-"

    return result


# Done total moving time
def get_activities_dur_result(filename, start_time, end_time):
    result = {
        "still": 0,
        "walking": 0,
        "running": 0,
        "on_bicycle": 0,
        "in_vehicle": 0,
        "total_moving_time": 0
    }

    still_sum = {}
    walking_sum = {}
    running_sum = {}
    on_bicycle_sum = {}
    in_vehicle_sum = {}

    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            lines = f.readlines()
            if lines.__len__() > 1:
                for line in lines:
                    values = re.sub('"', '', line.split(",")[1])
                    time, activity_name, type = values[:-1].split(" ")
                    if in_range(int(time), start_time, end_time):
                        if activity_name == 'STILL':
                            if type == "ENTER":
                                still_sum['enter'] += int(time)
                            elif type == "EXIT":
                                still_sum['exit'] += int(time)
                        elif activity_name == 'WALKING':
                            if type == "ENTER":
                                walking_sum['enter'] += int(time)
                            elif type == "EXIT":
                                walking_sum['exit'] += int(time)
                        elif activity_name == 'RUNNING':
                            if type == "ENTER":
                                running_sum['enter'] += int(time)
                            elif type == "EXIT":
                                running_sum['exit'] += int(time)
                        elif activity_name == 'ON_BICYCLE':
                            if type == "ENTER":
                                on_bicycle_sum['enter'] += int(time)
                            elif type == "EXIT":
                                on_bicycle_sum['exit'] += int(time)
                        elif activity_name == 'IN_VEHICLE':
                            if type == "ENTER":
                                in_vehicle_sum['enter'] += int(time)
                            elif type == "EXIT":
                                in_vehicle_sum['exit'] += int(time)

                if lines[0] == 'EXIT':
                    values = re.sub('"', '', lines[0].split(",")[1])
                    time, activity_name, type = values[:-1].split(" ")
                    if activity_name == 'STILL':
                        still_sum['enter'] += start_time
                    elif activity_name == 'WALKING':
                        walking_sum['enter'] += start_time
                    elif activity_name == 'RUNNING':
                        running_sum['enter'] += start_time
                    elif activity_name == 'ON_BICYCLE':
                        on_bicycle_sum['enter'] += start_time
                    elif activity_name == 'IN_VEHICLE':
                        in_vehicle_sum['enter'] += start_time

                if lines[lines.__len__() - 1] == 'ENTER':
                    values = re.sub('"', '', lines[0].split(",")[1])
                    time, activity_name, type = values[:-1].split(" ")
                    if activity_name == 'STILL':
                        still_sum['exit'] += end_time
                    elif activity_name == 'WALKING':
                        walking_sum['exit'] += end_time
                    elif activity_name == 'RUNNING':
                        running_sum['exit'] += end_time
                    elif activity_name == 'ON_BICYCLE':
                        on_bicycle_sum['exit'] += end_time
                    elif activity_name == 'IN_VEHICLE':
                        in_vehicle_sum['exit'] += end_time

    if still_sum['exit'] > still_sum['enter']:
        result['still'] = int((still_sum['exit'] - still_sum['enter']) / 1000)
    else:
        result['still'] = '-'

    if walking_sum['exit'] > walking_sum['enter']:
        result['walking'] = int((walking_sum['exit'] - walking_sum['enter']) / 1000)
    else:
        result['walking'] = '-'

    if running_sum['exit'] > running_sum['enter']:
        result['running'] = int((running_sum['exit'] - running_sum['enter']) / 1000)
    else:
        result['running'] = '-'

    if on_bicycle_sum['exit'] > on_bicycle_sum['enter']:
        result['on_bicycle'] = int((on_bicycle_sum['exit'] - on_bicycle_sum['enter']) / 1000)
    else:
        result['on_bicycle'] = '-'

    if in_vehicle_sum['exit'] > in_vehicle_sum['enter']:
        result['in_vehicle'] = int((in_vehicle_sum['exit'] - in_vehicle_sum['enter']) / 1000)
    else:
        result['in_vehicle'] = '-'

    result['total_moving_time'] = result['walking'] + result['running'] + result['on_bicycle']
    if result['total_moving_time'] == 0:
        result['total_moving_time'] = "-"

    return result


#  Done total notifs clicked, clicked/arrived, avg decision time
def get_notifs_data(filename, start_time, end_time):
    result = {
        "total_number_clicked": 0,
        "clicked_arrived_ratio": 0,
        "avg_decision_time": 0
    }

    arrived_cnt = 0
    decision_times = []
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                start, end, pckg_name, notif_type = values[:-1].split(" ")
                if in_range(int(start), start_time, end_time):
                    if notif_type == "CLICKED":
                        result['total_number_clicked'] += 1
                    if notif_type == "ARRIVED":
                        arrived_cnt += 1
                if in_range(int(start), start_time, end_time) and in_range(int(end), start_time, end_time):
                    if notif_type == "DECISION_TIME":
                        decision_times.append((end - start) / 1000)  # duration in seconds

    if arrived_cnt > 0:
        result['clicked_arrived_ratio'] = result['total_number_clicked'] / arrived_cnt
    else:
        result['clicked_arrived_ratio'] = '-'
        result['total_number_clicked'] = '-'

    result['avg_decision_time'] = statistics.mean(decision_times) if decision_times.__len__() > 0 else '-'

    return result


# Done total missed calls
def get_phonecall_result(filename, start_time, end_time):
    result = {
        "phone_calls_total_dur": 0,
        "phone_calls_total_number": 0,
        "phone_calls_ratio_in_out": 0,
        "missed_calls_total_number": 0
    }

    total_in = 0
    total_out = 0
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                start, end, call_type = values[:-1].split(" ")
                if in_range(int(start), start_time, end_time) and in_range(int(end), start_time, end_time):
                    result["phone_calls_total_dur"] += int((end - start) / 1000)
                    if call_type == "IN":
                        total_in += 1
                    elif call_type == "OUT":
                        total_out += 1

                if in_range(int(start), start_time, end_time):
                    if call_type == "MISSED":
                        result['missed_calls_total_number'] += 1

    if result["phone_calls_total_dur"] > 0:
        result["phone_calls_total_number"] = total_in + total_out
        result["phone_calls_ratio_in_out"] = total_in / total_out if total_out > 0 else "-"
    else:
        result["phone_calls_total_dur"] = "-"
        result["phone_calls_total_number"] = "-"
        result["phone_calls_ratio_in_out"] = "-"

    if result['missed_calls_total_number'] <= 0:
        result['missed_calls_total_number'] = '-'

    return result


# Done
def get_keystroke_data(filename_typing, filename_keystroke, start_time, end_time):
    result = {
        "avg_interkey_delay": -1,
        "backspace_ratio": -1,
        "autocorrect_rate": -1
    }

    key_presses_data = []
    backspace_presses = 0
    autocorrected_words = 0
    total_words = 0
    with open(filename_typing, 'r') as f_typing:
        if os.fstat(f_typing.fileno()).st_size > 0:
            for line in f_typing:
                values = re.sub('"', '', line.split(",")[1])
                start, end, pckg_name = values[:-1].split(" ")
                if in_range(int(start), start_time, end_time) and in_range(int(end), start_time, end_time):
                    with open(filename_keystroke, "r") as f_keystroke:
                        if os.fstat(f_keystroke.fileno()).st_size > 0:
                            for line_unlock in f_keystroke:
                                values = re.sub('"', '', line_unlock.split(",")[1])
                                time, autocorrect_yes_no, pckg_name, keystroke_type = values[:-1].split(" ")
                                if in_range(int(time), int(start), int(end)):
                                    if keystroke_type == "OTHER":
                                        key_presses_data.append(time)
                                    if keystroke_type == "BACKSPACE":
                                        backspace_presses += 1
                                    if keystroke_type == "AUTOCORRECT":
                                        if autocorrect_yes_no == "YES":
                                            autocorrected_words += 1
                                            total_words += 1
                                        elif autocorrect_yes_no == "NO":
                                            total_words += 1

    key_press_delays = []
    for i in range(1, key_presses_data.__len__()):
        key_press_delays.append(key_presses_data[i] - key_presses_data[i - 1])

    result['avg_interkey_delay'] = statistics.mean(key_press_delays)
    if key_presses_data.__len__() > 0:
        result['backspace_ratio'] = backspace_presses / key_presses_data.__len__()
    if total_words > 0:
        result['autocorrect_rate'] = autocorrected_words / total_words

    if result['avg_interkey_delay'] == -1:
        result['avg_interkey_delay'] = '-'
    if result['backspace_ratio'] == -1:
        result['backspace_ratio'] = '-'
    if result['autocorrect_rate'] == -1:
        result['autocorrect_rate'] = '-'
    return result


# Done sms characters and unique addresses
def get_sms_data(filename, start_time, end_time):
    result = {
        "unique_incoming_contacts_number": -1,
        "characters_in _sms": -1
    }

    addresses = []
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                time, address, characters = values[:-1].split(" ")
                if in_range(int(time), start_time, end_time):
                    result["characters_in"] += int(characters)
                    addresses.append(address)

    result['unique_incoming_contacts_number'] = (list(set(addresses))).__len__()

    if result["characters_in"] == -1:
        result["characters_in"] = '-'
    if result["unique_incoming_contacts_number"] == -1:
        result["unique_incoming_contacts_number"] = '-'

    return result


# Done incoming and outgoing traffic amount
def get_internet_traffic_data(filename, start_time, end_time):
    result = {
        "tx_bytes": -1,
        "rx_bytes": -1
    }

    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                start, end, bytes, usage_type = values[:-1].split(" ")
                if in_range(int(start), start_time, end_time):
                    if usage_type == 'TX':
                        result['tx_bytes'] += bytes
                    elif usage_type == 'RX':
                        result['rx_bytes'] += bytes

    result['tx_bytes'] = '-' if result['tx_bytes'] == -1 else result['tx_bytes']
    result['rx_bytes'] = '-' if result['rx_bytes'] == -1 else result['rx_bytes']

    return result


# Done sound pitch, energy, mfcc, jitter
def get_sound_data(filename, start_time, end_time):
    result = {
        "quiet_state_rate": -1,
        "noisy_state_rate": -1,
        "pitch_avg": -1,
        "energy_avg": -1,
        "jitter": -1,
        "mfcc1_avg": -1,
        "mfcc2_avg": -1,
        "mfcc3_avg": -1,
        "mfcc4_avg": -1,
        "mfcc5_avg": -1,
        "mfcc6_avg": -1,
        "mfcc7_avg": -1,
        "mfcc8_avg": -1,
        "mfcc9_avg": -1,
        "mfcc10_avg": -1,
        "mfcc11_avg": -1,
        "mfcc12_avg": -1,
        "mfcc13_avg": -1
    }

    pitch_data = []
    pitch_periods = []
    energy_data = []
    noisy_state_counter = 0
    quiet_state_counter = 0
    mfcc_data = []
    with open(filename, "r") as f:
        if is_file_not_empty(f):
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                time, value, sound_type = values[:-1].split(" ")
                if in_range(int(time), start_time, end_time):
                    if sound_type == 'PITCH':
                        pitch_data.append(float(value))
                        pitch_periods.append(math.floor(SOUND_SAMPLING_RATE / float(value)))
                    elif sound_type == 'ENERGY':
                        energy_data.append(float(value))
                        if float(value) > NOISY_STATE_THRESHOLD:
                            noisy_state_counter += 1
                        elif float(value) < QUIET_STATE_THRESHOLD:
                            quiet_state_counter += 1
                    elif sound_type == 'MFCC':
                        mfcc_data.append([float(i) for i in value[1:-1].split(",")])

    mfcc_data = np.array(mfcc_data)
    mfcc_avg = np.mean(mfcc_data, axis=0)

    jitter_numerator = 0
    for index, item in enumerate(pitch_periods):
        if index > 0:
            jitter_numerator += abs(pitch_periods[index] - pitch_periods[index - 1])

    if pitch_data.__len__() > 0:
        result['pitch_avg'] = statistics.mean(pitch_data)
        result['jitter'] = jitter_numerator / (pitch_periods.__len__() - 1)

    else:
        result['pitch_avg'] = '-'
        result['jitter'] = '-'

    if energy_data.__len__() > 0:
        result['energy_avg'] = statistics.mean(energy_data)
        result['quiet_state_rate'] = float(quiet_state_counter / energy_data.__len__())
        result['noisy_state_rate'] = float(noisy_state_counter / energy_data.__len__())
    else:
        result['energy_avg'] = '-'
        result['quiet_state_rate'] = '-'
        result['noisy_state_rate'] = '-'

    if mfcc_data.__len__() > 0:
        result['mfcc1_avg'] = mfcc_avg[0]
        result['mfcc2_avg'] = mfcc_avg[1]
        result['mfcc3_avg'] = mfcc_avg[2]
        result['mfcc4_avg'] = mfcc_avg[3]
        result['mfcc5_avg'] = mfcc_avg[4]
        result['mfcc6_avg'] = mfcc_avg[5]
        result['mfcc7_avg'] = mfcc_avg[6]
        result['mfcc8_avg'] = mfcc_avg[7]
        result['mfcc9_avg'] = mfcc_avg[8]
        result['mfcc10_avg'] = mfcc_avg[9]
        result['mfcc11_avg'] = mfcc_avg[10]
        result['mfcc12_avg'] = mfcc_avg[11]
        result['mfcc13_avg'] = mfcc_avg[12]
    else:
        result['mfcc1_avg'] = '-'
        result['mfcc2_avg'] = '-'
        result['mfcc3_avg'] = '-'
        result['mfcc4_avg'] = '-'
        result['mfcc5_avg'] = '-'
        result['mfcc6_avg'] = '-'
        result['mfcc7_avg'] = '-'
        result['mfcc8_avg'] = '-'
        result['mfcc9_avg'] = '-'
        result['mfcc10_avg'] = '-'
        result['mfcc11_avg'] = '-'
        result['mfcc12_avg'] = '-'
        result['mfcc13_avg'] = '-'

    return result


# TODO: finish all location data here
def get_gps_location_data(filename, start_time, end_time):
    result = {
        "total_distance": -1,
        "max_dist_two_location": -1,
        "gyration": -1,
        "max_dist_from_home": -1,
        "std_of_displacement": -1,
        "location_variance": -1
    }
    locations = []
    centroid = {
        "lat": 0,
        "lng": 0
    }
    total_time_in_locations = 0
    sum_gyration = 0
    sum_std = 0
    lat_data = []
    lng_data = []
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            lines = f.readlines()
            if lines > 0:
                for index, line in enumerate(lines):
                    values_current = re.sub('"', '', lines[index].split(",")[1])
                    values_next = re.sub('"', '', lines[index + 1].split(",")[1])
                    time1, lat1, lng1, speed1, accuracy1, altitude1 = values_current[:-1].split(" ")
                    time2, lat2, lng2, speed2, accuracy2, altitude2 = values_next[:-1].split(" ")
                    if in_range(int(time1), start_time, end_time) and in_range(int(time2), start_time, end_time):
                        # distance between current location and next one
                        lat_data.append(float(lat1))
                        lng_data.append(float(lng1))
                        distance = get_distance(float(lat1), float(lng1), float(lat2), float(lng2))
                        result['total_distance'] += distance  # total distance calculated
                        if distance > result['max_dist_two_location']:
                            result['max_dist_two_location'] = distance  # max dist between two locations calculated

                        # distance between home location and current location
                        distance_from_home = get_distance(home_lat, home_lng, float(lat1), float(lng1))
                        if distance > result['max_dist_from_home']:
                            result['max_dist_from_home'] = distance_from_home  # max dist from home calculated

                        centroid["lat"] += float(lat1)
                        centroid["lng"] += float(lng1)
                        total_time_in_locations += int((int(time2) - int(time1)) / 1000)
                        locations.append({"time": int(time1), "lat": float(lat1), "lng": float(lng1)})

                centroid["lat"] = centroid["lat"] / (locations.__len__() - 1)
                centroid["lng"] = centroid["lng"] / (locations.__len__() - 1)

                avg_displacement = result['total_distance'] / locations.__len__() - 1

                for i in range(0, locations.__len__() - 1):
                    distance_to_centroid = get_distance(locations[i]['lat'], locations[i]['lng'], centroid['lat'],
                                                        centroid['lng'])
                    sum_gyration += int((locations[i + 1]['time'] - locations[i]['time']) / 1000) * math.pow(
                        distance_to_centroid, 2)

                    distance_std = get_distance(locations[i]['lat'], locations[i]['lng'], locations[i + 1]['lat'],
                                                locations[i + 1]['lng'], )
                    sum_std += math.pow(distance_std - avg_displacement, 2)

                result['gyration'] = float(math.sqrt(sum_gyration / total_time_in_locations))
                result['std_of_displacement'] = float(math.sqrt(sum_std / locations.__len__() - 1))
                result['location_variance'] = statistics.variance(lat_data) + statistics.variance(lng_data)
        else:
            result = {
                "total_distance": '-',
                "max_dist_two_location": '-',
                "gyration": '-',
                "max_dist_from_home": '-',
                "std_of_displacement": '-',
                "location_variance": '-'
            }
    return result


# TODO: decide about removing the geofence and using the clustering instead
def get_time_at_geofence(filename, start_time, end_time, location_name):
    result = 0
    with open(filename, "r") as f:
        if os.fstat(f.fileno()).st_size > 0:
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                enter_time, exit_time, location_id = values[:-1].split(" ")
                if in_range(int(enter_time), start_time, end_time) and location_id == location_name:
                    result += (int(exit_time) - int(enter_time)) / 1000

    return result if result > 0 else "-"


# TODO change this function later
def get_unlock_duration_at_location(filename_geofence, filename_unlock, start_time, end_time, location_name):
    result = 0
    with open(filename_geofence, "r") as f_geofence:
        if os.fstat(f_geofence.fileno()).st_size > 0:
            for line_geofence in f_geofence:
                values = re.sub('"', '', line_geofence.split(",")[1])
                enter_time, exit_time, location_id = values[:-1].split(" ")
                if in_range(int(enter_time), start_time, end_time) and location_id == location_name:
                    with open(filename_unlock, "r") as f_unlock:
                        if os.fstat(f_unlock.fileno()).st_size > 0:
                            for line_unlock in f_unlock:
                                values = re.sub('"', '', line_unlock.split(",")[1])
                                start, end, duration = values[:-1].split(" ")
                                if in_range(int(start), int(enter_time), int(exit_time)) and in_range(int(end),
                                                                                                      int(enter_time),
                                                                                                      int(exit_time)):
                                    result += int(duration)

    return result if result > 0 else "-"


# TODO: any features during phone calls
def get_features_during_phone_call(filename_calls, filename_other_data, start_time, end_time):
    result = {
        "feature_1": -1,
        "feature_2": -1
    }
    with open(filename_calls, "r") as f_calls:
        if is_file_not_empty(f_calls):
            lines_calls = f_calls.readlines()
            for index in range(0, len(lines_calls)):
                values = re.sub('"', '', lines_calls[index].split(",")[1])
                call_start_time, call_end_time, call_type = values[:-1].split(" ")
                if in_range(int(call_start_time), start_time, end_time) and in_range(int(call_end_time), start_time,
                                                                                     end_time):

                    # TODO: open any other file here and check if the timestamp of that data is in range of phone calls
                    with open(filename_other_data, "r") as f_other:
                        if is_file_not_empty(f_other):
                            for line in f_other:
                                values = re.sub('"', '', line.split(",")[1])
                                # TODO: any data example
                                time, data1, data2, data3 = values[:-1].split(" ")
                                if in_range(int(time), int(call_start_time), int(call_end_time)):
                            # TODO: do the rest logic here
        else:
            result['feature_1'] = "-"
            result['feature_2'] = "-"

    return result


def get_app_usage_data(filename, start_time, end_time):
    result = {
        "num_of_different_apps_used": 0,
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
    used_apps_list = []
    with open(filename, "r") as f:
        if is_file_not_empty(f):
            for line in f:
                values = re.sub('"', '', line.split(",")[1])
                start, end, pckg_name = values[:-1].split(" ")
                duration = (int(end) - int(start)) / 1000
                if in_range(int(start), start_time, end_time) and in_range(int(end), start_time, end_time):

                    if not pckg_name in used_apps_list:
                        used_apps_list.append(pckg_name)

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

    result["num_of_different_apps_used"] = used_apps_list.__len__() if used_apps_list.__len__() > 0 else "-"
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


def get_sleep_duration(filename, start_time, end_time):
    result = 0
    durations = []
    with open(filename, "r") as f:
        if is_file_not_empty(f):
            lines = f.readlines()
            for index in range(0, len(lines)):
                try:
                    current_line_values = re.sub('"', '', lines[index].split(",")[1])
                    current_time, current_type = current_line_values[:-1].split(" ")

                    nex_line_values = re.sub('"', '', lines[index + 1].split(",")[1])
                    next_time, next_type = nex_line_values[:-1].split(" ")

                    if in_range(int(current_time), start_time, end_time):
                        if current_type == "OFF" and next_type == "ON":
                            durations.append(int((int(next_time) - int(current_time)) / 1000))
                except IndexError as err:
                    a = 1
                    # print("Skip this part: ", err)
    if durations:
        result = max(durations)
    return result if result > 0 else "-"


def extract_features(_url, _usernames):
    try:
        columns = [
            'User id',
            'phq1',
            'phq2',
            'phq3',
            'phq4',
            'phq5',
            'phq6',
            'phq7',
            'phq8',
            'phq9',
            'Responded time',
            'Day of week',
            'EMA order',
            'Unlock duration',
            'Phonecall duration',
            'Phonecall number',
            'Phonecall ratio',
            'Duration STILL',
            'Duration WALKING',
            'Duration RUNNING',
            'Duration BICYCLE',
            'Duration VEHICLE',
            'Duration ON_FOOT',
            'Duration TILTING',
            'Duration UNKNOWN',
            'Freq. STILL',
            'Freq. WALKING',
            'Freq. RUNNING',
            'Freq. BICYCLE',
            'Freq. VEHICLE',
            'Freq. ON_FOOT',
            'Freq. TILTING',
            'Freq. UNKNOWN',
            'Audio min.',
            'Audio max.',
            'Audio mean',
            'Total distance',
            'Num. of places',
            'Max. distance',
            'Gyration',
            'Max. dist.(HOME)',
            'STD. of displacement',
            'Duration(HOME)',
            'Unlock dur.(HOME)',
            'Diff. apps used',
            'Entertainment & Music',
            'Utilities',
            'Shopping',
            'Games & Comics',
            'Others',
            'Health & Wellness',
            'Social & Communication',
            'Education',
            'Travel',
            'Art & Design & Photo',
            'News & Magazine',
            'Food & Drink',
            'Unknown & Background',
            'Sleep dur.',
            'Phonecall audio min.',
            'Phonecall audio max.',
            'Phonecall audio mean'
        ]

        header = True
        for username in _usernames:
            print("Processing features for ", username, ".....")
            ema_responses = []  # Response.objects.filter(username=participant).order_by('day_num', 'ema_order')

            # take all the files of this user and store in one files array
            files_of_user = []
            for root, dirs, files in os.walk(_url):
                # all_files = files
                for filename in files:
                    if username in filename:
                        files_of_user.append("{0}{1}".format(_url, filename))  # append each file of this user

            with open(get_filename_for_data_src(files_of_user, SURVEY_EMA, username), 'r') as f:
                for line in f:
                    timestamp, value = line.split(",")
                    value = re.sub('"', '', value)
                    ema_responses.append(value)

            for index, ema_res in enumerate(ema_responses):
                print(index + 1, "/", ema_responses.__len__())
                time, ema_order, ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8, ans9 = ema_res.split(" ")
                end_time = int(time)
                start_time = end_time - 14400000  # 4hours before each EMA
                if start_time < 0:
                    continue

                unlock_data = get_unlock_result(get_filename_for_data_src(files_of_user, UNLOCK_STATE, username),
                                                start_time, end_time)
                activities_total_dur = get_activities_dur_result(
                    get_filename_for_data_src(files_of_user, ACTIVITY_RECOGNITION, username), start_time, end_time)
                notifications_data = get_notifs_data(get_filename_for_data_src(files_of_user, NOTIFICATIONS, username),
                                                     start_time, end_time)
                phonecall_data = get_phonecall_result(get_filename_for_data_src(files_of_user, CALLS, username),
                                                      start_time, end_time)
                keystroke_data = get_keystroke_data(get_filename_for_data_src(files_of_user, TYPING, username),
                                                    get_filename_for_data_src(files_of_user, KEYSTROKE_LOG, username),
                                                    start_time, end_time)
                sms_data = get_sms_data(get_filename_for_data_src(files_of_user, SMS, username),
                                        start_time, end_time)
                network_usage_data = get_internet_traffic_data(
                    get_filename_for_data_src(files_of_user, NETWORK_USAGE, username),
                    start_time, end_time)
                sound_data = get_sound_data(get_filename_for_data_src(files_of_user, SOUND_DATA, username),
                                            start_time, end_time)
                locations_data = get_gps_location_data(get_filename_for_data_src(files_of_user, LOCATION_GPS, username),
                                                       start_time, end_time)
                time_at_home = get_time_at_geofence(
                    get_filename_for_data_src(files_of_user, GEOFENCE, username),
                    start_time, end_time, LOCATION_HOME)

                # TODO: customize the following function and then use
                features_during_pc_data = get_features_during_phone_call(
                    get_filename_for_data_src(files_of_user, CALLS, username),
                    "ANY_OTHER_FILE",
                    start_time, end_time)

                app_usage = get_app_usage_data(get_filename_for_data_src(files_of_user, APPLICATION_USAGE, username),
                                               start_time, end_time)

                sleep_hour_start = 22  # start time boundary of sleep calculation
                sleep_hour_end = 10  # end time boundary of sleep calculation
                date_start = datetime.datetime.fromtimestamp(end_time / 1000)
                date_start = date_start - datetime.timedelta(days=1)
                date_start = date_start.replace(hour=sleep_hour_start, minute=0, second=0)

                date_end = datetime.datetime.fromtimestamp(int(time) / 1000)
                date_end = date_end.replace(hour=sleep_hour_end, minute=0, second=0)

                sleep_duration = get_sleep_duration(get_filename_for_data_src(files_of_user, SCREEN_STATE, username),
                                                    date_start.timestamp() * 1000, date_end.timestamp() * 1000)

                day_of_week = datetime.datetime.fromtimestamp(end_time / 1000).weekday()
                data = {'User id': username,
                        'phq1': ans1,
                        'phq2': ans2,
                        'phq3': ans3,
                        'phq4': ans4,
                        'phq5': ans5,
                        'phq6': ans6,
                        'phq7': ans7,
                        'phq8': ans8,
                        'phq9': ans9,
                        'Responded time': time,
                        'Day of week': day_of_week,
                        'EMA order': ema_order,
                        'Unlock duration': unlock_data['duration'],
                        'Unlock number': unlock_data['number'],
                        'Duration STILL': activities_total_dur["still"],
                        'Duration WALKING': activities_total_dur["walking"],
                        'Duration RUNNING': activities_total_dur["running"],
                        'Duration BICYCLE': activities_total_dur["on_bicycle"],
                        'Duration VEHICLE': activities_total_dur["in_vehicle"],
                        'Total moving duration': activities_total_dur["total_moving_time"],
                        "Notif. total num clicked": notifications_data['total_number_clicked'],
                        "Notif. click arrive ratio": notifications_data['clicked_arrived_ratio'],
                        "Notif. decision time": notifications_data['avg_decision_time'],
                        'Phonecall duration': phonecall_data["phone_calls_total_dur"],
                        'Phonecall number': phonecall_data["phone_calls_total_number"],
                        'Phonecall ratio': phonecall_data["phone_calls_ratio_in_out"],
                        'Phonecall missed number': phonecall_data["missed_calls_total_number"],
                        'Keystroke avg interkey delay': keystroke_data["avg_interkey_delay"],
                        'Keystroke backspace ratio': keystroke_data["backspace_ratio"],
                        'Keystroke autocorrect rate': keystroke_data["autocorrect_rate"],
                        'SMS unique contacts (IN)': sms_data["unique_incoming_contacts_number"],
                        'SMS characters (IN)': sms_data["characters_in"],
                        'Network TX bytes': network_usage_data["tx_bytes"],
                        'Network RX bytes': network_usage_data["rx_bytes"],
                        'Sound quiet state rate': sound_data["quiet_state_rate"],
                        'Sound noisy state rate': sound_data["noisy_state_rate"],
                        'Sound pitch avg': sound_data["pitch_avg"],
                        'Sound energy avg': sound_data["energy_avg"],
                        'Sound jitter': sound_data["jitter"],
                        'Sound avg_mfcc1': sound_data["mfcc1_avg"],
                        'Sound avg_mfcc2': sound_data["mfcc2_avg"],
                        'Sound avg_mfcc3': sound_data["mfcc3_avg"],
                        'Sound avg_mfcc4': sound_data["mfcc4_avg"],
                        'Sound avg_mfcc5': sound_data["mfcc5_avg"],
                        'Sound avg_mfcc6': sound_data["mfcc6_avg"],
                        'Sound avg_mfcc7': sound_data["mfcc7_avg"],
                        'Sound avg_mfcc8': sound_data["mfcc8_avg"],
                        'Sound avg_mfcc9': sound_data["mfcc9_avg"],
                        'Sound avg_mfcc10': sound_data["mfcc10_avg"],
                        'Sound avg_mfcc11': sound_data["mfcc11_avg"],
                        'Sound avg_mfcc12': sound_data["mfcc12_avg"],
                        'Sound avg_mfcc13': sound_data["mfcc13_avg"],
                        'Locations total distance': locations_data["total_distance"],
                        'Locations max dist btw two locations': locations_data["max_dist_two_location"],
                        'Locations gyration': locations_data["gyration"],
                        'Locations max dist from home': locations_data["max_dist_from_home"],
                        'Locations std of displacement': locations_data["std_of_displacement"],
                        'Locations variance': locations_data["location_variance"],
                        'Duration at HOME': time_at_home,
                        'Diff. apps used': app_usage['num_of_different_apps_used'],
                        'Entertainment & Music': app_usage['Entertainment & Music'],
                        'Utilities': app_usage['Utilities'],
                        'Shopping': app_usage['Shopping'],
                        'Games & Comics': app_usage['Games & Comics'],
                        'Others': app_usage['Others'],
                        'Health & Wellness': app_usage['Health & Wellness'],
                        'Social & Communication': app_usage['Social & Communication'],
                        'Education': app_usage['Education'],
                        'Travel': app_usage['Travel'],
                        'Art & Design & Photo': app_usage['Art & Design & Photo'],
                        'News & Magazine': app_usage['News & Magazine'],
                        'Food & Drink': app_usage['Food & Drink'],
                        'Unknown & Background': app_usage['Unknown & Background'],
                        'Sleep dur.': sleep_duration,
                        'Phonecall feature 1': features_during_pc_data['feature_1'],
                        'Phonecall feature 2': features_during_pc_data['feature_2']}

                df = pd.DataFrame(data, index=[0])
                df = df[columns]
                mode = 'w' if header else 'a'
                df.to_csv('features_stdd.csv', encoding='utf-8', mode=mode, header=header, index=False)
                header = False
    except Exception as e:
        print("Ex: ", e)


def main(argv):
    files_url = argv[0]
    usernames = argv[1].split(',')
    extract_features(files_url, usernames)


if __name__ == "__main__":
    main(sys.argv[1:])
