# -*- coding: utf-8 -*-
#The MIT License (MIT)

# Copyright (c) 2025 Rui Freitas, Ruben Queiros
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the right
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

__header__=r'''
  ___ ___   __   ___              _ _
 |_  )   \  \ \ / (_)____  _ __ _| (_)______ _ _
  / /| |) |  \ V /| (_-< || / _` | | |_ / -_) '_|
 /___|___/    \_/ |_/__/\_,_\__,_|_|_/__\___|_|
         ___________________________
        |             |             |
        |___          |          ___|
        |_  |         |         |  _|
       .| | |.       ,|.       .| | |.
       || | | )     ( | )     ( | | ||
       '|_| |'       `|'       `| |_|'
        |___|         |         |___|
        |             |             |
        |_____________|_____________|
'''


from haversine import haversine, Unit
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta

import pandas as pd
import sys
import os

import moviepy.editor as mp

root=''
background_img = mpimg.imread(root+'FootballField.png')
tformat="%H:%M:%S.%f"
df_lst=[]
game_data={}
total_frames=0
beg_time="00:00:00.00"
end_time="00:00:00.00"
#TODO Add Game 2nd part start time?
games_data = {
    "1": {
        'root_path': root + 'data/Dados - vs CD Mafra',
        'name': "Team1 vs Mafra",
        'start_time': '15:30:25.00',
        'field_coords': [
            [38.941212, -9.341908],
            [38.941244, -9.341159],
            [38.940294, -9.341865],
            [38.940325, -9.341109]]
    },
    "2": {
        'root_path': root + 'data/Dados - vs CD Trofense',
        'name': "Team1 vs Trofense",
        'start_time': '18:00:00.00',
        'field_coords': [
            [38.709282, -9.261246],
            [38.709291, -9.260478],
            [38.708335, -9.261230],
            [38.708344, -9.260461]]
    },
    "3": {
        'root_path': root + 'data/Dados - vs Famalicão FC',
        'name': "Team1 vs Famalicão",
        'start_time': '18:45:00.00',
        'field_coords': [
            [41.401015, -8.521879],
            [41.400870, -8.522665],
            [41.401929, -8.522175],
            [41.401789, -8.522965]]
    },
    "4": {
        'root_path': root + 'data/Dados - vs FC Boavista',
        'name': "Team1 vs Boavista",
        'start_time': '20:30:00.00',
        'field_coords': [
            [38.709282, -9.261246],
            [38.709291, -9.260478],
            [38.708335, -9.261230],
            [38.708344, -9.260461]]
    },
    "5": {
        'root_path': root + 'data/Dados - vs SC Covilhã',
        'name': "Team1 vs Covilhã",
        'start_time': '14:00:00.00',
        'field_coords': [
            [40.283733, -7.512215],
            [40.283514, -7.511525],
            [40.282901, -7.512666],
            [40.282680, -7.511972]]
    },
    "6": {
        'root_path': root + 'data/Dados - vs UD Vilafranquense',
        'name': "Team1 vs Vilafranquense",
        'start_time': '11:15:35.00',
        'field_coords': [
            [39.344006, -8.934984],
            [39.344174, -8.935733],
            [39.344910, -8.934638],
            [39.345078, -8.935387]]
    },
    "7": {
        'root_path': root + 'data/Dados - vs Vitória SC',
        'name': "Team1 vs Vitória",
        'start_time': '20:30:38.00',
        'field_coords': [
            [41.446300, -8.301432],
            [41.446398, -8.300632],
            [41.445368, -8.301232],
            [41.445466, -8.300429]]
    },
}

label_dict = {
    'Player1': 5,
    'Player2': 4,
    'Player3': 44,
    'Player4': 17,
    'Player5': 16,
    'Player6': 6,
    'Player7': 77,
    'Player8': 19,
    'Player9': 7,
    'Player10': 98,
    'Player11': 20,
    'Player12': 23,
    'Player13': 8,
    'Player14': 28,
    'Player15': 35,
    'Player16': 90,
    'Player17': 22,
    'Player18': 14,
    'Player19': 11,
    'Player20': 30,
    'Player21': 10,
    'Player22': 84,
    'Player23': 89,
    'Player23': 12,
}

def fetch_data(game_data, beg, end):

    root_path=game_data['root_path']

    for file in os.listdir(os.path.join(root_path)):
        if(file.endswith(".csv")):
            print("Fetching data from ", file, "...")
            df = pd.read_csv(os.path.join(root_path, file), skiprows=8, sep=";", index_col=False)

            player_name=file.partition(" vs")[0]
            df["Number"]=label_dict[player_name]

            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Timestamp'] = df['Timestamp'].dt.strftime(tformat)

            df['Time'] = pd.to_datetime(df['Timestamp'], format=tformat)

            #Filtered df
            df = df[(df['Time'].dt.time >= beg.time()) & (df['Time'].dt.time <= end.time())]

            df_lst.append(df)

def update_plot(frame):
    print("Processing frame: ", frame,"/",total_frames-1, " | ", (frame*100)/(total_frames-1),"%")

    plt.cla()

    clip_start = beg_time
    c_time= clip_start + timedelta(seconds=0.1*frame)

    maxdistance_x = haversine(game_data["field_coords"][0], game_data["field_coords"][2], unit=Unit.METERS)
    maxdistance_y= haversine(game_data["field_coords"][0], game_data["field_coords"][1], unit=Unit.METERS)

    plt.imshow(background_img, extent=[0, maxdistance_x, 0, maxdistance_y])

    for i in range(len(df_lst)):
        df = df_lst[i]

        closest_timestamp_row = None 
        closest_time_difference = 0.11
        found = False

        for index, row in df.iterrows():

            cts = datetime.strptime(str(row["Timestamp"]), tformat)
            time_diff = abs(cts-c_time)
            t_diff = float(str((time_diff.seconds*10e6+time_diff.microseconds)/10e6))
            if t_diff < closest_time_difference:
                closest_time_difference=t_diff
                closest_timestamp_row = row
                found=True
            if t_diff > closest_time_difference and found: 
                break

        if closest_timestamp_row is not None:

            lat=float(closest_timestamp_row["Latitude"].replace(',', '.'))
            lon=float(closest_timestamp_row["Longitude"].replace(',', '.'))
            p_coords=np.array([lat,lon])
            x, y = geo_to_xy(player_coords=p_coords)

            ax.scatter(x, y, marker='o',
                        color='red',
                        edgecolors='black',
                        s=50,
                        linestyle='None')
            ax.annotate(row["Number"], (x, y), textcoords="offset points", xytext=(0.5, -3), ha='center', fontsize=6, color='white', weight="bold")

    ctime_title=c_time.strftime(tformat)[:11]
    ax.set_title(game_data["name"] + ' | Timestamp: ' + ctime_title);
    ax.set_xlim([0, maxdistance_x])
    ax.set_ylim([0, maxdistance_y])
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)


def circle_intersection(x1, y1, r1, x2, y2, r2):
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if distance > r1 + r2 or distance < abs(r1 - r2):
        print("Error: No intersection or one circle contained within the other.")
        sys.exit()
    
    a = (r1**2 - r2**2 + distance**2) / (2 * distance)
    
    x_intersection = x1 + a * (x2 - x1) / distance
    y_intersection = y1 + a * (y2 - y1) / distance
    h = math.sqrt(r1**2 - a**2)
    intersection_1 = (x_intersection + h * (y2 - y1) / distance, y_intersection - h * (x2 - x1) / distance)
    intersection_2 = (x_intersection - h * (y2 - y1) / distance, y_intersection + h * (x2 - x1) / distance)
    if intersection_1[0] < 0 or intersection_1[1] < 0:
        intersection = intersection_2
    else:
        intersection = intersection_1

    return intersection[0], intersection[1] #x and y

def geo_to_xy(player_coords):

    corner_coords = np.array(game_data['field_coords'])
    maxdistance_x = haversine(corner_coords[0], corner_coords[2], unit=Unit.METERS)
    maxdistance_y= haversine(corner_coords[0], corner_coords[1], unit=Unit.METERS)

    distances = []
    for i in range(len(corner_coords)):
        distances.append(haversine(corner_coords[i], player_coords, unit=Unit.METERS))

    x1, y1, r1 = 0, 0, distances[0]
    x2, y2, r2 = 0, maxdistance_y, distances[1]

    # Calculate intersection points
    x, y = circle_intersection(x1, y1, r1, x2, y2, r2)
    return x, y

def gif_to_mp4(gif_path, mp4_file_name="video"):
    clip = mp.VideoFileClip(gif_path)
    file_name = mp4_file_name + ".mp4"
    clip.write_videofile(file_name)

def user_input():

    help=r'''
    ########################################################
    #                   2D Visualizer Menu                 #
    ########################################################
    | 1 : Game vs CD Mafra           | Start @ 15:30:25.00 |
    |--------------------------------|---------------------|
    | 2 : Game vs CD Trofense        | Start @ 18:00:00.00 |
    |--------------------------------|---------------------|
    | 3 : Game vs Famalicão FC       | Start @ 18:45:00.00 |
    |--------------------------------|---------------------|
    | 4 : Game vs FC Boavista        | Start @ 20:30:00.00 |
    |--------------------------------|---------------------|
    | 5 : Game vs SC Covilhã         | Start @ 14:00:00.00 |
    |--------------------------------|---------------------|
    | 6 : Game vs UD Vilafranquense  | Start @ 11:15:35.00 |
    |--------------------------------|---------------------|
    | 7 : Game vs Vitória SC         | Start @ 20:30:38.00 |
    |--------------------------------|---------------------|

                Example usage:
    What game you want to process? (valid options 1 to 7): 3 (enter)
    Clip starts at what time? 18:48:30.00 (enter)
    End of Clip at what time? 18:52:45.00 (enter)
    Clip file name (without extension)? example (enter)
    '''

    print(help)

    #Starting Values
    game_id = 0
    beg_time = datetime.strptime("00:00:00.00", tformat).time()
    end_time = datetime.strptime("00:00:00.00", tformat).time()

    while (game_id < 1) or (game_id > 7):
        game_id = int(input("What game you want to process? (valid options 1 to 7): "))
        print("")

    game_data = games_data[str(game_id)]
    game_start = datetime.strptime(game_data["start_time"], tformat).time()
    #game_end =   #TODO is it necessary?

    while (beg_time < game_start):
        print("Game ", game_id, " starts at ", games_data[str(game_id)]["start_time"])
        beg = input("Clip starts at what time? ")
        beg = datetime.strptime(beg, tformat)
        beg_time = beg.time()
        print("")

    while (end_time < beg_time):
        print("Beginning of Clip starts at ", beg_time.strftime(tformat)[:11])
        end = input("End of Clip at what time? ")
        end = datetime.strptime(end, tformat)
        end_time = end.time()
        print("")

    file_name = input("Clip file name? ")

    return game_data, beg, end, file_name

if __name__ == "__main__":

    print(__header__)
    
    generateClip=False
    
    game_data = games_data["1"]
    beg_time = datetime.strptime("15:37:03.85", tformat)
    end_time = datetime.strptime("15:37:05.78", tformat)
    file_name='TEST'

    fetch_data(game_data=game_data, beg=beg_time, end=end_time)

    for data in df_lst:
        if data.shape[0] > total_frames:
            total_frames=data.shape[0]

    if generateClip:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        update_plot(0)
        anim = FuncAnimation(fig, update_plot, frames=total_frames, interval = 100)
        anim.save(root+file_name+'.gif', writer='imagemagick')
        gif_to_mp4(root+file_name+'.gif', mp4_file_name=root+file_name)

    print("\n\nFinished Successfully!\n\n")
    sys.exit()