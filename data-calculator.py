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


from haversine import haversine, Unit
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statistics import mean
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

import pandas as pd
import sys
import os

root=''
tformat="%H:%M:%S.%f"
df_lst=[]
game_data={}
total_frames=0

beg_time="00:00:00.00"
end_time="00:00:00.00"

games_data = {
    "1": {
        'root_path': root + 'data/Dados - vs CD Mafra',
        'name': "Team1 vs Mafra",
        'start_time': '15:30:25.00',
        'field_coords': [
            [38.94121329569119, -9.341909306192129],
            [38.941244067115406, -9.341158287681568],
            [38.94029452130346, -9.341865274931349],
            [38.94032466747162, -9.341109340092814]],
        'field_length':102,
        'field_width':64,
        'attack_side_first_half': 'left',
        'attack_side_second_half': 'right'
    },
    "2": {
        'root_path': root + 'data/Dados - vs CD Trofense',
        'name': "Team1 vs Trofense",
        'start_time': '18:00:00.00',
        'field_coords': [
            [38.7092827087177, -9.261245134234308],
            [38.70929221561307, -9.260476715961246],
            [38.70833589588314, -9.261229915132168],
            [38.708343815105756, -9.260460143175647]],
        'field_length':110,
        'field_width':68,
        'attack_side_first_half': 'left',
        'attack_side_second_half': 'right'
    },
    "3": {
        'root_path': root + 'data/Dados - vs Famalicão FC',
        'name': "Team1 vs Famalicão",
        'start_time': '18:45:00.00',
        'field_coords': [
            [41.401023, -8.521895],
            [41.400880, -8.522683],
            [41.401942, -8.522190],
            [41.401799, -8.522980]],
        'field_length':105,
        'field_width':68,
        'attack_side_first_half': 'right',
        'attack_side_second_half': 'left'
    },
    "4": {
        'root_path': root + 'data/Dados - vs FC Boavista',
        'name': "Team1 vs Boavista",
        'start_time': '20:30:00.00',
        'field_coords': [
            [38.7092827087177, -9.261245134234308],
            [38.70929221561307, -9.260476715961246],
            [38.70833589588314, -9.261229915132168],
            [38.708343815105756, -9.260460143175647]],
        'field_length':110,
        'field_width':68,
        'attack_side_first_half': 'right',
        'attack_side_second_half': 'left'
    },
    "5": {
        'root_path': root + 'data/Dados - vs SC Covilhã',
        'name': "Team1 vs Covilhã",
        'start_time': '14:00:00.00',
        'field_coords': [
            [40.28373445813951, -7.512215282595129],
            [40.28351469182437, -7.511525130552143],
            [40.2829020031351, -7.512665887014353],
            [40.282679850448154, -7.511972963641041]],
        'field_length':105,
        'field_width':64,
        'attack_side_first_half': 'right',
        'attack_side_second_half': 'left'
    },
    "6": {
        'root_path': root + 'data/Dados - vs UD Vilafranquense',
        'name': "Team1 vs Vilafranquense",
        'start_time': '11:15:35.00',
        'field_coords': [
            [39.344008, -8.934985],
            [39.344175, -8.935733],
            [39.344911, -8.934638],
            [39.345075, -8.935391]],
        'field_length':104,
        'field_width':68,
        'attack_side_first_half': 'right',
        'attack_side_second_half': 'left'
    },
    "7": {
        'root_path': root + 'data/Dados - vs Vitória SC',
        'name': "Team1 vs Vitória",
        'start_time': '20:30:38.00',
        'field_coords': [
            [41.44630030469047, -8.301433104083392],
            [41.446398799383516, -8.300631367411661],
            [41.44536823773908, -8.301233205615898],
            [41.44546572469468, -8.300428731640585]],
        'field_length':105,
        'field_width':68,
        'attack_side_first_half': 'left',
        'attack_side_second_half': 'right'
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
    'Player24': 12,
}

def get_player_name(number):
    for key, value in label_dict.items():
        if number == value:
            return key
    return "Player with that number doesn't exist"

def circle_intersection(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    distance12 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    a12 = (r1**2 - r2**2 + distance12**2) / (2 * distance12)

    # Calculate the coordinates of the intersection points
    x_intersection12 = x1 + a12 * (x2 - x1) / distance12
    y_intersection12 = y1 + a12 * (y2 - y1) / distance12
    
    # Calculate the distance from the intersection point to each intersection point
    h12 = math.sqrt(r1**2 - a12**2)

    # Calculate the coordinates of the two intersection points
    intersection_1 = (x_intersection12 + h12 * (y2 - y1) / distance12, y_intersection12 - h12 * (x2 - x1) / distance12)
    intersection_2 = (x_intersection12 - h12 * (y2 - y1) / distance12, y_intersection12 + h12 * (x2 - x1) / distance12)
    
    if intersection_1[0] < -5.2 or intersection_1[1] < -5.2:
        intersection = intersection_2
        intersectionopt = intersection_1
    else:
        intersection = intersection_1
        intersectionopt = intersection_2

    return intersection[0], intersection[1], intersectionopt[0], intersectionopt[1] #x and y

def geo_to_xy(player_coords, corner_coords):

    #player_coords = np.array([40.283004, -7.512131])
    maxdistance_x = haversine(corner_coords[0], corner_coords[2], unit=Unit.METERS)
    maxdistance_y= haversine(corner_coords[0], corner_coords[1], unit=Unit.METERS)

    distances = []
    for i in range(len(corner_coords)):
        distances.append(haversine(corner_coords[i], player_coords, unit=Unit.METERS))



    x1, y1, r1 = 0, 0, distances[0]

    x2, y2, r2 = 0, maxdistance_y, distances[1]
    x3, y3, r3 = maxdistance_x, 0, distances[2]

    # Calculate intersection points
    x, y, xopt, yopt = circle_intersection(x1, y1, r1, x2, y2, r2, x3, y3, r3)
    return x, y, xopt, yopt

def fetch_transitions(source_file, result_file, debug=False):
    
    final_df = pd.DataFrame()
    transitions_df = pd.read_csv(source_file, sep=";", index_col=False)

    #Process Each Play individually
    for i, lance in transitions_df.iterrows():
        print("PROCESSING NEW PLAY... ", i+1,"/",len(transitions_df))
        
        final_row=pd.DataFrame([lance])
        data_lst=[]

        game_path=games_data[str(lance["Jogo no"])]['root_path']
        game_files=os.listdir(game_path)
        player_lst=[int(numero) for numero in lance["Jogadores em campo"].split(", ")]
        beg=datetime.strptime(lance['Inicio (GPS)'], tformat)
        end=datetime.strptime(lance['Fim (GPS)'], tformat)
        
        #Fetch and filter the relevant data rows of the players playing the game
        for player in player_lst:
            player_data_df=pd.DataFrame()
            for file in game_files:
                if get_player_name(player) in file:
                    print(file)
                    player_df = pd.read_csv(os.path.join(game_path, file), skiprows=8, sep=";", index_col=False)
                    player_df["Number"]=player

                    player_df['Timestamp'] = pd.to_datetime(player_df['Timestamp'], dayfirst=True)
                    player_df['Timestamp'] = player_df['Timestamp'].dt.strftime(tformat)
                    player_df['Time'] = pd.to_datetime(player_df['Timestamp'], format=tformat)
                    player_df = player_df[(player_df['Time'].dt.time >= beg.time()) & (player_df['Time'].dt.time <= end.time())]

                    player_df['Velocity'] = player_df['Velocity'].str.replace(',', '.')
                    player_df['Latitude'] = player_df['Latitude'].str.replace(',', '.')
                    player_df['Longitude'] = player_df['Longitude'].str.replace(',', '.')

                    for _, row in player_df.iterrows():
                        p_coords=np.array([float(row['Latitude']), float(row['Longitude'])])
                        x,y, xopt, yopt = geo_to_xy(p_coords,
                                        games_data[str(lance["Jogo no"])]['field_coords'])
                        
                        if debug:
                            if y > 70 or x > 110 or y < -3 or x < -3:
                                print("\n")
                                print("Timestamp: ", row['Timestamp'])
                                print("latitude, longitude: ", row['Latitude'], row['Longitude'])
                                print("X is :", x)                                
                                print("Y is :", y)
                                print("Other X is: ", xopt)
                                print("Other Y is: ", yopt)
                                
                                print(get_player_name(player))
                                print("\n")
                        
                        
                        player_data = pd.DataFrame(columns=[str(player)+"_velocity",
                                                            str(player)+"_x_position",
                                                            str(player)+"_y_position"],
                                                    data=[[row['Velocity'], x, y]])
                        
                        player_data_df = pd.concat([player_data_df, player_data], ignore_index=True)

                    data_lst.append(player_data_df)
                    if debug:
                        player_data_df.to_csv("data"+ str(player) +".csv")

        min_rows=9999
        finaldata_lst=[]
        for playerdata in data_lst:
            if playerdata.shape[0] < min_rows:
                min_rows = playerdata.shape[0]
        
        for playerdata in data_lst:
            difrows = playerdata.shape[0] - min_rows
            if difrows == 0:
                finaldata_lst.append(playerdata)
            else:
                finaldata_lst.append(playerdata.head(-difrows))
        
        old_comprimento=0
        old_largura=0
        old_cl_ratio=0
        old_surface_area=0
        old_speed=0
        old_dor=0
        old_do=0

        for amostra in range(min_rows): # min rows = amostras do lance atual
            v_lst=[]
            player_pos_lst=[]

            for p in finaldata_lst:
                v_lst.append(float(p.iloc[amostra][0]))
                player_pos_lst.append((float(p.iloc[amostra][1]),float(p.iloc[amostra][2])))

            player_pos=np.array(player_pos_lst)
            comprimento=abs(np.max(player_pos[:,0])-np.min(player_pos[:,0]))

            delta_comprimento=comprimento-old_comprimento
            if old_comprimento == 0: delta_comprimento=0
            old_comprimento=comprimento

            largura=abs(np.max(player_pos[:,1])-np.min(player_pos[:,1]))

            delta_largura=largura-old_largura
            if old_largura == 0: delta_largura=0
            old_largura=largura

            cl_ratio=comprimento/largura

            delta_cl_ratio=cl_ratio-old_cl_ratio
            if old_cl_ratio==0: delta_cl_ratio=0
            old_cl_ratio=cl_ratio
            
            convex_hull= ConvexHull(player_pos)
            surface_area=convex_hull.volume

            delta_surface_area=surface_area-old_surface_area
            if old_surface_area==0: delta_surface_area=0
            old_surface_area=surface_area

            colective_speed_of_displacement = mean(v_lst)

            delta_speed=colective_speed_of_displacement-old_speed
            if old_speed==0: delta_speed=0
            old_speed=colective_speed_of_displacement


            defensive_occupation_ratio= surface_area/((comprimento*games_data[str(lance["Jogo no"])]['field_width'])-surface_area)

            delta_dor=defensive_occupation_ratio-old_dor
            if old_dor==0: delta_dor=0
            old_dor=defensive_occupation_ratio

            if lance["Parte do jogo"] == 1:
                attack_side=games_data[str(lance["Jogo no"])]['attack_side_first_half']
            elif lance["Parte do jogo"] == 2:
                attack_side=games_data[str(lance["Jogo no"])]['attack_side_second_half']
            
            med_position=np.min(player_pos[:,0])+comprimento/2

            defensive_oscillation=0
            for pos in player_pos:
                if pos[0] < med_position and attack_side == 'left':
                    defensive_oscillation+= abs(pos[0] - med_position)
                elif pos[0] > med_position and attack_side == 'left':
                    defensive_oscillation+= -1*abs(pos[0] - med_position)
                elif pos[0] < med_position and attack_side == 'right':
                    defensive_oscillation+= -1*abs(pos[0] - med_position)
                elif pos[0] > med_position and attack_side == 'right':
                    defensive_oscillation+= abs(pos[0] - med_position)
            
            defensive_oscillation=defensive_oscillation/len(player_pos)
            
            delta_do=defensive_oscillation-old_do
            if old_do==0: delta_do=0
            old_do=defensive_oscillation


            final_row["Amostra"]=amostra+1

            final_row["Comprimento"]= comprimento
            final_row["delta Comprimento"]= delta_comprimento
            
            final_row["Largura"]= largura
            final_row["delta Largura"]= delta_largura
            
            final_row["Ratio Comprimento/Largura"]= cl_ratio
            final_row["delta Ratio Comprimento/Largura"]= delta_cl_ratio
            
            final_row["Surface Area"]= surface_area
            final_row["delta Surface Area"]= delta_surface_area

            final_row["Collective Speed of Displacement"]= colective_speed_of_displacement
            final_row["delta Collective Speed of Displacement"]= delta_speed
            
            final_row["Defensive Occupation Ratio"]= defensive_occupation_ratio
            final_row["delta Defensive Occupation Ratio"]= delta_dor

            final_row["Defensive Oscillation"]= defensive_oscillation
            final_row["delta Defensive Oscillation"]= delta_do

            final_df = pd.concat([final_df, final_row], ignore_index=True)

    final_df.to_csv(result_file, index=False)
    return

if __name__ == "__main__":

    results_df = pd.read_csv('resultados-final.csv', sep=",", index_col=False)
    print("Comprimento Summary: ", results_df["Comprimento"].describe())
    print("delta Comprimento Summary: ", results_df["delta Comprimento"].describe())
    
    print("Largura Summary: ", results_df["Largura"].describe())
    print("delta Largura Summary: ", results_df["delta Largura"].describe())

    print("Ratio C/L Summary: ", results_df["Ratio Comprimento/Largura"].describe())
    print("delta Ratio C/L Summary: ", results_df["delta Ratio Comprimento/Largura"].describe())

    print("Surface Area Summary: ", results_df["Surface Area"].describe())
    print("delta Surface Area Summary: ", results_df["Surface Area"].describe())
    
    print("Collective Speed of Displacement Summary: ", results_df["Collective Speed of Displacement"].describe())
    print("delta Collective Speed of Displacement Summary: ", results_df["delta Collective Speed of Displacement"].describe())

    print("Defensive Occupation Ratio", results_df["Defensive Occupation Ratio"].describe())
    print("delta Defensive Occupation Ratio", results_df["delta Defensive Occupation Ratio"].describe())

    print("Defensive Oscillation", results_df["Defensive Oscillation"].describe())
    print("delta Defensive Oscillation", results_df["delta Defensive Oscillation"].describe())

    print("\n\nFinished Successfully!\n\n")
    sys.exit()