# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
from classes_alg_fun import *

import math
import plotly.graph_objects as go
import time
import sys

from collections import deque
from utils import *

st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✈️",
)
# --------------------------------------------------------------------------------------------------------------------------#
# Code
# Load data
df = pd.read_csv("Dataset.csv")

# Build node and successors dictionary to pass it to the graph

world_dict = {}
for source_airport in list(df["SourceAirport"].unique()):
    source_to_destinations_df = df[df["SourceAirport"] == source_airport][
        ["SourceAirport", "DestinationAirport", "Distance"]
    ]

    destinations_dict = {}
    for index, row in source_to_destinations_df.iterrows():
        destinations_dict.update({row["DestinationAirport"]: row["Distance"]})

    world_dict.update({source_airport: destinations_dict})

# Instantiate Undirected Graph
world_map = UndirectedGraph(world_dict)

# Locations in Latitude, Longitude and Altitude for calculating Heuristic function
locations_dict = {}
for nod in list(df["SourceAirport"].unique()):
    nod_df = df[df["SourceAirport"] == nod][
        ["SourceAirport_Latitude", "SourceAirport_Longitude", "SourceAirport_Altitude"]
    ]
    locations_dict.update({nod: tuple(nod_df.iloc[0])})

# Handle all Airpots locations -- sources and destinations
for nod in list(df["DestinationAirport"].unique()):
    nod_df = df[df["DestinationAirport"] == nod][
        [
            "DestinationAirport_Latitude",
            "DestinationAirport_Longitude",
            "DestinationAirport_Altitude",
        ]
    ]
    if nod not in locations_dict.keys():
        locations_dict.update({nod: tuple(nod_df.iloc[0])})

# Pass a new attribute locations
world_map.locations = locations_dict


# --------------------------------------------------------------------------------------------------------------------------#


# sidebar
st.sidebar.header("Find The Best Flight Path")

# Filters
# Source Location
st.sidebar.subheader("Provide your current location:")
source_country = st.sidebar.selectbox(
    "Country", sorted(list(df["SourceAirport_Country"].unique()))
)
source_city = st.sidebar.selectbox(
    "City",
    sorted(list(
        df[df["SourceAirport_Country"] == source_country]["SourceAirport_City"].unique()
    )),
)
source_airPort = st.sidebar.selectbox(
    "Airport",
    sorted(list(
        df[
            (df["SourceAirport_Country"] == source_country)
            & (df["SourceAirport_City"] == source_city)
        ]["SourceAirport"].unique()
    )),
)

# Distination Location
st.sidebar.subheader("Provide your destination location:")
distination_country = st.sidebar.selectbox(
    "Country", sorted(list(df["DestinationAirport_Country"].unique()))
)
distination_city = st.sidebar.selectbox(
    "City",
    sorted(list(
        df[df["DestinationAirport_Country"] == distination_country][
            "DestinationAirport_City"
        ].unique()
    )),
)
distination_airPort = st.sidebar.selectbox(
    "Airport",
    sorted(list(
        df[
            (df["DestinationAirport_Country"] == distination_country)
            & (df["DestinationAirport_City"] == distination_city)
        ]["DestinationAirport"].unique()
    )),
)

# Pass initial and goal states  -----------------------------------------------------Our Games and filter here
airport_intial = source_airPort
airport_goal = distination_airPort
# Instantiate Our Problem
world_problem = GraphProblem(airport_intial, airport_goal, world_map)

st.sidebar.markdown(
    "Special thanks and acknowledgement for all contributers in : [Artificial Intelligence - A Modern Approach](https://aima.cs.berkeley.edu/)"
)
st.sidebar.markdown(
    "For any modifications or suggestions, do not hesitate to : [contact us](https://www.linkedin.com/in/ahmedfawzy-ko/)"
)


# Body
st.header("Optimized Flight paths according to each search algorithm")

# Plot Pathes
start_time = time.time()
breadth_node = breadth_first_graph_search(world_problem)
elapsed_time = time.time() - start_time


path_states = []
for node in breadth_node.path():
    path_states.append(node.state)

lat_list = [world_map.locations[state][0] for state in path_states]
lon_list = [world_map.locations[state][1] for state in path_states]

path_fig = draw_path(lat_list, lon_list)

# Greedy best-first search is accomplished by specifying f(n) = h(n)
greedy_best_first_graph_search = best_first_graph_search

algorithms_elapsed_times = {"breadth-first": elapsed_time}
search_algorthms = {
    "depth first": depth_first_graph_search,
    "depth limited": depth_limited_search,
    "iterative deepening": iterative_deepening_search,
    "uniform cost": uniform_cost_search,
    "greedy": greedy_best_first_graph_search,
    "A*": astar_search,
}

for name, search in search_algorthms.items():
    start_time = time.time()
    if search not in [greedy_best_first_graph_search, astar_search]:
        search_node = search(world_problem)
    else:
        search_node = search(world_problem, world_problem.h)
    elapsed_time = time.time() - start_time
    algorithms_elapsed_times.update({name: elapsed_time})

    path_states = []
    for node in search_node.path():
        path_states.append(node.state)

    lat_list = [world_map.locations[state][0] for state in path_states]
    lon_list = [world_map.locations[state][1] for state in path_states]

    path_fig = add_trace_path(path_fig, lat_list, lon_list, name)

st.plotly_chart(path_fig, use_container_width=True)

# Compare Execution time for each algorithm
fig_algorithms_elapsed_times = px.bar(
    x=list(algorithms_elapsed_times.keys()),
    y=list(algorithms_elapsed_times.values()),
    color=list(algorithms_elapsed_times.keys()),
    title="Algorithms Execution Elapsed Time",
    labels={"x": "Algorithm", "y": "Time in seconds"},
)
st.plotly_chart(fig_algorithms_elapsed_times, use_container_width=True)
