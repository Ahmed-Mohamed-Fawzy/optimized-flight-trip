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

# Page Header(Body)
st.title("Optimized Flight trip ✈️")


# sidebar
st.sidebar.header("Find The Best Flight Path")
st.sidebar.image("flight.jpg")
# Filters
# Source Location
st.sidebar.subheader("Provide your current location:")
source_country = st.sidebar.selectbox(
    "Country", sorted(list(df["SourceAirport_Country"].unique()))
)
source_city = st.sidebar.selectbox(
    "City",
    sorted(
        list(
            df[df["SourceAirport_Country"] == source_country][
                "SourceAirport_City"
            ].unique()
        ),
        reverse=True,
    ),
)
source_airPort = st.sidebar.selectbox(
    "Airport",
    list(
        df[
            (df["SourceAirport_Country"] == source_country)
            & (df["SourceAirport_City"] == source_city)
        ]["SourceAirport"].unique()
    ),
)

# Distination Location
st.sidebar.subheader("Provide your destination location:")
distination_country = st.sidebar.selectbox(
    "Country", sorted(list(df["DestinationAirport_Country"].unique()))
)
distination_city = st.sidebar.selectbox(
    "City",
    sorted(
        list(
            df[df["DestinationAirport_Country"] == distination_country][
                "DestinationAirport_City"
            ].unique(),
        )
    ),
)
distination_airPort = st.sidebar.selectbox(
    "Airport",
    list(
        df[
            (df["DestinationAirport_Country"] == distination_country)
            & (df["DestinationAirport_City"] == distination_city)
        ]["DestinationAirport"].unique()
    ),
)


btn = st.sidebar.button("Search")

st.sidebar.markdown(
    "Special thanks and acknowledgement for all contributers in : [Artificial Intelligence - A Modern Approach](https://aima.cs.berkeley.edu/)"
)
st.sidebar.markdown(
    "For any modifications or suggestions, do not hesitate to : [contact us](https://www.linkedin.com/in/ahmedfawzy-ko/)"
)


if btn:

    # Build node and successors dicationary for pass it to the graph

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

    # Locations in Latitude, Longitude and Altitude for calculate Heuristac function
    locations_dict = {}
    for nod in list(df["SourceAirport"].unique()):
        nod_df = df[df["SourceAirport"] == nod][
            [
                "SourceAirport_Latitude",
                "SourceAirport_Longitude",
                "SourceAirport_Altitude",
            ]
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

    # Pass intial and goal states  -----------------------------------------------------Our Games and filter here
    airport_intial = source_airPort
    airport_goal = distination_airPort
    # Instantiate Our Problem
    world_problem = GraphProblem(airport_intial, airport_goal, world_map)

    # Body

    # Upgrade
    start_time = time.time()
    breadth_node = breadth_first_graph_search(world_problem)
    elapsed_time_breadth = time.time() - start_time

    start_time = time.time()
    depth_first_node = depth_first_graph_search(world_problem)
    elapsed_time_depth_first = time.time() - start_time

    start_time = time.time()
    depth_limited_node = depth_limited_search(world_problem)
    elapsed_time_depth_limited = time.time() - start_time

    start_time = time.time()
    iterative_deepening_node = iterative_deepening_search(world_problem)
    elapsed_time_iterative = time.time() - start_time

    start_time = time.time()
    uniform_cost_node = uniform_cost_search(world_problem)
    elapsed_time_uniform = time.time() - start_time

    # Greedy best-first search is accomplished by specifying f(n) = h(n)
    greedy_best_first_graph_search = best_first_graph_search
    start_time = time.time()
    greedy_node = greedy_best_first_graph_search(world_problem, world_problem.h)
    elapsed_time_greedy = time.time() - start_time

    start_time = time.time()
    astar_node = astar_search(world_problem, world_problem.h)
    elapsed_time_astar = time.time() - start_time

    pathes_list = [
        breadth_node,
        depth_first_node,
        depth_limited_node,
        iterative_deepening_node,
        uniform_cost_node,
        greedy_node,
        astar_node,
    ]

    shortest_path = breadth_node
    for path in pathes_list[1:]:
        if path.path_cost < shortest_path.path_cost:
            shortest_path = path

    total_fly_time = 0
    total_trip_price = 0
    current_airport = airport_intial
    for airport in list(uniform_cost_node.solution()):
        next_airport = airport
        total_fly_time = (
            df[
                (df["SourceAirport"] == current_airport)
                & (df["DestinationAirport"] == next_airport)
            ]["FlyTime"].iloc[0]
            + total_fly_time
        )
        total_trip_price = (
            df[
                (df["SourceAirport"] == current_airport)
                & (df["DestinationAirport"] == next_airport)
            ]["Price"].iloc[0]
            + total_trip_price
        )
        current_airport = next_airport

    # Matrix

    col1, col2, col3 = st.columns(3)  # For Horizontal matrix
    col1.metric(
        "Total Fly Distance", "".join([str(round(shortest_path.path_cost)), " km"])
    )
    col2.metric("Total Fly Time", decimal_hours_to_time(total_fly_time))
    col3.metric("Total Trip Price", "".join([str(round(total_trip_price, 2)), " $"]))

    # Visualiztion the best path
    st.subheader("The Best Flight Path")

    path_states = []
    for node in shortest_path.path():
        path_states.append(node.state)

    lat_list = [world_map.locations[state][0] for state in path_states]
    lon_list = [world_map.locations[state][1] for state in path_states]

    shortest_path_fig = draw_path(lat_list, lon_list)
    st.plotly_chart(shortest_path_fig, use_container_width=True)

    st.write(f"### To travel from **{airport_intial}** to **{airport_goal}**")
    current_airport = airport_intial
    for airport in list(uniform_cost_node.solution()):
        next_airport = airport
        airline = df[
            (df["SourceAirport"] == current_airport)
            & (df["DestinationAirport"] == next_airport)
        ]["Airline"].iloc[0]
        st.markdown(
            f"Fly with **{airline}** from **{current_airport}** to **{next_airport}**"
        )
        current_airport = next_airport

    # Plot Pathes
    st.header("Optimized Flight paths according to each search algorithm")

    c1, c2 = st.columns((1, 1))
    with c1:
        st.markdown("**Breadth First**")
        path_states = []
        for node in breadth_node.path():
            path_states.append(node.state)

        lat_list_breadth = [world_map.locations[state][0] for state in path_states]
        lon_list_breadth = [world_map.locations[state][1] for state in path_states]

        path_fig = draw_path(lat_list_breadth, lon_list_breadth)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(breadth_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    with c2:
        st.markdown("**Depth First**")
        path_states = []
        for node in depth_first_node.path():
            path_states.append(node.state)

        lat_list_depth_first = [world_map.locations[state][0] for state in path_states]
        lon_list_depth_first = [world_map.locations[state][1] for state in path_states]

        path_fig = draw_path(lat_list_depth_first, lon_list_depth_first)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(depth_first_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    st.write("")
    c3, c4 = st.columns((1, 1))
    with c3:
        st.markdown("**Depth Limited**")
        path_states = []
        for node in depth_limited_node.path():
            path_states.append(node.state)

        lat_list_depth_limited = [
            world_map.locations[state][0] for state in path_states
        ]
        lon_list_depth_limited = [
            world_map.locations[state][1] for state in path_states
        ]

        path_fig = draw_path(lat_list_depth_limited, lon_list_depth_limited)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(depth_limited_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    with c4:
        st.markdown("**Iterative Deepening**")
        path_states = []
        for node in iterative_deepening_node.path():
            path_states.append(node.state)

        lat_list_iteritive_deepening = [
            world_map.locations[state][0] for state in path_states
        ]
        lon_list_iteritive_deepening = [
            world_map.locations[state][1] for state in path_states
        ]

        path_fig = draw_path(lat_list_iteritive_deepening, lon_list_iteritive_deepening)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(iterative_deepening_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    st.write("")
    c5, c6 = st.columns((1, 1))
    with c5:
        st.markdown("**Uniform Cost**")
        path_states = []
        for node in uniform_cost_node.path():
            path_states.append(node.state)

        lat_list_uniform = [world_map.locations[state][0] for state in path_states]
        lon_list_uniform = [world_map.locations[state][1] for state in path_states]

        path_fig = draw_path(lat_list_uniform, lon_list_uniform)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(uniform_cost_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    with c6:
        st.markdown("**Greedy**")
        path_states = []
        for node in greedy_node.path():
            path_states.append(node.state)

        lat_list_greedy = [world_map.locations[state][0] for state in path_states]
        lon_list_greedy = [world_map.locations[state][1] for state in path_states]

        path_fig = draw_path(lat_list_greedy, lon_list_greedy)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(greedy_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    st.write("")
    c7, c8 = st.columns((1, 1))
    with c7:
        st.markdown("**A star**")
        path_states = []
        for node in astar_node.path():
            path_states.append(node.state)

        lat_list_astar = [world_map.locations[state][0] for state in path_states]
        lon_list_astar = [world_map.locations[state][1] for state in path_states]

        path_fig = draw_path(lat_list_astar, lon_list_astar)
        st.plotly_chart(path_fig, use_container_width=True)
        rounded = round(astar_node.path_cost)
        st.text(f"Total Distance : {rounded}km")

    with c8:
        search_algorithms_ditances = {
            "Breadth First": breadth_node.path_cost,
            "Depth First": depth_first_node.path_cost,
            "Depth Limited": depth_limited_node.path_cost,
            "Iterative Deepening": iterative_deepening_node.path_cost,
            "Uniform Cost": uniform_cost_node.path_cost,
            "Greedy": greedy_node.path_cost,
            "A*": astar_node.path_cost,
        }

        search_algorithms_ditances = sorted(
            search_algorithms_ditances.items(), key=lambda x: x[1]
        )
        search_algorithms_ditances = dict(search_algorithms_ditances)

        fig_distance = px.bar(
            x=search_algorithms_ditances.values(),
            y=search_algorithms_ditances.keys(),
            title="Algorithms' Distances",
            color=search_algorithms_ditances.keys(),
            labels={"x": "Distance (km)", "y": "Algorithms"},
        )
        st.plotly_chart(fig_distance, use_container_width=True)

    
    st.subheader("Comparison of seven different search algorithms")
    st.markdown("*Don't forget to use legend filter and interactive 3D map*")

    search_algorithms_lat_lon = {
        "Depth First": [lat_list_depth_first, lon_list_depth_first],
        "Depth Limited": [lat_list_depth_limited, lon_list_depth_limited],
        "Iterative Deepening": [lat_list_iteritive_deepening, lon_list_iteritive_deepening],
        "Uniform Cost": [lat_list_uniform, lon_list_uniform],
        "Greedy": [lat_list_greedy, lon_list_greedy],
        "A*": [lat_list_astar, lon_list_astar],
    }

    path_fig = draw_path(lat_list_breadth, lon_list_breadth,"Breadth First")
    for name,lat_lon_list in search_algorithms_lat_lon.items():
        path_fig=add_trace_path(path_fig,lat_lon_list[0],lat_lon_list[1],name)
    st.plotly_chart(path_fig, use_container_width=True)


    search_algorithms_elapsed_times = {
        "Breadth First": elapsed_time_breadth,
        "Depth First": elapsed_time_depth_first,
        "Depth Limited": elapsed_time_depth_limited,
        "Iterative Deepening": elapsed_time_iterative,
        "Uniform Cost": elapsed_time_uniform,
        "Greedy": elapsed_time_greedy,
        "A*": elapsed_time_astar,
    }

    # Compare Execution time for each algorithm
    fig_algorithms_elapsed_times = px.bar(
        x=list(search_algorithms_elapsed_times.keys()),
        y=list(search_algorithms_elapsed_times.values()),
        color=list(search_algorithms_elapsed_times.keys()),
        title="Algorithms Execution Elapsed Time",
        labels={"x": "Algorithm", "y": "Time in seconds"},
    )
    st.plotly_chart(fig_algorithms_elapsed_times, use_container_width=True)
