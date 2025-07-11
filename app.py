#2025.07.11 updates agent names o lite detaljer

import streamlit as st
from PIL import Image, ExifTags
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable #nytt 2025.07.11
import folium
import tempfile
import os
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

import requests
from bs4 import BeautifulSoup

import pandas as pd

import time

import json

import reverse_geocoder as rg




import openai #old code

from openai import OpenAI

# Set up OpenAI API
openai_api_key = ""  # Replace with your actual API key


#password secrets handling
import os
from dotenv import load_dotenv
load_dotenv(".env")



rapidApiKey = os.getenv("rapidApiKey")
yelp_api_key = os.getenv("yelp_api_key")
ocm_api_key = os.getenv("ocm_api_key")
api_key =  os.getenv("googleMaps_api_key")
X_RapidAPI_Key = os.getenv("X-RapidAPI-Key")



st.set_page_config(page_title="Simple Image Locator", page_icon=None, layout="centered")#, initial_sidebar_state="expanded")



# Function to calculate the distance between two GPS coordinates using the Haversine formula
def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # approximate radius of the Earth in km

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Function to calculate time taken for each mode of transportation
def calculate_travel_time(distance, mode="walking"):
    # Average speeds in km/h for different modes of transportation
    speed_mapping = {
        "walking": 3,
        "biking": 14,
        "car": 70,
    }
    speed = speed_mapping.get(mode.lower(), 5)  # Default to walking speed if mode is not recognized

    time_taken = distance / speed
    return time_taken

# Function to extract the date and time from the EXIF data
def extract_datetime(exif_data):
    if exif_data is not None:
        for tag, value in exif_data.items():
            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "DateTimeOriginal":
                return value
    return None


# Function to scrape Wikipedia information for a given location name
def scrape_wikipedia(location_name):
    wikipedia_url = f"https://en.wikipedia.org/wiki/{location_name.replace(' ', '_')}"
    response = requests.get(wikipedia_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find("div", {"id": "mw-content-text"})
        paragraphs = content.find_all("p")
        wiki_info = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
        return wiki_info
    else:
        return None




def get_address_by_location(latitude, longitude, language="en"):
    """This function returns an address as raw from a location
    will repeat until success"""
    # build coordinates string to pass to reverse() function
    coordinates = f"{latitude}, {longitude}"
    # sleep for a second to respect Usage Policy
    time.sleep(1)
    try:
        return app.reverse(coordinates, language=language).raw
        time.sleep(1)
    except:
        return get_address_by_location(latitude, longitude)





# Set up the geolocator
geolocator = Nominatim(user_agent="thomastestar-image-locator")






# Streamlit app header
st.title("Simple Image Locator")
st.info("Upload  photos and see where and when they were taken and get some Info about the location from Wikipedia and OpenAI. If you upload several photos, you can see the travel distances (walking, driving, biking)")

# Image upload
uploaded_files = st.sidebar.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png","heic"], accept_multiple_files=True)

# List to store image info (filename, latitude, longitude, datetime)
image_info_list = []

#en Dataframe för värdena ur geocoder reverse search
df_searchLokalInfo = pd.DataFrame()

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            tmp_file.write(uploaded_file.read())

        # Load the image from the temporary location
        img = Image.open(tmp_filename)

        # Extract GPS coordinates from the image (if available)
        exif_data = img._getexif()
        datetime_taken = extract_datetime(exif_data)

        if exif_data is None:
            st.warning("No exif-data found for the image " + uploaded_file.name)

        if exif_data is not None:
            for tag, value in exif_data.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "GPSInfo":

                    latitude = float(value[2][0]) + float(value[2][1]) / 60 + float(value[2][2]) / 3600
                    longitude = float(value[4][0]) + float(value[4][1]) / 60 + float(value[4][2]) / 3600

                    # Reverse geocoding to get the location
                    location = geolocator.reverse((latitude, longitude))
                    address = location.address
                    time.sleep(1)

                    df_searchLokalInfo_Zwischen = pd.DataFrame()

                    geolocator = Nominatim(user_agent="thomastestar-nearest-town-finder")

                    #ny kod 2025.07.11
                    location = None

                    for _ in range(3):
                        try:
                            location = geolocator.reverse((latitude, longitude), exactly_one=True, timeout=10)
                            break  # success, exit loop
                        except (GeocoderTimedOut, GeocoderUnavailable):
                            time.sleep(1)
                    
                    #location = geolocator.reverse((latitude, longitude), exactly_one=True)
                    #time.sleep(1)
                    
                    if location:
                        nearest_town = location.address.split(",")[2].strip()

                        #Variante 3 - Test av search variante - funzt besser! name enthält stadt!
                        coordinates = (latitude, longitude)
                        searchLokalInfo = rg.search(coordinates)
                        
    
                        #st.write("searchLokalInfo", searchLokalInfo)
                        


                        searchLokalInfo_name = [x.get('name') for x in searchLokalInfo]
                        Town = searchLokalInfo_name[0]
                        #st.write("Town:", Town)
                        #st.write("searchLokalInfo_name: ",searchLokalInfo_name[0])
                        
                        
                        
                        df_searchLokalInfo_Zwischen['Town'] = searchLokalInfo_name
                        


                    # Append image info to the list
                    image_info_list.append((img, address, latitude, longitude, datetime_taken,nearest_town, Town))

                    break  # Stop processing once GPSInfo is found

        # Clean up the temporary file
        os.remove(tmp_filename)

with st.expander("Enter OpenAI key to get AI infos about the locations", expanded=False):
    openai_api_key = st.text_input("Enter your OpenAI Key to fetch location info from OpenAI", value="")
    
    # Google Maps API key
    #api_key = ""
    #import os
    #api_key = os.getenv('api_key')

Sortierung = False

# Sort the image_info_list based on the datetime_taken value
anzeigenSortierung = st.checkbox("Show newest Photos first")
if anzeigenSortierung == True:
    Sortierung = True




#image_info_list = sorted(image_info_list, key=lambda x: x[4],reverse=Sortierung)

#ny sorteringsfunktion 2025.07.11
image_info_list = sorted(
    image_info_list,
    key=lambda x: x[4] if len(x) > 4 and x[4] is not None else '',
    reverse=Sortierung
)

image_info_list_df = pd.DataFrame(image_info_list)

if len(image_info_list) >0:
    show_image_info_list = st.checkbox("Show table with image infos")
    if show_image_info_list:

        # Rename the columns
        image_info_list_df_formatted = image_info_list_df.rename(columns={1: 'Address', 2: 'Latitude', 3: 'Longitude', 4: 'Date', 5: 'Location1', 6: 'Location2'})

        # Drop the column named 0
        image_info_list_df_formatted = image_info_list_df_formatted.drop(columns=[0])

        # Reorder the columns
        new_order = ['Date'] + [col for col in image_info_list_df_formatted.columns if col != 'Date']
        image_info_list_df_formatted = image_info_list_df_formatted[new_order]

        st.write("List of Images with with date and location:")
        st.dataframe(image_info_list_df_formatted)

        # Create the dataframe
        #image_info_list_df_formatted = pd.DataFrame(image_info_list_df_formatted)

        # Convert the 'Date' column to datetime format with specific format
        #image_info_list_df_formatted['Date'] = pd.to_datetime(image_info_list_df_formatted['Date'],format='%Y:%m:%d %H:%M:%S')


        # Calculate the total time difference between the last and first rows
        #total_time_difference = image_info_list_df_formatted['Date'].iloc[-1] - image_info_list_df_formatted['Date'].iloc[0]

        #st.write(image_info_list_df_formatted)
        #st.write("Total time difference:", total_time_difference)





# Display the images and their locations on the map
imageKey = 0

if image_info_list:




    # Display the images -  en efter en
    st.header("Images and Locations")
    st.subheader("")

    for img, address, _, _, datetime_taken,nearest_town,Town in image_info_list:
        imageKey = imageKey+1
        st.write("")
        st.caption(datetime_taken)
        st.image(img, caption=f" {nearest_town}, {address}")

        col1, col2, col3 = st.columns([1, 1,2])

        visaWiki = col1.toggle("Show Wikipedia Info", key=imageKey*1000)
        if visaWiki:

            wiki_info1 = scrape_wikipedia(Town)
            if wiki_info1 == None:
                st.info("Found no Info an Wikipedia")
            if wiki_info1 != None:
                st.subheader("")
                st.info("Info from Wikipedia:")
                with st.container(height=300):
                    st.subheader(f"{Town}")
                    st.markdown(wiki_info1)

                    #st.sidebar.write(wiki_info1)



                    #Här zusätzlich nearest town som möjlighet

                    time.sleep(1)
                    st.divider()

                    locationInfoList = address.split(",")
                    locationInfoList.insert(0, str(nearest_town))
                    #locationInfoListAuswahl = str(nearest_town) +","+locationInfoList

                    #Annan approach för att hitta orstnamn
                    adressEintragAuswahl = len(locationInfoList) - 5

                    st.write("Location:" , locationInfoList[adressEintragAuswahl])

                    wiki_info2 = scrape_wikipedia(locationInfoList[adressEintragAuswahl])
                    st.subheader(locationInfoList[adressEintragAuswahl])
                    st.write(wiki_info2)




        #st.sidebar.divider()


        #CHAT OPEN AI ###################################################
        visaOpenAI = col3.toggle("Show Chat OpenAI Info", key=imageKey * 2222)
        if visaOpenAI:
            client = OpenAI(
                # This is the default and can be omitted
                api_key=openai_api_key,
            )

            # User input
            pre_Input = "Please give me a summary of the demographical, historical, cultural, meteroligical, polticial and touristic information about this location and the closest town of following location, and if some famous people have lived here and there are some interesting tourist attractions, shopping, restaurants and bars: "
            user_input = address

            prompt_input = pre_Input + user_input

            # Use ChatGPT to generate a response
            if user_input:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Use GPT-3.5 or GPT-4 (e.g., "gpt-4")
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt_input}
                        ],
                        max_tokens=800,  # Adjust the length of the response as needed
                    )
                    if response and response.choices:
                        bot_response = response.choices[0].message.content
                        st.subheader("")
                        st.info("Info by ChatOpenAI:")
                        with st.container(height=300):
                            st.write(bot_response)
                    else:
                        st.warning("OpenAI: I'm sorry, I couldn't generate a response at the moment.")
                except Exception as e:
                    st.warning("OpenAI: An error occurred while processing your request.")
                    st.write("Error Message:", str(e))

            # END CHAT OPEN AI ###################################################




            st.divider()


            _="""
            selectedLocation = st.sidebar.selectbox("Choose other nearby location",locationInfoList)
            #selectedLocationButton = st.sidebar.checkbox("Change location")
            #if selectedLocationButton:
            if selectedLocation:
                st.sidebar.subheader(selectedLocation)  
                wiki_infoSelected = scrape_wikipedia(selectedLocation)
                #st.checkbox("Check location")
                st.sidebar.write(wiki_infoSelected)
            """


            st.sidebar.divider()

            st.sidebar.write("latitude", latitude)
            st.sidebar.write("longitude", longitude)

            st.sidebar.divider()

            #st.sidebar.write("location.raw address:",nearest_town)
            

        visaExif = col2.toggle("Show exif data", key=imageKey)
        if visaExif:
            st.subheader("")
            st.info("Raw Exif Data:")
            with st.container(height=300):
                st.write(exif_data)

        st.subheader("")
        st.divider()
        
    # Create a map centered around the locations
    #map = folium.Map(location=[image_info_list[0][2], image_info_list[0][3]], zoom_start=10)
    #st.write("image_info_list[0][2]: ",image_info_list[0][2])
    #st.write("image_info_list[0][3]: ", image_info_list[0][3])

    if len(image_info_list) > 1:
        st.subheader("Overview of locations")

        routingAuswahlOverview = ['drive', 'truck', 'bicycle', 'walk']

        routingModeSelectionOverview  = st.selectbox("Choose routing", routingAuswahlOverview , key="routingAuswahlOverview")

        #Trying to create an overview map that shows the routing from point to point #####
        point_coordinates = [(latitude, longitude) for img, address, latitude, longitude, datetime_taken, nearest_town, Town
                            in image_info_list]

        OverviewSumDistance = 0
        OverviewSumTime = 0

        # Initialize the map outside the loop
        OverViewMap = folium.Map()
        for p in range(len(point_coordinates) - 1):
            lon1 = str(point_coordinates[p][1])
            lat1 = str(point_coordinates[p][0])

            lon2 = str(point_coordinates[p + 1][1])
            lat2 = str(point_coordinates[p + 1][0])

            #st.write("point p #######: ", p)
            #st.write("lon1:", lon1)
            #st.write("lat1:", lat1)
            #st.write("lon2:", lon2)
            #st.write("lat2:", lat2)

            url = "https://route-and-directions.p.rapidapi.com/v1/routing"
            querystring = {"waypoints": f"{lat1},{lon1}|{lat2},{lon2}", "mode": routingModeSelectionOverview}
            headers = {
                "X-RapidAPI-Key": X_RapidAPI_Key,
                "X-RapidAPI-Host": "route-and-directions.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=querystring)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if 'features' in response_data:
                        mls = response_data['features'][0]['geometry']['coordinates']
                        points = [(i[1], i[0]) for i in mls[0]]

                        # Add markers for the start and ending points
                        folium.Marker(points[0]).add_to(OverViewMap)
                        folium.Marker(points[-1]).add_to(OverViewMap)

                        # Add the line between points
                        folium.PolyLine(points, weight=5, opacity=1).add_to(OverViewMap)

                        # Add markers for each location
                        for _, _, latitude, longitude, datetime_taken, _, Town in image_info_list:
                            folium.Marker([latitude, longitude], tooltip=datetime_taken + " in " + Town).add_to(OverViewMap)

                        # Calculate the bounds of the data
                        sw = image_info_list_df[[2, 3]].min().values.tolist()
                        ne = image_info_list_df[[2, 3]].max().values.tolist()


                        # Adjust the map bounds to include the current route
                        df = pd.DataFrame(mls[0]).rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]
                        #sw = df[['Lat', 'Lon']].min().values.tolist()
                        #ne = df[['Lat', 'Lon']].max().values.tolist()
                        OverViewMap.fit_bounds([sw, ne])

                        mlsTabelle = response.json()['features'][0]['properties']['legs'][0]['steps']
                        df_mlsTabelle = pd.json_normalize(
                            mlsTabelle)  # .rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]

                        #st.write(df_mlsTabelle)

                        sumDistance = df_mlsTabelle['distance'].sum() / 1000
                        sumTime = df_mlsTabelle['time'].sum() / 60

                        OverviewSumDistance = OverviewSumDistance + sumDistance
                        OverviewSumTime = OverviewSumTime + sumTime

                    else:
                        st.error(f"API response does not contain 'features' key for points {p} to {p + 1}.")
                except Exception as e:
                    st.error(f"An error occurred while processing the API response: {e}")
            else:
                st.error(f"API request failed with status code {response.status_code} for points {p} to {p + 1}.")

            # Display the map
        st_data = st_folium(OverViewMap, width=725, key="overview_map")

        Overviewcol1, Overviewcol2 = st.columns(2)

        Overviewcol1.metric(label="Distance (km)", value=OverviewSumDistance)

        if sumTime < 180:
            Overviewcol2.metric(label="Duration (min)", value=OverviewSumTime)
        else:
            OverviewSumTime = OverviewSumTime / 60
            Overviewcol2.metric(label="Duration (hours)", value=OverviewSumTime)



        #Straightline Overview ####################

        with st.expander("Show Straighline Distance"):
            # Create a folium map
            map = folium.Map()

            # Draw a line connecting the locations and add distance and travel time information as popups
            #line_coordinates = [(latitude, longitude) for _, _, latitude, longitude, _ in image_info_list] #funkar, men försöker smuggla med orter..
            line_coordinates = [(latitude, longitude) for img, address, latitude, longitude, datetime_taken,nearest_town,Town in image_info_list]

            #st.write("line_coordinates:",line_coordinates)


            total_distance = 0.0
            for i in range(len(line_coordinates) - 1):
                coord1 = line_coordinates[i]
                coord2 = line_coordinates[i + 1]


                #st.write("coord1:",coord1)
                #st.write("coord2:", coord2)
                #st.write("nearest_town:",nearest_town)

                #coord1_str = str(coord1)
                #coord2_str = str(coord2)
                #st.write("coord2_str:", coord2_str)

                distance = calculate_distance(coord1, coord2)
                total_distance += distance


                #walking_time = calculate_travel_time(distance, mode="walking")
                #biking_time = calculate_travel_time(distance, mode="biking")
                #car_time = calculate_travel_time(distance, mode="car")

                #popup_text = f"Distance: {distance:.2f} km\n"
                #popup_text += f"Estimated Walking Time: {walking_time:.1f} hours\n"
                #popup_text += f"Estimated Biking Time: {biking_time:.1f} hours\n"
                #popup_text += f"Estimated Car Driving Time: {car_time:.1f} hours"


                folium.PolyLine(locations=[coord1, coord2], color='blue').add_to(map)
                #folium.Marker(coord1, popup=datetime_taken).add_to(map)

            # Add markers for each location
            for _, _, latitude, longitude, datetime_taken, _ , Town in image_info_list:
                folium.Marker([latitude, longitude], tooltip=datetime_taken + " in " + Town).add_to(map)

            # Calculate the bounds of the data
            sw = image_info_list_df[[2, 3]].min().values.tolist()
            ne = image_info_list_df[[2, 3]].max().values.tolist()

            # Fit the map to the bounds
            map.fit_bounds([sw, ne])



            # Display the map with straightline ####################
            st.subheader("Straightline")
            st_data = st_folium(map, width=725)
            st.write("Total Straightline Distance:")
            st.write(f"{total_distance:.2f} km")

    if len(image_info_list_df)>1:

        # Display the total straight line distance



        #visa time estimates

        zeitSchäetzungExpander = st.expander("Show distances and travel times of Segments")
        with zeitSchäetzungExpander:

            total_distance = 0.0
            Route = 0
            for i in range(len(line_coordinates) - 1): #funkar
            #for _, _, latitude, longitude, datetime_taken, _, Town in image_info_list: #funkar inte

                coord1 = line_coordinates[i]
                coord2 = line_coordinates[i + 1]
                town1 = image_info_list_df._get_value(i, 6)
                town2 = image_info_list_df._get_value(i+1, 6)

                distance = calculate_distance(coord1, coord2)
                total_distance += distance
                walking_time = calculate_travel_time(distance, mode="walking")
                biking_time = calculate_travel_time(distance, mode="biking")
                car_time = calculate_travel_time(distance, mode="car")
                #st.write("coord1: ",coord1)
                #st.write("coord2: ", coord2)


                Route = Route + 1
                st.write("")
                st.subheader("Segment " +str(Route)+": " + "From " + town1 + " to " +town2)
                #st.write("From " + town1 + " to " +town2)

                routingAuswahl = ['drive', 'truck', 'bicycle', 'walk']

                routingModeSelection = st.selectbox("Choose routing", routingAuswahl, key="rapidApitransportmodecheck" + str(i))

                #st.divider()


                #Google Routing - does not work (anymore) #############
                # Transportation mode dropdown
                #transport_mode = st.selectbox("Select Google Maps Transportation Mode:",
                                              #["driving", "walking", "bicycling"], key="transportmodecheck" + str(i))
                # Define the API endpoint
                #base_url = "https://maps.googleapis.com/maps/api/directions/json?"


                lat1, lon1 = coord1
                lat2, lon2 = coord2

                # Create the origin and destination strings
                origin = f"{lat1},{lon1}"
                destination = f"{lat2},{lon2}"

                #st.write("origin: ",origin)
                #st.write("destination: ", destination)

                #wandle lat lon in str um
                lat1 = str(lat1)
                lon1 = str(lon1)
                lat2 = str(lat2)
                lon2 = str(lon2)

                #st.write("lat1: ", lat1)
                #st.write("lon1: ", lon1)
                #st.write("lat2: ", lat2)
                #st.write("lon2: ", lon2)

                #rapidapi routing #####################################
                url = "https://route-and-directions.p.rapidapi.com/v1/routing"

                querystring = {"waypoints": f"{lat1},{lon1}|{lat2},{lon2}", "mode": routingModeSelection}

                #querystring = {"waypoints": f"{str(origin)}|{str(destination)}",
                #               "mode": routingModeSelection}

                #querystring = {"waypoints": f"{str(lat1)},{str(lon1)}|{destination}",
                 #          "mode": routingModeSelection}

                headers = {
                        "X-RapidAPI-Key": X_RapidAPI_Key,
                        "X-RapidAPI-Host": "route-and-directions.p.rapidapi.com"
                    }

                response = requests.get(url, headers=headers, params=querystring)

                #st.write("response: ",response)

                mls = response.json()['features'][0]['geometry']['coordinates']

                # st.write(mls)
                points = [(i[1], i[0]) for i in mls[0]]

                #st.write("points:",points)

                m = folium.Map()
                # add marker for the start and ending points
                for point in [points[0], points[-1]]:
                    folium.Marker(point).add_to(m)
                # add the lines
                folium.PolyLine(points, weight=5, opacity=1).add_to(m)
                # create optimal zoom
                df = pd.DataFrame(mls[0]).rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]
                # st.write(df)
                sw = df[['Lat', 'Lon']].min().values.tolist()
                ne = df[['Lat', 'Lon']].max().values.tolist()
                m.fit_bounds([sw, ne])

                # Display the map
                st_data = st_folium(m, width=725)

                #m = create_map(response)

                # st.write("origin: ", origin)
                # st.write("destination: ", destination)

                # thomastestar

                thomasLatLonTabelle = response.json()['features'][0]['geometry']['coordinates']
                points = [(i[1], i[0]) for i in thomasLatLonTabelle[0]]

                st.divider()

                df_thomasLatLonTabelle = pd.DataFrame(thomasLatLonTabelle[0]).rename(columns={0: 'Lon', 1: 'Lat'})[
                    ['Lat', 'Lon']]

                last_lat = df_thomasLatLonTabelle['Lat'].iloc[-1]
                last_lon = df_thomasLatLonTabelle['Lon'].iloc[-1]

                # st.write(last_lon)
                # st.write(last_lat)

                mlsTabelle = response.json()['features'][0]['properties']['legs'][0]['steps']
                df_mlsTabelle = pd.json_normalize(mlsTabelle)  # .rename(columns={0: 'Lon', 1: 'Lat'})[['Lat', 'Lon']]

                drivingInstructionAsText = df_mlsTabelle['instruction.text'].to_string(index=False)

                if st.checkbox("Show Navigation-Table", key="Navigationstabelle" + str(i)):
                    st.write(df_mlsTabelle)

                if st.checkbox("Show Routedescription", key="Routedescription" + str(i)):
                    st.info(drivingInstructionAsText)



                col1, col2 = st.columns(2)

                sumDistance = df_mlsTabelle['distance'].sum() / 1000

                # st.write("sumDistance (km):",sumDistance )

                col1.metric(label="Distance (km)", value=sumDistance.round(0))

                sumTime = df_mlsTabelle['time'].sum() / 60

                if sumTime < 180:
                    col2.metric(label="Duration (min)", value=sumTime.round(0))
                else:
                    sumTime = sumTime / 60
                    col2.metric(label="Duration (hours)", value=sumTime.round(0))









                # Define the parameters for the google  API request, including transportation mode
                _="""
                params = {
                    "origin": origin,
                    "destination": destination,
                    "mode": transport_mode,
                    "key": api_key,
                }
                

                # Make Google the API request
                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()

                    # Extract the driving distance in kilometers and duration in minutes from the API response
                    if data["status"] == "OK":
                        distance_meters = data["routes"][0]["legs"][0]["distance"]["value"]
                        distance_km = distance_meters / 1000  # Convert meters to kilometers
                        duration_seconds = data["routes"][0]["legs"][0]["duration"]["value"]
                        duration_minutes = duration_seconds / 60  # Convert seconds to minutes

                        st.write(f"Google Maps Distance: {distance_km:.2f} km")
                        st.write(f"Google Maps Duration (min): {duration_minutes:.2f} minutes " + transport_mode)
                        if duration_minutes > 60:
                            st.write(f"Google Maps Duration (hours): {duration_minutes / 60:.1f} hours " + transport_mode)

                    else:
                        st.error("Error: Unable to calculate Google Maps distance and duration.")
                else:
                    st.error(
                        "Error: Unable to connect to the Google Maps API. Please check your API key and try again.")

                st.write("")
                st.write(f"Straightline - Distance: {distance:.2f} km\n")
                st.write(f"Straightline - Estimated Walking Time: {walking_time:.1f} hours\n")
                st.write(f"Straightline - Estimated Biking Time: {biking_time:.1f} hours\n")
                st.write(f"Straightline - Estimated Car Driving Time: {car_time:.1f} hours")
                st.write("")

                """

                st.divider()
