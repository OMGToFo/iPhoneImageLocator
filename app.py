import streamlit as st
from PIL import Image, ExifTags
from geopy.geocoders import Nominatim
import folium
import tempfile
import os
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

import requests
from bs4 import BeautifulSoup

import pandas as pd

import time


import reverse_geocoder as rg

import openai
# Set up OpenAI API
#openai.api_key = ""  # Replace with your actual API key





st.set_page_config(page_title="Image Locator", page_icon=None, layout="centered", initial_sidebar_state="expanded")



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
    except:
        return get_address_by_location(latitude, longitude)





# Set up the geolocator
geolocator = Nominatim(user_agent="image-locator")






# Streamlit app header
st.title("iPhone Image Locator App")
st.info("Upload iPhone photos from your Computer and see where and when they were taken and get some Info about the location from Wikipedia and OpenAI. If you upload several photos, you can see the travel distances (walking, driving, biking) calculated by Google Maps")

# Image upload
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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
        if exif_data is not None:
            for tag, value in exif_data.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "GPSInfo":

                    latitude = float(value[2][0]) + float(value[2][1]) / 60 + float(value[2][2]) / 3600
                    longitude = float(value[4][0]) + float(value[4][1]) / 60 + float(value[4][2]) / 3600

                    # Reverse geocoding to get the location
                    location = geolocator.reverse((latitude, longitude))
                    address = location.address

                    df_searchLokalInfo_Zwischen = pd.DataFrame()

                    geolocator = Nominatim(user_agent="nearest-town-finder")
                    location = geolocator.reverse((latitude, longitude), exactly_one=True)
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


openai.api_key = st.text_input("Enter your OpenAI Key to fetch location info from OpenAI", value="")
# Google Maps API key (replace with your own key)
api_key = "AIzaSyDrxPuhRPNObb7BiT6CoEvEunw3Vo3g5-I"

Sortierung = False

# Sort the image_info_list based on the datetime_taken value
anzeigenSortierung = st.checkbox("Show newest Photos first")
if anzeigenSortierung == True:
    Sortierung = True

image_info_list = sorted(image_info_list, key=lambda x: x[4],reverse=Sortierung)
image_info_list_df = pd.DataFrame(image_info_list)

show_image_info_list = st.checkbox("Show image_info_list")
if show_image_info_list:  
    st.dataframe(image_info_list_df)



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
        st.image(img, caption=f" {nearest_town}, {address}", use_column_width='always')

        col1, col2, col3 = st.columns([1, 1,2])

        visaWiki = col1.button("Wikipedia Info", key=imageKey*1000)
        if visaWiki:

            wiki_info1 = scrape_wikipedia(Town)
            if wiki_info1 != None:
                st.sidebar.subheader(f"{Town}")
      

                
            
            #st.sidebar.write("address: ",locationInfoList)
                st.sidebar.write(wiki_info1)
            #st.sidebar.write(len(wiki_info))


            #Här zusätzlich nearest town som möjlighet

            time.sleep(1)
            st.sidebar.divider()

            locationInfoList = address.split(",")
            locationInfoList.insert(0, str(nearest_town))
            #locationInfoListAuswahl = str(nearest_town) +","+locationInfoList
            
            #Annan approach för att hitta orstnamn
            adressEintragAuswahl = len(locationInfoList) - 5
            
            st.sidebar.write("Location:" , locationInfoList[adressEintragAuswahl])
                    
            wiki_info2 = scrape_wikipedia(locationInfoList[adressEintragAuswahl])
            st.sidebar.subheader(locationInfoList[adressEintragAuswahl])        
            st.sidebar.write(wiki_info2)

            #st.sidebar.write("searchLokalInfo", searchLokalInfo)


        st.sidebar.divider()

        visaOpenAI = col3.button("Chat OpenAI Info", key=imageKey * 2222)
        if visaOpenAI:

            # User input
            pre_Input = "Please give me a summary of the demographical, historical, cultural, meteroligical, polticial and touristic information about this location and the closest town of following location, and if some famous people have lived here and there are some interesting tourist attractions: "
            user_input = address

            prompt_input = pre_Input + user_input

            # Use ChatGPT to generate a response
            if user_input:
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-002",  # Use GPT-3.5 engine
                        prompt=prompt_input,
                        max_tokens=800,  # Adjust the length of the response as needed
                    )
                    if response and "choices" in response and response["choices"]:
                        bot_response = response["choices"][0]["text"]
                        st.sidebar.write(bot_response)
                    else:
                        st.sidebar.write("Bot: I'm sorry, I couldn't generate a response at the moment.")
                except Exception as e:
                    st.sidebar.write("Bot: An error occurred while processing your request.")
                    st.sidebar.write("Error Message:", str(e))


            st.sidebar.divider()


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
            

        visaExif = col2.button("exif", key=imageKey)
        if visaExif:
            st.sidebar.write(exif_data)

        st.subheader("")
        st.divider()
        
    # Create a map centered around the locations
    map = folium.Map(location=[image_info_list[0][2], image_info_list[0][3]], zoom_start=10)

    # Draw a line connecting the locations and add distance and travel time information as popups
    #line_coordinates = [(latitude, longitude) for _, _, latitude, longitude, _ in image_info_list] #funkar, men försöker smuggla med orter..
    line_coordinates = [(latitude, longitude) for img, address, latitude, longitude, datetime_taken,nearest_town,Town in image_info_list]

    total_distance = 0.0
    for i in range(len(line_coordinates) - 1):
        coord1 = line_coordinates[i]
        coord2 = line_coordinates[i + 1]
        #st.write("coord1:",coord1)
        #st.write("coord2:", coord2)
        #st.write("nearest_town:",nearest_town)


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

    

    # Display the map
    st_data = st_folium(map, width=725)

    if len(image_info_list_df)>1:

        # Display the total straight line distance



        #visa time estimates

        zeitSchäetzungExpander = st.expander("Show distances and travel times")
        with zeitSchäetzungExpander:

            st.write("Total Straightline Distance:")
            st.write(f"{total_distance:.2f} km")


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

                # Transportation mode dropdown
                transport_mode = st.selectbox("Select Google Maps Transportation Mode:",
                                              ["driving", "walking", "bicycling"], key="transportmodecheck" + str(i))
                # Define the API endpoint
                base_url = "https://maps.googleapis.com/maps/api/directions/json?"
                lat1, lon1 = coord1
                lat2, lon2 = coord2

                # Create the origin and destination strings
                origin = f"{lat1},{lon1}"
                destination = f"{lat2},{lon2}"

                # st.write("origin: ", origin)

                # st.write("destination: ", destination)

                # Define the parameters for the API request, including transportation mode
                params = {
                    "origin": origin,
                    "destination": destination,
                    "mode": transport_mode,
                    "key": api_key,
                }

                # Make the API request
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
                        st.error("Error: Unable to calculate distance and duration.")
                else:
                    st.error(
                        "Error: Unable to connect to the Google Maps API. Please check your API key and try again.")

                st.write("")
                st.write(f"Straightline - Distance: {distance:.2f} km\n")
                st.write(f"Straightline - Estimated Walking Time: {walking_time:.1f} hours\n")
                st.write(f"Straightline - Estimated Biking Time: {biking_time:.1f} hours\n")
                st.write(f"Straightline - Estimated Car Driving Time: {car_time:.1f} hours")
                st.write("")


                st.divider()