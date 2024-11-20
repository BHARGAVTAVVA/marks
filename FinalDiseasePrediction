import os
import streamlit as st
import google.generativeai as genai
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd

# Set up Google API Key
GOOGLE_API_KEY = "AIzaSyCWmWlwM4R3Otqp0Go51z9EVCNfEgWa2rM"  # Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Azure Maps Subscription Key
AZURE_MAPS_KEY = "1xjlGxP2T6R2tLskhR2KtjzM2HHrs2mGUxhk17sX3kb3wgRJhHgfJQQJ99AKACYeBjFWKof7AAAgAZMPZ0rE"

# Function to search nearby places
def search_nearby_places(location, category):
    endpoint = "https://atlas.microsoft.com/search/address/json"
    query_params = {
        "api-version": "1.0",
        "query": location,
        "subscription-key": AZURE_MAPS_KEY,
    }

    response = requests.get(endpoint, params=query_params)
    response_json = response.json()

    if not response_json.get('results'):
        st.error("Location not found.")
        return None

    lat = response_json['results'][0]['position']['lat']
    lon = response_json['results'][0]['position']['lon']

    # Search for nearby places
    search_endpoint = "https://atlas.microsoft.com/search/poi/json"
    search_params = {
        "api-version": "1.0",
        "query": category,
        "lat": lat,
        "lon": lon,
        "subscription-key": AZURE_MAPS_KEY,
        "radius": 10000,
        "orderby": "Distance",
        "top": 10,
    }

    search_response = requests.get(search_endpoint, params=search_params)
    search_json = search_response.json()

    if 'results' not in search_json:
        st.error("No results found for the category.")
        return None

    return search_json['results']

# Streamlit Application
def main():
    st.title("Early Disease Prediction & Nearby Places Search")

    # Tabs for separate functionalities
    tab1, tab2 = st.tabs(["Early Disease Prediction", "Nearby Places Search"])

    # Tab 1: Early Disease Prediction
    with tab1:
        st.header("Early Disease Prediction")

        # User Input
        symptoms = st.text_input("Enter your symptoms (e.g., headache, fever, fatigue):")

        # Submit Button
        if st.button("Submit Symptoms"):
            if symptoms.strip():
                # Format prompt to include request for probabilities
                prompt = f"""
                You are a highly intelligent medical assistant. A user has described their symptoms as follows:
                Symptoms: {symptoms}

                Please:
                1. Predict the most likely diseases based on the symptoms along with their probability scores.
                2. Provide precautions to prevent worsening of the diseases.
                3. Suggest possible medications (common over-the-counter or prescription medications).
                """

                # Generate response
                st.write("Analyzing your symptoms...")
                try:
                    model = genai.GenerativeModel('gemini-pro')  # Use a valid model name here
                    response = model.generate_content(prompt)  # Use generate_content instead of generate_text
                    
                    # Display results
                    st.success("Prediction Results:")
                    result_text = response.text  # Adjust based on actual response structure
                    
                    # Displaying the response text
                    st.write(result_text)
                    
                    # Add a warning about medications
                    st.warning("**Note:** Medications suggested are optional. Please use cautiously and consult a healthcare professional before starting any new medication.")

                except Exception as e:
                    st.error(f"Error generating response: {e}")
            else:
                st.error("Please enter symptoms to proceed.")

    # Tab 2: Nearby Places Search
    with tab2:
        st.header("Nearby Places Search")

        location = st.text_input("Enter a location (e.g., Eluru)")
        category = st.text_input("Enter a category (e.g., Hospitals)")

        # Store the map and list in session state
        if 'map' not in st.session_state:
            st.session_state.map = None
        if 'places' not in st.session_state:
            st.session_state.places = None

        if st.button("Search Places"):
            if not location or not category:
                st.warning("Please enter both location and category.")
                return

            results = search_nearby_places(location, category)

            if not results:
                st.info("No results found.")
                return

            # Extract lat and lon from the first result
            lat = results[0]['position']['lat']
            lon = results[0]['position']['lon']

            # Create a Folium map
            m = folium.Map(location=[lat, lon], zoom_start=12)

            # Add markers for each nearby place
            for place in results:
                folium.Marker([place['position']['lat'], place['position']['lon']],
                              popup=place['poi']['name']).add_to(m)

            # Store map and places list in session state to avoid re-rendering
            st.session_state.map = m
            st.session_state.places = results

        # Display the map if it exists in session state
        if st.session_state.map:
            st_folium(st.session_state.map, width=700, height=500)

        # Display the list of places in a clean table format
        if st.session_state.places:
            st.subheader("Nearby Places:")
            places_data = []
            
            # Prepare data for the table
            for place in st.session_state.places:
                name = place['poi']['name']
                address = place['address']['freeformAddress']
                lat = place['position']['lat']
                lon = place['position']['lon']
                places_data.append([name, address, lat, lon])

            # Create a pandas DataFrame
            df = pd.DataFrame(places_data, columns=["Place Name", "Address", "Latitude", "Longitude"])

            # Display the table
            st.table(df)

if __name__ == "__main__":
    main()
