import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import pandas as pd
import io
import zipfile
import time
from datetime import datetime

# Streamlit Setup
st.set_page_config(page_title="ODK Audio Test Downloader", layout="centered")
st.title("🔊 ODK Audio Test Download (All Files from Each Form)")

# ODK Configurations for each server
ODK_CONFIGS = {
    "Server 1": {
        "ODK_USERNAME": "rushi@tnodk01.ii.com",
        "ODK_PASSWORD": "rushi2025&",
        "BASE_URL": "https://tnodk01.indiaintentions.com"
    },
    "Server 2": {
        "ODK_USERNAME": "rushi@tnodk01.ii.com",
        "ODK_PASSWORD": "rushi2025&",
        "BASE_URL": "https://tnodk02.indiaintentions.com"
    }
}

# Form mappings
# Form mappings (truncated for brevity - include your full form mappings here)
forms = {
    "Server 1": {
        "TN Master": {
            "Master Landscape Survey 04-2025 ODK XLSForm": {
                "project_id": 3,
                "form_id": "Master Landscape Survey 04-2025 ODK XLSForm"
            }
        },
        "BK TN AC Landscape": {
            "201-Cumbum": {"project_id": 4, "form_id": "201-Cumbum Landscape Survey 04-2025"},
            "39-Sholingur": {"project_id": 4, "form_id": "39-Sholingur Landscape Survey 04-2025"},
            "155-Cuddalore": {"project_id": 4, "form_id": "155-Cuddalore Landscape Survey 04-2025"},
            "66-Polur": {"project_id": 4, "form_id": "66-Polur Landscape Survey 04-2025"},
            "144-Manachanallur": {"project_id": 4, "form_id": "144-Manachanallur Landscape Survey 04-2025"},
            "145-Musiri": {"project_id": 4, "form_id": "145-Musiri Landscape Survey 04-2025"},
            "56-Thalli": {"project_id": 4, "form_id": "56-Thalli Landscape Survey 04-2025"},
            "3-Thirutthani": {"project_id": 4, "form_id": "3-Thirutthani Landscape Survey 04-2025"},
            "146-Thuraiyur": {"project_id": 4, "form_id": "146. Thuraiyur (SC) Landscape Survey 04-2025"},
            "182-Alangudi": {"project_id": 4, "form_id": "182-Alangudi Landscape Survey 04-2025"},
            "137-Kulithalai": {"project_id": 4, "form_id": "137-Kulithalai Landscape Survey 04-2025"},
            "12-Perambur": {"project_id": 4, "form_id": "12-Perambur Landscape Survey 04-2025"},
            "171-Kumbakonam": {"project_id": 4, "form_id": "171-Kumbakonam Landscape Survey 04-2025"},
            "150-Jayankondam": {"project_id": 4, "form_id": "150-Jayankondam Landscape Survey 04-2025"},
            "132-Dindigul": {"project_id": 4, "form_id": "132 Dindigul Landscape Survey 04-2025"},
            "20-Thousand Lights": {"project_id": 4, "form_id": "20-Thousand Lights Landscape Survey 04-2025"},
            "37-Kancheepuram": {"project_id": 4, "form_id": "37-Kancheepuram Landscape Survey 04-2025"},
            "136-Krishnarayapuram": {"project_id": 4, "form_id": "136-Krishnarayapuram (SC) Landscape Survey 04-2025"},
            "38-Arakkonam": {"project_id": 4, "form_id": "38-Arakkonam (SC) Landscape Survey 04-2025"},
            "200-Bodinayakanur": {"project_id": 4, "form_id": "200-Bodinayakanur Landscape Survey 04-2025"},
            "118-Coimbatore North": {"project_id": 4, "form_id": "118-Coimbatore North Landscape Survey 04-2025"},
            "189-Madurai East": {"project_id": 4, "form_id": "189-Madurai East Landscape Survey 04-2025"},
            "103-Perundurai": {"project_id": 4, "form_id": "103-Perundurai Landscape Survey 04-2025"},
            "199-Periyakulam": {"project_id": 4, "form_id": "199. Periyakulam (SC) Landscape Survey 04-2025"},
            "149-Ariyalur": {"project_id": 4, "form_id": "149-Ariyalur Landscape Survey 04-2025"},
            "135-Karur": {"project_id": 4, "form_id": "135-Karur Landscape Survey 04-2025"},
            "138-Manapparai": {"project_id": 4, "form_id": "138-Manapparai Landscape Survey 04-2025"},
            "152-Vriddhachalam": {"project_id": 4, "form_id": "152-Vriddhachalam Landscape Survey 04-2025"},
            "153-Neyveli": {"project_id": 4, "form_id": "153-Neyveli Landscape Survey 04-2025"},
            "156-Kurinjipadi": {"project_id": 4, "form_id": "156-Kurinjipadi Landscape Survey 04-2025"},
            "161-Mayiladuthurai": {"project_id": 4, "form_id": "161-Mayiladuthurai Landscape Survey 04-2025"},
            "166-Thiruthuraipoondi": {"project_id": 4, "form_id": "166-Thiruthuraipoondi (SC) Landscape Survey 04-2025"},
            "178-Gandharvakottai": {"project_id": 4, "form_id": "178-Gandharvakottai (SC) Landscape Survey 04-2025 copy 3"},
            "198-Andipatti": {"project_id": 4, "form_id": "198-Andipatti Landscape Survey 04-2025"},
            "51-Uthangarai": {"project_id": 4, "form_id": "51-Uthangarai (SC) Landscape Survey 04-2025"},
            "60-Pappireddippatti": {"project_id": 4, "form_id": "60-Pappireddippatti Landscape Survey 04-2025"},
            "61-Harur": {"project_id": 4, "form_id": "61-Harur (SC) Landscape Survey 04-2025"},
            "180-Pudukkottai": {"project_id": 4, "form_id": "180-Pudukkottai Landscape Survey 04-2025"},
            "158-Chidambaram": {"project_id": 4, "form_id": "158-Chidambaram Landscape Survey 04-2025"},
            "216-Srivaikuntam Landscape Survey 04-2025": {"project_id": 4, "form_id": "216-Srivaikuntam Landscape Survey 04-2025"},
            "133-Vedasandur Landscape Survey 04-2025": {"project_id": 4, "form_id": "133-Vedasandur Landscape Survey 04-2025"},
            "110-Coonoor Landscape Survey 04-2025": {"project_id": 4, "form_id": "110-Coonoor Landscape Survey 04-2025"},
            "109-Gudalur (SC) Landscape Survey 04-2025": {"project_id": 4, "form_id": "109-Gudalur (SC) Landscape Survey 04-2025"},
            "62-Chengam (SC)" : {"project_id":4,"form_id":"62-Chengam (SC) Landscape Survey 04-2025"}
        },
        "03 TN AC Landscape": {
            "110-Coonoor": {"project_id": 5, "form_id": "110-Coonoor Landscape Survey 04-2025"},
            "222-Tenkasi": {"project_id": 5, "form_id": "222-Tenkasi Landscape Survey 04-2025"},
            "223-Alangulam": {"project_id": 5, "form_id": "223-Alangulam Landscape Survey 04-2025"},
            "48-Ambur": {"project_id": 5, "form_id": "48-AmburLandscape Survey 04-2025"},
            "49-Jolarpet": {"project_id": 5, "form_id": "49-Jolarpet Landscape Survey 04-2025"},
            "63-Tiruvannamalai": {"project_id": 5, "form_id": "63-Tiruvannamalai Landscape Survey 04-2025"},
            "64-Kilpennathur": {"project_id": 5, "form_id": "64-KilpennathurLandscape Survey 04-2025"},
            "65-Kalasapakkam": {"project_id": 5, "form_id": "65-Kalasapakkam Landscape Survey 04-2025"},
            "67-Arani": {"project_id": 5, "form_id": "67-Arani Landscape Survey 04-2025"},
            "68-Cheyyar": {"project_id": 5, "form_id": "68-Cheyyar Landscape Survey 04-2025"},
            "100-Modakkurichi": {"project_id":5,"form_id":"100-Modakkurichi Landscape Survey 04-2025"},
            "74-Villupuram" : {"project_id":5,"form_id":"74-Villupuram Landscape Survey 04-2025"},
            "55-Hosur" : {"project_id":5,"form_id":"55-Hosur Landscape Survey 04-2025"}
        },
        "04 TN AC Landscape": {
            "183-Aranthangi": {"project_id": 6, "form_id": "183-Aranthangi Landscape Survey 04-2025"},
            "181-Thirumayam": {"project_id": 6, "form_id": "181-Thirumayam Landscape Survey 04-2025"},
            "159-Kattumannarkoil (SC)" : {"project_id":6,"form_id":"159-Kattumannarkoil (SC) Landscape Survey 04-2025"}
        }
    },
    "Server 2": {
        "01 FMRS TN LandScape Survey": {
            "139-Srirangam": {"project_id": 3, "form_id": "139-SrirangamLandscape Survey 04-2025"},
            "141-Tiruchirappalli (East)": {"project_id": 3, "form_id": "141-Tiruchirappalli (East) Landscape Survey 04-2025"},
            "163-Nagapattinam": {"project_id": 3, "form_id": "163-Nagapattinam Landscape Survey 04-2025"},
            "172-Papanasam": {"project_id": 3, "form_id": "172-Papanasam Landscape Survey 04-2025"},
            "174-Thanjavur": {"project_id": 3, "form_id": "174-Thanjavur Landscape Survey 04-2025"},
            "177-Peravurani": {"project_id": 3, "form_id": "177-Peravurani Landscape Survey 04-2025"},
            "205-Sivakasi": {"project_id": 3, "form_id": "205-Sivakasi Landscape Survey 04-2025"},
            "209-Paramakudi (SC)": {"project_id": 3, "form_id": "209-Paramakudi (SC) Landscape Survey 04-2025"},
            "213-Vilathikulam": {"project_id": 3, "form_id": "213-Vilathikulam Landscape Survey 04-2025"},
            "216-Srivaikuntam": {"project_id": 3, "form_id": "216-Srivaikuntam Landscape Survey 04-2025"},
            "217-Ottapidaram (SC)": {"project_id": 3, "form_id": "217 - Ottapidaram (SC) Landscape Survey 04-2025"},
            "65-Kalasapakkam": {"project_id": 3, "form_id": "65-Kalasapakkam Landscape Survey 04-2025"},
            "68-Cheyyar": {"project_id": 3, "form_id": "68-Cheyyar Landscape Survey 04-2025"},
            "77-Ulundurpettai": {"project_id": 3, "form_id": "77-Ulundurpettai Landscape Survey 04-2025"},
            "8-Ambattur": {"project_id": 3, "form_id": "8-Ambattur Landscape Survey 04-2025"},
            "Test Landscape Survey 04-2025 copy": {"project_id": 3, "form_id": "Test Landscape Survey 04-2025 copy"},
            "V3 Master Landscape Survey 04-2025 ODK XLSForm": {"project_id": 3, "form_id": "V3 Master Landscape Survey 04-2025 ODK XLSForm"}
        },
        "01 Bikas TN Landscape Survey": {
            "48-Ambur": {"project_id": 4, "form_id": "48-Ambur Landscape Survey 04-2025"},
            "24-Thiyagarayanagar": {"project_id": 4, "form_id": "24-Thiyagarayanagar Landscape Survey 04-2025"},
            "14-Villivakkam": {"project_id": 4, "form_id": "14-Villivakkam Landscape Survey 04-2025"},
            "46-Gudiyattam (SC)": {"project_id": 4, "form_id": "46-Gudiyattam (SC) Landscape Survey 04-2025"},
            "147-Perambalur (SC)": {"project_id": 4, "form_id": "147-Perambalur (SC) Landscape Survey 04-2025"},
            "173-Thiruvaiyaru": {"project_id": 4, "form_id": "173-Thiruvaiyaru Landscape Survey 04-2025"},
            "223-Alangulam": {"project_id": 4, "form_id": "223-Alangulam Landscape Survey 04-2025"},
            "150-Jayankondam": {"project_id": 4,"form_id": "150-Jayankondam Landscape Survey 04-2025"},
            "179-Viralimalai" : {"project_id":4,"form_id":"179-Viralimalai Landscape Survey 04-2025"}
        },
        "02 Bikas TN Landscape Survey" : {
            "157-Bhuvanagiri" : {"project_id":5,"form_id": "157-Bhuvanagiri Landscape Survey 04-2025"},
            "171-Kumbakonam" : {"project_id":5,"form_id": "171-Kumbakonam Landscape Survey 04-2025 copy 11"},
            "66-Polur" : {"project_id":5,"form_id": "66-Polur Landscape Survey 04-2025"},
            "132-Dindigul" : {"project_id":5,"form_id":"132-Dindigul Landscape Survey 04-2025"},
            "135-Karur" : {"project_id":5,"form_id":"135-Karur Landscape Survey 04-2025"},
            "64-Kilpennathur":{"project_id":5,"form_id":"64-Kilpennathur Landscape Survey 04-2025"},
            "67-Arani" : {"project_id":5,"form_id":"67-Arani Landscape Survey 04-2025"},
            "155-Cuddalore" : {"project_id":5,"form_id":"155-Cuddalore Landscape Survey 04-2025"},
            "140-Tiruchirappalli (West)":{"project_id":5,"form_id": "140-Tiruchirappalli (West) Landscape Survey 04-2025"},
            "12-Perambur": {"project_id":5,"form_id":"12-Perambur Landscape Survey 04-2025"},
            "134-Aravakurichi":{"project_id":5,"form_id":"134-Aravakurichi Landscape Survey 04-2025"},
            "138-Manapparai": {"project_id":5,"form_id":"138-Manapparai Landscape Survey 04-2025"},
            "144-Manachanallur" : {"project_id":5,"form_id":"144-Manachanallur Landscape Survey 04-2025"}
        },
        "01 Shashi TN Landscape Survey":{
            "90-Salem (South)" : {"project_id":6,"form_id":"90-Salem (South) Landscape Survey 04-2025"},
            "81-Gangavalli (SC)" : {"project_id":6,"form_id":"81-Gangavalli (SC) Landscape Survey 04-2025"},
            "82-Attur (SC)" : {"project_id":6,"form_id":"82-Attur (SC) Landscape Survey 04-2025"},
            "83-Yercaud (ST)" : {"project_id":6,"form_id":"83-Yercaud (ST) Landscape Survey 04-2025"},
            "84-Omalur" : {"project_id":6,"form_id":"84-Omalur Landscape Survey 04-2025"},
            "85-Mettur" : {"project_id":6,"form_id":"85-Mettur Landscape Survey 04-2025"},
            "86-Edappadi" : {"project_id":6,"form_id":"86-Edappadi Landscape Survey 04-2025"},
            "87-Sankari" : {"project_id":6,"form_id":"87-Sankari Landscape Survey 04-2025 copy 8"},
            "88-Salem (West)" : {"project_id":6,"form_id":"88-Salem (West) Landscape Survey 04-2025"},
            "89-Salem (North)" : {"project_id":6,"form_id":"89-Salem (North) Landscape Survey 04-2025"},
            "91-Veerapandi" : {"project_id":6,"form_id":"91-Veerapandi Landscape Survey 04-2025"}
        },
        "02 Shashi TN Landscape ":{
            "11-Dr.Radhakrishnan Nagar" : {"project_id":8,"form_id":"11-Dr.Radhakrishnan Nagar Landscape Survey 04-2025"},
            "17-Royapuram" : {"project_id":8,"form_id":"17-Royapuram Landscape Survey 04-2025"},
            "22-Virugampakkam" : {"project_id":8,"form_id":"22-Virugampakkam Landscape Survey 04-2025"},
            "25-Mylapore" : {"project_id":8,"form_id":"25-Mylapore Landscape Survey 04-2025"}
        },
        "02 FMRS TN Landscape Survey":{
            "192-Madurai South" : {"project_id":7,"form_id":"192-Madurai South Landscape Survey 04-2025"},
            "193-Madurai Central" : {"project_id":7,"form_id":"193-Madurai Central Landscape Survey 04-2025"},
            "188-Melur": {"project_id":7,"form_id":"188-Melur Landscape Survey 04-2025"},
            "190-Sholavandan (SC)" : {"project_id":7,"form_id":"190-Sholavandan (SC) Landscape Survey 04-2025"},
            "191-Madurai North" : {"project_id":7,"form_id":"191-Madurai North Landscape Survey 04-2025"},
            "194-Madurai West": {"project_id":7,"form_id":"194-Madurai West Landscape Survey 04-2025"},
            "195-Thiruparankundram" : {"project_id":7,"form_id":"195-Thiruparankundram Landscape Survey 04-2025"},
            "196-Thirumangalam" : {"project_id":7,"form_id":"196-Thirumangalam Landscape Survey 04-2025"}
        }
    }
}

def fetch_submissions(selected_server, project_id, form_id):
    config = ODK_CONFIGS[selected_server]
    submission_url = f"{config['BASE_URL']}/v1/projects/{project_id}/forms/{form_id}.svc/Submissions"
    try:
        response = requests.get(
            submission_url,
            auth=HTTPBasicAuth(config['ODK_USERNAME'], config['ODK_PASSWORD']),
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("value", [])
    except Exception as e:
        st.error(f"Error fetching submissions: {str(e)}")
        return []

def filter_submissions_by_date(submissions, selected_date):
    if selected_date:
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        return [submission for submission in submissions if submission['__system']['submissionDate'].startswith(selected_date_str)]
    return submissions

def download_audio_files(selected_server, form_name, project_id, form_id, audio_submissions):
    config = ODK_CONFIGS[selected_server]
    zip_buffer = io.BytesIO()
    download_status = []
    
    total_size = 0  # Track the total size of the ZIP
    max_size = 200 * 1024 * 1024  # 200MB limit for Cloud Run memory
    
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, (_, row) in enumerate(audio_submissions.iterrows()):
                audio_file = row['bg_audio']
                submission_id = row['__id']
                
                # Handle case where submitted_by might be in a different location or format
                submitted_by = "unknown"
                if 'group_six' in row and isinstance(row['group_six'], dict) and 'submittedBy' in row['group_six']:
                    submitted_by = row['group_six']['submittedBy']
                elif 'submittedBy' in row:
                    submitted_by = row['submittedBy']
                
                # Ensure the submitted_by value is safe for filenames
                submitted_by = str(submitted_by).replace(" ", "_").replace("/", "_").replace("\\", "_")
                
                with st.spinner(f"Downloading {i+1}/{len(audio_submissions)}: {submitted_by}_{audio_file}..."):
                    try:
                        audio_url = f"{config['BASE_URL']}/v1/projects/{project_id}/forms/{form_id}/submissions/{submission_id}/attachments/{audio_file}"
                        audio_response = requests.get(
                            audio_url,
                            auth=HTTPBasicAuth(config['ODK_USERNAME'], config['ODK_PASSWORD']),
                            timeout=60  # Increased timeout for larger files
                        )
                        audio_response.raise_for_status()
                        
                        # Check if adding this file would exceed our size limit
                        file_size = len(audio_response.content)
                        if total_size + file_size > max_size:
                            download_status.append(f"⚠️ Skipping remaining files: ZIP file size limit reached")
                            break
                            
                        total_size += file_size

                        # Create filename as requested: submittedBy_bg_audio
                        # Ensure the filename is valid
                        clean_name = f"{submitted_by}_{audio_file}"
                        clean_name = clean_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        
                        zip_file.writestr(clean_name, audio_response.content)
                        download_status.append(f"✅ Downloaded: {clean_name}")

                    except Exception as e:
                        download_status.append(f"❌ Failed to download {audio_file}: {str(e)}")
                        continue
    except Exception as e:
        download_status.append(f"❌ Error creating ZIP file: {str(e)}")
        return None, download_status
        
    # Ensure we're at the beginning of the buffer before returning
    zip_buffer.seek(0)
    return zip_buffer, download_status

def main():
    try:
        # Sidebar for server, project, and form selection and date input
        st.sidebar.header("Filter Options")
        selected_server = st.sidebar.selectbox("Select Server", list(forms.keys()))
        if selected_server:
            selected_project = st.sidebar.selectbox("Select Project", list(forms[selected_server].keys()))
            if selected_project:
                selected_form = st.sidebar.selectbox("Select Form", list(forms[selected_server][selected_project].keys()))
                selected_date = st.sidebar.date_input("Select Date", None)

                if selected_form:
                    form_name = selected_form
                    project_id = forms[selected_server][selected_project][selected_form]['project_id']
                    form_id = forms[selected_server][selected_project][selected_form]['form_id']

                    # Fetch submissions for the selected form
                    with st.spinner(f"Fetching submissions for {form_name}..."):
                        data = fetch_submissions(selected_server, project_id, form_id)

                    if not data:
                        st.warning(f"No submissions found for {form_name}.")
                    else:
                        # Filter submissions by date
                        filtered_data = filter_submissions_by_date(data, selected_date)

                        if not filtered_data:
                            st.warning(f"No submissions found for {form_name} on {selected_date}.")
                        else:
                            df = pd.DataFrame(filtered_data)
                            
                            # Check if 'bg_audio' column exists
                            if 'bg_audio' not in df.columns:
                                st.error("No 'bg_audio' column found in the data. Available columns: " + ", ".join(df.columns))
                            else:
                                audio_submissions = df[df['bg_audio'].notna()]  # Take all audio files

                                if audio_submissions.empty:
                                    st.warning(f"No audio files found in submissions for {form_name}.")
                                else:
                                    st.success(f"Found {len(audio_submissions)} audio files to download for {form_name}")
                                    
                                    # Show a preview of the first few files
                                    st.write(f"Sample files to download for {form_name}:")
                                    preview_df = audio_submissions[['__id', 'bg_audio']].head(5)
                                    st.dataframe(preview_df)

                                    # Show file size warning
                                    st.warning(f"Note: Downloading large numbers of audio files may take time and require substantial memory.")
                                    
                                    # Add option to limit number of files
                                    max_files = st.slider("Maximum number of files to download", 
                                                         min_value=1, 
                                                         max_value=min(100, len(audio_submissions)), 
                                                         value=min(20, len(audio_submissions)))
                                    
                                    # Download button
                                    if st.button(f"🚀 Download Audio Files from {form_name} (Max: {max_files})"):
                                        # Limit to the selected number of files
                                        limited_submissions = audio_submissions.head(max_files)
                                        
                                        zip_buffer, download_status = download_audio_files(
                                            selected_server, form_name, project_id, form_id, limited_submissions
                                        )
                                        
                                        if zip_buffer is not None:
                                            timestamp = time.strftime('%Y%m%d_%H%M%S')
                                            clean_form_name = form_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                                            
                                            try:
                                                st.success(f"🎉 Download prepared for {len(limited_submissions)} files from {form_name}!")
                                                st.download_button(
                                                    label=f"⬇️ Download {form_name} Audio Files (ZIP)",
                                                    data=zip_buffer.getvalue(),
                                                    file_name=f"{clean_form_name}_AUDIOS_{timestamp}.zip",
                                                    mime="application/zip"
                                                )
                                            except Exception as e:
                                                st.error(f"Error with download button: {str(e)}")

                                        # Display download status messages
                                        st.write("Download Status:")
                                        for status in download_status:
                                            st.write(status)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()