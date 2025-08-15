import requests
import os

def download_data(url, output_folder="data", filename="premier_league_2324.csv"):
    """
    Downloads a file from a URL and saves it to a specified folder.

    Args:
        url (str): The URL of the file to download.
        output_folder (str): The folder to save the file in.
        filename (str): The name to save the file as.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, filename)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # The content is text (CSV), so we write it in text mode.
        # It's important to handle the encoding correctly. football-data.co.uk uses 'latin1' or 'iso-8859-1'.
        with open(output_path, 'w', encoding='latin1') as f:
            f.write(response.text)

        print(f"Data successfully downloaded and saved to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the data: {e}")
        return None

if __name__ == "__main__":
    # URL for the 2023-2024 Premier League season data
    DATA_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

    print("Starting data download...")
    download_data(DATA_URL)
