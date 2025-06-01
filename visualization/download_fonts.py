#!/usr/bin/env python3

import os
import requests
import zipfile
import io
import shutil
from pathlib import Path


def download_dejavu_fonts(output_dir="fonts"):
    """
    Downloads and extracts DejaVu fonts which support Cyrillic characters
    
    Args:
        output_dir (str): Directory to save the font files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for DejaVu fonts
    dejavu_url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
    
    print(f"Downloading DejaVu fonts from {dejavu_url}...")
    
    try:
        # Download the zip file
        response = requests.get(dejavu_url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Extract from the zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # List of font files we need
            needed_fonts = [
                "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
                "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf",
                "dejavu-fonts-ttf-2.37/ttf/DejaVuSerif.ttf",
                "dejavu-fonts-ttf-2.37/ttf/DejaVuSerif-Bold.ttf",
                "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Oblique.ttf"
            ]
            
            # Extract only the needed files
            for file in needed_fonts:
                target_filename = os.path.basename(file)
                print(f"Extracting {target_filename}...")
                
                # Extract the file to the output directory
                with zip_ref.open(file) as source, open(os.path.join(output_dir, target_filename), "wb") as target:
                    shutil.copyfileobj(source, target)
        
        print(f"Successfully downloaded and extracted DejaVu fonts to {output_dir}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading fonts: {e}")
        return False
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fonts_dir = os.path.join(script_dir, "fonts")
    
    download_dejavu_fonts(fonts_dir)