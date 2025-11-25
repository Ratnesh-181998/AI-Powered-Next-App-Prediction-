"""
Screenshot Capture Script
Captures a screenshot of the web UI and saves it to the web folder
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import os

def capture_ui_screenshot():
    """Capture screenshot of the UI"""
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-gpu')
    
    try:
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to the UI
        ui_path = os.path.abspath('web/index.html')
        driver.get(f'file:///{ui_path}')
        
        # Wait for page to load
        time.sleep(2)
        
        # Take screenshot
        screenshot_path = os.path.join('web', 'app_predictor_ui_complete.png')
        driver.save_screenshot(screenshot_path)
        
        print(f"Screenshot saved to: {screenshot_path}")
        
        # Close driver
        driver.quit()
        
        return screenshot_path
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires Chrome and ChromeDriver to be installed")
        print("Alternative: Use browser's built-in screenshot feature (F12 > Ctrl+Shift+P > 'screenshot')")
        return None

if __name__ == '__main__':
    capture_ui_screenshot()
