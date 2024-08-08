# dota_match_three_solver
This project is a bot for playing "three in a row". The bot automatically detects icons on the playing field, recognizes them and makes optimal moves to achieve the best results.
Installation and Usage Guide
Step 1: Install Python

Download and install Python:
Go to the official Python website and download the latest version of Python for your operating system (Windows, macOS, Linux).
Install Python by following the on-screen instructions. Be sure to check the "Add Python to PATH" option during installation.

Check Python installation:
Open a terminal or command prompt.
Enter the command:

python --version

You should see the installed version of Python.

Step 2: Install the required libraries

Create a virtual environment (optional):
A virtual environment will help isolate your project's dependencies.
Enter the command:

python -m venv myenv

Activate the virtual environment:

On Windows:

myenv\Scripts\activate

On macOS and Linux:

source myenv/bin/activate

Install the necessary libraries:

Enter the command:

pip install opencv-python-headless numpy pillow mss pyautogui keyboard

Create a working directory:
Create a folder for your project and move the downloaded code into it.

python main.py
