
# OCR

There are two scripts

**Manual Script**
The script is located in rotate.py file, use the following command to run it

    python3 rotate.py <path to image file>

It will open a window to you, so you can select text from image.
If you selected wrong area in image and want to undo it, press "c".

**Autonomous Script**
The script is located in main.py file, use the following command to run it

    python3 main.py <path to image file>

This script automatically detect locations of text in image, select them and type the detected text over them.
This script is weak in detecting long line of text, but it can detect one or two words in specific banner.