# EvaluateQRReaderPaper
Code to generate QR codes on a background image with distortions to evaluate their read performance

This works together with a bulk QR code generator program: https://github.com/markwhittyunsw/GenerateQRCode as well as the standalone QR code image renaming program: https://github.com/markwhittyunsw/QR_image_renamer 
- Author: Gareth N Hill PFR NZ and Mark Whitty UNSW (the original version of this code was written while on sabbatical at Plant and Food Research New Zealand)
- garethnhill@gmail.com | m.whitty@unsw.edu.au
- Source code: https://github.com/markwhittyunsw/EvaluateQRReaderPaper
- MIT licence

## To compile from source
 - install python 3.7.2 64 bit edition
 - update pip: "python -m pip install --upgrade pip" from location of Python installation using Administrator privileged CMD shell.
 - pip install pyzbar
 - pip install numpy
 - pip install opencv-contrib-python
 - run EvaluateQRCodePerformance.py to install OpenCV (cv2)
 - pip install scikit-image  # For random noise
 - pip install matplotlib
 - python EvaluateQRCodePerformance
HINT: Use Terminal in PyCharm to run commands for installing packages using pip (not Python Console)
