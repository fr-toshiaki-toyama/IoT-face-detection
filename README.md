## Dependencies
- cv2 (python)
  - OS X
    - `$ brew tap homebrew/science`
    - `$ brew install opencv`
    - setup [virtualenvs](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
    ```bash
    $ pip install virtualenv
    $ cd my_project_folder
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```
    - Link your opencv files in your `venv`'s `site-packages`
    ```bash
    cd venv/lib/python2.7/site-packages
    ln -s /usr/local/Cellar/opencv/2.4.13_3/lib/python2.7/site-packages/cv.py cv.py
    ln -s /usr/local/Cellar/opencv/2.4.13_3/lib/python2.7/site-packages/cv2.so cv2.so
    ```
  - Ubuntu
    - `$ apt-get install python-opencv` 
    - `$ pip install -r requirements.txt`

## Source of detectors:
- Rainer Lienhart (face)
- Modesto Castrillon-Santana (eyes)
