# Face Expression Detection App


## General Information
Real time recognition of facial expressions using Tensorflow Keras, OpenCV and Django. 
This is a simple face recognition application.
model was trained for 50 epochs and got an sccuracy of 64.9%.
The model was embedded in the django app to run on the web so users can upload pictures or take photos with camera and make prediction.

To run this app, simple create a venv, install required packages and and run the camera file..


## Installation
Create a virtual env, then
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all required packages (see requirements.txt)

```bash
pip install <package_name>
```

Activate virtual env and `cd` into project directory.

To use the app. `cd` into face_app, then run `python manage.py runserver` to start server (http://127.0.0.1:8000/
), open that in browser.

To run the notebook, simply open it on colab or jupyter notebook.


## Technologies Used
- Tensorflow
- Keras
- OpenCV
- Django

## Contributing
Pull requests are welcome. Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request.
You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

## Contributors
- Benjamin

See model prediction samples below:

![neutral](https://user-images.githubusercontent.com/32393357/157032517-f92cc202-ad34-4a82-8e71-f496f55cfb07.png)

![happy](https://user-images.githubusercontent.com/32393357/157031643-c4f35a60-6b99-444f-8e63-eaff7786f8b1.png)

## Check out demo video here:
[Video](https://github.com/benjaminogbonna/face_expression_detection/blob/master/sample_predictions/myface.mp4)

## Contact Information
[Linkedin](https://www.linkedin.com/in/onyedikachi-benjamin-ogbonna-218404177/)


## License
[MIT](https://choosealicense.com/licenses/mit/)
