import pyrebase
config = {
      "apiKey": "AIzaSyBEdoDCZxSAAaEhUGhSMvAW_FOWNh2JOjw",
      "authDomain": "foodmanager-c10fd.firebaseapp.com",
      "projectId": "foodmanager-c10fd",
      "storageBucket": "foodmanager-c10fd.appspot.com",
      "messagingSenderId": "134814754014",
      "appId": "1:134814754014:web:0c70bdf5924a0bab074e4f",
      "measurementId": "G-L6TDYKLRZ7"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

path_in_cloud = "food.jpg"
path_local = "food.png"
storage.child(path_on_cloud).put(path_local)