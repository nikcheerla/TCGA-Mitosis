A cloud server for crowdsourcing mitosis detection in pathological images.
Built on Flask and Unity.
Hosted currently with Heroku at https://mitosis-cloud-server.herokuapp.com.

This server can be easily modified to work with other types of data!
All labeled images are stored as jpgs inside the app/static/train directory. These jpgs have corresponding csvs indicating the location of mitotic figures. The file app/static/intermediate_coords.npy stores a numpy array of coordinates indicating the desired normal coordinates to sample from in the slide images. Lastly, the directory app/static/uncat contains uncategorized images which user data is collected for.

