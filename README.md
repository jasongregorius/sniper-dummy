# SNIPER Dummy Website

app.py is a Flask App written in Python:
	•	Live preview of the camera feed
	•	Displays the detected license plate along with the captured GPS coordinates
	•	Features for remote reboot and monitoring temperature

best.pt is the trained result of the “Plate Recognition” model using YOLOv8 (Ultralytics Python).
It has already gone through training for slanted plates.

The problem with this model is that it still doesn’t work well at close range — the car still needs to be visible.
For demo purposes with cars, the accuracy is fairly high. But it needs to work from a motorcycle’s perspective (perhaps closer to other vehicles, so the full car image is not captured).

Jason
Aug 27
