Building Drone Image Analysis tool to be used onsite in DJI controller
https://spicy-bear.github.io/image/


- it's still a bit buggy, I pivoted while testing the python to work and cram this into html/js. the dbscan is not working yet. 🤷
- it's been a bit of challenge to get detections/false positives low to be usable but not have false negatives, but these basic statistics/hsv values works pretty well./
- i had a live version that worked really well and cool but it was on a small video, as soon as it was over like 5 minutes it was too slow and unusable.
- the 'smart terrain' was where I was creating a basseline of colors to exclude, based off the large amount of natrual terrain vs samaller percentage of less natural colors.
- the 'global filter' was for a basic color analysis of the whole video and make detections on the stds/outliers of those values, which works if you dont introduce a lot of other colors/keep the video in scope.
- python script still needs work, used more of the manual settings less baseline or stds, was slightly abandonded to pivot work on the html version, it is sluggish on phone
- tldr theres a lot of junk to clean up from goofing around to see what worked and find a balance to deploy in field without getting overwhelmed.

- current main index for in field use
- https://github.com/spicy-bear/image/blob/main/index.html

- if you want to serve html the file and use phone as a server/hotspot and access non a controller, jsut run app.py and index.html from folder. Or just transfer the video from the drone to the phone to process.
