# A collection of mini tutorials

Most of them discussed in posts on my LinkedIn: https://www.linkedin.com/in/edoardo-camerinelli/
Contact me at: edoardo.camerinelli@usi.ch



## If you want to import a script from this repository:

```
import requests
url = 'https://raw.githubusercontent.com/username/repository/branch/filename.py' #URL of the raw file on GitHub
```

Fetch the content of the file and execute it. This way python automatically loads the script with all the respective functions/libraries into your enviroment.
```
exec(requests.get(url).text)
```


