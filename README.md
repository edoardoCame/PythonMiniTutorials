# A collection of mini tutorials

⚫️ Most of them discussed in posts on my LinkedIn: https://www.linkedin.com/in/edoardo-camerinelli/

⚫️ Contact me at: edoardo.camerinelli@usi.ch



## If you want to import a script from this repository:

```
import requests
url = 'https://raw.githubusercontent.com/username/repository/branch/filename.py' #URL of the raw file on GitHub
```

Fetch the content of the file and execute it. This way python automatically loads the script with all the respective functions/libraries into your enviroment.
```
exec(requests.get(url).text)
```



## Struttura del repository

Qui sotto trovi una panoramica delle cartelle principali e di cosa contengono:

```
PythonMiniTutorials/
├─ Data Downloading/
│  └─ Management tutorials/    -> notebook su download e analisi dati
├─ Data Processing/            -> notebook e script su manipolazione dati
├─ data analytics and visualization/ -> notebook di analisi/visualizzazione
├─ learn python/               -> notebook introduttivi e script vari
├─ time series models/         -> notebook teorici e pratici sulle serie temporali
└─ trading strategies/         -> materiale piu' avanzato su backtesting e trading
```

- **Data Downloading / Management tutorials**: notebook su download e analisi dati.
- **Data Processing**: esempi di manipolazione e pulizia dei dati.
- **data analytics and visualization**: analisi dei dati e visualizzazioni.
- **learn python**: materiale introduttivo e script utili per iniziare.
- **time series models**: modelli e tecniche per serie temporali.
- **trading strategies**: script di backtesting e strategie di trading.
