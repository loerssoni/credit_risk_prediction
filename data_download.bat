mkdir bk
cd bk
powershell -Command "Invoke-WebRequest http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip -OutFile data_berka.zip"
powershell Expand-Archive data_berka.zip
del data_berka.zip
cd ..