# LSTM_CHAR

### ALLGEMEIN

`--help | -h` zeigt die Gebrauchsanweisung an, also genau diese hier (doc/help.txt)

`--verbose | -v` zeichnet den Programmverlauf ausführlicher nach

`--ascii | -a` konvertiert den Text nach ASCII; ansonsten wird der Text nach Unicode dekodiert

`--expanded | -p` konvertiert den Text nach ASCII + äöüÄÖÜß; ansonsten wird der Text nach Unicode dekodiert

`--input <Textdatei> | -i <Textdatei>` der Ausgangstext, dieser sollte im Verzeichnis /input abgelegt sein; Beispiel: `-i wahlverwandschaften.txt`

### TEXT GENERIEREN

`--generate <Ganzzahl> | -g <Ganzzahl>` es wird mithilfe der gespeicherten Gewichtung_nur ausgegeben_. Die geladene Matrix ist, wenn kein Limit festgelegt wurde, immer die zuletzt gespeicherte; die Zahl legt die Länge [in Zeichen] der Ausgabe fest; standardmäßig 500; Beispiel: `-g 4200`

`--seed | -s` es wird von einem später anzugebenden Seed aus weitergeschrieben

`--diversity <Gleitkommazahl> | -d <Gleitkommazahl>` legt den Grad fest, zu dem abgewichen werden darf; standardmäßig 0.2; Höchstwert 2.0; Beispiel: `-d 1.2`

### NETZWERK TRAINIEREN

`--new | -n` das Netzwerk wird von Null an trainiert, die Gewichtung wird _für das festgelegte Zeichenbeschränkung_ gespeichert

`--resume | -r` das Training wird mithilfe der _zuletzt_ gespeicherten Konfiguration (Gewichtung, Zeichenbeschränkung, Anzahl der Epochen) wiederaufgenommen, die Gewichtung wird gespeichert

`--epochs <Ganzzahl> | -e <Ganzzahl>` legt die Anzahl der Durchläufe fest, bevor die Matrix gespeichert wird; standardmäßig 10; Beispiel: `-e 5`

`--limit <Ganzzahl> | -l <Ganzzahl> ODER -s max` legt die Zeichenbeschränkung fest; standardmäßig 10000; für jedes Limit wird eine eigene Gewichtung gespeichert; sie sind jedoch untereinander kompatibel; Beispiel: `-l 500`

`--dynamic | -y` verschiebt den Anfang des Textausschnitts um einen zufälligen Betrag

## BEISPIELE

1. Du willst ein neues Netzwerk mit einem Text in voller Länge trainieren und möglichst alle Zeichen berücksichtigen. Da bei langen Texten jede Epoche viel Zeit in Anspruch nimmt, könnte man die Anzahl der Epochen heruntersetzen:
  ```
  python lstm_char.py --input roomwithaview.txt --new --epochs 5 --limit max --experimental
  ```

2. Du willst dieses Netzwerk nach Abschluss des Trainings in derselben Konfiguration zu einem beliebigen Zeitpunkt erneut trainieren:
  ```
  python lstm_char.py --input roomwithaview.txt --resume
  ```

3. Du willst dieses Netzwerk (durch die Zeichenbeschränkung identifiziert) nach Abschluss des Trainings mit einer anderen Anzahl an Epochen erneut trainieren:
  ```
  python lstm_char.py --input roomwithaview.txt --limit max --epochs 20 --experimental
  ```

4. Du willst ein Netzwerk möglichst schnell trainieren. Dafür könntest du ein niedriges Limit setzen und nur den ASCII-Zeichensatz verwenden:
  ```
  python lstm_char.py --input roomwithaview.txt --new --epochs 20 --limit 30000 --ascii
  ```

5. Du willst einen 2000 Zeichen langen Text ausgeben. Durch die Zeichenbeschränkung (und ggf. den Zeichensatz) identifizierst du das zu verwendende Netzwerk:
  ```
  python lstm_char.py --input roomwithaview.txt --limit 30000 --generate 2000
  ```

6. Du willst mit diesem Netzwerk, _ausgehend von einem anderen Text_, einen 5000 Zeichen langen Text ausgeben. Die Zeichenbeschränkung (und ggf. der Zeichensatz) muss angegeben werden, damit die richtige Matrix geladen wird:
  ```
  python lstm_char.py --input ulysses.txt --limit 30000 --generate 2000 --other > roomwithaview
  ```

7. Du willst mit einem trainierten Netzwerk einen Text ausgeben, der im Stil eines Texts vom gegebenen Seed weiterschreibt:
  ```
  python lstm_char.py --input ulysses.txt --limit 100000 --generate 5000 --seed > I don't think Bloom is Stephen's father, do you?
  ```

Netzwerke, die mit verschiedenen Limits trainiert wurden, sollten trotzdem untereinander kompatibel sein. Allerdings wäre es z. B. wenig sinnvoll, ein Netzwerk mit Limit 500000 zu trainieren, und das Training dann mit einem viel niedrigeren fortzusetzen.
Sollte dies trotzdem erwünscht sein, muss das trainierte Netzwerk zuerst von Hand in den Ordner des neuen Limits kopiert werden.

---

# LSTM_WORD

`--help | -h` zeigt die Gebrauchsanweisung an, also genau diese hier (docs/readme_word.txt)

`--verbose | -v` zeichnet den Programmverlauf ausführlicher nach

`--input <Textdatei> | -i <Textdatei>` der Ausgangstext, dieser sollte im Verzeichnis /input abgelegt sein; Beispiel: `-i mobydick.txt`

### TEXT GENERIEREN

`--generate <Ganzzahl> | -g <Ganzzahl>` es wird mithilfe der gespeicherten Gewichtung_nur ausgegeben_. Die geladene Matrix ist, wenn kein Limit festgelegt wurde, immer die zuletzt gespeicherte; die Zahl legt die Länge [in Wörtern] der Ausgabe fest; standardmäßig 50; Beispiel: `-g 400`

`--seed | -s` es wird von einem später anzugebenden Seed aus weitergeschrieben

`--diversity <Gleitkommazahl> | -d <Gleitkommazahl>` legt den Grad fest, zu dem abgewichen werden darf; standardmäßig 0.2; Höchstwert 2.0; Beispiel: `-d 1.2`

### NETZWERK TRAINIEREN

`--new | -n` das Netzwerk wird von Null an trainiert, die Gewichtung wird _für das festgelegte Zeichenbeschränkung_ gespeichert

`--resume | -r` das Training wird mithilfe der _zuletzt_ gespeicherten Konfiguration (Gewichtung, Wortbeschränkung, Anzahl der Epochen) wiederaufgenommen, die Gewichtung wird gespeichert

`--epochs <Ganzzahl> | -e <Ganzzahl>` legt die Anzahl der Durchläufe fest, bevor die Matrix gespeichert wird; standardmäßig 10; Beispiel: `-e 5`

`--limit <Ganzzahl> | -l <Ganzzahl> ODER -s max` legt die Wortbeschränkung fest; standardmäßig 1000; für jedes Limit wird eine eigene Gewichtung gespeichert

`--dynamic | -y` verschiebt den Anfang des Textausschnitts um einen zufälligen Betrag

## ABWEICHUNG VON LSTM_CHAR #

Bisher sind die trainierten Netzwerke untereinander _nicht kompatibel_. Das liegt daran, dass die Dimensionalität des Vektors von der Anzahl eindeutiger Wörter abhängig ist. Diese werden als nicht geordnetes Set verwendet. D. h. unter anderem auch, dass das gleiche Wort in verschiedenen Texten eine andere Stelle einnehmen kann. Also auch zwei Texte, die die gleiche Anzahl eindeutiger Wörter, sogar die genau gleichen Wörter in anderer Reihenfolge haben, produzieren inkompatible Netzwerke.
