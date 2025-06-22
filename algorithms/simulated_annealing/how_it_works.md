### Before you start

Adapt cooling_kwargs for selected cooling function.

for example: 
```python
# For adaptive cooling
cooling_kwargs = {
    "learning_rate": 0.5,
    "cooling_rate": 0.95,
}

best_sol, best_val, history, acceptance_rate, acceptance_rate_worse = simulated_annealing(
    # main parameters
    cooling_schedule="adaptive",    
    cooling_kwargs=cooling_kwargs,
    # rest of the parameters
    )
```

### How Simulated Annealing Algorithm Works?

#### Hr

- Inicijaliziramo početnu temperaturu, broj iteracija, step size te zeljenu funkciju hlađenja i iterirammo,
u našem slučaju sve dok ne dođemo do n iteracija.

- U svakoj iteraciji kreiramo novog kandidata koji ovisi o trenutnom rješenju i step size-u. Koji
je uvijek u obliku `[num] * dimensions` - to su koordiante novog kandidata u prostoru.

- Nakon toga pozovemo zeljenu objective funckiju s novim kandidatom (koordiante), te dobijemo vrijednost te točke.

- Nakon toga algoritam odlučuje želi li prihvatiti kandidata tako što će napraviti usporedbu između trenutne najbolje
vrijednosti i trenutne vrijednosti koju smo dobili od objective funckije - singularne vrijednosti.

- Ukoliko je, u našem slučaju, delta manja od 0, automatksi znamo da smo našli bolju točku tj. vrijednost i 
automatski ćemo ju prihvatiti.

- Ukoliko je delta veća od 0. Tada nam simulated annealing dolzai sa svojom logikom te radimo usporedbu
nekog random broja `[0, 1)` i `exp(-delta / T)` - gdje je `delta` razlika izmedu vrijednosti kandidata i trenutne najbolje
vrijednosti, a `T` je trenutna temperatura.

- Što je delta manja i temperatura veća, to je veća vjerojatnost da ćemo prihvatiti lošijeg kandidata. 

- Ukoliko je temperatura jako visoka doći ćemo do faze da svaki puta prihvaćamo kandidate bez obzira na njegovu vrijednost. 

- -Opet ako je temperatura premala nećemo više vjerjatno prihvaćati niti jednog kandidata. Kod ekstrema u temperaturi delta nam
više ne igra veliku ulogu.

#### En

TBD