
## Brent Burns is having a historic year
### Which players are most similar to him this year?
#### Hint: It's a couple of Russian Snipers

Brent Burns, as we all know, is having a historic year for a defensemen with 27 goals and 36 assists for 63 points in 59 games (currently on pace for 38 goals and 50 assists for 88 points).  I am curious if we can use machine learning to see which players are most similar to Brent Burns this year.  

### The Data

I got my data from [stats.hockeyanalysis.com](http://stats.hockeyanalysis.com/ratings.php?db=201617&sit=5v5&type=individual&teamid=0&pos=skaters&minutes=400&disp=1&sort=PCT&sortdir=DESC) and put the following filters on:

- Season: 2016-17 - I only wanted to look at data for this year
- Situation: 5 on 5 - I only wanted to look at even strength numbers
- Select Position: Defensemen - Since Brent is a defensemen, I want to see which defensemen is most similar to him
- Minutes Played: 400 - I didn't want to look at players that had only played a handful of games 

Let's get started!

### The Process

Let's load the package that I will be using right off the bat, pandas, and load my dataset in.


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

defense = pd.read_csv('defensemen.csv')
defense.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Team</th>
      <th>GP</th>
      <th>TOI</th>
      <th>G</th>
      <th>A</th>
      <th>FirstA</th>
      <th>Points</th>
      <th>Shots</th>
      <th>iFenwick</th>
      <th>...</th>
      <th>G/60</th>
      <th>A/60</th>
      <th>FirstA/60</th>
      <th>Points/60</th>
      <th>Shots/60</th>
      <th>iFenwick/60</th>
      <th>iCorsi/60</th>
      <th>IGP</th>
      <th>IAP</th>
      <th>IPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BURNS, BRENT</td>
      <td>San Jose</td>
      <td>58</td>
      <td>1062:04:00</td>
      <td>17</td>
      <td>23</td>
      <td>12</td>
      <td>40</td>
      <td>156</td>
      <td>234</td>
      <td>...</td>
      <td>0.96</td>
      <td>1.30</td>
      <td>0.68</td>
      <td>2.26</td>
      <td>8.81</td>
      <td>13.22</td>
      <td>20.11</td>
      <td>29.3</td>
      <td>39.7</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SKJEI, BRADY</td>
      <td>NY Rangers</td>
      <td>56</td>
      <td>827:05:00</td>
      <td>2</td>
      <td>18</td>
      <td>9</td>
      <td>20</td>
      <td>66</td>
      <td>91</td>
      <td>...</td>
      <td>0.15</td>
      <td>1.31</td>
      <td>0.65</td>
      <td>1.45</td>
      <td>4.79</td>
      <td>6.60</td>
      <td>8.92</td>
      <td>4.4</td>
      <td>40.0</td>
      <td>44.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAMILTON, DOUGIE</td>
      <td>Calgary</td>
      <td>57</td>
      <td>876:31:00</td>
      <td>6</td>
      <td>15</td>
      <td>9</td>
      <td>21</td>
      <td>122</td>
      <td>162</td>
      <td>...</td>
      <td>0.41</td>
      <td>1.03</td>
      <td>0.62</td>
      <td>1.44</td>
      <td>8.35</td>
      <td>11.09</td>
      <td>15.61</td>
      <td>18.2</td>
      <td>45.5</td>
      <td>63.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KARLSSON, ERIK</td>
      <td>Ottawa</td>
      <td>55</td>
      <td>1058:05:00</td>
      <td>7</td>
      <td>18</td>
      <td>10</td>
      <td>25</td>
      <td>85</td>
      <td>146</td>
      <td>...</td>
      <td>0.40</td>
      <td>1.02</td>
      <td>0.57</td>
      <td>1.42</td>
      <td>4.82</td>
      <td>8.28</td>
      <td>13.04</td>
      <td>17.5</td>
      <td>45.0</td>
      <td>62.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MARKOV, ANDREI</td>
      <td>Montreal</td>
      <td>39</td>
      <td>615:33:00</td>
      <td>3</td>
      <td>11</td>
      <td>7</td>
      <td>14</td>
      <td>44</td>
      <td>61</td>
      <td>...</td>
      <td>0.29</td>
      <td>1.07</td>
      <td>0.68</td>
      <td>1.36</td>
      <td>4.29</td>
      <td>5.95</td>
      <td>10.04</td>
      <td>10.3</td>
      <td>37.9</td>
      <td>48.3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



We're looking good so far - first I want to set the players name as the index, as that will allow for easier searching my players name later.


```python
df = defense.set_index('Player Name')
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>GP</th>
      <th>TOI</th>
      <th>G</th>
      <th>A</th>
      <th>FirstA</th>
      <th>Points</th>
      <th>Shots</th>
      <th>iFenwick</th>
      <th>iCorsi</th>
      <th>...</th>
      <th>G/60</th>
      <th>A/60</th>
      <th>FirstA/60</th>
      <th>Points/60</th>
      <th>Shots/60</th>
      <th>iFenwick/60</th>
      <th>iCorsi/60</th>
      <th>IGP</th>
      <th>IAP</th>
      <th>IPP</th>
    </tr>
    <tr>
      <th>Player Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BURNS, BRENT</th>
      <td>San Jose</td>
      <td>58</td>
      <td>1062:04:00</td>
      <td>17</td>
      <td>23</td>
      <td>12</td>
      <td>40</td>
      <td>156</td>
      <td>234</td>
      <td>356</td>
      <td>...</td>
      <td>0.96</td>
      <td>1.30</td>
      <td>0.68</td>
      <td>2.26</td>
      <td>8.81</td>
      <td>13.22</td>
      <td>20.11</td>
      <td>29.3</td>
      <td>39.7</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>SKJEI, BRADY</th>
      <td>NY Rangers</td>
      <td>56</td>
      <td>827:05:00</td>
      <td>2</td>
      <td>18</td>
      <td>9</td>
      <td>20</td>
      <td>66</td>
      <td>91</td>
      <td>123</td>
      <td>...</td>
      <td>0.15</td>
      <td>1.31</td>
      <td>0.65</td>
      <td>1.45</td>
      <td>4.79</td>
      <td>6.60</td>
      <td>8.92</td>
      <td>4.4</td>
      <td>40.0</td>
      <td>44.4</td>
    </tr>
    <tr>
      <th>HAMILTON, DOUGIE</th>
      <td>Calgary</td>
      <td>57</td>
      <td>876:31:00</td>
      <td>6</td>
      <td>15</td>
      <td>9</td>
      <td>21</td>
      <td>122</td>
      <td>162</td>
      <td>228</td>
      <td>...</td>
      <td>0.41</td>
      <td>1.03</td>
      <td>0.62</td>
      <td>1.44</td>
      <td>8.35</td>
      <td>11.09</td>
      <td>15.61</td>
      <td>18.2</td>
      <td>45.5</td>
      <td>63.6</td>
    </tr>
    <tr>
      <th>KARLSSON, ERIK</th>
      <td>Ottawa</td>
      <td>55</td>
      <td>1058:05:00</td>
      <td>7</td>
      <td>18</td>
      <td>10</td>
      <td>25</td>
      <td>85</td>
      <td>146</td>
      <td>230</td>
      <td>...</td>
      <td>0.40</td>
      <td>1.02</td>
      <td>0.57</td>
      <td>1.42</td>
      <td>4.82</td>
      <td>8.28</td>
      <td>13.04</td>
      <td>17.5</td>
      <td>45.0</td>
      <td>62.5</td>
    </tr>
    <tr>
      <th>MARKOV, ANDREI</th>
      <td>Montreal</td>
      <td>39</td>
      <td>615:33:00</td>
      <td>3</td>
      <td>11</td>
      <td>7</td>
      <td>14</td>
      <td>44</td>
      <td>61</td>
      <td>103</td>
      <td>...</td>
      <td>0.29</td>
      <td>1.07</td>
      <td>0.68</td>
      <td>1.36</td>
      <td>4.29</td>
      <td>5.95</td>
      <td>10.04</td>
      <td>10.3</td>
      <td>37.9</td>
      <td>48.3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Next, I am going to drop 'Team', 'GP', and 'TOI' as I do not want to find players that have played similar games or total time on ice, but similar playing numbers. 


```python
df.drop(['Team', 'GP', 'TOI'], axis = 1, inplace = True)
```

I am going to be using the Nearest Neighbors algorithm and this algorithm likes all the features to be normalized, so now I will normalize all the features.


```python
df_norm = (df - df.mean()) / df.std()
```


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>G</th>
      <th>A</th>
      <th>FirstA</th>
      <th>Points</th>
      <th>Shots</th>
      <th>iFenwick</th>
      <th>iCorsi</th>
      <th>ShPct</th>
      <th>G/60</th>
      <th>A/60</th>
      <th>FirstA/60</th>
      <th>Points/60</th>
      <th>Shots/60</th>
      <th>iFenwick/60</th>
      <th>iCorsi/60</th>
      <th>IGP</th>
      <th>IAP</th>
      <th>IPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
      <td>183.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.256831</td>
      <td>7.983607</td>
      <td>3.907104</td>
      <td>10.240437</td>
      <td>57.546448</td>
      <td>83.732240</td>
      <td>126.617486</td>
      <td>3.739454</td>
      <td>0.162404</td>
      <td>0.582295</td>
      <td>0.286011</td>
      <td>0.744699</td>
      <td>4.234590</td>
      <td>6.167432</td>
      <td>9.340820</td>
      <td>7.178142</td>
      <td>26.086339</td>
      <td>33.267213</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.074080</td>
      <td>3.892054</td>
      <td>2.442100</td>
      <td>5.123433</td>
      <td>22.194861</td>
      <td>30.967182</td>
      <td>44.312156</td>
      <td>2.745815</td>
      <td>0.134290</td>
      <td>0.237446</td>
      <td>0.165303</td>
      <td>0.298264</td>
      <td>1.220487</td>
      <td>1.663611</td>
      <td>2.367674</td>
      <td>5.713235</td>
      <td>9.960018</td>
      <td>11.927907</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>19.000000</td>
      <td>28.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.190000</td>
      <td>1.780000</td>
      <td>2.860000</td>
      <td>4.400000</td>
      <td>0.000000</td>
      <td>5.900000</td>
      <td>11.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>44.000000</td>
      <td>63.500000</td>
      <td>99.000000</td>
      <td>1.755000</td>
      <td>0.070000</td>
      <td>0.395000</td>
      <td>0.160000</td>
      <td>0.530000</td>
      <td>3.445000</td>
      <td>5.060000</td>
      <td>7.760000</td>
      <td>3.150000</td>
      <td>18.800000</td>
      <td>24.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>55.000000</td>
      <td>80.000000</td>
      <td>124.000000</td>
      <td>3.570000</td>
      <td>0.140000</td>
      <td>0.540000</td>
      <td>0.280000</td>
      <td>0.710000</td>
      <td>4.030000</td>
      <td>5.840000</td>
      <td>9.150000</td>
      <td>6.200000</td>
      <td>25.000000</td>
      <td>32.400000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>66.500000</td>
      <td>101.500000</td>
      <td>154.000000</td>
      <td>5.335000</td>
      <td>0.235000</td>
      <td>0.740000</td>
      <td>0.385000</td>
      <td>0.910000</td>
      <td>4.815000</td>
      <td>7.055000</td>
      <td>10.605000</td>
      <td>10.300000</td>
      <td>32.100000</td>
      <td>40.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>23.000000</td>
      <td>12.000000</td>
      <td>40.000000</td>
      <td>156.000000</td>
      <td>234.000000</td>
      <td>356.000000</td>
      <td>13.330000</td>
      <td>0.960000</td>
      <td>1.310000</td>
      <td>0.740000</td>
      <td>2.260000</td>
      <td>8.810000</td>
      <td>13.220000</td>
      <td>20.110000</td>
      <td>29.300000</td>
      <td>57.100000</td>
      <td>69.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_norm.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>G</th>
      <th>A</th>
      <th>FirstA</th>
      <th>Points</th>
      <th>Shots</th>
      <th>iFenwick</th>
      <th>iCorsi</th>
      <th>ShPct</th>
      <th>G/60</th>
      <th>A/60</th>
      <th>FirstA/60</th>
      <th>Points/60</th>
      <th>Shots/60</th>
      <th>iFenwick/60</th>
      <th>iCorsi/60</th>
      <th>IGP</th>
      <th>IAP</th>
      <th>IPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
      <td>1.830000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-2.450984e-16</td>
      <td>8.493509e-18</td>
      <td>1.007088e-16</td>
      <td>1.092023e-16</td>
      <td>3.518740e-17</td>
      <td>1.152691e-16</td>
      <td>-7.765494e-17</td>
      <td>2.196179e-16</td>
      <td>1.100516e-15</td>
      <td>1.369882e-15</td>
      <td>-9.003120e-16</td>
      <td>2.759177e-15</td>
      <td>-1.067755e-15</td>
      <td>4.440892e-16</td>
      <td>3.300335e-16</td>
      <td>1.213358e-17</td>
      <td>-1.324987e-15</td>
      <td>-1.367455e-15</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.088112e+00</td>
      <td>-1.794324e+00</td>
      <td>-1.599895e+00</td>
      <td>-1.608382e+00</td>
      <td>-1.736729e+00</td>
      <td>-1.799719e+00</td>
      <td>-1.977279e+00</td>
      <td>-1.361874e+00</td>
      <td>-1.209358e+00</td>
      <td>-2.031181e+00</td>
      <td>-1.730218e+00</td>
      <td>-1.859761e+00</td>
      <td>-2.011157e+00</td>
      <td>-1.988104e+00</td>
      <td>-2.086782e+00</td>
      <td>-1.256406e+00</td>
      <td>-2.026737e+00</td>
      <td>-1.858433e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.059702e-01</td>
      <td>-7.665893e-01</td>
      <td>-7.809280e-01</td>
      <td>-6.324738e-01</td>
      <td>-6.103417e-01</td>
      <td>-6.533446e-01</td>
      <td>-6.232485e-01</td>
      <td>-7.227194e-01</td>
      <td>-6.880968e-01</td>
      <td>-7.887915e-01</td>
      <td>-7.623010e-01</td>
      <td>-7.198307e-01</td>
      <td>-6.469470e-01</td>
      <td>-6.656795e-01</td>
      <td>-6.676678e-01</td>
      <td>-7.050545e-01</td>
      <td>-7.315588e-01</td>
      <td>-7.434006e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.238287e-01</td>
      <td>-2.527217e-01</td>
      <td>3.803947e-02</td>
      <td>-4.692892e-02</td>
      <td>-1.147314e-01</td>
      <td>-1.205224e-01</td>
      <td>-5.906926e-02</td>
      <td>-6.171340e-02</td>
      <td>-1.668360e-01</td>
      <td>-1.781253e-01</td>
      <td>-3.636301e-02</td>
      <td>-1.163381e-01</td>
      <td>-1.676300e-01</td>
      <td>-1.968199e-01</td>
      <td>-8.059373e-02</td>
      <td>-1.712063e-01</td>
      <td>-1.090700e-01</td>
      <td>-7.270455e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.583128e-01</td>
      <td>7.750133e-01</td>
      <td>4.475232e-01</td>
      <td>5.386160e-01</td>
      <td>4.034065e-01</td>
      <td>5.737609e-01</td>
      <td>6.179459e-01</td>
      <td>5.810831e-01</td>
      <td>5.405893e-01</td>
      <td>6.641728e-01</td>
      <td>5.988327e-01</td>
      <td>5.542092e-01</td>
      <td>4.755561e-01</td>
      <td>5.335191e-01</td>
      <td>5.339334e-01</td>
      <td>5.464256e-01</td>
      <td>6.037802e-01</td>
      <td>5.896078e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.108294e+00</td>
      <td>3.858219e+00</td>
      <td>3.313909e+00</td>
      <td>5.808520e+00</td>
      <td>4.435872e+00</td>
      <td>4.852484e+00</td>
      <td>5.176514e+00</td>
      <td>3.492787e+00</td>
      <td>5.939362e+00</td>
      <td>3.064723e+00</td>
      <td>2.746399e+00</td>
      <td>5.080403e+00</td>
      <td>3.748841e+00</td>
      <td>4.239313e+00</td>
      <td>4.548422e+00</td>
      <td>3.872037e+00</td>
      <td>3.113816e+00</td>
      <td>2.995730e+00</td>
    </tr>
  </tbody>
</table>
</div>



Look at the means and the standard deviations now - means are approximately 0 and standard deviations are approximately 1.  If we did not do this, the algorithm would weigh some features more than the others.  Let's import Nearest Neighbors from sklearn and get into the fun stuff! 


```python
from sklearn.neighbors import NearestNeighbors
```


```python
neighbors = NearestNeighbors(n_neighbors = 11)
```

I am going to look at the 10 closest neighbors to Brent Burns.  I selected 11 because in theory Brent Burns' closest neighbor will be himself.  I like to leave that in the dataset to use as a sanity check, his closest neighbor should be himself with a score of 0.  Let's see if we're right.


```python
neighbors.fit(df_norm)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
             metric_params=None, n_jobs=1, n_neighbors=11, p=2, radius=1.0)




```python
bb = df_norm.ix['BURNS, BRENT']
score, name = neighbors.kneighbors(bb)
```

All right, I've tested my algorithm with Brent Burns' data and now I want to view the results.  The results are in numpy arrays, so I'm going to convert it to Pandas DataFrames for easier viewing. 


```python
import numpy as np
score1 = score.tolist()
score2 = score1[0]

name1 = name.tolist()
name2 = name1[0]

player_names = []
for x in name2:
    names = defense.iloc[x]['Player Name']
    player_names.append(names)
    
nearest_neighbors = pd.DataFrame(
                        {'Player Name': player_names,
                         'Similarity Score': score2})
nearest_neighbors
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Similarity Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BURNS, BRENT</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAMILTON, DOUGIE</td>
      <td>10.194288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KARLSSON, ERIK</td>
      <td>11.040524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BYFUGLIEN, DUSTIN</td>
      <td>11.795462</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KLEFBOM, OSCAR</td>
      <td>12.589581</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TROUBA, JACOB</td>
      <td>12.982133</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PETRY, JEFF</td>
      <td>13.032122</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HEDMAN, VICTOR</td>
      <td>13.874861</td>
    </tr>
    <tr>
      <th>8</th>
      <td>JONES, SETH</td>
      <td>13.899590</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CARLSON, JOHN</td>
      <td>14.100756</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NISKANEN, MATT</td>
      <td>14.125928</td>
    </tr>
  </tbody>
</table>
</div>



Number 1, we passed our sanity check the most similar player to Brent Burns is Brent Burns.  Number 2, Dougie Hamilton is the most similar player to Brent Burns; however his similarity score 10.19 which is pretty high (read: not highly similar).  To compare I'm going to look at the similarity scores of the 10 most similar players to Erik Karlsson.


```python
ek = df_norm.ix['KARLSSON, ERIK']
score, name = neighbors.kneighbors(ek)

score1 = score.tolist()
score2 = score1[0]

name1 = name.tolist()
name2 = name1[0]

player_names = []
for x in name2:
    names = defense.iloc[x]['Player Name']
    player_names.append(names)
    
nearest_neighbors = pd.DataFrame(
                        {'Player Name': player_names,
                         'Similarity Score': score2})
nearest_neighbors
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Similarity Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KARLSSON, ERIK</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HEDMAN, VICTOR</td>
      <td>3.183491</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BARRIE, TYSON</td>
      <td>4.036872</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TROUBA, JACOB</td>
      <td>4.153143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BYFUGLIEN, DUSTIN</td>
      <td>4.191766</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PROVOROV, IVAN</td>
      <td>4.306723</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HAMILTON, DOUGIE</td>
      <td>4.317378</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NISKANEN, MATT</td>
      <td>4.401412</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SHATTENKIRK, KEVIN</td>
      <td>4.620814</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CARLSON, JOHN</td>
      <td>4.767609</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PETRY, JEFF</td>
      <td>4.863925</td>
    </tr>
  </tbody>
</table>
</div>



The most similar player to Erik Karlsson is Victor Kedman with a similarity score 3.18.  The 10th most similar player is Jeff Petry with a similarity score of 4.86.  As you can see the players that are "similar", really are not all that similar.  This is mainly due to Brent Burns having a truly historic year for a defensemen, so lets use a dataset off all skaters and see which players he is most similar to.


```python
all_skaters = pd.read_csv('all_players.csv')
all_skaters.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Team</th>
      <th>GP</th>
      <th>TOI</th>
      <th>G</th>
      <th>A</th>
      <th>FirstA</th>
      <th>Points</th>
      <th>Shots</th>
      <th>iFenwick</th>
      <th>...</th>
      <th>G/60</th>
      <th>A/60</th>
      <th>FirstA/60</th>
      <th>Points/60</th>
      <th>Shots/60</th>
      <th>iFenwick/60</th>
      <th>iCorsi/60</th>
      <th>IGP</th>
      <th>IAP</th>
      <th>IPP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SHEARY, CONOR</td>
      <td>Pittsburgh</td>
      <td>42</td>
      <td>564:06:00</td>
      <td>12</td>
      <td>17</td>
      <td>6</td>
      <td>29</td>
      <td>84</td>
      <td>106</td>
      <td>...</td>
      <td>1.28</td>
      <td>1.81</td>
      <td>0.64</td>
      <td>3.08</td>
      <td>8.93</td>
      <td>11.27</td>
      <td>13.93</td>
      <td>32.4</td>
      <td>45.9</td>
      <td>78.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VANEK, THOMAS</td>
      <td>Detroit</td>
      <td>44</td>
      <td>494:14:00</td>
      <td>9</td>
      <td>15</td>
      <td>8</td>
      <td>24</td>
      <td>65</td>
      <td>82</td>
      <td>...</td>
      <td>1.09</td>
      <td>1.82</td>
      <td>0.97</td>
      <td>2.91</td>
      <td>7.89</td>
      <td>9.95</td>
      <td>12.63</td>
      <td>31.0</td>
      <td>51.7</td>
      <td>82.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CROSBY, SIDNEY</td>
      <td>Pittsburgh</td>
      <td>51</td>
      <td>748:23:00</td>
      <td>18</td>
      <td>18</td>
      <td>15</td>
      <td>36</td>
      <td>114</td>
      <td>139</td>
      <td>...</td>
      <td>1.44</td>
      <td>1.44</td>
      <td>1.20</td>
      <td>2.89</td>
      <td>9.14</td>
      <td>11.14</td>
      <td>14.11</td>
      <td>40.0</td>
      <td>40.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ZUCKER, JASON</td>
      <td>Minnesota</td>
      <td>57</td>
      <td>776:36:00</td>
      <td>14</td>
      <td>23</td>
      <td>15</td>
      <td>37</td>
      <td>109</td>
      <td>163</td>
      <td>...</td>
      <td>1.08</td>
      <td>1.78</td>
      <td>1.16</td>
      <td>2.86</td>
      <td>8.42</td>
      <td>12.59</td>
      <td>14.83</td>
      <td>28.0</td>
      <td>46.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MALKIN, EVGENI</td>
      <td>Pittsburgh</td>
      <td>50</td>
      <td>700:29:00</td>
      <td>14</td>
      <td>19</td>
      <td>16</td>
      <td>33</td>
      <td>90</td>
      <td>115</td>
      <td>...</td>
      <td>1.20</td>
      <td>1.63</td>
      <td>1.37</td>
      <td>2.83</td>
      <td>7.71</td>
      <td>9.85</td>
      <td>12.33</td>
      <td>32.6</td>
      <td>44.2</td>
      <td>76.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
all_df = all_skaters.set_index('Player Name')
all_df.drop(['Team', 'GP', 'TOI'], axis = 1, inplace = True)
all_norm = (all_df - all_df.mean()) / all_df.std()

neighbors = NearestNeighbors(n_neighbors = 11)
neighbors.fit(all_norm)
bb = all_norm.ix['BURNS, BRENT']
score, name = neighbors.kneighbors(bb)

score1 = score.tolist()
score2 = score1[0]

name1 = name.tolist()
name2 = name1[0]

player_names = []
for x in name2:
    names = all_skaters.iloc[x]['Player Name']
    player_names.append(names)
    
nearest_neighbors = pd.DataFrame(
                        {'Player Name': player_names,
                         'Similarity Score': score2})
nearest_neighbors
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Similarity Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BURNS, BRENT</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OVECHKIN, ALEX</td>
      <td>3.857538</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TARASENKO, VLADIMIR</td>
      <td>4.284213</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KANE, PATRICK</td>
      <td>4.353555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PACIORETTY, MAX</td>
      <td>4.659067</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KADRI, NAZEM</td>
      <td>4.735868</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SEGUIN, TYLER</td>
      <td>4.758538</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SAAD, BRANDON</td>
      <td>4.809207</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ARVIDSSON, VIKTOR</td>
      <td>4.866129</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MARCHAND, BRAD</td>
      <td>5.039093</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PASTRNAK, DAVID</td>
      <td>5.146107</td>
    </tr>
  </tbody>
</table>
</div>



The most smilar player to Brent Burns this year is Alex Ovechkin.  You know you are having a great year when the 3 most similar players to you are Alex Ovechkin, Vladimir Tarasenko, and Patrick Kane.  Remember, Brent Burns is currently (as of 2/19) 3rd in the league in points and tied for 6th in goals.
