# Data Science in Socio-Techno Economic Society
## Investigating the Link Between Ozempic Online Discussion and Mental Health in Young People

Data Science project investigating the ties between Ozempic social media discussion and mental health in young people. Our initial hypothesis is that the Ozempic "trend" has likely led to increased concern over personal aesthetic and presentation and even possibly worsened body dysmorphia amongst certain people. In addition, we are interested in trying to develop some model that can forecast mental health trends from 

### Data

Our study largely relies on independently source Reddit data to perform our analysis and correlation. The data scraped from this and other similar websites will be made available on this repository in the `/data` folder in due time. However, other data sources might also be useful in this study. In particular...

1. **Google Trends Data**: trends on "Ozempic", "Mental Health", "Body Dysmorphia" and related could be helpful as a preliminary test to determine if there is some kind of correlation between these variables
2. **NHS Mental Health Service Response Data**: NHS England provides a decently extensive repository of data on Mental Health Service Response from 2016 to 2025 which could be useful for our correlation study [see here](https://digital.nhs.uk/data-and-information/publications/statistical/mental-health-services-monthly-statistics)

### Methods

As part of our analysis, we are interested in establishing some kind of statistical significance (or lack thereof) between the Ozempic trend and mental health in social media or among young people. If possible, we would like to use models that can determine if these two variables track each other well over time (time-series modelling). Additionally, we are also interested in performing some topic modelling and other NLP analysis on the Ozempic discussion data collected to try to determine its nature. Some of the models we could consider using are

1. **BERTopic**: pre-trained LM with topic-modelling fine-tuning that can be used to extract sentiments/discussion topics from Reddit comments on Ozempic or Ozempic threads
2. **RoBERTa**: pre-trained LM that can be used to embed comments in latent-space and possibly do some further analysis (such as clustering) to determine major groups of discussion topics (consider using 1 first)
3. **DiD**: given enough data points through time, using a difference-in-difference model would help us establish whether the Ozempic trend and mental health discussion tracks well or not
4. **LSTM/GRU**: should we want to pursue the predictive modelling objective, we could train a small recurrent neural network to learn to predict mental health trends from Ozempic discussion 

### Miscellaneous

Note that application of time-series models is subject to being able to obtain sufficient data points through time. Should we be unable to do this, we should instead pivot to trying to determine a statistical significance (e.g. t-test) between the two different periods (before and after Ozempic was introduced).
