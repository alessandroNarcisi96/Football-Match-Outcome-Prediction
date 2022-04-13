# Football-Match-Outcome-Prediction

## Milestone 0 Data cleaning and data preparation
The data are spread among several files so first of all we need to import them and create one dataset.
The first problem to address is to define the outcome of the match as we have only the result espressed in a string like that "score hometeam - score away team"
The function is parseResult on charge to parse the result and it returns 0 if the winner is the hometeam,1 if the winner is the awayteam and 2 in case of draw.
As a result there is a column more called OutCome.
If a row lacks of the result it cannot be used so it is simply dropped from the dataset.

### Overall Comments
One thing to note immediately is that the dataset in quite balanced.This a good news to classify the outcome properly.

### Capacity
The column Capacity contains dirty data like 32,500.In order to clean them lest's replace ',' with ''
Since there are a lot of outliers to fill the null values there will be used the median.

### Pitch
This column has many values to express the same one.In the notebook is chosen a value and the other ones are replaced
Since it is a categorical value and more than 95% of values are natural the null values are replaces by the mode

### Other null values
Elo: 8% of Elo values are null.Since it is quite important instead of infer a value from the other ones I decided to drop this rows directly

Cards: more than 70% are null in the dataset.As the plot drawn by missingno shows the missing values are randomly.
    Since the present values belong to different teams over the Seasons a could choice could replace the null values with a neutral one like zero


## Milestone 1 EDA
After Data Cleaning and the import of all the files the data set has the following 14 features:
 0   Home_Team     
 1   Away_Team    
 2   Result        
 3   Season        
 4   League       
 5   Pitch     
 6   ELO_Home     
 7   ELO_Away     
 8   Home_Yellow  
 9   Home_Red     
 10  Away_Yellow  
 11  Away_Red     
 12  Capacity    
 13  OutCome     

The first step involves getting a big picture about which feature seems to be relevant to predict the outcome of the matches.
How to do that?
Let's split the dataset by the sigle outcome(HomeWin,AwayWin and Draw) and see if there is any significant difference in the features.

### Capacity
As the boxplots show Capacity is not indipendent from the Outcome.Infact the mean is different in each group.
It looks that the more the stadium is bigger more likely the home will win.
Probably is due to the fact that the number of supporter is higher

### Elo
Needless to say if a team is stronger than other has a higher probability to win.But how to demostrate that?
I built a column called EloDiff which contains the difference between EloHome and EloAway.
It turns out that in HomeWin the EloDiff is skewed toward positive values(EloHome is higher than EloHome ) same for AwayWin but toward negative values
For drawa as we expected the values are around zero(Similar Elo lead to draw)

### Season
The plot in the notebook shows that the distribuition of wins change a little by year so it is a relevant factor in take in account

### League
The plot in the notebook shows that the distribuition of wins change a little by league so it is a relevant factor in take in account

### Pitch
More than 95% of matches belong to grass so it is not going to be a relevant feature

## Milestone 2 First baseline score
Let's train a simple model like Logistic Regression in order to see what is our initial score.
The accuracy is 0.48 and it is already a good result as a random classifier performs at 0.33 on 3 outcome to predict.

The confusion matrix is the following:
[[11094  1047   240]
 [ 5122  1699   141]
 [ 6352  1111   168]]

And the metrix these ones:
Accuracy: 0.48049974049084304
Precision: 0.48049974049084304
Recall: 0.48049974049084304

It suggests that the model is too sensitive to the homewin(first column) and as a result it misclassifies the other ones.

 By checking the confusion matrix on the training set and the metrix it turns out that there is not any overfitting

 44681  4200   952
 20500  6705   597
 25342  4245   673

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| 44681         | 4200          | 952   |
| 20500         | 6705          |   597 |
| 25342         | 4245          |   673 |

Accuracy: 0.4824968719588489
Precision: 0.4824968719588489
Recall: 0.4824968719588489

How can we improve the model performance?
A solution could be adding new features and this leads to the next Milestone

## Milestone 3 Feature Engineering
Let's try to figure out what could be a relevant feature to add by checkig the most importance features in the dataset.
By applying the random forest model we can draw easily a plot to see that.
I
It turns out that ELO is the most important features.
It suggests that the information about the team perfomance are more important than general information as Capacity o League

Let's try to take into account the amount of goal for each Season of each Team as well as red or yellow cards.

Now each match has 8 colums more:
Goal_Scored_HomeTeam,Goal_Collected_HomeTeam,Yellow_Collected_HomeTeam,Red_Collected_HomeTeam
Goal_Scored_AwayTeam,Goal_Collected_AwayTeam,Yellow_Collected_AwayTeam,Red_Collected_AwayTeam

It allows to the model to check if a team is more aggressive than another one and to get a more precise vision of the teams.

Ok now let's apply the random forest again.
As the plot show the new features have a high score.

## Milestone 4 Training the model

It seems that there is need a more complex model as well.
A good attempt could be the AdaBoost as it is shared in part the same logic of Random Forest:Information Gain.

The score are better now:
Accuracy: 0.5156446948913769
Precision: 0.5156446948913769
Recall: 0.5156446948913769

Can we do better?
Our goal is to predict the next matches.
The dataset contains even matches of 30 years ago.
As the plot for Season shows the result change a little and of course the style of game has changed a lot over the time
So maybe by taking into account only the most recent matches we could get a better score.

Since the dataset is smaller we'are going to apply KFold in order to get a more reliable score:
Accuracy: 0.530 (0.007)

## Milestone 5 Conclusion

The final result is quite better than a random classifier.
It turns out that the more we get data about the team along the season the more we will be able to predict its performance.
Another important factor to point out is the Capacity of the Statium.Probably the supporters can affect the result of a match more than we can imagine.
This could be a good point to take into account if a coach wants to improve the performance of his team.