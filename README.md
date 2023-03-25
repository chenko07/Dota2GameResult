# Dota2GameResult
Multi-Layer Perceptron Model for Dota 2 Game Result Dataset from UCI Machine Learning Repository

#Model
There is 2 types of MLP models using Sequential and MLPClassifier. All predictions are using label that you have to edit the raw csv files. 

MLPClassifier Model get :
- Highest Score Train: Accuracy : 84.10%
- Highest Score Test : Accuracy : 78.56%

Sequential Model get :
- Highest Score of test : Accuracy : 100% (already identify anomali using different metrcis)
- Another Score of test : Accuracy : 54.35%

#Dataset Information 
Dota 2 is a popular computer game with two teams of 5 players. At the start of the game each player chooses a unique hero with different strengths and weaknesses. The dataset is reasonably sparse as only 10 of 113 possible heroes are chosen in a given game. All games were played in a space of 2 hours on the 13th of August, 2016

Each row of the dataset is a single game with the following features (in the order in the vector):
1. Team won the game (1 or -1)
2. Cluster ID (related to location)
3. Game mode (eg All Pick)
4. Game type (eg. Ranked)
5 - end: Each element is an indicator for a hero. Value of 1 indicates that a player from team '1' played as that hero and '-1' for the other team. Hero can be selected by only one player each game. This means that each row has five '1' and five '-1' values.

RESOURCE LINK AND REFERENCES :
- https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
- https://github.com/kronusme/dota2-api/blob/master/data/heroes.json
- https://gist.github.com/da-steve101/1a7ae319448db431715bd75391a66e1b
- Akhmedov, K., & Phan, A. H. (2021). Machine learning models for DOTA 2 outcomes prediction. arXiv preprint arXiv:2106.017821.
