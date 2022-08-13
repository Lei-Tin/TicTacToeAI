# TicTacToeAI - v1.1

My personal attempt of creating a TicTacToe game and a bot associated with it

A lightweight AI written in roughly 500 lines.

# Features

- Creates a CSV file that is used to store game trees, contains all the valid moves possible
- Has multiple bots that has a ultimate goal of making 'O' lose or tie

# File Directory Structure

- main.py, the main file that is responsible for the whole project
- README.md, the file you are reading at the moment
- data.csv, the file that stores all the possible moves

# Types of Bots included

- RandomBotPlayer
  - The bot that plays completely randomly, picks from all available moves possible currently on the board. 
- ExplorationBotPlayer
  - The bot with a probability of exploring new moves that hasn't been played before
  - If decides to not explore new moves, will play the option with the highest winning probability at the moment
- BotPlayer
  - The bot that plays with the highest probability of winning

# Libraries Used

- typing
- random
- csv
- \_\_future__
- plotly
